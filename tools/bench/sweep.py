#!/usr/bin/env python3
"""A long prefill/decode sweep of qwasar on this Mac, run unattended.

For each point: wait for the SoC to cool, start a 1 Hz temperature sampler,
run qwbench (fresh process: load, prefill exactly N tokens, decode 128 at that
depth), stop the sampler, and append one JSON line to results.jsonl with the
max temperatures seen during prefill and during decode.  summary.md is
rewritten after every point, so the results so far are always readable, and a
rerun resumes, skipping finished points.

Order is interleaved -- a whole pass over the sizes, then the next -- so a
sweep stopped early still covers every size.

  make bench
  caffeinate -is python3 tools/bench/sweep.py --out sweep-out [--model DIR]
      [--sizes 512,1024,...] [--reps 3] [--decode 128]

The corpus is this repository's own sources, so the prompt is real text.
"""
import argparse, glob, json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BENCH = os.path.join(HERE, "qwbench")
TEMPMON = os.path.join(HERE, "tempmon")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="directory for results, summary, log, samples")
ap.add_argument("--model", default=os.environ.get("QWASAR_MODEL") or os.path.join(ROOT, "qwasar-model"))
ap.add_argument("--sizes", default="512,1024,2048,4096,8192,16384,32768,65536")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--decode", type=int, default=128)
ap.add_argument("--cool-to", type=float, default=42.0, help="die max, C, before a run starts")
ARGS = ap.parse_args()

OUT = os.path.abspath(ARGS.out)
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.realpath(ARGS.model)
CORPUS = os.path.join(OUT, "corpus.txt")
RESULTS = os.path.join(OUT, "results.jsonl")
SUMMARY = os.path.join(OUT, "summary.md")
LOG = os.path.join(OUT, "log.txt")
DECODE = ARGS.decode

SIZES = [int(x) for x in ARGS.sizes.split(",")]
PLAN = [(n, rep) for rep in range(1, ARGS.reps + 1) for n in SIZES]

COOL_TARGET = ARGS.cool_to
COOL_MIN, COOL_MAX = 60, 900   # seconds

if not os.path.exists(CORPUS):
    with open(CORPUS, "w") as f:
        for pat in ("*.c", "*.h", "*.m", "*.md", "metal/*.metal", "metal4/*.metal"):
            for p in sorted(glob.glob(os.path.join(ROOT, pat))):
                f.write(open(p, errors="replace").read())


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def temps():
    t, die, dev, bat, state = subprocess.check_output([TEMPMON], text=True).split()
    return float(die), float(dev), float(bat), int(state)


def cool_down():
    t0 = time.time()
    while True:
        die, dev, bat, state = temps()
        waited = time.time() - t0
        if waited >= COOL_MIN and die <= COOL_TARGET:
            return die, dev, bat, waited
        if waited >= COOL_MAX:
            log(f"  cooldown capped at {COOL_MAX}s with die at {die:.1f}C")
            return die, dev, bat, waited
        time.sleep(15)


def window_max(samples, a, b):
    w = [s for s in samples if a <= s[0] <= b]
    if not w:
        return None
    return {"die": max(s[1] for s in w), "dev": max(s[2] for s in w),
            "battery": max(s[3] for s in w), "thermal_state": max(s[4] for s in w)}


def swap_used_mb():
    out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    return float(out.split("used = ")[1].split("M")[0])


def run_point(n, rep):
    die0, dev0, bat0, waited = cool_down()
    swap0 = swap_used_mb()
    log(f"n={n} rep={rep}: start, die {die0:.1f}C after {waited:.0f}s cooling, swap {swap0:.0f} MB")
    samp_path = os.path.join(OUT, "samples", f"n{n}_r{rep}.txt")
    os.makedirs(os.path.dirname(samp_path), exist_ok=True)
    with open(samp_path, "w") as sf:
        mon = subprocess.Popen([TEMPMON, "1"], stdout=sf)
        t_start = time.time()
        p = subprocess.run([BENCH, MODEL, CORPUS, str(n), str(DECODE)],
                           capture_output=True, text=True)
        mon.terminate(); mon.wait()
    wall = time.time() - t_start
    samples = []
    for line in open(samp_path):
        parts = line.split()
        if len(parts) == 5:
            samples.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])))
    rec = {"n": n, "rep": rep, "start": t_start, "wall_s": round(wall, 1), "cooled_s": round(waited),
           "start_temp": {"die": die0, "dev": dev0, "battery": bat0}, "swap_mb_start": swap0,
           "swap_mb_end": swap_used_mb(), "exit": p.returncode}
    try:
        r = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        r = {"error": (p.stderr or p.stdout)[-500:]}
    rec.update(r)
    if "t_prefill" in r:
        rec["temp_prefill"] = window_max(samples, *r["t_prefill"])
        rec["temp_decode"] = window_max(samples, *r["t_decode"])
    rec["temp_run"] = window_max(samples, t_start, t_start + wall)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if "error" in rec:
        log(f"n={n} rep={rep}: FAILED: {rec['error']}")
    else:
        log(f"n={n} rep={rep}: prefill {rec['prefill_tps']:.1f} t/s, decode {rec['decode_tps']:.2f} t/s, "
            f"die max {rec['temp_run']['die']:.1f}C, wall {wall/60:.1f} min")
    return rec


def summarize():
    rows = [json.loads(l) for l in open(RESULTS)] if os.path.exists(RESULTS) else []
    by = {}
    for r in rows:
        by.setdefault(r["n"], []).append(r)
    out = [f"# qwasar sweep: {os.path.basename(MODEL)}", "",
           f"Updated {time.strftime('%Y-%m-%d %H:%M')}.  Each point is a fresh process: prefill exactly N",
           f"tokens of real text (256-token chunks), then {DECODE} greedy decode tokens at that depth.",
           "Medians over repetitions; temperatures are the maxima seen (C), die = SoC.", "",
           "| N | reps | prefill t/s | decode t/s | max die, prefill | max die, decode | max battery | thermal state |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in sorted(by):
        ok = [r for r in by[n] if "prefill_tps" in r]
        if not ok:
            out.append(f"| {n} | 0 | failed: {by[n][-1].get('error', '?')[:60]} | | | | | |")
            continue
        med = lambda f: statistics.median(f(r) for r in ok)
        out.append(f"| {n} | {len(ok)} | {med(lambda r: r['prefill_tps']):.1f} | {med(lambda r: r['decode_tps']):.2f} | "
                   f"{max(r['temp_prefill']['die'] for r in ok):.1f} | {max(r['temp_decode']['die'] for r in ok):.1f} | "
                   f"{max(r['temp_run']['battery'] for r in ok):.1f} | {max(r['temp_run']['thermal_state'] for r in ok)} |")
    out += ["", "results.jsonl holds every run, with per-chunk prefill times (prefill rate at each depth)",
            "and per-token decode times; samples/ holds the 1 Hz temperature traces."]
    with open(SUMMARY, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    done = set()
    if os.path.exists(RESULTS):
        for l in open(RESULTS):
            r = json.loads(l); done.add((r["n"], r["rep"]))
    failed_sizes = set()
    log(f"sweep starting: {len(PLAN)} points, {len(done)} already done")
    for n, rep in PLAN:
        if (n, rep) in done or n in failed_sizes:
            continue
        rec = run_point(n, rep)
        if "error" in rec:
            failed_sizes.add(n)
        summarize()
    log("sweep finished")


if __name__ == "__main__":
    main()
