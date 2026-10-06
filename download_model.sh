#!/bin/sh
# qwasar model downloader.
#
# Fetches the weights qwasar needs and links them where the binaries look by
# default, so that after
#
#     ./download_model.sh model
#
# this works from the project directory with no arguments:
#
#     ./qwasar -p "Hello"
#
# Three models, one engine: Qwen3.8 27B (dense, ~16 GB, runs on a 32 GB Mac),
# Qwen3.8 Flash-Next (a 125B mixture of experts with 6B active, ~111 GB,
# which needs a 128 GB Mac), and Qwen3.5 9B (dense, ~6 GB, for 16 GB Macs --
# the previous generation, but the same architecture, tokenizer and chat
# format).  The engine tells them apart by config.json.
#
# Run with no arguments to have it suggest one for this Mac's memory.
#
# Only the files qwasar actually reads are downloaded -- config.json, the
# safetensors index, the shards, and tokenizer.json.  The rest of each
# repository is a chat template qwasar embeds verbatim (see THIRD-PARTY.md)
# and preprocessor configs whose few constants are compiled in.
#
# Downloads resume: run the same command again after an interruption.
set -e

MODEL_REPO="lmstudio-community/Qwen3.8-27B-MLX-4bit"
MODEL_REV="6067b15cf581666a4aecf6af3afaba4bb5efc20c"
MODEL_NAME="Qwen3.8-27B-MLX-4bit"
MODEL_FILES="config.json
model.safetensors.index.json
tokenizer.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors"
MODEL_GB=16
# The 27B holds ~16 GB of weights plus its context; below this (a 24 GB Mac)
# the engine would refuse to load it.  Memory sizes here are as Macs are sold.
MODEL_MIN_MEM_GB=32

# Qwen3.5 9B, mlx-community's 4-bit build (group 64, A_log kept in F32, which
# the loader rounds to BF16).  ~5.5 GB of weights, ~8 GB with a 32K context.
NINEB_REPO="mlx-community/Qwen3.5-9B-MLX-4bit"
NINEB_REV="938d8919941c6e7efd3c7150eff7fe9d12afa631"
NINEB_NAME="Qwen3.5-9B-MLX-4bit"
NINEB_FILES="config.json
model.safetensors.index.json
tokenizer.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors"
NINEB_GB=6
NINEB_MIN_MEM_GB=12

# mlx-community's 4-bit build, made with mlx-vlm: group 32, the router and
# shared-expert gates at 8 bits, the engram table 4-bit and mixed into the
# model files.  The engine reads that format as it is (PLAN-flash-next.md,
# "the real model runs").
FLASH_REPO="mlx-community/Qwen3.8-Flash-Next-4bit"
FLASH_REV="07b5dc6c54600a359b87f1e53e7adf6351c72a2c"
FLASH_NAME="Qwen3.8-Flash-Next-MLX-4bit"
FLASH_FILES="config.json
model.safetensors.index.json
tokenizer.json"
i=1
while [ $i -le 22 ]; do
    FLASH_FILES="$FLASH_FILES
$(printf 'model-%05d-of-00022.safetensors' $i)"
    i=$((i + 1))
done
FLASH_GB=112
# Below this much memory the engine refuses to load Flash-Next (it holds
# ~80 GB of weights wired), so the download is refused too unless --force.
FLASH_MIN_MEM_GB=96

MTP_REPO="EigenLabs/Qwen3.8-27B-MTP-bf16"
MTP_REV="26a328e070875b0314d652a039b6b59902690f03"
MTP_NAME="Qwen3.8-27B-MTP-bf16"
MTP_FILES="config.json
model.safetensors.index.json
model.safetensors"
MTP_GB=1

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${QWASAR_MODEL_DIR:-"$ROOT/models"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}
VERIFY=0
FORCE=0
AS_DEFAULT=0

usage() {
    cat <<EOF
qwasar model downloader

Usage:
  ./download_model.sh            (suggest a model for this Mac's memory)
  ./download_model.sh model      [--force] [--verify] [--token TOKEN]
  ./download_model.sh 9b         [--default] [--force] [--verify] [--token TOKEN]
  ./download_model.sh flash-next [--default] [--force] [--verify] [--token TOKEN]
  ./download_model.sh mtp-head   [--verify] [--token TOKEN]
  ./download_model.sh all        [--verify] [--token TOKEN]

Targets:

  model
       Qwen3.8 27B, dense, MLX affine 4-bit, group 64.  About 16 GB on disk
       and in memory; runs on a 32 GB Mac.  Other quantisations of the same
       model (8-bit, 6-bit, FP8, GGUF) will not load: the kernels are built
       around the MLX affine 4-bit format on purpose (PLAN.md section 1.2).

       Repository: $MODEL_REPO
       Links ./qwasar-model, which every qwasar binary uses when -m is absent.

  9b
       Qwen3.5 9B, dense, MLX affine 4-bit, group 64.  About 6 GB on disk and
       ~8 GB in memory with a 32K context, so it runs on a 16 GB Mac.  The
       previous Qwen generation, but the same hybrid architecture, tokenizer
       and chat format.  Faster than the 27B and weaker on long agent tasks.

       Repository: $NINEB_REPO
       Links ./qwasar-9b.  With --default, ./qwasar-model points at it too,
       so the binaries use it when -m is absent.

  flash-next
       Qwen3.8 Flash-Next: a 125B mixture of experts with 6B active per
       token, MLX affine 4-bit (group 32).  About 111 GB on disk; ~80 GB of
       it is held in memory and the engram table stays on disk, so it needs
       a 128 GB Mac.  Decodes at ~68 t/s on an M5 Max.

       Repository: $FLASH_REPO
       Links ./qwasar-flash-next.  With --default, ./qwasar-model points at
       it too, so the binaries use it when -m is absent.

  mtp-head
       The 27B's multi-token-prediction draft head, bf16.  About 850 MB.
       Optional: speculative decoding with it is about 1.5x on the 27B
       (qwasar --mtp ./qwasar-mtp --spec).  Flash-Next does not use it.

       Repository: $MTP_REPO
       Links ./qwasar-mtp, which is what to pass to --mtp.

  all
       The 27B and its draft head -- what runs on any supported Mac.
       Flash-Next is its own target: it is 111 GB and needs 128 GB.

Options:
  --default      (9b, flash-next) Also point ./qwasar-model at it.
  --force        Download even on a machine with too little memory to run
                 it, e.g. to copy it to another Mac.
  --verify       Check the SHA-256 of every downloaded file.  Sizes are always
                 checked; this additionally re-reads every file, so it is
                 opt-in.
  --token TOKEN  Hugging Face token.  None of the repositories needs one;
                 HF_TOKEN and the local token cache are used if present.

Environment:
  QWASAR_MODEL_DIR  Where downloads are kept.  Default: ./models
  QWASAR_MODEL      Checked by the binaries before ./qwasar-model, so an
                    existing copy elsewhere can be used without downloading.

Every repository is pinned to a revision, so a re-run fetches the same bytes
this was tested against rather than whatever main has become.
EOF
}

# Physical memory in binary GB, so a "16 GB" Mac reads 16, or 0 if unknown.
# QWASAR_MEM_GB pretends otherwise, for trying the suggestions and refusals.
mem_gb() {
    if [ -n "${QWASAR_MEM_GB:-}" ]; then echo "$QWASAR_MEM_GB"; return; fi
    b=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    echo $((b / 1073741824))
}

# With no target: the largest model this Mac runs, and how to get it.
suggest() {
    mem=$(mem_gb)
    if [ "$mem" -eq 0 ]; then
        echo "Could not read this machine's memory size."
        echo
        usage
        return
    fi
    echo "This Mac has $mem GB of unified memory."
    echo
    if [ "$mem" -ge "$FLASH_MIN_MEM_GB" ]; then
        cat <<EOF
Suggested: Qwen3.8 Flash-Next -- ~111 GB download, ~80 GB in memory, the
fastest model and the one the agent is best on.

    ./download_model.sh flash-next --default

The 27B also runs here (./download_model.sh model), at ~16 GB.
EOF
    elif [ "$mem" -ge "$MODEL_MIN_MEM_GB" ]; then
        cat <<EOF
Suggested: Qwen3.8 27B -- ~16 GB download and in memory.

    ./download_model.sh model

For speed over quality, Qwen3.5 9B runs here too (./download_model.sh 9b).
Flash-Next needs a 128 GB Mac.
EOF
    elif [ "$mem" -ge "$NINEB_MIN_MEM_GB" ]; then
        cat <<EOF
Suggested: Qwen3.5 9B -- ~6 GB download, ~8 GB in memory.

    ./download_model.sh 9b --default

The 27B needs a 32 GB Mac and Flash-Next a 128 GB one.
EOF
    else
        cat <<EOF
None of the models qwasar is tested with fits: the smallest, Qwen3.5 9B,
needs ~8 GB free, which a Mac of this size does not leave beside macOS.
(./download_model.sh 9b --force downloads it anyway.)
EOF
    fi
    echo
    echo "Every target: ./download_model.sh --help"
}

if [ $# -eq 0 ]; then
    suggest
    exit 0
fi

TARGET=$1
shift
case "$TARGET" in
    model|9b|flash-next|mtp-head|all) ;;
    -h|--help|help) usage; exit 0 ;;
    *)
        echo "Unknown target: $TARGET" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            [ $# -gt 0 ] || { echo "Missing value after --token" >&2; exit 1; }
            TOKEN=$1
            ;;
        --verify) VERIFY=1 ;;
        --force) FORCE=1 ;;
        --default) AS_DEFAULT=1 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

if [ "$AS_DEFAULT" -eq 1 ] && [ "$TARGET" != "flash-next" ] && [ "$TARGET" != "9b" ]; then
    echo "--default applies to 9b and flash-next; model already links ./qwasar-model" >&2
    exit 1
fi

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

# ---- before a download that cannot be used ------------------------------------

# Free space in GB where the downloads go (the nearest existing parent).
free_gb() {
    d=$OUT_DIR
    while [ ! -d "$d" ]; do d=$(dirname "$d"); done
    kb=$(df -k "$d" | awk 'NR==2 {print $4}')
    echo $((kb / 1000000))
}

# GB already on disk for a model, so a resumed download is not refused for
# space it has already used.
have_gb() {
    dir="$OUT_DIR/$1"
    [ -d "$dir" ] || { echo 0; return; }
    kb=$(du -sk "$dir" | cut -f1)
    echo $((kb / 1000000))
}

check_space() {
    name=$1; need=$2
    left=$((need - $(have_gb "$name")))
    [ "$left" -gt 0 ] || return 0
    free=$(free_gb)
    if [ "$free" -lt $((left + 5)) ]; then
        echo "Not enough disk space for $name: about $left GB still to download," >&2
        echo "$free GB free under $OUT_DIR.  Set QWASAR_MODEL_DIR to put it elsewhere." >&2
        exit 1
    fi
}

# Refuses a model this Mac cannot run, and says which one it can.
too_small() {
    what=$1; why=$2
    echo "$why" >&2
    echo >&2
    suggest >&2
    echo >&2
    echo "To download $what anyway (for another Mac), add --force." >&2
    exit 1
}

if [ "$FORCE" -eq 0 ] && [ "$(mem_gb)" -gt 0 ]; then
    case "$TARGET" in
        flash-next)
            [ "$(mem_gb)" -ge "$FLASH_MIN_MEM_GB" ] || too_small Flash-Next \
                "Flash-Next holds ~80 GB of weights in memory, so it needs a 128 GB Mac." ;;
        model|all)
            [ "$(mem_gb)" -ge "$MODEL_MIN_MEM_GB" ] || too_small "the 27B" \
                "The 27B holds ~16 GB of weights plus its context, so it needs a 32 GB Mac." ;;
        9b)
            [ "$(mem_gb)" -ge "$NINEB_MIN_MEM_GB" ] || too_small "the 9B" \
                "The 9B needs ~8 GB free with a 32K context, so it needs a 16 GB Mac." ;;
    esac
fi

# Asking the server for the expected size and digest rather than pinning them
# here means the check cannot go stale against the pinned revision.
#
# Redirects have to be followed to learn either: a HEAD on the repository URL
# answers with the length of the redirect body, not of the file, which is a
# trap worth naming because the number it returns looks perfectly plausible.
# The LAST content-length in the chain is the real one.
#
# x-linked-etag is a SHA-256 for the LFS files and a git blob SHA-1 for the
# small ones, so only the 64-character form is usable as a digest.
remote_meta() {
    if [ -n "$TOKEN" ]; then
        curl -sIL -H "Authorization: Bearer $TOKEN" --max-time 120 "$1"
    else
        curl -sIL --max-time 120 "$1"
    fi | tr -d '\r' | awk 'BEGIN{IGNORECASE=1}
             /^content-length:/{s=$2}
             /^x-linked-etag:/{gsub(/"/,"",$2); if (length($2)==64) e=$2}
             END{print s, e}'
}

file_size() {
    wc -c < "$1" | tr -d ' '
}

download_one() {
    repo=$1; rev=$2; file=$3; dir=$4
    out="$dir/$file"
    part="$out.part"
    url="https://huggingface.co/$repo/resolve/$rev/$file"

    meta=$(remote_meta "$url")
    want_size=$(printf '%s' "$meta" | cut -d' ' -f1)
    want_sha=$(printf '%s' "$meta" | cut -d' ' -f2)

    if [ -s "$out" ]; then
        if [ -n "$want_size" ] && [ "$(file_size "$out")" != "$want_size" ]; then
            echo "Size mismatch, re-downloading: $out" >&2
            rm -f "$out"
        else
            echo "Already downloaded: $file"
            [ "$VERIFY" -eq 1 ] || return 0
        fi
    fi

    if [ ! -s "$out" ]; then
        echo "Downloading $file from $repo"
        if [ -n "$TOKEN" ]; then
            curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
        else
            curl -fL --progress-meter -C - -o "$part" "$url"
        fi
        mv "$part" "$out"

        # A truncated download is the failure this catches, and it is worth
        # catching here: qwasar would otherwise report it as a malformed
        # safetensors header several gigabytes into a load.
        if [ -n "$want_size" ] && [ "$(file_size "$out")" != "$want_size" ]; then
            echo "Downloaded $out is $(file_size "$out") bytes, expected $want_size." >&2
            echo "Run this command again to resume." >&2
            exit 1
        fi
    fi

    if [ "$VERIFY" -eq 1 ] && [ -n "$want_sha" ]; then
        got=$(shasum -a 256 "$out" | cut -d' ' -f1)
        if [ "$got" != "$want_sha" ]; then
            echo "SHA-256 mismatch for $out" >&2
            echo "  expected $want_sha" >&2
            echo "  got      $got" >&2
            exit 1
        fi
        echo "  verified $file"
    fi
}

fetch() {
    repo=$1; rev=$2; name=$3; files=$4; link=$5; gb=$6
    dir="$OUT_DIR/$name"
    check_space "$name" "$gb"
    mkdir -p "$dir"
    # A `for` loop rather than `read` from a pipe: the pipe would put the body
    # in a subshell, where a failed download's `exit` stops only the subshell.
    for f in $files; do
        download_one "$repo" "$rev" "$f" "$dir"
    done
    ln -sfn "$dir" "$ROOT/$link"
    echo "Linked ./$link -> $dir"
}

fetch_model() { fetch "$MODEL_REPO" "$MODEL_REV" "$MODEL_NAME" "$MODEL_FILES" qwasar-model "$MODEL_GB"; }
fetch_mtp()   { fetch "$MTP_REPO"   "$MTP_REV"   "$MTP_NAME"   "$MTP_FILES"   qwasar-mtp   "$MTP_GB"; }
fetch_9b() {
    fetch "$NINEB_REPO" "$NINEB_REV" "$NINEB_NAME" "$NINEB_FILES" qwasar-9b "$NINEB_GB"
    if [ "$AS_DEFAULT" -eq 1 ]; then
        ln -sfn "$OUT_DIR/$NINEB_NAME" "$ROOT/qwasar-model"
        echo "Linked ./qwasar-model -> $OUT_DIR/$NINEB_NAME"
    fi
}
fetch_flash() {
    fetch "$FLASH_REPO" "$FLASH_REV" "$FLASH_NAME" "$FLASH_FILES" qwasar-flash-next "$FLASH_GB"
    if [ "$AS_DEFAULT" -eq 1 ]; then
        ln -sfn "$OUT_DIR/$FLASH_NAME" "$ROOT/qwasar-model"
        echo "Linked ./qwasar-model -> $OUT_DIR/$FLASH_NAME"
    fi
}

case "$TARGET" in
    model)      fetch_model ;;
    9b)         fetch_9b ;;
    flash-next) fetch_flash ;;
    mtp-head)   fetch_mtp ;;
    all)        fetch_model; fetch_mtp ;;
esac

echo
case "$TARGET" in
    model|all)
        echo "Ready.  From this directory:"
        echo "  ./qwasar -p \"Hello\""
        echo "  ./qwasar-agent"
        echo "  ./qwasar-server"
        ;;
    flash-next)
        echo "Ready.  From this directory:"
        if [ "$AS_DEFAULT" -eq 1 ]; then
            echo "  ./qwasar -p \"Hello\""
            echo "  ./qwasar-server"
        else
            echo "  ./qwasar -m ./qwasar-flash-next -p \"Hello\""
            echo "  ./qwasar-server -m ./qwasar-flash-next"
            echo "or make it the default: ./download_model.sh flash-next --default"
        fi
        echo "In Qwasar.app, choose models/$FLASH_NAME in the Model menu."
        ;;
    9b)
        echo "Ready.  From this directory:"
        if [ "$AS_DEFAULT" -eq 1 ]; then
            echo "  ./qwasar -p \"Hello\""
            echo "  ./qwasar-server"
        else
            echo "  ./qwasar -m ./qwasar-9b -p \"Hello\""
            echo "  ./qwasar-server -m ./qwasar-9b"
            echo "or make it the default: ./download_model.sh 9b --default"
        fi
        echo "In Qwasar.app, choose models/$NINEB_NAME in the Model menu."
        ;;
esac
case "$TARGET" in
    mtp-head|all)
        echo
        echo "Draft head at ./qwasar-mtp; speculative decoding with the 27B:"
        echo "  ./qwasar --mtp ./qwasar-mtp --spec -p \"Hello\""
        ;;
esac
