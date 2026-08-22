/* qwasar command line. */

#include "qwasar.h"
#include "qwasar_gpu.h"
#include "qwasar_model.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Prints one token, honouring the reasoning block.  Factored out because the
 * speculative path emits several per round and they have to look the same. */
static void emit_token(const qwasar_tokenizer *tok, int32_t id, bool *in_reasoning,
                       int32_t think_close, bool show_think) {
    if (id == think_close && *in_reasoning) {
        *in_reasoning = false;
        if (show_think) printf("\n--- answer ---\n");
        return;
    }
    if (*in_reasoning && !show_think) return;
    size_t len = 0;
    bool special = false;
    const char *text = qwasar_token_bytes(tok, id, &len, &special);
    if (text && len && !special) { fwrite(text, 1, len, stdout); fflush(stdout); }
}

static void usage(FILE *out) {
    fprintf(out,
        "qwasar -- Qwen3.8 inference on macOS Metal\n"
        "\n"
        "usage: qwasar [-m <model-dir>] [options]\n"
        "\n"
        "  -m, --model <dir>     model directory (config.json + *.safetensors);\n"
        "                        default ./qwasar-model, or $QWASAR_MODEL\n"
        "  -c, --context <n>     context size in tokens (default 32768)\n"
        "      --chunk <n>       prefill tokens per forward pass (default 256)\n"
        "      --image <path>    an image to look at (jpeg, png, bmp, gif)\n"
        "      --mtp <dir>       multi-token-prediction draft head directory\n"
        "      --mtp-depth <n>   drafts per round; default adapts, 0 disables\n"
        "      --spec            decode speculatively; without it, --mtp-depth\n"
        "                        only measures what the head would have proposed\n"
        "  -n, --predict <n>     tokens to generate (default 512)\n"
        "  -p, --prompt <text>   user message\n"
        "  -s, --system <text>   system message\n"
        "      --effort <level>  reasoning effort: xhigh (default), medium, low\n"
        "      --no-think        skip the reasoning block\n"
        "      --show-think      print the reasoning block (hidden by default)\n"
        "      --tokens <ids>    comma-separated prompt token ids, instead of -p\n"
        "      --info            print the parsed architecture and exit\n"
        "  -v, --verbose         verbose loading\n"
        "  -h, --help            this message\n");
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int32_t *parse_tokens(const char *spec, int32_t *out_n) {
    int32_t cap = 16, n = 0;
    int32_t *v = malloc((size_t)cap * sizeof *v);
    const char *p = spec;
    while (*p) {
        char *end = NULL;
        long id = strtol(p, &end, 10);
        if (end == p) break;
        if (n == cap) { cap *= 2; v = realloc(v, (size_t)cap * sizeof *v); }
        v[n++] = (int32_t)id;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    *out_n = n;
    return v;
}

/* Greedy argmax on the CPU.  At ~250k logits this is a fraction of a
 * millisecond against a decode step measured in hundreds, so it stays here
 * until sampling proper arrives. */
static int32_t argmax(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

static bool resolve_model(qwasar_options *opts, const char *prog) {
    if (opts->model_path) return true;
    opts->model_path = qwasar_default_model_path();
    if (opts->model_path) return true;
    fprintf(stderr,
        "%s: no model given and none found.\n"
        "\n"
        "Download it once:\n"
        "    ./download_model.sh model\n"
        "\n"
        "or point at an existing copy with -m <dir>, or set QWASAR_MODEL.\n", prog);
    return false;
}

int main(int argc, char **argv) {
    qwasar_options opts = { 0 };
    bool want_info = false;
    const char *token_spec = NULL;
    const char *prompt_text = NULL;
    const char *system_text = NULL;
    const char *effort = "xhigh";
    bool thinking = true, show_think = false;
    int n_predict = 512;   /* xhigh reasoning alone often exceeds 128 */
    int mtp_depth = -1;    /* -1 = adaptive, 0 = off, n = fixed */
    const char *image_path = NULL;
    bool use_spec = false; /* --spec: decode speculatively rather than measure */

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if ((!strcmp(a, "-m") || !strcmp(a, "--model")) && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if ((!strcmp(a, "-c") || !strcmp(a, "--context")) && i + 1 < argc) {
            opts.context_size = atoi(argv[++i]);
        } else if (!strcmp(a, "--chunk") && i + 1 < argc) {
            opts.prefill_chunk = atoi(argv[++i]);
        } else if (!strcmp(a, "--mtp") && i + 1 < argc) {
            opts.mtp_path = argv[++i];
        } else if (!strcmp(a, "--mtp-depth") && i + 1 < argc) {
            mtp_depth = atoi(argv[++i]);
        } else if (!strcmp(a, "--spec")) {
            use_spec = true;
        } else if (!strcmp(a, "--image") && i + 1 < argc) {
            image_path = argv[++i];
        } else if ((!strcmp(a, "-n") || !strcmp(a, "--predict")) && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if ((!strcmp(a, "-p") || !strcmp(a, "--prompt")) && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if ((!strcmp(a, "-s") || !strcmp(a, "--system")) && i + 1 < argc) {
            system_text = argv[++i];
        } else if (!strcmp(a, "--effort") && i + 1 < argc) {
            effort = argv[++i];
        } else if (!strcmp(a, "--no-think")) {
            thinking = false;
        } else if (!strcmp(a, "--show-think")) {
            show_think = true;
        } else if (!strcmp(a, "--tokens") && i + 1 < argc) {
            token_spec = argv[++i];
        } else if (!strcmp(a, "--info")) {
            want_info = true;
        } else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) {
            opts.verbose = true;
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "qwasar: unknown argument '%s'\n\n", a);
            usage(stderr);
            return 2;
        }
    }

    if (!resolve_model(&opts, "qwasar")) return 2;

    char err[512] = "";
    double t0 = now_sec();
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "qwasar: %s\n", err); return 1; }
    double t_load = now_sec() - t0;

    /* The image is encoded before any text is rendered, because how many
     * <|image_pad|> tokens the prompt needs is a property of its patch grid. */
    qw_image img = { 0 };
    float *img_rows = NULL;
    int32_t n_img_rows = 0;
    if (image_path) {
        if (!qw_image_load(&img, image_path, qwasar_engine_config(e), err, sizeof err)) {
            fprintf(stderr, "qwasar: %s\n", err);
            return 1;
        }
        double ti = now_sec();
        img_rows = qwasar_encode_image(e, &img, &n_img_rows, err, sizeof err);
        if (!img_rows) { fprintf(stderr, "qwasar: %s\n", err); return 1; }
        fprintf(stderr, "image %dx%d -> %d patches -> %d tokens in %.2fs\n",
                img.src_w, img.src_h, img.n_patches, n_img_rows, now_sec() - ti);
    }

    if (want_info) {
        qwasar_engine_print_info(e, stdout);
        qwasar_engine_free(e);
        qw_gpu_shutdown();
        return 0;
    }
    if (!token_spec && !prompt_text) {
        fprintf(stderr, "qwasar: nothing to do; pass -p, --tokens, or --info\n");
        qwasar_engine_free(e);
        qw_gpu_shutdown();
        return 2;
    }

    qwasar_tokenizer *tok = qwasar_tokenizer_load(opts.model_path, err, sizeof err);
    if (!tok) { fprintf(stderr, "qwasar: %s\n", err); return 1; }

    int32_t n_prompt = 0;
    int32_t *prompt = NULL;
    if (token_spec) {
        prompt = parse_tokens(token_spec, &n_prompt);
    } else {
        qwasar_message msgs[2];
        int32_t n_msgs = 0;
        if (system_text) msgs[n_msgs++] = (qwasar_message){ "system", system_text, NULL, NULL };
        msgs[n_msgs++] = (qwasar_message){ "user", prompt_text, NULL, NULL, n_img_rows };
        qwasar_chat_options chat = { .enable_thinking = thinking,
                                     .reasoning_effort = effort,
                                     .add_generation_prompt = true };
        prompt = qwasar_apply_chat_template(tok, msgs, n_msgs, &chat, &n_prompt,
                                            err, sizeof err);
        if (!prompt) { fprintf(stderr, "qwasar: %s\n", err); return 1; }
    }
    if (n_prompt <= 0) { fprintf(stderr, "qwasar: empty prompt\n"); return 2; }

    /* The first forward pass through a fresh process faults in ~10 GB of
     * mapped weights, which would otherwise be charged to prefill and make its
     * reported rate several times worse than the steady-state one.  Pay it here
     * on a throwaway session so the numbers below mean what they say. */
    {
        double tw = now_sec();
        qwasar_session *warm = qwasar_session_new(e, err, sizeof err);
        if (warm) {
            int32_t one = prompt[0];
            qwasar_session_eval(warm, &one, 1, err, sizeof err);
            qwasar_session_free(warm);
        }
        t_load += now_sec() - tw;
    }

    qwasar_session *s = qwasar_session_new(e, err, sizeof err);
    if (!s) { fprintf(stderr, "qwasar: %s\n", err); return 1; }

    fprintf(stderr, "ready in %.2fs | prompt %d tokens\n", t_load, n_prompt);

    t0 = now_sec();
    const float *logits;
    if (n_img_rows > 0) {
        const qwasar_image_input in = { img_rows, n_img_rows,
                                        img.grid_t, img.grid_h, img.grid_w };
        logits = qwasar_session_eval_images(s, prompt, n_prompt, &in, 1,
                                            err, sizeof err);
    } else {
        logits = qwasar_session_eval(s, prompt, n_prompt, err, sizeof err);
    }
    if (!logits) { fprintf(stderr, "qwasar: %s\n", err); return 1; }
    double t_prefill = now_sec() - t0;

    const int32_t vocab = qwasar_vocab_size(e);
    const int32_t think_close = qwasar_token_id(tok, "</think>");
    int generated = 0;
    double t_decode = 0.0;

    /* ---- draft acceptance ------------------------------------------------
     *
     * Measured against ordinary decoding, with no rollback anywhere: the target
     * decides every token exactly as it would have, and the head is asked what
     * it WOULD have proposed.  Nothing here can change the output, which is the
     * point -- the acceptance rate has to be known before any of the machinery
     * that depends on it is worth building.
     *
     * A round drafts `mtp_depth` tokens, then the target generates until one of
     * them is wrong, which is exactly how a real verify would advance; the next
     * round starts from there.  Position i is only counted when every draft
     * before it was accepted, so these are the conditional probabilities the
     * depth schedule needs, not marginal ones. */
    int32_t drafts[QWASAR_MAX_DRAFT];
    int64_t drafted[QWASAR_MAX_DRAFT] = { 0 }, accepted[QWASAR_MAX_DRAFT] = { 0 };
    int64_t rounds = 0, round_tokens = 0;
    int32_t n_drafted = 0, draft_at = 0;
    double t_draft = 0.0;
    const bool have_mtp = qwasar_session_has_mtp(s);
    const bool spec    = use_spec && mtp_depth != 0 && have_mtp;
    const bool measure = !use_spec && mtp_depth > 0 && have_mtp;
    if (mtp_depth > 0 && !have_mtp)
        fprintf(stderr, "qwasar: --mtp-depth needs --mtp <dir>\n");
    int64_t depth_sum = 0;

    /* The generation prompt leaves <think> open, so everything up to </think>
     * is reasoning.  Hidden unless asked for, but always consumed. */
    bool in_reasoning = thinking && !token_spec;

    int32_t next = argmax(logits, vocab);
    while (generated < n_predict) {
        if (qwasar_is_eos(e, next)) break;

        if (measure) {
            if (n_drafted > 0) {
                /* One draft stands for this position.  A miss ends the round,
                 * exactly as a rejected token would end a verify. */
                const bool ok = (drafts[draft_at] == next);
                drafted[draft_at]++;
                if (ok) accepted[draft_at]++;
                round_tokens++;
                if (!ok || ++draft_at >= n_drafted) n_drafted = 0;
            }
            if (n_drafted == 0) {
                double td = now_sec();
                int32_t got = qwasar_session_draft(s, next, drafts, mtp_depth,
                                                   err, sizeof err);
                t_draft += now_sec() - td;
                if (got < 0) { fprintf(stderr, "\nqwasar: %s\n", err); return 1; }
                n_drafted = got;
                draft_at = 0;
                rounds++;
            }
        }

        emit_token(tok, next, &in_reasoning, think_close, show_think);
        generated++;

        if (spec) {
            /* Propose a block, settle all of it in one pass over the weights,
             * and keep the longest correct prefix.  The last committed token
             * takes the place `next` had: it is the one this round leaves
             * undecided, exactly as an ordinary step would. */
            int32_t blk[1 + QWASAR_MAX_DRAFT], got[1 + QWASAR_MAX_DRAFT];
            blk[0] = next;
            const int32_t want = mtp_depth > 0 ? mtp_depth
                                               : qwasar_session_draft_depth(s);
            if (want == 0) {
                /* The head has been wrong often enough that a round would cost
                 * more than it returns.  A plain step is the cheaper answer. */
                double t0d = now_sec();
                logits = qwasar_session_eval(s, &next, 1, err, sizeof err);
                t_decode += now_sec() - t0d;
                if (!logits) { fprintf(stderr, "\nqwasar: %s\n", err); return 1; }
                next = argmax(logits, vocab);
                continue;
            }
            depth_sum += want;
            double td = now_sec();
            int32_t nd = qwasar_session_draft(s, next, blk + 1, want,
                                              err, sizeof err);
            t_draft += now_sec() - td;
            if (nd < 0) { fprintf(stderr, "\nqwasar: %s\n", err); return 1; }

            double t1 = now_sec();
            int32_t nc = qwasar_session_verify(s, blk, nd + 1, got, err, sizeof err);
            t_decode += now_sec() - t1;
            if (nc < 0) { fprintf(stderr, "\nqwasar: %s\n", err); return 1; }
            rounds++;
            round_tokens += nc;

            bool stop = false;
            for (int32_t i = 0; i + 1 < nc && generated < n_predict; i++) {
                if (qwasar_is_eos(e, got[i])) { stop = true; break; }
                emit_token(tok, got[i], &in_reasoning, think_close, show_think);
                generated++;
            }
            if (stop) break;
            next = got[nc - 1];
            continue;
        }

        double t1 = now_sec();
        logits = qwasar_session_eval(s, &next, 1, err, sizeof err);
        t_decode += now_sec() - t1;
        if (!logits) { fprintf(stderr, "\nqwasar: %s\n", err); return 1; }
        next = argmax(logits, vocab);
    }
    printf("\n");

    if (spec && rounds > 0)
        fprintf(stderr, "\nmtp  %lld rounds, %.2f tokens per round, mean depth %.2f, "
                        "%.1fs drafting\n", (long long)rounds,
                (double)round_tokens / (double)rounds,
                (double)depth_sum / (double)rounds, t_draft);

    if (measure && rounds > 0) {
        fprintf(stderr, "\nmtp  %lld rounds, %.2f tokens per round, %.1fs drafting\n",
                (long long)rounds, (double)round_tokens / (double)rounds, t_draft);
        double reach = 1.0;
        for (int i = 0; i < mtp_depth && i < QWASAR_MAX_DRAFT; i++) {
            if (!drafted[i]) break;
            double p = (double)accepted[i] / (double)drafted[i];
            reach *= p;
            fprintf(stderr, "     draft %d: %lld/%lld accepted (%.1f%%), "
                            "reach %.1f%%\n",
                    i, (long long)accepted[i], (long long)drafted[i],
                    100.0 * p, 100.0 * reach);
        }
    }

    /* Silence here would be indistinguishable from a finished answer, and with
     * reasoning hidden the visible output can be a fraction of the budget. */
    if (generated == n_predict)
        fprintf(stderr, "\nqwasar: stopped at the %d-token budget%s; raise it with -n\n",
                n_predict, in_reasoning ? " while still reasoning" : "");

    fprintf(stderr, "\nprefill %d tokens in %.2fs (%.1f tok/s) | "
                    "decode %d tokens in %.2fs (%.2f tok/s)\n",
            n_prompt, t_prefill, n_prompt / t_prefill,
            generated, t_decode, generated > 0 ? generated / t_decode : 0.0);

    free(prompt);
    qwasar_session_free(s);
    qwasar_tokenizer_free(tok);
    qwasar_engine_free(e);
    qw_gpu_shutdown();
    return 0;
}
