/* qwasar command line. */

#include "qwasar.h"
#include "qwasar_gpu.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(FILE *out) {
    fprintf(out,
        "qwasar -- Qwen3.8 inference on macOS Metal\n"
        "\n"
        "usage: qwasar -m <model-dir> [options]\n"
        "\n"
        "  -m, --model <dir>     model directory (config.json + *.safetensors)\n"
        "  -c, --context <n>     context size in tokens (default 32768)\n"
        "      --chunk <n>       prefill tokens per forward pass (default 256)\n"
        "  -n, --predict <n>     tokens to generate (default 128)\n"
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

int main(int argc, char **argv) {
    qwasar_options opts = { 0 };
    bool want_info = false;
    const char *token_spec = NULL;
    const char *prompt_text = NULL;
    const char *system_text = NULL;
    const char *effort = "xhigh";
    bool thinking = true, show_think = false;
    int n_predict = 128;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if ((!strcmp(a, "-m") || !strcmp(a, "--model")) && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if ((!strcmp(a, "-c") || !strcmp(a, "--context")) && i + 1 < argc) {
            opts.context_size = atoi(argv[++i]);
        } else if (!strcmp(a, "--chunk") && i + 1 < argc) {
            opts.prefill_chunk = atoi(argv[++i]);
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

    if (!opts.model_path) {
        fprintf(stderr, "qwasar: -m/--model is required\n\n");
        usage(stderr);
        return 2;
    }

    char err[512] = "";
    double t0 = now_sec();
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "qwasar: %s\n", err); return 1; }
    double t_load = now_sec() - t0;

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
        if (system_text) msgs[n_msgs++] = (qwasar_message){ "system", system_text, NULL };
        msgs[n_msgs++] = (qwasar_message){ "user", prompt_text, NULL };
        qwasar_chat_options chat = { .enable_thinking = thinking,
                                     .reasoning_effort = effort,
                                     .add_generation_prompt = true };
        prompt = qwasar_apply_chat_template(tok, msgs, n_msgs, &chat, &n_prompt,
                                            err, sizeof err);
        if (!prompt) { fprintf(stderr, "qwasar: %s\n", err); return 1; }
    }
    if (n_prompt <= 0) { fprintf(stderr, "qwasar: empty prompt\n"); return 2; }

    qwasar_session *s = qwasar_session_new(e, err, sizeof err);
    if (!s) { fprintf(stderr, "qwasar: %s\n", err); return 1; }

    fprintf(stderr, "loaded in %.2fs | prompt %d tokens\n", t_load, n_prompt);

    t0 = now_sec();
    const float *logits = qwasar_session_eval(s, prompt, n_prompt, err, sizeof err);
    if (!logits) { fprintf(stderr, "qwasar: %s\n", err); return 1; }
    double t_prefill = now_sec() - t0;

    const int32_t vocab = qwasar_vocab_size(e);
    const int32_t think_close = qwasar_token_id(tok, "</think>");
    int generated = 0;
    double t_decode = 0.0;

    /* The generation prompt leaves <think> open, so everything up to </think>
     * is reasoning.  Hidden unless asked for, but always consumed. */
    bool in_reasoning = thinking && !token_spec;

    for (; generated < n_predict; generated++) {
        int32_t next = argmax(logits, vocab);
        if (qwasar_is_eos(e, next)) break;

        if (next == think_close && in_reasoning) {
            in_reasoning = false;
            if (show_think) printf("\n--- answer ---\n");
        } else if (!in_reasoning || show_think) {
            size_t len = 0;
            bool special = false;
            const char *text = qwasar_token_bytes(tok, next, &len, &special);
            if (text && len && !special) { fwrite(text, 1, len, stdout); fflush(stdout); }
        }

        double t1 = now_sec();
        logits = qwasar_session_eval(s, &next, 1, err, sizeof err);
        t_decode += now_sec() - t1;
        if (!logits) { fprintf(stderr, "\nqwasar: %s\n", err); return 1; }
    }
    printf("\n");

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
