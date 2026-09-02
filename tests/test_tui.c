/* The footer under a mid-turn question.
 *
 * Runs without a model: the TUI is driven from a pseudo-terminal, the way a
 * person drives it, and the bytes it paints are read back and judged.  The
 * scenario is the one that used to go wrong: a tool asks "proceed? [y/N]"
 * while the user has a message queued and another half typed, and once the
 * answer is given the footer must go back to the prompt it showed before --
 * with the half-typed text still in it -- and the queued message must reach
 * the next real prompt untouched, never the question.
 *
 * Two processes.  The child is a puppet that runs the scenario against the
 * real tui_* calls and reports what each read returned over a pipe; the
 * parent types into the pty on a schedule and captures everything the child
 * painted.  Timing is by delays generous enough that a loaded machine does
 * not fail this, not by racing. */

#include "qwasar_tui.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <util.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- the puppet ---------------------------------------------------------- */

static void report(int fd, const char *key, const char *val) {
    char b[256];
    int n = snprintf(b, sizeof b, "RESULT %s=%s\n", key, val ? val : "(null)");
    if (n > 0) write(fd, b, (size_t)n);
}

static void puppet(int rep) {
    qw_tui *t = tui_new();
    if (!tui_is_tty(t)) { report(rep, "tty", "0"); _exit(0); }

    /* The main prompt, once: this is the resting prompt everything after
     * must come back to. */
    char *l = tui_readline(t, "> ");
    report(rep, "first", l);
    free(l);

    /* "Generation": poll for input the way the decode loop does, long enough
     * for the driver to queue a message and start another. */
    double until = now_sec() + 0.6;
    while (now_sec() < until) {
        tui_interrupted(t);
        usleep(10000);
    }

    /* The question.  Whatever was typed ahead is not the answer to it. */
    char *a = tui_ask(t, "  proceed? [y/N] ");
    report(rep, "ask", a);
    free(a);

    /* Output after the answer is where the footer gets repainted, and where
     * the stale question used to show up. */
    tui_puts(t, "AFTERMARK\n");

    /* The next prompt: the queued message first, then the half-typed one
     * once it is finished. */
    l = tui_readline(t, "> ");
    report(rep, "queued", l);
    free(l);
    l = tui_readline(t, "> ");
    report(rep, "partial", l);
    free(l);

    tui_free(t);
    _exit(0);
}

/* ---- the driver ---------------------------------------------------------- */

static char captured[1 << 16];
static size_t captured_len;
static char results[4096];
static size_t results_len;

static void type(int fd, const char *s) { write(fd, s, strlen(s)); }

static bool have_result(const char *key) {
    char want[64];
    snprintf(want, sizeof want, "RESULT %s=", key);
    return strstr(results, want) != NULL;
}

static const char *result(const char *key) {
    static char val[256];
    char want[64];
    snprintf(want, sizeof want, "RESULT %s=", key);
    const char *p = strstr(results, want);
    if (!p) return NULL;
    p += strlen(want);
    size_t n = strcspn(p, "\n");
    if (n >= sizeof val) n = sizeof val - 1;
    memcpy(val, p, n);
    val[n] = 0;
    return val;
}

static void pump(int master, int rep, double seconds) {
    double until = now_sec() + seconds;
    for (;;) {
        double left = until - now_sec();
        if (left <= 0) return;
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(master, &fds);
        FD_SET(rep, &fds);
        struct timeval tv = { (time_t)left, (suseconds_t)((left - (double)(time_t)left) * 1e6) };
        int nfds = (master > rep ? master : rep) + 1;
        if (select(nfds, &fds, NULL, NULL, &tv) <= 0) continue;
        if (FD_ISSET(master, &fds)) {
            ssize_t n = read(master, captured + captured_len,
                             sizeof captured - 1 - captured_len);
            if (n > 0) { captured_len += (size_t)n; captured[captured_len] = 0; }
        }
        if (FD_ISSET(rep, &fds)) {
            ssize_t n = read(rep, results + results_len, sizeof results - 1 - results_len);
            if (n > 0) { results_len += (size_t)n; results[results_len] = 0; }
        }
    }
}

/* Pumps until `key` has been reported, or the deadline passes. */
static bool pump_until(int master, int rep, const char *key, double seconds) {
    double until = now_sec() + seconds;
    while (now_sec() < until) {
        if (have_result(key)) return true;
        pump(master, rep, 0.05);
    }
    return have_result(key);
}

int main(void) {
    int pipefd[2];
    if (pipe(pipefd) != 0) { perror("pipe"); return 1; }

    struct winsize ws = { .ws_row = 24, .ws_col = 80 };
    int master = -1;
    pid_t pid = forkpty(&master, NULL, NULL, &ws);
    if (pid < 0) { perror("forkpty"); return 1; }
    if (pid == 0) {
        close(pipefd[0]);
        /* linenoise refuses raw mode for TERM=dumb, and make's environment
         * is allowed to be that; the test is about the real terminal path. */
        setenv("TERM", "xterm-256color", 1);
        puppet(pipefd[1]);
        _exit(0);
    }
    close(pipefd[1]);
    int rep = pipefd[0];
    fcntl(master, F_SETFL, O_NONBLOCK);
    fcntl(rep, F_SETFL, O_NONBLOCK);

    /* The main prompt is up; answer it. */
    pump(master, rep, 0.3);
    type(master, "hello\r");
    CHECK(pump_until(master, rep, "first", 3.0), "the first read never returned");
    if (result("first")) CHECK(!strcmp(result("first"), "hello"), "first=%s", result("first"));

    /* The puppet is now "generating".  Queue a whole message, then start
     * another and leave it unfinished. */
    pump(master, rep, 0.1);
    type(master, "ahead\r");
    pump(master, rep, 0.1);
    type(master, "part");

    /* Let the question appear, then answer it -- with input that is not
     * the queued line and not the partial one. */
    pump(master, rep, 0.8);
    type(master, "y\r");
    CHECK(pump_until(master, rep, "ask", 3.0), "the question was never answered");
    if (result("ask")) CHECK(!strcmp(result("ask"), "y"),
                             "the question took %s as its answer", result("ask"));

    /* The queued message goes to the next real prompt, immediately. */
    CHECK(pump_until(master, rep, "queued", 3.0), "the queued line never arrived");
    if (result("queued")) CHECK(!strcmp(result("queued"), "ahead"),
                                "queued=%s, wanted the typed-ahead line", result("queued"));

    /* Finish the half-typed message.  If the reset dropped it, this reads
     * "ial" alone. */
    pump(master, rep, 0.2);
    type(master, "ial\r");
    CHECK(pump_until(master, rep, "partial", 3.0), "the partial line never completed");
    if (result("partial")) CHECK(!strcmp(result("partial"), "partial"),
                                 "partial=%s, wanted the half-typed text kept", result("partial"));

    pump(master, rep, 0.3);
    int status = 0;
    if (waitpid(pid, &status, WNOHANG) == 0) { kill(pid, SIGKILL); waitpid(pid, &status, 0); }

    /* What the terminal saw.  After AFTERMARK the footer must show the
     * resting prompt with the partial text in it, and never the question. */
    const char *after = strstr(captured, "AFTERMARK");
    CHECK(after != NULL, "AFTERMARK was never painted");
    if (after) {
        CHECK(strstr(after, "proceed?") == NULL,
              "the question stayed in the footer after it was answered");
        CHECK(strstr(after, "> part") != NULL,
              "the resting prompt (with the half-typed text) did not come back");
    }
    /* And the transcript records the question with its answer, once. */
    const char *echo = strstr(captured, "proceed? [y/N] y");
    CHECK(echo != NULL && (after == NULL || echo < after),
          "the answered question was not echoed into the transcript");

    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    printf("tui: all checks pass\n");
    return 0;
}
