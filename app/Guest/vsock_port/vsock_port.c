/* vsock_port -- AF_VSOCK <-> stdio, for a BEAM that cannot open a vsock.
 *
 * PLAN.md 6.4. The guest's control channel is a virtio socket, and Erlang's
 * gen_tcp has no AF_VSOCK support.  socat would do it, at 400 KB and a
 * dependency; this is the same job in a page of C.
 *
 * The BEAM opens it as a port:
 *
 *     open_port({spawn_executable, "/usr/local/bin/vsock_port"},
 *               [{packet, 4}, binary, exit_status])
 *
 * and gets whole framed messages with no parsing of its own.  This program
 * knows nothing about the framing: {packet, 4} writes a 4-byte big-endian
 * length ahead of each message on the port side, and the host reads the same
 * four bytes off the socket.  Here it is a byte pipe and nothing more, which is
 * exactly why it can stay this short.
 *
 * One connection at a time, deliberately.  The bridge is a single control
 * channel; a second connection would mean two things believe they own the
 * guest.
 *
 *     cc -O2 -static -o vsock_port vsock_port.c
 */

/* sys/socket.h must precede linux/vm_sockets.h: the kernel header uses
 * sa_family_t and sizeof(struct sockaddr) without defining either, so the
 * alphabetical ordering everything else in this tree uses does not compile. */
#include <sys/socket.h>
#include <linux/vm_sockets.h>

#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PORT_DEFAULT 1024
#define BUF 65536

/* Writes all of `n` bytes or fails.  A short write on a pipe is ordinary, and
 * treating one as success is how a framed protocol loses a header and
 * desynchronises for the rest of the session. */
static int write_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (w == 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

/* Pumps until either side closes.  Both directions are polled together so a
 * large host->guest message cannot starve the reply travelling the other way. */
static void pump(int sock) {
    struct pollfd fds[2] = {
        { .fd = STDIN_FILENO, .events = POLLIN },
        { .fd = sock,         .events = POLLIN },
    };
    char buf[BUF];

    for (;;) {
        if (poll(fds, 2, -1) < 0) {
            if (errno == EINTR) continue;
            return;
        }
        /* BEAM -> host */
        if (fds[0].revents & (POLLIN | POLLHUP)) {
            ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
            if (n <= 0) return;
            if (write_all(sock, buf, (size_t)n) < 0) return;
        }
        /* host -> BEAM */
        if (fds[1].revents & (POLLIN | POLLHUP)) {
            ssize_t n = read(sock, buf, sizeof buf);
            if (n <= 0) return;
            if (write_all(STDOUT_FILENO, buf, (size_t)n) < 0) return;
        }
        if ((fds[0].revents | fds[1].revents) & (POLLERR | POLLNVAL)) return;
    }
}

int main(int argc, char **argv) {
    unsigned int port = argc > 1 ? (unsigned int)strtoul(argv[1], NULL, 10) : PORT_DEFAULT;

    int srv = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket(AF_VSOCK)"); return 1; }

    struct sockaddr_vm addr;
    memset(&addr, 0, sizeof addr);
    addr.svm_family = AF_VSOCK;
    addr.svm_cid    = VMADDR_CID_ANY;
    addr.svm_port   = port;

    if (bind(srv, (struct sockaddr *)&addr, sizeof addr) < 0) { perror("bind"); return 1; }
    if (listen(srv, 1) < 0) { perror("listen"); return 1; }

    /* Announced on stderr, which the guest's console carries, so a host that
     * connects too early can see that the listener did come up. */
    fprintf(stderr, "vsock_port: listening on port %u\n", port);

    int c = accept(srv, NULL, NULL);
    if (c < 0) { perror("accept"); return 1; }
    fprintf(stderr, "vsock_port: connected\n");

    pump(c);

    close(c);
    close(srv);
    fprintf(stderr, "vsock_port: closed\n");
    return 0;
}
