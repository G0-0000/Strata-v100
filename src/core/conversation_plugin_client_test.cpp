// Host-only tests for the conversation cache plugin client's protocol v3 handshake: a fake plugin server on a
// unix socket answers the hello - with the memfd attached, with a refusal, with junk, without the memfd, or
// not at all - and the client's outcome is checked.  The success case also maps the received memfd and reads
// the control block the fake plugin wrote into it.  Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA,
// no model.
#include "strata/core/conversation_plugin_client.hpp"
#include "strata/core/shared_control.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

std::string socket_path(int n) { return "/tmp/strata-plugin-client-test-" + std::to_string(::getpid()) + "-" + std::to_string(n) + ".sock"; }

int make_listener(const std::string& path) {
    const int l = ::socket(AF_UNIX, SOCK_STREAM, 0);
    check(l >= 0, "listener socket");
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    check(path.size() < sizeof(addr.sun_path), "socket path fits");
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    ::unlink(path.c_str());
    check(::bind(l, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0, "listener bind");
    check(::listen(l, 4) == 0, "listener listen");
    return l;
}

// What the fake plugin does for one connection.
struct Behavior {
    const char* reply = nullptr;   // the hello reply line (nullptr: nothing goes out)
    bool attach_fd = false;        // one memfd (with a written control block) rides the same sendmsg
    int delay_ms = 0;              // sleep before answering (the timeout case)
};

// The request line the fake plugin read, for the request-shape checks.
std::string g_request;

void serve_once(int listener, const Behavior& b) {
    const int c = ::accept(listener, nullptr, nullptr);
    if (c < 0) return;
    char req[2048];
    size_t len = 0;
    while (len < sizeof(req) - 1) {
        const ssize_t n = ::recv(c, req + len, sizeof(req) - 1 - len, 0);
        if (n <= 0) break;
        len += (size_t) n;
        if (std::memchr(req, '\n', len) != nullptr) break;
    }
    req[len] = '\0';
    g_request = req;
    if (b.delay_ms > 0) {
        timespec ts{b.delay_ms / 1000, (b.delay_ms % 1000) * 1000000L};
        ::nanosleep(&ts, nullptr);
    }
    if (b.reply != nullptr) {
        const size_t rlen = std::strlen(b.reply);
        if (b.attach_fd) {
            const int m = ::memfd_create("strata-plugin-client-test", MFD_CLOEXEC);
            if (m >= 0) {
                if (::ftruncate(m, (off_t) SharedControl::kControlBytes) != 0) { ::close(c); ::close(m); return; }
                SharedControl ctl{};
                ctl.magic = SharedControl::kMagic;
                ctl.version = SharedControl::kVersion;
                ctl.state = SharedControl::kStatePlugin;
                ctl.bytes = SharedControl::kControlBytes;
                ctl.table_offset = SharedControl::kTableOffset;
                ctl.table_capacity = SharedControl::kTableCapacity;
                if (::pwrite(m, &ctl, sizeof(ctl), 0) != (ssize_t) sizeof(ctl)) { ::close(c); ::close(m); return; }
                iovec iov{const_cast<char*>(b.reply), rlen};
                alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))];
                msghdr msg{};
                msg.msg_iov = &iov;
                msg.msg_iovlen = 1;
                msg.msg_control = control;
                msg.msg_controllen = sizeof(control);
                cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
                cmsg->cmsg_level = SOL_SOCKET;
                cmsg->cmsg_type = SCM_RIGHTS;
                cmsg->cmsg_len = CMSG_LEN(sizeof(int));
                *reinterpret_cast<int*>(CMSG_DATA(cmsg)) = m;
                (void) ::sendmsg(c, &msg, MSG_NOSIGNAL);
                ::close(m);
            }
        } else {
            (void) ::send(c, b.reply, rlen, MSG_NOSIGNAL);
        }
    }
    ::close(c);
}

// One handshake round: listener, a fake plugin thread, the client, the verdict.
bool run(int n, const std::string& engine, const Behavior& b, PluginShare& share,
         PluginClientError& err, std::string& detail) {
    const std::string path = socket_path(n);
    const int listener = make_listener(path);
    std::thread server(serve_once, listener, b);
    const bool ok = plugin_hello(path, engine, "{}", share, err, detail);
    server.join();
    ::close(listener);
    ::unlink(path.c_str());
    return ok;
}

}   // namespace

int main() {
    // The hello the client sends: protocol v3, op hello, the engine version, an empty identity (S1).
    {
        PluginShare share;
        PluginClientError err = PluginClientError::none;
        std::string detail;
        (void) run(0, "0.2.10-test", Behavior{}, share, err, detail);   // no reply: the request line is all this case needs
        check(g_request.find("\"op\":\"hello\"") != std::string::npos, "the hello names the op");
        check(g_request.find("\"protocol\":3") != std::string::npos, "the hello speaks protocol 3");
        check(g_request.find("\"engine\":\"0.2.10-test\"") != std::string::npos, "the hello names the engine");
        check(g_request.find("\"identity\":{}") != std::string::npos, "the hello carries the identity object");
        check(g_request.find("\"id\":\"1\"") != std::string::npos, "the hello carries an id");
    }

    // unreachable: nothing listens at the path.
    {
        PluginShare share;
        PluginClientError err = PluginClientError::none;
        std::string detail;
        const bool ok = plugin_hello("/tmp/strata-plugin-client-test-nowhere.sock", "0.2.10-test", "{}", share, err, detail);
        check(!ok, "unreachable: the handshake fails");
        check(err == PluginClientError::unreachable, "unreachable: the error kind");
        check(share.fd == -1, "unreachable: no fd");
    }

    // success: the reply plus one memfd (with the control block written), on the same sendmsg.
    {
        const char* reply = "{\"id\":\"1\",\"ok\":true,\"result\":{\"protocol\":3,\"plugin\":\"0.1.0\","
                            "\"share_base\":\"0x600000000000\",\"share_bytes\":4096}}\n";
        PluginShare share;
        PluginClientError err = PluginClientError::none;
        std::string detail;
        const bool ok = run(1, "0.2.10-test", Behavior{reply, true, 0}, share, err, detail);
        check(ok, "success: the handshake completes");
        if (ok) {
            check(err == PluginClientError::none, "success: the error kind");
            check(detail.empty(), "success: no detail");
            check(share.share_base == kPluginShareBase, "success: the announced base");
            check(share.share_bytes == SharedControl::kControlBytes, "success: the announced bytes");
            check(share.fd >= 0, "success: one memfd");
            // The fd is a real mapping: the control block the fake plugin wrote is there.
            void* map = ::mmap(nullptr, (size_t) share.share_bytes, PROT_READ, MAP_SHARED, share.fd, 0);
            check(map != MAP_FAILED, "success: the memfd maps");
            if (map != MAP_FAILED) {
                SharedControl ctl{};
                std::memcpy(&ctl, map, sizeof(ctl));
                check(ctl.magic == SharedControl::kMagic, "success: the control block's magic");
                check(ctl.version == SharedControl::kVersion, "success: the control block's version");
                check(ctl.table_offset == SharedControl::kTableOffset, "success: the control block's table offset");
                ::munmap(map, (size_t) share.share_bytes);
            }
            plugin_share_close(share);
            check(share.fd == -1, "success: plugin_share_close closed the fd");
            plugin_share_close(share);   // idempotent
        }
    }

    // version: the plugin refuses the hello (ok:false, code version) and sends no fd.
    {
        const char* reply = "{\"id\":\"1\",\"ok\":false,\"error\":{\"code\":\"version\",\"message\":\"plugin speaks protocol 2\"}}\n";
        PluginShare share;
        PluginClientError err = PluginClientError::none;
        std::string detail;
        const bool ok = run(2, "0.2.10-test", Behavior{reply, false, 0}, share, err, detail);
        check(!ok, "version: the handshake is refused");
        check(err == PluginClientError::version, "version: the error kind");
        check(detail.find("protocol 2") != std::string::npos, "version: the plugin's message is the detail");
        check(share.fd == -1, "version: no fd leaks");
    }

    // timeout: the plugin reads the hello and answers 3 s later - past the client's 2 s.
    {
        const char* reply = "{\"id\":\"1\",\"ok\":true,\"result\":{\"protocol\":3,\"share_base\":\"0x600000000000\",\"share_bytes\":4096}}\n";
        PluginShare share;
        PluginClientError err = PluginClientError::none;
        std::string detail;
        const bool ok = run(3, "0.2.10-test", Behavior{reply, true, 3000}, share, err, detail);
        check(!ok, "timeout: the handshake fails");
        check(err == PluginClientError::timeout, "timeout: the error kind");
        check(share.fd == -1, "timeout: no fd leaks");
    }

    // bad reply: ok:true but not the protocol v3 shape (no result at all), one fd riding along.
    {
        const char* reply = "{\"id\":\"1\",\"ok\":true}\n";
        PluginShare share;
        PluginClientError err = PluginClientError::none;
        std::string detail;
        const bool ok = run(4, "0.2.10-test", Behavior{reply, true, 0}, share, err, detail);
        check(!ok, "bad reply: the handshake fails");
        check(err == PluginClientError::bad_reply, "bad reply: the error kind");
        check(share.fd == -1, "bad reply: the riding fd does not leak");
    }

    // no fd: a perfectly shaped reply whose sendmsg carries no memfd.
    {
        const char* reply = "{\"id\":\"1\",\"ok\":true,\"result\":{\"protocol\":3,\"plugin\":\"0.1.0\","
                            "\"share_base\":\"0x600000000000\",\"share_bytes\":4096}}\n";
        PluginShare share;
        PluginClientError err = PluginClientError::none;
        std::string detail;
        const bool ok = run(5, "0.2.10-test", Behavior{reply, false, 0}, share, err, detail);
        check(!ok, "no fd: the handshake fails");
        check(err == PluginClientError::bad_reply, "no fd: the error kind");
        check(detail.find("no memfd") != std::string::npos, "no fd: the detail says so");
        check(share.fd == -1, "no fd: no fd");
    }

    std::fprintf(stderr, "conversation_plugin_client_test: %d checks passed\n", checks);
    return 0;
}
