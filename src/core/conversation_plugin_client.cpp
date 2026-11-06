#include "strata/core/conversation_plugin_client.hpp"
#include "strata/core/shared_control.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace strata::core {

namespace {

constexpr int kConnectTimeoutMs = 1000;   // S1v3: 1 s to connect
constexpr int kHelloTimeoutMs = 2000;     // ... and 2 s for the hello's round trip
constexpr size_t kReplyMax = 4096;        // one NDJSON line; the plugin's message fits easily
constexpr int kMaxFds = 4;                // the protocol sends ONE memfd; more is a bad reply (all closed)

const char* error_name(PluginClientError e) {
    switch (e) {
        case PluginClientError::none: return "none";
        case PluginClientError::unreachable: return "unreachable";
        case PluginClientError::timeout: return "timeout";
        case PluginClientError::version: return "version";
        case PluginClientError::bad_reply: return "bad_reply";
    }
    return "?";
}

void close_received_fds(int* fds, int count) {
    for (int i = 0; i < count; ++i)
        if (fds[i] >= 0) {
            ::close(fds[i]);
            fds[i] = -1;
        }
}

// A flat scanner is enough: the reply's keys are unique per line (the plugin's message text is the only free
// text and sits after the keys it could shadow), so find `"key"`, step over the colon, return the value's
// first byte.  A match inside an escaped string is skipped via the backslash check (the byte before a real
// key's quote is `{`, `,`, `:` or space, never `\`).
const char* find_value(const char* json, const char* key) {
    char pattern[64];
    if (std::snprintf(pattern, sizeof(pattern), "\"%s\"", key) >= (int) sizeof(pattern)) return nullptr;
    const size_t plen = std::strlen(pattern);
    for (const char* p = json; (p = std::strstr(p, pattern)) != nullptr; p += plen) {
        if (p > json && p[-1] == '\\') continue;   // inside a string literal (an escaped quote)
        const char* v = p + plen;
        while (*v == ' ' || *v == ':') ++v;
        if (*v != '\0') return v;
    }
    return nullptr;
}

// Numbers: bare (share_bytes, protocol) or a quoted string with an integer in it (share_base as "0x...").
bool parse_u64(const char* json, const char* key, uint64_t& out) {
    const char* v = find_value(json, key);
    if (v == nullptr) return false;
    if (*v == '"') {
        ++v;
        char* end = nullptr;
        const unsigned long long n = std::strtoull(v, &end, 0);   // base 0: "0x..." reads as hex
        if (end == v || (*end != '"' && *end != ' ' && *end != ',' && *end != '}')) return false;
        out = (uint64_t) n;
        return true;
    }
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v, &end, 10);
    if (end == v) return false;
    out = (uint64_t) n;
    return true;
}

bool parse_bool(const char* json, const char* key, bool& out) {
    const char* v = find_value(json, key);
    if (v == nullptr) return false;
    if (std::strncmp(v, "true", 4) == 0) { out = true; return true; }
    if (std::strncmp(v, "false", 5) == 0) { out = false; return true; }
    return false;
}

bool parse_string(const char* json, const char* key, std::string& out) {
    const char* v = find_value(json, key);
    if (v == nullptr || *v != '"') return false;
    ++v;
    const char* end = v;
    while (*end != '\0' && *end != '"') ++end;
    if (*end != '"') return false;
    out.assign(v, (size_t) (end - v));
    return true;
}

}   // namespace

const char* plugin_client_error_name(PluginClientError e) {
    return e == PluginClientError::none ? "none" : error_name(e);
}

void plugin_share_close(PluginShare& share) {
    if (share.fd >= 0) {
        ::close(share.fd);
        share.fd = -1;
    }
}

bool plugin_hello(const std::string& socket_path, const std::string& engine_version,
                  const std::string& identity_json, PluginShare& share, PluginClientError& error,
                  std::string& detail) {
    share = PluginShare{};
    error = PluginClientError::unreachable;
    detail.clear();

    if (socket_path.empty()) {
        detail = "no socket path given";
        return false;
    }

    // --- connect (nonblocking + poll, 1 s) ---
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        detail = std::string("socket: ") + std::strerror(errno);
        return false;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        detail = "socket path too long";
        return false;
    }
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
    const int flags = ::fcntl(fd, F_GETFL);
    if (flags >= 0) (void) ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int rc = ::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
    if (rc != 0 && errno == EINPROGRESS) {
        pollfd p{fd, POLLOUT, 0};
        rc = ::poll(&p, 1, kConnectTimeoutMs);
        if (rc == 0) {
            ::close(fd);
            error = PluginClientError::timeout;
            detail = "connect timed out (1 s)";
            return false;
        }
        if (rc < 0) {
            const int saved = errno;
            ::close(fd);
            detail = std::string("connect poll: ") + std::strerror(saved);
            return false;
        }
        int soerr = 0;
        socklen_t slen = sizeof(soerr);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &slen) != 0) soerr = errno;
        if (soerr != 0) {
            ::close(fd);
            detail = std::string("connect: ") + std::strerror(soerr);
            return false;
        }
        rc = 0;
    }
    if (rc != 0) {
        const int saved = errno;
        ::close(fd);
        detail = std::string("connect: ") + std::strerror(saved);
        return false;
    }

    // --- send the hello (one NDJSON line) ---
    // The engine version and the identity object come from this repo's own code, so neither carries a byte
    // that needs JSON escaping (digits, dots, braces; no quotes, no backslashes).
    const std::string hello = std::string("{\"id\":\"1\",\"op\":\"hello\",\"protocol\":") +
                              std::to_string(SharedControl::kVersion) + ",\"engine\":\"" + engine_version +
                              "\",\"identity\":" + identity_json + "}\n";
    size_t sent = 0;
    while (sent < hello.size()) {
        pollfd p{fd, POLLOUT, 0};
        const int pr = ::poll(&p, 1, kHelloTimeoutMs);
        if (pr == 0) {
            ::close(fd);
            error = PluginClientError::timeout;
            detail = "the hello send timed out (2 s)";
            return false;
        }
        if (pr < 0 && errno != EINTR) {
            const int saved = errno;
            ::close(fd);
            detail = std::string("send poll: ") + std::strerror(saved);
            return false;
        }
        const ssize_t n = ::write(fd, hello.data() + sent, hello.size() - sent);
        if (n < 0) {
            if (errno == EINTR) continue;
            const int saved = errno;
            ::close(fd);
            detail = std::string("send: ") + std::strerror(saved);
            return false;
        }
        sent += (size_t) n;
    }

    // --- receive the reply AND the memfd on the same socket (SCM_RIGHTS), 2 s ---
    char reply[kReplyMax];
    size_t len = 0;
    bool line_done = false;
    int fds[kMaxFds];
    int nfd = 0;
    for (int i = 0; i < kMaxFds; ++i) fds[i] = -1;
    while (!line_done) {
        pollfd p{fd, POLLIN, 0};
        const int pr = ::poll(&p, 1, kHelloTimeoutMs);
        if (pr == 0) {
            ::close(fd);
            close_received_fds(fds, kMaxFds);
            error = PluginClientError::timeout;
            detail = "the hello reply timed out (2 s)";
            return false;
        }
        if (pr < 0 && errno != EINTR) {
            const int saved = errno;
            ::close(fd);
            close_received_fds(fds, kMaxFds);
            detail = std::string("recv poll: ") + std::strerror(saved);
            return false;
        }
        alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * kMaxFds)];
        iovec iov{reply + len, sizeof(reply) - len};
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        const ssize_t n = ::recvmsg(fd, &msg, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            const int saved = errno;
            ::close(fd);
            close_received_fds(fds, kMaxFds);
            detail = std::string("recvmsg: ") + std::strerror(saved);
            return false;
        }
        if (n == 0) {
            ::close(fd);
            close_received_fds(fds, kMaxFds);
            detail = "the plugin closed the connection before the reply";
            return false;
        }
        len += (size_t) n;
        for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c)) {
            if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
                const size_t count = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                const int* pfd = reinterpret_cast<const int*>(CMSG_DATA(c));
                for (size_t i = 0; i < count && nfd < kMaxFds; ++i) fds[nfd++] = pfd[i];
                for (size_t i = kMaxFds; i < count; ++i) ::close(pfd[i]);   // over the cap: close on sight
            }
        }
        if (std::memchr(reply, '\n', len) != nullptr) line_done = true;
        else if (len >= sizeof(reply)) {
            ::close(fd);
            close_received_fds(fds, kMaxFds);
            error = PluginClientError::bad_reply;
            detail = "the hello reply exceeds the 4 KiB line cap";
            return false;
        }
    }
    ::close(fd);
    if (len >= sizeof(reply)) len = sizeof(reply) - 1;   // keep the NUL inside the buffer
    reply[len] = '\0';

    // --- parse the reply (the fd check comes after ok: a refusal carries no memfd) ---
    bool ok = false;
    if (!parse_bool(reply, "ok", ok)) {
        close_received_fds(fds, kMaxFds);
        error = PluginClientError::bad_reply;
        detail = "the reply has no ok";
        return false;
    }
    if (!ok) {
        std::string code, message;
        parse_string(reply, "code", code);
        parse_string(reply, "message", message);
        close_received_fds(fds, kMaxFds);
        if (code == "version") {
            error = PluginClientError::version;
            detail = message.empty() ? "the plugin refused the hello (code version)" : message;
        } else {
            error = PluginClientError::bad_reply;
            detail = "ok:false, code " + (code.empty() ? std::string("(none)") : code);
        }
        return false;
    }
    if (nfd != 1) {
        close_received_fds(fds, kMaxFds);
        error = PluginClientError::bad_reply;
        detail = nfd == 0 ? "no memfd came with the hello reply"
                          : "more than one fd came with the hello reply";
        return false;
    }
    uint64_t protocol = 0;
    if (!parse_u64(reply, "protocol", protocol)) {
        ::close(fds[0]);
        fds[0] = -1;
        error = PluginClientError::bad_reply;
        detail = "the reply's result lacks a protocol";
        return false;
    }
    if (protocol != SharedControl::kVersion) {
        ::close(fds[0]);
        fds[0] = -1;
        error = PluginClientError::version;
        detail = "the plugin speaks protocol " + std::to_string(protocol);
        return false;
    }
    if (!parse_u64(reply, "share_base", share.share_base) || !parse_u64(reply, "share_bytes", share.share_bytes) ||
        share.share_base == 0 || share.share_bytes == 0) {
        ::close(fds[0]);
        fds[0] = -1;
        error = PluginClientError::bad_reply;
        detail = "the reply's result lacks a share_base/share_bytes";
        return false;
    }
    share.fd = fds[0];
    fds[0] = -1;
    error = PluginClientError::none;
    return true;
}

}   // namespace strata::core
