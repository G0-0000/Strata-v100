// The engine's client for the conversation cache plugin's control socket (protocol v3, S1 skeleton): one
// handshake - connect (1 s), send the hello (protocol, engine, identity), and receive the reply AND the
// share's memfd on the same sendmsg (SCM_RIGHTS, 2 s) - and nothing else.  No swap, no other ops: later steps
// add acquire/release over the same connection's protocol.
//
// Host code only (no CUDA).
#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

enum class PluginClientError {
    none,        // the handshake completed: share holds the base, the bytes and the memfd
    unreachable, // the socket path could not be connected (nothing listening, a stale path, a dead plugin)
    timeout,     // connect took over 1 s, or the hello reply took over 2 s
    version,     // the plugin refused the hello (an ok:false whose code is "version") or answered another protocol
    bad_reply,   // the reply is not the protocol v3 hello reply: the wrong shape, an ok:false of another code,
                 // no memfd with it, or more than one
};

const char* plugin_client_error_name(PluginClientError e);   // "unreachable", "timeout", "version", ...

// One memfd plus the numbers the hello reply announces.  fd is the caller's to dispose of
// (plugin_share_close, or plain close() once the share is mapped: the mapping outlives the fd).
struct PluginShare {
    uint64_t share_base = 0;
    uint64_t share_bytes = 0;
    int fd = -1;
};

// The handshake.  'identity_json' is the hello's identity object, verbatim: "{}" is allowed (S1 leaves it
// empty; v10 sends {"model":"<hex16>","config":"<hex16>"}).  On false, 'error' says why and 'detail' carries
// one line (the plugin's message, or what the reply was missing); on true, 'share' is filled and fd >= 0.
bool plugin_hello(const std::string& socket_path, const std::string& engine_version,
                  const std::string& identity_json, PluginShare& share, PluginClientError& error,
                  std::string& detail);

// Closes share.fd (idempotent: a second call is a no-op).  Never unmaps anything: closing the fd keeps a
// mapped share alive.
void plugin_share_close(PluginShare& share);

}   // namespace strata::core
