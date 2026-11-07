#include <cstdio>
#include <string>
#include "store_index.hpp"
using namespace strata::cache;
int main() {
  IndexEntry e;
  e.name = "strata-conv-7.sess";
  e.path = "/tmp/s3/store/strata-conv-7.sess";
  e.model = "0123456789abcdef";
  e.config = "fedcba9876543210";
  e.tokens = 6; e.digest = "0011223344556677"; e.bytes = 1234; e.saved_unix = 42;
  const std::string line = SerializeIndexLine(e);
  std::string_view view(line);
  std::size_t at = 0;
  SkipIndexSpace(view, &at);
  ++at;  // {
  for (;;) {
    SkipIndexSpace(view, &at);
    std::string key;
    if (!protocol::ParseJsonString(view, &at, &key)) { std::printf("key fail at %zu\n", at); return 1; }
    SkipIndexSpace(view, &at);
    if (at >= view.size() || view[at] != ':') { std::printf("colon fail at %zu after %s\n", at, key.c_str()); return 1; }
    ++at;
    SkipIndexSpace(view, &at);
    std::uint64_t n = 0;
    if (!ParseIndexUnsigned(view, &at, &n)) {
      std::string s;
      std::size_t probe = at;
      if (!protocol::ParseJsonString(view, &at, &s)) { std::printf("value fail at %zu after %s\n", at, key.c_str()); return 1; }
      std::printf("key %s = \"%s\"\n", key.c_str(), s.c_str());
    } else {
      std::printf("key %s = %llu\n", key.c_str(), (unsigned long long)n);
    }
    SkipIndexSpace(view, &at);
    if (at < view.size() && view[at] == ',') { ++at; continue; }
    if (at < view.size() && view[at] == '}') { ++at; break; }
    std::printf("tail fail at %zu char '%c'\n", at, view[at]);
    return 1;
  }
  SkipIndexSpace(view, &at);
  std::printf("end: at=%zu size=%zu\n", at, view.size());
  return 0;
}
