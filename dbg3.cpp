#include <cstdio>
#include <cstring>
#include <string>
#include "store_index.hpp"
using namespace strata::cache;
int main() {
  const std::string store = "/tmp/s3/store2";
  IndexEntry e;
  e.name = "strata-conv-7.sess";
  e.path = store + "/strata-conv-7.sess";
  e.model = "0123456789abcdef";
  e.config = "fedcba9876543210";
  e.tokens = 6; e.digest = "0011223344556677"; e.bytes = 1234; e.saved_unix = 42;
  std::string error;
  if (!RecordSave(store, e, &error)) { std::printf("record fail: %s\n", error.c_str()); return 1; }
  std::FILE* f = std::fopen((store + "/index.jsonl").c_str(), "rb");
  char buf[4096]; size_t got = std::fread(buf, 1, sizeof buf, f); std::fclose(f);
  std::printf("raw (%zu bytes):\n", got);
  for (size_t i = 0; i < got; ++i) {
    unsigned char c = buf[i];
    std::printf("%c", (c >= 0x20 && c < 0x7f) ? c : (c == '\n' ? '\n' : '?'));
  }
  std::vector<IndexEntry> entries; bool missing = false;
  bool ok = ReadIndex(store, &entries, &missing);
  std::printf("read=%d missing=%d count=%zu\n", ok, missing, entries.size());
  // parse the exact slice ReadIndex would hand over
  std::string text(buf, got);
  std::size_t nl = text.find('\n');
  std::string_view line(text.data(), nl);
  IndexEntry back;
  std::printf("slice parse: %d\n", ParseIndexLine(line, &back));
  IndexEntry back2;
  std::printf("full-with-nl parse: %d\n", ParseIndexLine(text, &back2));
  return 0;
}
