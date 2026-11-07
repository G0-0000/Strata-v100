#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "store_index.hpp"
using namespace strata::cache;
int main() {
  const std::string store = "/tmp/s3/store2";
  std::FILE* file = std::fopen(IndexPath(store).c_str(), "rb");
  if (!file) { std::printf("open fail: %s\n", std::strerror(errno)); return 1; }
  std::string text;
  char chunk[65536];
  std::size_t got = 0;
  while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0) text.append(chunk, got);
  bool read_ok = (std::ferror(file) == 0);
  std::printf("text=%zu ferror=%d read_ok=%d\n", text.size(), std::ferror(file), read_ok);
  std::fclose(file);
  if (!read_ok) return 1;
  std::size_t at = 0;
  while (at < text.size()) {
    const std::size_t newline = text.find('\n', at);
    const std::string_view line = text.substr(at, newline == std::string::npos ? std::string::npos : newline - at);
    at = newline == std::string::npos ? text.size() : newline + 1;
    std::printf("line len=%zu blank=%d\n", line.size(), line.find_first_not_of(" \t\r") == std::string::npos);
    IndexEntry entry;
    bool ok = ParseIndexLine(line, &entry);
    std::printf("parse=%d name=%s\n", ok, entry.name.c_str());
  }
  return 0;
}
