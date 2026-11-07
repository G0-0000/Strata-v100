#include <cstdio>
#include <cstring>
#include <string>
#include "store_index.hpp"
using namespace strata::cache;
int main() {
  const std::string store = "/tmp/s3/store2";
  std::FILE* file = std::fopen(IndexPath(store).c_str(), "rb");
  std::string text;
  char chunk[65536];
  size_t got = 0;
  while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0) text.append(chunk, got);
  std::fclose(file);
  std::size_t nl = text.find('\n');
  const char* td = text.data();
  std::string_view a(td, nl);
  std::string_view b = text.substr(0, nl);
  std::printf("nl=%zu text.data=%p a.data=%p b.data=%p\n", nl, td, (void*)a.data(), (void*)b.data());
  std::printf("a[0..15]: %.15s\n", a.data());
  std::printf("b[0..15]: %.15s\n", b.data());
  std::printf("a tail: %.30s\n", a.data() + a.size() - 30);
  std::printf("b tail: %.30s\n", b.data() + b.size() - 30);
  return 0;
}
