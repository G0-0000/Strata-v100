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
  std::string_view a(text.data(), nl);                 // dbg3 style
  std::string_view b = text.substr(0, nl);             // dbg4 style
  std::printf("a.size=%zu b.size=%zu data equal=%d\n", a.size(), b.size(),
              a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()) == 0);
  IndexEntry e1, e2;
  std::printf("a parse=%d\n", ParseIndexLine(a, &e1));
  std::printf("b parse=%d\n", ParseIndexLine(b, &e2));
  std::string_view c = a;  // copy of the dbg3-style view
  std::printf("c parse=%d\n", ParseIndexLine(c, &e2));
  // and a fresh std::string copy
  std::string copy(a);
  std::string_view d(copy);
  std::printf("d parse=%d\n", ParseIndexLine(d, &e2));
  return 0;
}
