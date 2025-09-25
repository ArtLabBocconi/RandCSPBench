#include <cstdio>
#include <vector>

// old variadic macro usage 'ARGS...' / '#ARGS' discouraged in g++-7...
// new variadic macro usage '...' / '__VA_ARGS__' available in C99/C++11
//
#define MACRO(FMT, ...) printf (FMT "\n", __VA_ARGS__)

// we use ranged for loops which became available in gcc 4.6 and for
// the gcc 4.6 as well as 4.8 requires '-std=c++11' too.
//
unsigned f (const std::vector<unsigned> & a) {
  unsigned res = 0;
  for (auto i : a) res += i;
  return res;
}

int main () { MACRO ("%d", 42); return 0; }
