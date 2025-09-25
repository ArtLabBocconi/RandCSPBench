#include <cstdlib>
struct S {
  int size;
  int flexible_array_member[];
};
int main () {
  struct S * s = (struct S*) malloc (12);
  s->size = 2;
  s->flexible_array_member[0] = 1;
  s->flexible_array_member[1] = -1;
  int res = 0;
  for (int i = 0; i != s->size; i++)
    res += s->flexible_array_member[i];
  return res;
}
