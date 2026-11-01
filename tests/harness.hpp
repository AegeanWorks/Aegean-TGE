#pragma once

#include <cmath>
#include <cstdio>
#include <vector>

namespace tgetest {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline int& failure_count() {
  static int count = 0;
  return count;
}

inline void fail(const char* expression, const char* file, int line) {
  std::fprintf(stderr, "    FAIL %s\n         at %s:%d\n", expression, file,
               line);
  ++failure_count();
}

}  // namespace tgetest

#define TGE_CHECK(expr)                           \
  do {                                            \
    if (!(expr)) {                                \
      ::tgetest::fail(#expr, __FILE__, __LINE__); \
    }                                             \
  } while (false)

#define TGE_REQUIRE(expr)                         \
  do {                                            \
    if (!(expr)) {                                \
      ::tgetest::fail(#expr, __FILE__, __LINE__); \
      return;                                     \
    }                                             \
  } while (false)

#define TGE_CHECK_NEAR(actual, expected, tolerance)                          \
  do {                                                                       \
    const double tge_actual = static_cast<double>(actual);                   \
    const double tge_expected = static_cast<double>(expected);               \
    const double tge_tolerance = static_cast<double>(tolerance);             \
    if (!(std::fabs(tge_actual - tge_expected) <= tge_tolerance)) {          \
      std::fprintf(stderr,                                                   \
                   "    %s = %.9g, expected %s = %.9g (tolerance %.3g)\n",   \
                   #actual, tge_actual, #expected, tge_expected,             \
                   tge_tolerance);                                           \
      ::tgetest::fail(#actual " near " #expected, __FILE__, __LINE__);       \
    }                                                                        \
  } while (false)

#define TGE_TEST(name)                                           \
  static void name();                                            \
  static ::tgetest::Registrar tge_registrar_##name(#name, name); \
  static void name()
