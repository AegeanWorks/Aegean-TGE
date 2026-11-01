#include <cstdio>

#include "harness.hpp"

int main() {
  int failed_cases = 0;

  for (const tgetest::Case& test_case : tgetest::registry()) {
    const int before = tgetest::failure_count();
    std::printf("[ RUN  ] %s\n", test_case.name);
    // Flush before running: a hanging case must name itself in the log.
    std::fflush(stdout);
    test_case.fn();
    if (tgetest::failure_count() == before) {
      std::printf("[  OK  ] %s\n", test_case.name);
    } else {
      std::printf("[ FAIL ] %s\n", test_case.name);
      ++failed_cases;
    }
    std::fflush(stdout);
  }

  const std::size_t total = tgetest::registry().size();
  std::printf("\n%zu cases, %d failed, %d assertions failed\n", total,
              failed_cases, tgetest::failure_count());
  return failed_cases == 0 ? 0 : 1;
}
