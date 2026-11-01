cmake_minimum_required(VERSION 3.25)

project(aegean_tge
  VERSION 0.1.0
  DESCRIPTION "Aegean Token Generation Engine - native Windows hybrid CPU/GPU inference"
  LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(TGE_BUILD_TESTS "Build the correctness test suite" ON)

if(MSVC)
  add_compile_options(/W4 /permissive- /EHsc /utf-8 /Zc:__cplusplus)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Ship self-contained executables: the engine is deployed as a standalone
# Windows binary, so the C++ runtime is linked in rather than resolved by PATH.
if(MINGW)
  add_link_options(-static -static-libgcc -static-libstdc++)
endif()

# Runtime dispatch requires that ISA-specific translation units are the only
# ones compiled with extended instruction sets.
function(tge_set_isa_flags isa)
  if(MSVC)
    set(_flags "/arch:AVX2")
  else()
    set(_flags "-mavx2;-mfma")
  endif()
  set_source_files_properties(${ARGN} PROPERTIES COMPILE_OPTIONS "${_flags}")
endfunction()

add_library(tge_core STATIC
  src/core/cpu.cpp
  src/core/dispatch.cpp
  src/core/thread_pool.cpp
  src/memory/system_ram.cpp
  src/memory/allocator.cpp
  src/memory/budget.cpp
  src/tensor/reference.cpp
  src/cpu/avx2/gemm_avx2.cpp
  src/cpu/avx2/mem_bandwidth_avx2.cpp
)

target_include_directories(tge_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

tge_set_isa_flags(AVX2
  src/cpu/avx2/gemm_avx2.cpp
  src/cpu/avx2/mem_bandwidth_avx2.cpp)

find_package(Threads REQUIRED)
target_link_libraries(tge_core PUBLIC Threads::Threads)

target_compile_definitions(tge_core PUBLIC TGE_VERSION="${PROJECT_VERSION}")

if(WIN32)
  target_compile_definitions(tge_core PUBLIC _WIN32_WINNT=0x0A00)
endif()

add_executable(tge apps/cli/main.cpp)
target_link_libraries(tge PRIVATE tge_core)

if(TGE_BUILD_TESTS)
  enable_testing()
  add_executable(tge_tests
    tests/run_tests.cpp
    tests/test_cpu.cpp
    tests/test_memory.cpp
    tests/test_tensor.cpp
    tests/test_thread_pool.cpp
  )
  target_link_libraries(tge_tests PRIVATE tge_core)
  add_test(NAME tge_tests COMMAND tge_tests)
endif()
