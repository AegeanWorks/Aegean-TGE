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
option(TGE_ENABLE_VULKAN "Vulkan GPU backend (needs SDK headers + glslc)" ON)

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
    set(_flags "-mavx2;-mfma;-mf16c")
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
  src/models/gguf.cpp
  src/models/qwen2.cpp
  src/models/sampling.cpp
  src/models/tokenizer.cpp
  src/models/plan.cpp
  src/gpu/noop.cpp
  src/cpu/avx2/gemm_avx2.cpp
  src/cpu/avx2/gemm_epilogue_avx2.cpp
  src/cpu/avx2/gemm_unpacked.cpp
  src/cpu/avx2/gemv_avx2.cpp
  src/cpu/avx2/q8_gemv_avx2.cpp
  src/cpu/avx2/q8_batch_avx2.cpp
  src/cpu/avx2/q8_batch_vnni.cpp
  src/cpu/avx2/dequant_batch_avx2.cpp
  src/cpu/avx2/dequant_batch_kq8_avx2.cpp
  src/cpu/avx2/dequant_avx2.cpp
  src/cpu/avx2/attention_avx2.cpp
  src/cpu/avx2/norm_rope_avx2.cpp
  src/cpu/avx2/sampler_exp_avx2.cpp
  src/cpu/avx2/mem_bandwidth_avx2.cpp
  src/cpu/scalar/mem_bandwidth_scalar.cpp
  src/cpu/scalar/q8_gemv_scalar.cpp
  src/cpu/scalar/q8_batch_scalar.cpp
  src/cpu/scalar/dequant.cpp
  src/cpu/scalar/dequant_batch_scalar.cpp
  src/cpu/scalar/attention_scalar.cpp
  src/cpu/scalar/norm_rope_scalar.cpp
)

target_include_directories(tge_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

tge_set_isa_flags(AVX2
  src/cpu/avx2/gemm_avx2.cpp
  src/cpu/avx2/gemm_epilogue_avx2.cpp
  src/cpu/avx2/gemm_unpacked.cpp
  src/cpu/avx2/gemv_avx2.cpp
  src/cpu/avx2/q8_gemv_avx2.cpp
  src/cpu/avx2/q8_batch_avx2.cpp
  src/cpu/avx2/dequant_batch_avx2.cpp
  src/cpu/avx2/dequant_batch_kq8_avx2.cpp
  src/cpu/avx2/dequant_avx2.cpp
  src/cpu/avx2/attention_avx2.cpp
  src/cpu/avx2/norm_rope_avx2.cpp
  src/cpu/avx2/sampler_exp_avx2.cpp
  src/cpu/avx2/mem_bandwidth_avx2.cpp)

# AVX-VNNI is a separate tier. Reachable from the benchmark harness only;
# dispatch still selects the float kernel.
if(MSVC)
  set_source_files_properties(src/cpu/avx2/q8_batch_vnni.cpp
    PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(src/cpu/avx2/q8_batch_vnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mavxvnni")
endif()

find_package(Threads REQUIRED)
target_link_libraries(tge_core PUBLIC Threads::Threads)

# Portable GPU backend: the Vulkan SDK (headers + glslc) is optional. Without
# it the engine builds and runs CPU-only; src/gpu/noop.cpp reports
# Unavailable and every eligibility check keeps the CPU kernels.
if(TGE_ENABLE_VULKAN)
  find_package(Vulkan QUIET)
endif()
if(Vulkan_FOUND)
  target_sources(tge_core PRIVATE src/gpu/vulkan_context.cpp)
  target_link_libraries(tge_core PUBLIC Vulkan::Vulkan)
  target_compile_definitions(tge_core PUBLIC TGE_HAS_VULKAN=1)

  # Shader pipeline: glslc compiles each .comp to SPIR-V, then bin2h.cmake
  # embeds it as an aligned byte array so the binary stays single-file.
  # The SDK installer does not put Bin on PATH, so look there explicitly.
  find_program(TGE_GLSLC glslc
    PATHS "$ENV{VULKAN_SDK}/Bin" "C:/VulkanSDK/1.4.363.0/Bin"
    NO_DEFAULT_PATH)
  find_program(TGE_GLSLC glslc)
  if(NOT TGE_GLSLC)
    message(FATAL_ERROR "TGE_ENABLE_VULKAN is on and Vulkan was found, but "
                        "glslc is missing; install the SDK Bin tools or set "
                        "TGE_ENABLE_VULKAN=OFF")
  endif()
  set(TGE_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/tge/gpu")
  file(MAKE_DIRECTORY "${TGE_SPV_DIR}")
  set(TGE_GPU_SHADERS "")
  foreach(shader noop q8_gemv q8_gemm)
    # Embedded symbol is k + CamelCase + Spv (kNoopSpv, kQ8GemvSpv).
    string(REPLACE "_" ";" name_parts "${shader}")
    set(capped "")
    foreach(part IN LISTS name_parts)
      string(SUBSTRING "${part}" 0 1 first_letter)
      string(TOUPPER "${first_letter}" FIRST_LETTER)
      string(SUBSTRING "${part}" 1 -1 shader_rest)
      set(capped "${capped}${FIRST_LETTER}${shader_rest}")
    endforeach()
    set(spv_var "k${capped}Spv")
    set(comp "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders/${shader}.comp")
    set(spv "${TGE_SPV_DIR}/${shader}.spv")
    # Flat in the generated root: the TU includes it bare and tge_core gets
    # the generated root on its private include path.
    set(header "${CMAKE_CURRENT_BINARY_DIR}/generated/tge_gpu_${shader}_spv.h")
    add_custom_command(
      OUTPUT "${spv}" "${header}"
      COMMAND "${TGE_GLSLC}" -fshader-stage=comp -O -o "${spv}" "${comp}"
      COMMAND "${CMAKE_COMMAND}" -DINPUT="${spv}" -DOUTPUT="${header}"
              -DVAR=${spv_var} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake"
      DEPENDS "${comp}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake"
      COMMENT "Compiling + embedding GPU shader ${shader}.comp")
    list(APPEND TGE_GPU_SHADERS "${spv}" "${header}")
  endforeach()
  add_custom_target(tge_gpu_shaders ALL DEPENDS ${TGE_GPU_SHADERS})
  add_dependencies(tge_core tge_gpu_shaders)
  target_include_directories(tge_core PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
else()
  target_compile_definitions(tge_core PUBLIC TGE_HAS_VULKAN=0)
endif()

# MinGW's ar can leave the archive index stale when only some members change;
# refresh it so the linker sees every kernel symbol after an incremental build.
if(CMAKE_RANLIB)
  add_custom_command(TARGET tge_core POST_BUILD
    COMMAND "${CMAKE_RANLIB}" "$<TARGET_FILE:tge_core>"
    COMMENT "Refreshing tge_core archive index")
endif()

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
    tests/test_dequant.cpp
    tests/test_cpu.cpp
    tests/test_gemm_epilogue.cpp
    tests/test_gguf.cpp
    tests/test_memory.cpp
    tests/test_q8.cpp
    tests/test_sampling.cpp
    tests/test_tensor.cpp
    tests/test_thread_pool.cpp
    tests/test_attention.cpp
    tests/test_block_table.cpp
    tests/test_io_advice.cpp
    tests/test_norm_rope.cpp
    tests/test_gpu_fallback.cpp
    tests/test_cli_options.cpp
  )
  target_link_libraries(tge_tests PRIVATE tge_core)
  target_include_directories(tge_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/apps/cli")
  add_test(NAME tge_tests COMMAND tge_tests)
endif()
