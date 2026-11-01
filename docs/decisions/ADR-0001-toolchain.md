# ADR-0001: GCC first, MSVC when an admin install is available

- Status: accepted
- Date: 2026-11-01

## Context

The specification names MSVC as the primary compiler with Clang as an
alternative. The development machine had no C++ compiler at all: Visual Studio
Build Tools 2022 was present but carried only MSBuild, with no VC tools and no
Windows SDK. The shell is not elevated, so a system-wide MSVC install would
require UAC that an automated agent cannot answer.

## Decision

Build the engine with MinGW-w64 GCC 16.2 (UCRT, POSIX threads) installed
per-user through winget, and keep the source MSVC-clean so a Build Tools install
can be added later without code changes.

Measures taken to keep both compilers viable:

- CPUID goes through a small wrapper that uses `__cpuidex`/`_xgetbv` on MSVC and
  `<cpuid.h>`/inline asm on GCC.
- ISA flags are applied per translation unit through `tge_set_isa_flags`, which
  emits `/arch:AVX2` or `-mavx2 -mfma` as appropriate.
- Executables link statically under MinGW so they do not depend on runtime DLLs
  being on `PATH`.

## Consequences

- The engine builds and its tests run today, which was otherwise impossible.
- HIP requires MSVC on Windows, so GPU milestones stay blocked until an elevated
  install happens.
- MSVC-specific code generation is untested. Any performance claim is a GCC
  claim until MSVC builds exist.
- GCC is a supported rather than preferred configuration; the benchmark numbers
  in `docs/benchmarks.md` are labelled accordingly.
