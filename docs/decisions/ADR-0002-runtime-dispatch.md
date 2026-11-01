# ADR-0002: Runtime dispatch gated on XCR0, not on CPUID alone

- Status: accepted
- Date: 2026-11-01

## Context

A CPU can report AVX2 in CPUID while the operating system has not enabled the
YMM register state. Executing AVX2 in that configuration is not a slow path, it
is an illegal instruction.

Environments where this occurs in practice: hypervisors that mask the upper
register file, older kernels, and virtual machines with a mismatched CPU model.
Both are plausible for a tool users will run on arbitrary hardware.

## Decision

Report a vector feature only when both conditions hold:

1. CPUID advertises it.
2. `OSXSAVE` is set and XCR0 has the corresponding bits enabled - bits 1 and 2
   for AVX, bits 5, 6 and 7 additionally for AVX-512.

Detection stores the filtered result. Dispatch reads only the filtered result,
so no downstream code has to re-check.

## Consequences

- The reference kernel is selected on machines that advertise but cannot execute
  AVX2. `tge info` shows which kernel was chosen, so the situation is visible
  rather than mysterious.
- Kernel selection happens once and is exposed in diagnostics. There is no
  silent per-call fallback; a caller that requested GPU or AVX2 execution is
  never quietly downgraded.
- The same filtering will be applied to AMX, which has an analogous OS-managed
  state (XTILECFG).
