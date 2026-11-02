# Captures the standard measurement baseline (guideline G-001).
#
# Run before the first edit of a task and again after every change, then diff
# the two outputs:
#
#   .\scripts\measure.ps1 > before.txt
#   ... change ...
#   .\scripts\measure.ps1 > after.txt
#   Compare-Object (Get-Content before.txt) (Get-Content after.txt)
#
# The output is one line per measurement so the diff stays readable.

$ErrorActionPreference = 'Stop'

# Sourcing the toolchain script is a side effect on PATH; its banner and version
# lines would otherwise pollute the snapshot. All streams are discarded; the
# PATH change still applies.
. (Join-Path $PSScriptRoot 'env.ps1') *>$null

$exe = Join-Path $PSScriptRoot '..\build\release\bin\tge.exe'
if (-not (Test-Path $exe)) {
    throw 'no binary found; build first:  cmake --build --preset release'
}

$exe = (Resolve-Path $exe).Path

# Which kernels this machine selected. A kernel change shows up here first.
$kernels = & $exe info | Select-String 'gemm f32|gemv f32|attention|streaming|threads'
foreach ($line in $kernels) {
    "kernel  $($line.Line.Trim())"
}

# Prefill kernel, per shape the model actually uses.
$shapes = @(
    @(896, 896, 'attn'),
    @(4864, 896, 'ffn'),
    @(151936, 896, 'output')
)
foreach ($shape in $shapes) {
    $out = & $exe bench batch --rows $shape[0] --cols $shape[1] --tokens 64 --repeat 3
    $result = $out | Select-String 'GFLOPS'
    $value = $result.Line.Trim()
    "batch   {0,-7} {1}" -f $shape[2], $value
}

# Decode-shaped K-quant kernels, 1 token: the orientation each format
# actually takes in the model (gate/up Q5_0, down-proj Q4_K/Q6_K). Scalar
# fallbacks would show up here first.
$kshapes = @(
    @(4864, 896, 'q50', 'gate'),
    @(896, 4864, 'q4k', 'down4'),
    @(896, 4864, 'q6k', 'down6')
)
foreach ($shape in $kshapes) {
    $out = & $exe bench batch --rows $shape[0] --cols $shape[1] --tokens 1 --repeat 3 --format $shape[2]
    $result = $out | Select-String 'GB/s'
    $value = $result.Line.Trim()
    "kbatch  {0,-5} {1}" -f $shape[3], $value
}

# Decode primitive and the bandwidth ceiling.
$gemv = & $exe bench gemv --rows 4096 --cols 4096 --repeat 3 | Select-String 'avx2-fma\s+16\s'
"gemv    {0}" -f $gemv.Line.Trim()

$mem = & $exe bench mem --size-mb 256 --repeat 3
$read = $mem | Select-String 'read \(sum\)'
"mem     {0}" -f $read.Line.Trim()
