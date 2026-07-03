param([ValidateRange(1, 100000)][int]$Runs = 100,
      [ValidateRange(1, 100000)][int]$Warmup = 10,
      [string]$OutputPath = 'evaluation\cuda_benchmark.json')
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Executable = Join-Path $ProjectRoot 'build\BenchmarkPreprocess.exe'
if (-not (Test-Path -LiteralPath $Executable)) { throw 'Build first: powershell -File scripts\build_cuda.ps1' }
if (-not [IO.Path]::IsPathRooted($OutputPath)) { $OutputPath = Join-Path $ProjectRoot $OutputPath }
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $OutputPath) {
    throw "Report already exists: $OutputPath. Select a new -OutputPath to preserve measured reports."
}
$Json = & $Executable $Runs $Warmup
if ($LASTEXITCODE -ne 0) { throw "CUDA benchmark failed with exit code $LASTEXITCODE; no report saved." }
$Report = ($Json -join [Environment]::NewLine) | ConvertFrom-Json
if ($Report.cases.Count -ne 4 -or @($Report.cases | Where-Object { -not $_.comparison_passed }).Count) {
    throw 'Benchmark output is incomplete or comparisons failed.'
}
$Report | Add-Member NoteProperty measured_at_utc ([DateTime]::UtcNow.ToString('o'))
New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($OutputPath)) -Force | Out-Null
$Report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Output "Measured report saved: $OutputPath"
$Report.cases | Format-Table width,height,cpu_ms,h2d_ms,kernel_ms,d2h_ms,complete_ms,maximum_absolute_error
