param(
    [string]$CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4',
    [string]$CompilerVersion = '14.38'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Nvcc = Join-Path $CudaRoot 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $Nvcc)) { throw "CUDA 12.4 nvcc not found at $Nvcc" }
$Version = & $Nvcc --version
if ($LASTEXITCODE -ne 0 -or ($Version -join '\n') -notmatch 'release 12\.4,') {
    throw 'This build requires CUDA Toolkit 12.4. It will not switch toolkits silently.'
}
if ($CompilerVersion -notmatch '^\d+\.\d+$') { throw 'CompilerVersion must be a version such as 14.38.' }
$VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $VsWhere)) { throw 'Visual Studio vswhere.exe was not found.' }
$VsRoot = & $VsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$VcVars = Join-Path $VsRoot 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $VcVars)) { throw 'Visual Studio vcvars64.bat was not found.' }
# Load the supported MSVC environment without changing the user's Python environment.
$EnvLines = & $env:ComSpec /d /s /c ('call "' + $VcVars + '" -vcvars_ver=' + $CompilerVersion + ' >nul && set')
if ($LASTEXITCODE -ne 0) { throw "Failed to initialize MSVC $CompilerVersion" }
foreach ($Line in $EnvLines) {
    if ($Line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
}
$BuildDir = Join-Path $ProjectRoot 'build'
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
$Preprocess = Join-Path $ProjectRoot 'src\cuda\ImagePreprocess.cu'
$Defect = Join-Path $ProjectRoot 'src\cuda\DefectAnalysis.cu'
if (-not (Test-Path -LiteralPath $Defect)) { throw 'Custom classifier source DefectAnalysis.cu is missing; refusing to build an incomplete application DLL.' }
$Benchmark = Join-Path $ProjectRoot 'src\cuda\BenchmarkPreprocess.cu'
$Flags = @('-O3', '-std=c++14', '-gencode', 'arch=compute_61,code=sm_61')
Push-Location $BuildDir
try {
    $DllSources = @($Preprocess, $Defect)
    & $Nvcc @Flags --shared @DllSources -o (Join-Path $BuildDir 'fruitguard_preprocess.dll')
    if ($LASTEXITCODE -ne 0) { throw "CUDA DLL build failed with exit code $LASTEXITCODE" }
    & $Nvcc @Flags $Preprocess $Benchmark -o (Join-Path $BuildDir 'BenchmarkPreprocess.exe')
    if ($LASTEXITCODE -ne 0) { throw "CUDA benchmark build failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
Write-Output 'Built build\fruitguard_preprocess.dll and build\BenchmarkPreprocess.exe with CUDA 12.4, sm_61.'
