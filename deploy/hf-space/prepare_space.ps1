# Assembles a ready-to-push Hugging Face Space folder in deploy/hf-space/space-build/
# Run from anywhere:  powershell -File deploy\hf-space\prepare_space.ps1

$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # project root (elsevier_downloader)
$rag   = Join-Path $root "rag"
$hf    = $PSScriptRoot
$build = Join-Path $hf "space-build"

Write-Host "Project root: $root"
if (-not (Test-Path (Join-Path $rag "chromadb"))) {
    throw "rag/chromadb not found. Build the index first (rag/02_build_index.py)."
}
if (-not (Test-Path (Join-Path $rag "meta_index.json"))) {
    throw "rag/meta_index.json not found. Run rag/build_metadata.py first."
}

# Fresh build dir
if (Test-Path $build) { Remove-Item -Recurse -Force $build }
New-Item -ItemType Directory -Force $build | Out-Null

# 1) Runtime app code
Copy-Item (Join-Path $rag "app.py")        $build
Copy-Item (Join-Path $rag "synthesize.py") $build
Copy-Item (Join-Path $rag "templates")     (Join-Path $build "templates") -Recurse

# 2) Prebuilt index + metadata
Copy-Item (Join-Path $rag "chromadb")        (Join-Path $build "chromadb") -Recurse
Copy-Item (Join-Path $rag "meta_index.json") $build

# 3) Deployment files
Copy-Item (Join-Path $hf "Dockerfile")       $build
Copy-Item (Join-Path $hf "requirements.txt") $build
Copy-Item (Join-Path $hf "README.md")        $build
Copy-Item (Join-Path $hf ".gitattributes")   $build

$size = [math]::Round(((Get-ChildItem $build -Recurse | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host ""
Write-Host "Space folder ready: $build  ($size MB)"
Write-Host "Next: follow deploy/hf-space/DEPLOY.md to push it to your Space."
