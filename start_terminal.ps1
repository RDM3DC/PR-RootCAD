# AdaptiveCAD Terminal Starter
# This script sets up the terminal environment for AdaptiveCAD

# Navigate to the AdaptiveCAD directory
Set-Location -Path "D:\SuperCAD\AdaptiveCAD"
Write-Host "Changed directory to: $(Get-Location)" -ForegroundColor Green

# Check if conda is available
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
    Write-Host "Conda not found in PATH. Make sure conda is installed correctly." -ForegroundColor Red
    return
}

# Initialize conda for PowerShell
& conda shell.powershell hook | Out-String | Invoke-Expression

# Activate the adaptivecad environment
Write-Host "Activating adaptivecad conda environment..." -ForegroundColor Cyan
conda activate adaptivecad

# Verify environment activation
if ($env:CONDA_DEFAULT_ENV -eq "adaptivecad") {
    Write-Host "Successfully activated adaptivecad environment" -ForegroundColor Green
    Write-Host "Python path: $(Get-Command python | Select-Object -ExpandProperty Source)" -ForegroundColor Green
    
    # Check dependencies (use here-string to avoid quoting issues)
    Write-Host "Checking for key dependencies:" -ForegroundColor Cyan
    $pyBlock = @"
import importlib, sys
def chk(name, label=None):
    label = label or name
    try:
        m = importlib.import_module(name)
        ver = getattr(m, '__version__', 'present')
        print(f'  \u2713 {label}: {ver}')
    except Exception:
        print(f'  \u2717 {label}: Not installed')

chk('numpy','numpy')
chk('PySide6','PySide6')
try:
    import OCC.Core  # noqa
    print('  \u2713 pythonocc-core: Installed')
except Exception:
    print('  \u2717 pythonocc-core: Not installed')
"@
    $pyBlock | & python -
} else {
    Write-Host "Failed to activate adaptivecad environment. Current environment: $env:CONDA_DEFAULT_ENV" -ForegroundColor Red
}

# Display helpful commands
Write-Host "`nHelpful commands:" -ForegroundColor Yellow
Write-Host "  python -m adaptivecad.gui.playground    # Start the AdaptiveCAD GUI" -ForegroundColor Yellow
Write-Host "  .\check_environment.ps1                 # Run full environment check" -ForegroundColor Yellow
Write-Host "  .\test_import.ps1                       # Test import functionality" -ForegroundColor Yellow
Write-Host "  .\start_adaptivecad.ps1                 # Start AdaptiveCAD GUI with environment setup" -ForegroundColor Yellow
