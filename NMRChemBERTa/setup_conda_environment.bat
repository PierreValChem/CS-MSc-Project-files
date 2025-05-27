@echo off
echo Setting up NMR-ChemBERTa with Conda...
echo.

REM Check if conda is available
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: Conda not found! Please install Anaconda or Miniconda first.
    echo Download from: https://www.anaconda.com/products/individual
    pause
    exit /b 1
)

REM Force deactivate any active environment
call conda deactivate 2>nul
call conda deactivate 2>nul

REM Check if environment exists and remove it
echo Checking for existing environment...
conda env list | findstr /C:"nmr-chemberta" >nul 2>nul
if not errorlevel 1 (
    echo Found existing nmr-chemberta environment.
    echo Removing it (this may take a moment)...
    echo.
    REM Force remove without confirmation
    conda env remove -n nmr-chemberta -y --all >nul 2>&1
    timeout /t 2 >nul
)

REM Create conda environment with Python 3.11
echo Creating new conda environment with Python 3.11...
echo This may take a few minutes...
echo.
call conda create -n nmr-chemberta python=3.11 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment
    echo Trying alternative approach...
    call conda create -n nmr-chemberta python=3.11 --force -y
    if errorlevel 1 (
        echo ERROR: Still failed. Please try:
        echo   1. Close all Python/Conda windows
        echo   2. Run: conda clean --all
        echo   3. Try this script again
        pause
        exit /b 1
    )
)

REM Activate environment
echo.
echo Activating environment...
call conda activate nmr-chemberta
if errorlevel 1 (
    echo ERROR: Failed to activate environment
    echo Trying to fix conda init...
    call conda init cmd.exe
    echo Please close this window and run the script again.
    pause
    exit /b 1
)

REM Verify we're in the right environment
echo.
echo Current conda environment:
conda info --envs | findstr "*"
echo.

REM Verify Python version
echo Python version in environment:
python --version
echo.

REM Update conda and pip first
echo Updating conda and pip...
conda update -n base conda -y >nul 2>&1
python -m pip install --upgrade pip

REM Install PyTorch
echo.
echo Installing PyTorch (this may take several minutes)...
echo Trying CUDA 11.8 version first...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo.
    echo CUDA version failed, installing CPU-only PyTorch...
    pip install torch torchvision torchaudio
)

REM Install RDKit via pip first (often faster than conda-forge)
echo.
echo Installing RDKit...
pip install rdkit
if errorlevel 1 (
    echo pip install failed, trying conda-forge...
    conda install -c conda-forge rdkit -y
)

REM Install core scientific packages
echo.
echo Installing core scientific packages...
pip install numpy>=1.24.0 pandas>=2.0.0 scipy>=1.10.0 matplotlib>=3.7.0

REM Install ML packages
echo.
echo Installing ML packages...
pip install transformers>=4.30.0
pip install tokenizers>=0.13.0

REM Install utilities
echo.
echo Installing utilities...
pip install psutil>=5.9.0 tqdm>=4.65.0 pyyaml>=6.0 tensorboard>=2.13.0

REM Optional packages
echo.
echo Installing optional packages...
pip install wandb pytest black

REM Quick verification
echo.
echo ========================================
echo Verifying installations...
echo ========================================
python -c "import sys; print(f'Python: {sys.version}')" 2>nul
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul || echo PyTorch: Failed to import
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>nul || echo Transformers: Failed to import
python -c "import rdkit; print(f'RDKit installed successfully')" 2>nul || echo RDKit: Failed to import
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>nul || echo NumPy: Failed to import

REM Check CUDA availability
echo.
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 0 (
    python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Environment name: nmr-chemberta
echo.
echo To use this environment:
echo     conda activate nmr-chemberta
echo     python train.py
echo.
echo To see installed packages:
echo     conda activate nmr-chemberta
echo     pip list
echo.
echo If you encounter any issues, try:
echo     conda clean --all
echo     Then run this script again
echo.
echo ========================================
pause