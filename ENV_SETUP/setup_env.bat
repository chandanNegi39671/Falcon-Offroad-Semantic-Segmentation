@echo off
echo ============================================
echo   Duality AI — EDU Environment Setup
echo ============================================
echo.

call conda create -n EDU python=3.9 -y
call conda activate EDU

pip install torch torchvision
pip install transformers
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install matplotlib
pip install pillow
pip install tqdm

echo.
echo ============================================
echo   EDU environment ready!
echo   Run: conda activate EDU
echo ============================================
pause
