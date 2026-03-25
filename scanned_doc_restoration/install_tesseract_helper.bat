@echo off
echo ========================================
echo Tesseract OCR Installation Helper
echo ========================================
echo.

echo Checking if Tesseract is installed...
where tesseract >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Tesseract is installed and in PATH!
    tesseract --version
    echo.
    goto :check_python
) else (
    echo [X] Tesseract not found in PATH
    echo.
    echo Please install Tesseract OCR:
    echo 1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
    echo 2. Run the installer
    echo 3. Add to PATH during installation
    echo 4. Restart this script
    echo.
    pause
    exit /b 1
)

:check_python
echo Checking Python packages...
python -c "import pytesseract" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] pytesseract is installed
) else (
    echo [X] pytesseract not installed
    echo Installing pytesseract...
    pip install pytesseract python-Levenshtein
)

echo.
echo ========================================
echo Installation Check Complete!
echo ========================================
echo.
echo You can now run the app with:
echo   streamlit run app.py
echo.
pause
