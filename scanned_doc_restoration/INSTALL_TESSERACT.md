# How to Install Tesseract OCR

## For Windows Users

### Step 1: Download Tesseract

1. Go to: **https://github.com/UB-Mannheim/tesseract/wiki**
2. Download the latest installer: `tesseract-ocr-w64-setup-5.3.3.20231005.exe` (or newer)

### Step 2: Install Tesseract

1. Run the downloaded `.exe` file
2. Follow the installation wizard
3. **Important**: Note the installation path (usually `C:\Program Files\Tesseract-OCR`)
4. Complete the installation

### Step 3: Add to PATH (if not automatic)

**Option A: During Installation**
- Check the box "Add to PATH" if available

**Option B: Manual PATH Setup**
1. Open **System Properties** → **Environment Variables**
2. Under "System variables", find and select **Path**
3. Click **Edit**
4. Click **New**
5. Add: `C:\Program Files\Tesseract-OCR`
6. Click **OK** on all windows

**Option C: PowerShell Command (Run as Administrator)**
```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\Tesseract-OCR", "Machine")
```

### Step 4: Install Python Package

Open Command Prompt or PowerShell and run:
```bash
pip install pytesseract python-Levenshtein
```

### Step 5: Verify Installation

Open a new Command Prompt and run:
```bash
tesseract --version
```

You should see version information like:
```
tesseract 5.3.3
```

### Step 6: Restart Your App

1. Close the Streamlit app (Ctrl+C in terminal)
2. Restart it:
   ```bash
   streamlit run app.py
   ```

## Troubleshooting

### "tesseract is not recognized"
- Make sure Tesseract is added to PATH
- Restart your terminal/command prompt
- Restart your computer if needed

### "pytesseract not installed"
```bash
pip install pytesseract
```

### Still Not Working?

The app will automatically try these paths:
- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- `C:\Users\[YourUsername]\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`

If Tesseract is installed elsewhere, you can manually set the path in the code.

## For Linux Users

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
pip install pytesseract python-Levenshtein
```

## For macOS Users

```bash
brew install tesseract
pip install pytesseract python-Levenshtein
```

## Verify It's Working

After installation, the app sidebar should show:
```
✅ OCR (Tesseract) is available
```

And when you process a document, you'll see extracted text instead of an error message!

## Quick Links

- **Download**: https://github.com/UB-Mannheim/tesseract/wiki
- **Documentation**: https://tesseract-ocr.github.io/
- **Python Package**: https://pypi.org/project/pytesseract/