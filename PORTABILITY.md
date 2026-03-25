# Setup for New PC

To run this application on another computer, follow these steps:

## 1. Prerequisites
- **Python 3.9+**: Install from [python.org](https://www.python.org/)
- **Git**: Install from [git-scm.com](https://git-scm.com/)
- **Tesseract OCR**: **CRITICAL** for text extraction. 
  - Windows: Download and run the installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
  - Add `C:\Program Files\Tesseract-OCR` to your System PATH.
  - Follow instructions in `scanned_doc_restoration/INSTALL_TESSERACT.md`.

## 2. Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dennis5644r56t7/project-guy.git
   cd project-guy
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r scanned_doc_restoration/requirements.txt
   ```

## 3. Running the App
The app includes the **pre-trained models**, so you don't need to retrain it!
```bash
cd scanned_doc_restoration
streamlit run app.py
```

## Troubleshooting
- **"Tesseract is not found"**: Ensure Tesseract is installed and its path is in the System Environment Variables.
- **"Module not found"**: Double-check that you ran `pip install -r requirements.txt`.
