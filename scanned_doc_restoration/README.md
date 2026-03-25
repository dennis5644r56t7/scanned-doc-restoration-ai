# Scanned Document Restoration Pipeline

A complete implementation of the document restoration pipeline based on the research by Cosmas Kiptoo Sang (Kisii University, March 2026).

## Project Overview

This project implements a three-stage pipeline for restoring scanned documents:
1. **Denoising**: Removes speckle noise from scanned documents
2. **Deskewing**: Corrects document rotation using a CNN model
3. **OCR**: Extracts text using Tesseract OCR

Based on the research paper: "Scanned Document Restoration for Improved Optical Character Recognition"

## Project Structure

```
scanned_doc_restoration/
├── data/                    # Dataset (images and labels)
├── src/                     # Source code
│   ├── denoising.py         # Denoising module
│   ├── deskewing.py         # Deskewing module with CNN
│   ├── ocr_module.py        # OCR processing
│   └── pipeline.py          # Main pipeline
├── models/                  # Trained models
├── results/                 # Results and visualizations
├── scripts/                 # Training and evaluation scripts
├── notebooks/               # Jupyter notebooks for exploration
├── config.py                # Configuration
├── requirements.txt         # Dependencies
└── run_pipeline.py          # Main execution script
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - **Windows**: Download installer from GitHub releases
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

## Dataset

The dataset should be placed in the `data/` directory with the following structure:
```
data/
├── images/
│   ├── train/          # Training images (000-499)
│   └── test/           # Test images (500-599)
├── labels/              # Angle labels (.txt files)
├── train_list.json      # Training image list
└── test_list.json       # Test image list
```

## Usage

### 1. Train the deskewing model:
```bash
python scripts/train_deskew_model.py
```

### 2. Run the complete pipeline:
```bash
python run_pipeline.py --all
```

### 3. Run specific components:
```bash
# Train only
python run_pipeline.py --train

# Run pipeline only
python run_pipeline.py --run-pipeline

# Generate visualizations
python run_pipeline.py --visualize
```

### 4. Explore with Jupyter:
```bash
jupyter notebook notebooks/explore_dataset.ipynb
```

## Pipeline Components

### 1. Denoising Module
Implements 6 denoising methods:
- Gaussian Blur
- Median Filter
- Bilateral Filter
- Non-Local Means (Best performer)
- Morphological Opening
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

### 2. Deskewing Module
- CNN-based angle prediction
- Regression model for continuous angle prediction
- Trained on 500 labeled images

### 3. OCR Module
- Tesseract OCR integration
- Character Error Rate (CER) calculation
- Ground truth comparison

## Results

Based on the research paper:
- Baseline CER (no preprocessing): 34.2%
- After denoising only: 21.7% CER
- After deskewing only: 18.3% CER
- Full pipeline (NLM + Deskewing + CLAHE): 8.6% CER

## Configuration

Edit `config.py` to adjust:
- Image processing parameters
- Model hyperparameters
- File paths
- Evaluation metrics

## Citation

If you use this code in your research, please cite:

```
@article{sang2026scanned,
  title={Scanned Document Restoration for Improved Optical Character Recognition},
  author={Sang, Cosmas Kiptoo},
  year={2026},
  institution={Kisii University}
}
```

## License

MIT License - see LICENSE file for details.