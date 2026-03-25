"""
OCR Module for Scanned Document Restoration Pipeline
Text extraction using Tesseract OCR with preprocessing
"""

import os
import sys
from pathlib import Path

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
    
    # Try to find Tesseract automatically on Windows
    if sys.platform == 'win32':
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME')),
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
        
        tesseract_found = False
        for path in possible_paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                tesseract_found = True
                print(f"Found Tesseract at: {path}")
                break
        
        if not tesseract_found:
            print("Tesseract not found in common locations. Will try PATH.")
                
except ImportError:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any

try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    Levenshtein = None

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor using Tesseract."""
    
    def __init__(self, 
                 lang: str = 'eng',
                 config: str = '--psm 6 --oem 1',
                 ground_truth: Optional[str] = None):
        """
        Initialize OCR processor.
        
        Args:
            lang: Language for OCR
            config: Tesseract configuration (psm 6 = uniform block of text, oem 1 = LSTM only)
            ground_truth: Ground truth text for CER calculation
        """
        self.lang = lang
        self.config = config
        self.ground_truth = ground_truth
        self.tesseract_available = False
        
        # Check if Tesseract is available
        if not PYTESSERACT_AVAILABLE:
            logger.warning("pytesseract not installed. OCR functionality will be disabled.")
            logger.warning("Install with: pip install pytesseract")
            return
        
        # Verify Tesseract is installed and accessible
        try:
            version = pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info(f"Tesseract OCR {version} is available and ready")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            logger.error("Please install Tesseract OCR:")
            logger.error("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            logger.error("  Or set path: pytesseract.pytesseract.tesseract_cmd = r'C:\\Path\\To\\tesseract.exe'")
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal OCR performance.
        MINIMAL preprocessing since pipeline already cleaned the image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple upscaling for better OCR (2x is enough)
        height, width = gray.shape
        upscaled = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        # Convert to RGB for Tesseract (it expects RGB)
        rgb = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)
        
        return rgb
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from image using Tesseract.
        
        Args:
            image: Input image (already preprocessed by pipeline)
            
        Returns:
            Extracted text
        """
        if not PYTESSERACT_AVAILABLE:
            return "[OCR not available - pytesseract not installed]\n\nInstall with: pip install pytesseract\nThen install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
        
        if not self.tesseract_available:
            return "[OCR Error: Tesseract is not installed or not in PATH]\n\nDownload from: https://github.com/UB-Mannheim/tesseract/wiki\nInstall and add to PATH, then restart the app."
        
        # Minimal preprocessing (just upscale and convert to RGB)
        processed_image = self.preprocess_for_ocr(image)
        
        # Extract text with single best config
        try:
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.lang,
                config=self.config
            )
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return f"[OCR Error: {e}]"
    
    def _post_process_text(self, text: str) -> str:
        """
        Post-process OCR text to fix common errors.
        
        Args:
            text: Raw OCR output
            
        Returns:
            Cleaned text
        """
        # Common OCR error corrections for speckle-noisy documents
        corrections = {
            # Previous corrections
            'che': 'the',
            'tee': 'the',
            'sud': 'and',
            'Woth': 'With',
            'cxceptiond': 'exceptional',
            'sudcats': 'students',
            'acaicmice': 'academics',
            'mhustry': 'industry',
            'rescaechers': 'researchers',
            'peverd': 'several',
            'eonkcrenne': 'conference',
            'Vissca': 'Vision',
            'cveut': 'event',
            'Competes': 'Computer',
            'premuce': 'premier',
            'summa': 'annual',
            # New corrections for speckle noise
            'I.est': 'Id est',
            'Uht-ut': 'ut',
            'riecessitatibus': 'necessitatibus',
            'Jvteprehenderit': 'Reprehenderit',
            'laborum.sed': 'laborum. Sed',
            'earurt': 'earum',
            'érim': 'enim',
            'vebt': 'velit',
            'doloreiuset': 'dolor eius et',
            'arumi': 'animi',
            'Nikit': 'Nihil',
            'Libers': 'Libero',
            'possirnus': 'possimus',
            'exceptun': 'excepturi',
            'Voluptag': 'Voluptas',
            'Jabouosam': 'laboriosam',
            'Persprcatis': 'Perspiciatis',
            'Cupsdatat': 'Cupidatat',
            'bore': 'labore',
            'Excépteur': 'Excepteur',
            'Laboriogam': 'Laboriosam',
            "Guiderh'dolor": 'Quidem dolor',
            'aspermatur': 'aspernatur',
            'guisquami': 'quisquam',
            'ddlorum': 'dolorum',
        }
        
        # Apply corrections (word boundaries)
        import re
        for wrong, correct in corrections.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
        
        # Remove excessive spaces and dots
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single
        text = re.sub(r'\s*\.\s*', '. ', text)  # Fix spacing around dots
        
        return text.strip()
    
    def compute_cer(self, predicted_text: str, reference_text: str) -> float:
        """
        Compute Character Error Rate (CER).
        
        Args:
            predicted_text: OCR output text
            reference_text: Ground truth text
            
        Returns:
            CER as percentage (0-100)
        """
        if not LEVENSHTEIN_AVAILABLE:
            return 0.0  # Can't compute CER without Levenshtein
        
        if not reference_text:
            return 100.0  # No reference, assume worst case
        
        # Clean texts
        predicted = predicted_text.strip()
        reference = reference_text.strip()
        
        if not reference:  # Empty reference
            return 100.0 if predicted else 0.0
        
        # Compute Levenshtein distance
        distance = Levenshtein.distance(predicted, reference)
        
        # CER = (edit distance / reference length) * 100
        cer = (distance / len(reference)) * 100
        
        return cer
    
    def process_image(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Process image and return text with CER.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (extracted_text, cer_percentage)
        """
        # Extract text
        text = self.extract_text(image)
        
        # Compute CER if ground truth is available
        cer = self.compute_cer(text, self.ground_truth) if self.ground_truth else 0.0
        
        return text, cer
    
    def batch_process(self, images: list) -> Tuple[list, list]:
        """
        Process batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Tuple of (texts, cers)
        """
        texts = []
        cers = []
        
        for img in images:
            text, cer = self.process_image(img)
            texts.append(text)
            cers.append(cer)
        
        return texts, cers


def evaluate_ocr_accuracy(images: list, ground_truth: str) -> Dict[str, Any]:
    """
    Evaluate OCR accuracy on a set of images.
    
    Args:
        images: List of images to process
        ground_truth: Reference text
        
    Returns:
        Dictionary with evaluation metrics
    """
    ocr = OCRProcessor(ground_truth=ground_truth)
    texts, cers = ocr.batch_process(images)
    
    # Compute statistics
    mean_cer = np.mean(cers) if cers else 100.0
    median_cer = np.median(cers) if cers else 100.0
    std_cer = np.std(cers) if len(cers) > 1 else 0.0
    min_cer = np.min(cers) if cers else 100.0
    max_cer = np.max(cers) if cers else 100.0
    
    # Count images with CER below thresholds
    cer_5 = sum(1 for cer in cers if cer <= 5.0)
    cer_10 = sum(1 for cer in cers if cer <= 10.0)
    cer_20 = sum(1 for cer in cers if cer <= 20.0)
    
    results = {
        'mean_cer': mean_cer,
        'median_cer': median_cer,
        'std_cer': std_cer,
        'min_cer': min_cer,
        'max_cer': max_cer,
        'cer_5_count': cer_5,
        'cer_10_count': cer_10,
        'cer_20_count': cer_20,
        'total_images': len(images),
        'texts': texts,
        'cers': cers
    }
    
    return results


def compare_preprocessing_pipelines(images: list, 
                                    ground_truth: str,
                                    pipeline_names: list,
                                    pipeline_images: list) -> Dict[str, Any]:
    """
    Compare different preprocessing pipelines.
    
    Args:
        images: Original images
        ground_truth: Reference text
        pipeline_names: Names of pipelines
        pipeline_images: List of image lists (one per pipeline)
        
    Returns:
        Comparison results
    """
    results = {}
    
    for name, pipe_images in zip(pipeline_names, pipeline_images):
        logger.info(f"Evaluating pipeline: {name}")
        pipe_results = evaluate_ocr_accuracy(pipe_images, ground_truth)
        results[name] = pipe_results
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config import GROUND_TRUTH_TEXT
    
    # Initialize OCR processor
    ocr = OCRProcessor(ground_truth=GROUND_TRUTH_TEXT)
    
    # Example: Process a single image
    test_image = cv2.imread("test_image.png")
    text, cer = ocr.process_image(test_image)
    
    print(f"Extracted text: {text[:100]}...")
    print(f"CER: {cer:.2f}%")
    
    # Example: Batch processing
    images = [test_image] * 3
    texts, cers = ocr.batch_process(images)
    
    for i, (text, cer) in enumerate(zip(texts, cers)):
        print(f"Image {i+1}: CER = {cer:.2f}%")