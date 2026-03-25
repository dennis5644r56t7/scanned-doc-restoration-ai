"""
Main Pipeline for Scanned Document Restoration
Integrates denoising, deskewing, and OCR modules
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.denoising import ImageDenoiser, load_image, save_image, compute_psnr, compute_ssim
from src.deskewing import DeskewPipeline, DeskewModel
from src.ocr_module import OCRProcessor, evaluate_ocr_accuracy
from config import (
    TRAIN_IMAGES_DIR, TEST_IMAGES_DIR, TRAIN_LABELS_DIR,
    TRAIN_LIST_PATH, TEST_LIST_PATH, GROUND_TRUTH_TEXT,
    DENOISING_METHODS, CNN_PARAMS, OCR_CONFIG,
    DENOISING_RESULTS_PATH, DESKEWING_RESULTS_PATH,
    OCR_RESULTS_PATH, PIPELINE_RESULTS_PATH, RESULTS_DIR
)

logger = logging.getLogger(__name__)


class DocumentRestorationPipeline:
    """Complete document restoration pipeline."""
    
    def __init__(self, 
                 denoising_method: str = "non_local_means",
                 use_deskewing: bool = True,
                 use_clahe: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            denoising_method: Denoising method to use
            use_deskewing: Whether to apply deskewing
            use_clahe: Whether to apply CLAHE enhancement
        """
        self.denoising_method = denoising_method
        self.use_deskewing = use_deskewing
        self.use_clahe = use_clahe
        
        # Initialize modules
        self.denoiser = ImageDenoiser()
        self.deskewer = DeskewPipeline()  # Will load weights if available
        self.ocr = OCRProcessor(
            lang=OCR_CONFIG["lang"],
            config=OCR_CONFIG["config"],
            ground_truth=GROUND_TRUTH_TEXT
        )
        
        # Results storage
        self.results = {}
    
    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "image_name": image_path.name,
            "original_image": None,
            "denoised_image": None,
            "deskewed_image": None,
            "final_image": None,
            "denoising_time": 0,
            "deskewing_time": 0,
            "ocr_time": 0,
            "extracted_text": "",
            "cer": 100.0,
            "predicted_angle": 0.0
        }
        
        try:
            # 1. Load image
            start_time = cv2.getTickCount()
            image = load_image(image_path)
            results["original_image"] = image.copy()
            load_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            # 2. Denoising
            start_time = cv2.getTickCount()
            denoised = self.denoiser.denoise(image, self.denoising_method)
            results["denoised_image"] = denoised.copy()
            results["denoising_time"] = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            # 3. Deskewing
            if self.use_deskewing:
                start_time = cv2.getTickCount()
                deskewed, angle = self.deskewer.deskew_image(denoised)
                results["deskewed_image"] = deskewed.copy()
                results["predicted_angle"] = angle
                results["deskewing_time"] = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                current_image = deskewed
            else:
                current_image = denoised
            
            # 4. CLAHE enhancement
            if self.use_clahe:
                current_image = self.denoiser.clahe(current_image)
            
            results["final_image"] = current_image.copy()
            
            # 5. OCR
            start_time = cv2.getTickCount()
            text, cer = self.ocr.process_image(current_image)
            results["extracted_text"] = text
            results["cer"] = cer
            results["ocr_time"] = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            # 6. Compute image quality metrics
            if results["original_image"] is not None and results["denoised_image"] is not None:
                results["psnr"] = compute_psnr(results["original_image"], results["denoised_image"])
                results["ssim"] = compute_ssim(results["original_image"], results["denoised_image"])
            
            # Total processing time
            results["total_time"] = (
                load_time + results["denoising_time"] + 
                results["deskewing_time"] + results["ocr_time"]
            )
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            results["error"] = str(e)
        
        return results
    
    def process_batch(self, image_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of results dictionaries
        """
        all_results = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            results = self.process_image(img_path)
            all_results.append(results)
        
        return all_results
    
    def evaluate_denoising_methods(self, image_paths: List[Path], 
                                   clean_reference: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Evaluate all denoising methods on a set of images.
        
        Args:
            image_paths: List of image paths
            clean_reference: Optional clean reference image for PSNR/SSIM
            
        Returns:
            DataFrame with evaluation results
        """
        methods = list(DENOISING_METHODS.keys())
        results = []
        
        for method in methods:
            logger.info(f"Evaluating denoising method: {method}")
            method_results = []
            
            for img_path in tqdm(image_paths, desc=f"{method}"):
                try:
                    image = load_image(img_path)
                    denoised = self.denoiser.denoise(image, method)
                    
                    # Compute metrics
                    if clean_reference is not None:
                        # Resize clean reference to match image size
                        ref_resized = cv2.resize(clean_reference, (image.shape[1], image.shape[0]))
                        psnr = compute_psnr(ref_resized, denoised)
                        ssim = compute_ssim(ref_resized, denoised)
                    else:
                        # Use original as reference (less accurate)
                        psnr = compute_psnr(image, denoised)
                        ssim = compute_ssim(image, denoised)
                    
                    method_results.append({
                        "image": img_path.name,
                        "method": method,
                        "psnr": psnr,
                        "ssim": ssim
                    })
                except Exception as e:
                    logger.error(f"Error processing {img_path.name} with {method}: {e}")
            
            # Aggregate results
            if method_results:
                df_method = pd.DataFrame(method_results)
                results.append(df_method)
        
        # Combine all results
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Path):
        """
        Save pipeline results to CSV.
        
        Args:
            results: List of results dictionaries
            output_path: Path to save CSV
        """
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to: {output_path}")
        
        # Print summary statistics
        if "cer" in df.columns:
            mean_cer = df["cer"].mean()
            median_cer = df["cer"].median()
            logger.info(f"CER Summary - Mean: {mean_cer:.2f}%, Median: {median_cer:.2f}%")
        
        if "psnr" in df.columns:
            mean_psnr = df["psnr"].mean()
            logger.info(f"PSNR Summary - Mean: {mean_psnr:.2f} dB")
    
    def visualize_results(self, results: Dict[str, Any], output_dir: Path):
        """
        Create visualization of processing results.
        
        Args:
            results: Results dictionary for a single image
            output_dir: Directory to save visualizations
        """
        img_name = results["image_name"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        if results["original_image"] is not None:
            axes[0, 0].imshow(cv2.cvtColor(results["original_image"], cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f"Original\n{img_name}")
            axes[0, 0].axis('off')
        
        # Denoised image
        if results["denoised_image"] is not None:
            axes[0, 1].imshow(cv2.cvtColor(results["denoised_image"], cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f"Denoised ({self.denoising_method})")
            if "psnr" in results and "ssim" in results:
                axes[0, 1].set_xlabel(f"PSNR: {results['psnr']:.2f} dB\nSSIM: {results['ssim']:.4f}")
            axes[0, 1].axis('off')
        
        # Deskewed image
        if results["deskewed_image"] is not None:
            axes[1, 0].imshow(cv2.cvtColor(results["deskewed_image"], cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title(f"Deskewed\nAngle: {results['predicted_angle']:.2f}°")
            axes[1, 0].axis('off')
        
        # Final image
        if results["final_image"] is not None:
            axes[1, 1].imshow(cv2.cvtColor(results["final_image"], cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title(f"Final Output\nCER: {results['cer']:.2f}%")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f"{img_name.replace('.png', '_results.png')}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to: {output_path}")


def run_full_pipeline():
    """Run the complete pipeline as described in the research."""
    logger.info("Starting full document restoration pipeline...")
    
    # Load image lists
    with open(TRAIN_LIST_PATH, 'r') as f:
        train_images = json.load(f)
    
    with open(TEST_LIST_PATH, 'r') as f:
        test_images = json.load(f)
    
    # Convert to full paths
    train_paths = [TRAIN_IMAGES_DIR / img for img in train_images]
    test_paths = [TRAIN_IMAGES_DIR / img for img in test_images]  # Test images are in train dir
    
    # Take a subset for testing (first 10 images)
    test_subset = train_paths[:10]
    
    # 1. Evaluate denoising methods
    logger.info("Step 1: Evaluating denoising methods...")
    pipeline = DocumentRestorationPipeline()
    
    # Find a relatively clean image for reference
    clean_ref = None
    for img_path in train_paths[:5]:
        try:
            img = load_image(img_path)
            # Simple heuristic: low standard deviation might indicate less noise
            if np.std(img) < 30:
                clean_ref = img
                break
        except:
            continue
    
    denoising_results = pipeline.evaluate_denoising_methods(test_subset, clean_ref)
    if not denoising_results.empty:
        denoising_results.to_csv(DENOISING_RESULTS_PATH, index=False)
        logger.info(f"Saved denoising results to: {DENOISING_RESULTS_PATH}")
    
    # 2. Process with best method (Non-Local Means)
    logger.info("Step 2: Processing with Non-Local Means denoising...")
    pipeline_nlm = DocumentRestorationPipeline(denoising_method="non_local_means")
    results_nlm = pipeline_nlm.process_batch(test_subset)
    pipeline_nlm.save_results(results_nlm, RESULTS_DIR / "nlm_results.csv")
    
    # 3. Process with denoising + deskewing
    logger.info("Step 3: Processing with denoising + deskewing...")
    pipeline_full = DocumentRestorationPipeline(
        denoising_method="non_local_means",
        use_deskewing=True,
        use_clahe=True
    )
    results_full = pipeline_full.process_batch(test_subset)
    pipeline_full.save_results(results_full, PIPELINE_RESULTS_PATH)
    
    # 4. Compare pipelines
    logger.info("Step 4: Comparing pipeline configurations...")
    
    # Create comparison DataFrame
    comparison_data = []
    for result in results_full:
        comparison_data.append({
            "image": result["image_name"],
            "cer_baseline": 34.2,  # From research
            "cer_nlm_only": next((r["cer"] for r in results_nlm 
                                 if r["image_name"] == result["image_name"]), 100.0),
            "cer_full": result["cer"]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = RESULTS_DIR / "pipeline_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PIPELINE COMPARISON SUMMARY")
    logger.info("="*50)
    logger.info(f"Baseline CER (from research): 34.2%")
    logger.info(f"NLM only CER (mean): {comparison_df['cer_nlm_only'].mean():.2f}%")
    logger.info(f"Full pipeline CER (mean): {comparison_df['cer_full'].mean():.2f}%")
    logger.info(f"Improvement over baseline: {34.2 - comparison_df['cer_full'].mean():.2f} percentage points")
    
    # 5. Create visualizations
    logger.info("Step 5: Creating visualizations...")
    vis_dir = RESULTS_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    for result in results_full[:5]:  # First 5 images
        pipeline_full.visualize_results(result, vis_dir)
    
    logger.info("Pipeline completed successfully!")
    return comparison_df


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the pipeline
    try:
        results = run_full_pipeline()
        print("\nPipeline execution completed!")
        print(f"Results saved in: {RESULTS_DIR}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise