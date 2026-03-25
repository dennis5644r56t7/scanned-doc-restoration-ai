"""
Denoising Module for Scanned Document Restoration Pipeline
Implements six classical denoising methods as described in the research
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageDenoiser:
    """
    Denoising class implementing six classical methods:
    1. Gaussian Blur
    2. Median Filter
    3. Bilateral Filter
    4. Non-Local Means
    5. Morphological Opening
    6. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, ksize: tuple = (5, 5)) -> np.ndarray:
        """
        Apply Gaussian blur for general noise reduction.
        
        Args:
            image: Input BGR image (uint8)
            ksize: Gaussian kernel size (width, height)
            
        Returns:
            Denoised BGR image
        """
        if len(image.shape) == 2:
            # Grayscale image
            return cv2.GaussianBlur(image, ksize, 0)
        else:
            # Color image
            return cv2.GaussianBlur(image, ksize, 0)
    
    @staticmethod
    def median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
        """
        Apply median filter for isolated speckle removal.
        
        Args:
            image: Input BGR image (uint8)
            ksize: Aperture linear size (must be odd and > 1)
            
        Returns:
            Denoised BGR image
        """
        if len(image.shape) == 2:
            return cv2.medianBlur(image, ksize)
        else:
            # Apply median filter to each channel separately for color images
            channels = cv2.split(image)
            denoised_channels = [cv2.medianBlur(ch, ksize) for ch in channels]
            return cv2.merge(denoised_channels)
    
    @staticmethod
    def bilateral_filter(
        image: np.ndarray, 
        d: int = 9, 
        sigma_color: float = 75, 
        sigma_space: float = 75
    ) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving noise reduction.
        
        Args:
            image: Input BGR image (uint8)
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Denoised BGR image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def non_local_means(
        image: np.ndarray, 
        h: float = 10, 
        template_window_size: int = 7, 
        search_window_size: int = 21
    ) -> np.ndarray:
        """
        Apply Non-Local Means denoising (highest quality, texture-aware).
        
        Args:
            image: Input BGR image (uint8)
            h: Parameter regulating filter strength
            template_window_size: Size in pixels of the template patch
            search_window_size: Size in pixels of the window for searching
            
        Returns:
            Denoised BGR image
        """
        if len(image.shape) == 2:
            # Grayscale
            return cv2.fastNlMeansDenoising(
                image, 
                h=h, 
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
        else:
            # Color
            return cv2.fastNlMeansDenoisingColored(
                image, 
                h=h, 
                hColor=h,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
    
    @staticmethod
    def morphological_opening(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply morphological opening for small speckle mark removal.
        
        Args:
            image: Input BGR image (uint8)
            kernel_size: Size of the structuring element
            
        Returns:
            Denoised BGR image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if len(image.shape) == 2:
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        else:
            # Apply to each channel
            channels = cv2.split(image)
            opened_channels = [cv2.morphologyEx(ch, cv2.MORPH_OPEN, kernel) for ch in channels]
            return cv2.merge(opened_channels)
    
    @staticmethod
    def clahe(
        image: np.ndarray, 
        clip_limit: float = 3.0, 
        tile_grid_size: tuple = (8, 8)
    ) -> np.ndarray:
        """
        Apply CLAHE for contrast enhancement (usually applied after denoising).
        
        Args:
            image: Input BGR image (uint8)
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            Contrast-enhanced BGR image
        """
        if len(image.shape) == 2:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)
        else:
            # Convert to LAB color space, apply CLAHE to L channel only
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    @classmethod
    def denoise(
        cls, 
        image: np.ndarray, 
        method: str = "non_local_means", 
        **kwargs
    ) -> np.ndarray:
        """
        Apply specified denoising method to image.
        
        Args:
            image: Input BGR image (uint8)
            method: Denoising method name
            **kwargs: Method-specific parameters
            
        Returns:
            Denoised BGR image
            
        Raises:
            ValueError: If method is not supported
        """
        method_map = {
            "gaussian_blur": cls.gaussian_blur,
            "median_filter": cls.median_filter,
            "bilateral_filter": cls.bilateral_filter,
            "non_local_means": cls.non_local_means,
            "morphological_opening": cls.morphological_opening,
            "clahe": cls.clahe,
        }
        
        if method not in method_map:
            raise ValueError(f"Unsupported denoising method: {method}. "
                           f"Supported methods: {list(method_map.keys())}")
        
        logger.info(f"Applying {method} denoising")
        return method_map[method](image, **kwargs)
    
    @classmethod
    def denoise_all_methods(cls, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply all denoising methods to the same image for comparison.
        
        Args:
            image: Input BGR image (uint8)
            
        Returns:
            Dictionary mapping method names to denoised images
        """
        results = {}
        
        # Apply each method with default parameters
        for method in [
            "gaussian_blur", "median_filter", "bilateral_filter",
            "non_local_means", "morphological_opening", "clahe"
        ]:
            try:
                results[method] = cls.denoise(image, method)
            except Exception as e:
                logger.error(f"Error applying {method}: {e}")
                results[method] = image.copy()  # Return original on error
        
        return results


def load_image(image_path: Path) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        BGR image array
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Path) -> None:
    """
    Save image to file.
    
    Args:
        image: BGR image array
        output_path: Path to save image
    """
    cv2.imwrite(str(output_path), image)
    logger.info(f"Saved image to: {output_path}")


def compute_psnr(original: np.ndarray, denoised: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        original: Original image
        denoised: Denoised image
        
    Returns:
        PSNR value in dB
    """
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(original: np.ndarray, denoised: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        original: Original image
        denoised: Denoised image
        
    Returns:
        SSIM value between 0 and 1
    """
    # Simple SSIM implementation
    # For more accurate results, use scikit-image's ssim function
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu_x = np.mean(original)
    mu_y = np.mean(denoised)
    sigma_x = np.std(original)
    sigma_y = np.std(denoised)
    sigma_xy = np.cov(original.flatten(), denoised.flatten())[0, 1]
    
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)
    
    return numerator / denominator


if __name__ == "__main__":
    # Test the denoising module
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config import TRAIN_IMAGES_DIR
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample image
    test_image_path = TRAIN_IMAGES_DIR / "scan_000.png"
    if test_image_path.exists():
        print(f"Testing denoising module with: {test_image_path}")
        
        # Load image
        image = load_image(test_image_path)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        # Test each method
        denoiser = ImageDenoiser()
        methods = ["gaussian_blur", "median_filter", "non_local_means"]
        
        for method in methods:
            print(f"\nTesting {method}...")
            try:
                denoised = denoiser.denoise(image, method)
                print(f"  Success! Output shape: {denoised.shape}")
                
                # Compute metrics
                psnr = compute_psnr(image, denoised)
                ssim = compute_ssim(image, denoised)
                print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print(f"Test image not found: {test_image_path}")