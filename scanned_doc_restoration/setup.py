"""
Setup script for Scanned Document Restoration Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="scanned-doc-restoration",
    version="1.0.0",
    author="Cosmas Kiptoo Sang",
    author_email="[email]",
    description="Document restoration pipeline for improved OCR accuracy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scanned-doc-restoration",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Text Processing :: Optical Character Recognition (OCR)",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "opencv-python>=4.8.0",
        "tensorflow>=2.12.0",
        "pytesseract>=0.3.0",
        "python-Levenshtein>=0.21.0",
        "tqdm>=4.65.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "doc-restore=run_pipeline:main",
            "train-deskew=scripts.train_deskew_model:main",
        ],
    },
)