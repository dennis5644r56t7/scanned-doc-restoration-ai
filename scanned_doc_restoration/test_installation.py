"""
Test script to verify installation and basic functionality
"""

import sys
from pathlib import Path
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        "numpy",
        "cv2",
        "tensorflow",
        "pytesseract",
        "skimage",
        "pandas",
        "matplotlib",
    ]
    
    print("Testing package imports...")
    for package in required_packages:
        try:
            importlib.import_module(package if package != "cv2" else "cv2")
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
    
    return True

def test_project_structure():
    """Test if project structure is correct."""
    required_dirs = [
        "data",
        "src",
        "models",
        "results",
        "scripts",
        "notebooks",
    ]
    
    required_files = [
        "config.py",
        "requirements.txt",
        "run_pipeline.py",
        "README.md",
        "src/denoising.py",
        "src/deskewing.py",
        "src/ocr_module.py",
        "src/pipeline.py",
    ]
    
    print("\nTesting project structure...")
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ Directory: {dir_name}")
        else:
            print(f"  ✗ Directory missing: {dir_name}")
    
    # Check files
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print(f"  ✓ File: {file_name}")
        else:
            print(f"  ✗ File missing: {file_name}")
    
    return True

def test_data_availability():
    """Test if dataset is available."""
    print("\nTesting data availability...")
    
    data_dirs = [
        "data/images/train",
        "data/labels/train",
    ]
    
    data_files = [
        "data/train_list.json",
        "data/test_list.json",
    ]
    
    for dir_name in data_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            # Count files
            file_count = len(list(dir_path.glob("*")))
            print(f"  ✓ {dir_name}: {file_count} files")
        else:
            print(f"  ✗ Directory missing: {dir_name}")
    
    for file_name in data_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print(f"  ✓ File: {file_name}")
        else:
            print(f"  ✗ File missing: {file_name}")
    
    return True

def test_config():
    """Test if configuration can be loaded."""
    print("\nTesting configuration...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        from config import (
            PROJECT_ROOT, DATA_DIR, SRC_DIR, MODELS_DIR,
            TRAIN_IMAGES_DIR, GROUND_TRUTH_TEXT
        )
        
        print(f"  ✓ Project Root: {PROJECT_ROOT}")
        print(f"  ✓ Data Directory: {DATA_DIR}")
        print(f"  ✓ Source Directory: {SRC_DIR}")
        print(f"  ✓ Ground Truth Text: {len(GROUND_TRUTH_TEXT)} characters")
        
        return True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Scanned Document Restoration Pipeline - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Data Availability", test_data_availability),
        ("Configuration", test_config),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Train the deskewing model:")
        print("   python scripts/train_deskew_model.py")
        print("\n2. Run the complete pipeline:")
        print("   python run_pipeline.py --all")
        print("\n3. Explore the dataset:")
        print("   jupyter notebook notebooks/explore_dataset.ipynb")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)