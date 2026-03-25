"""
Main script to run the complete document restoration pipeline.
Based on Cosmas Kiptoo Sang's research project.
"""

import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import run_full_pipeline
from scripts.train_deskew_model import main as train_model

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Document Restoration Pipeline')
    parser.add_argument('--train', action='store_true', 
                       help='Train the deskewing model')
    parser.add_argument('--run-pipeline', action='store_true',
                       help='Run the complete pipeline')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the pipeline on test data')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (train + run)')
    
    return parser.parse_args()

def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Document Restoration Pipeline")
    logger.info("Based on research by Cosmas Kiptoo Sang")
    logger.info("Kisii University, March 2026")
    logger.info("=" * 60)
    
    try:
        if args.train or args.all:
            logger.info("Starting model training...")
            # Import and run training
            from scripts.train_deskew_model import main as train_main
            train_main()  # This would need to be adapted from train_deskew_model.py
            logger.info("Model training completed")
        
        if args.run_pipeline or args.all:
            logger.info("Starting document restoration pipeline...")
            results = run_full_pipeline()
            logger.info(f"Pipeline completed. Results saved.")
            
        if args.evaluate:
            logger.info("Evaluating pipeline...")
            # Add evaluation code here
            pass
            
        if args.visualize:
            logger.info("Generating visualizations...")
            # Add visualization code here
            pass
            
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise
    
    logger.info("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()