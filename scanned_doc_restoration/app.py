"""
Streamlit Web App for Document Restoration Pipeline
Upload scanned documents → AI processes → Extract copyable text
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set page config
st.set_page_config(
    page_title="Document Restoration AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .image-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    .image-box {
        text-align: center;
        padding: 1rem;
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
    }
    .text-output {
        background-color: #1F2937;
        color: #F9FAFB;
        padding: 1.5rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 300px;
        overflow-y: auto;
        margin: 1rem 0;
    }
    .copy-button {
        background-color: #3B82F6;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        cursor: pointer;
        font-size: 0.9rem;
    }
    .copy-button:hover {
        background-color: #2563EB;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">📄 Document Restoration AI</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>Based on research by Cosmas Kiptoo Sang</strong><br>
    This AI system restores scanned documents by:
    1. Removing speckle noise
    2. Correcting rotation (deskewing)
    3. Extracting text with high accuracy
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/document.png", width=100)
    st.markdown("### 🛠️ Pipeline Settings")
    
    # Denoising method selection
    denoising_method = st.selectbox(
        "Denoising Method",
        ["non_local_means", "median_filter", "bilateral_filter", "gaussian_blur"],
        help="Choose the denoising algorithm"
    )
    
    # Processing options
    use_deskewing = st.checkbox("Apply Deskewing", value=True, 
                                help="Correct document rotation using trained AI model")
    use_clahe = st.checkbox("Enhance Contrast (CLAHE)", value=True,
                           help="Improve text visibility")
    
    # Model status
    st.markdown("---")
    st.markdown("### 🤖 AI Model Status")
    
    # Check if model is trained (check both old and new format)
    model_path_new = project_root / "models" / "deskew_model_best.weights.h5"
    model_path_old = project_root / "models" / "deskew_model_best.h5"
    
    if model_path_new.exists() or model_path_old.exists():
        st.success("✅ Deskewing model is trained and ready")
    else:
        st.warning("⚠️ Model not trained yet. Run training first.")
    
    # Check OCR availability
    try:
        import pytesseract
        st.success("✅ OCR (Tesseract) is available")
    except ImportError:
        st.warning("⚠️ OCR not available (pytesseract not installed)")
        st.info("Install: `pip install pytesseract`")
    
    # Training button
    if st.button("🔄 Train AI Model", type="secondary"):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import and run training with progress callback
            from scripts.train_model import train_model
            
            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)
            
            status_text.text("Starting training...")
            success = train_model(progress_callback=update_progress)
            
            if success:
                progress_bar.progress(1.0)
                status_text.empty()
                st.success("✅ Model training completed!")
                st.balloons()
            else:
                status_text.empty()
                st.error("Training failed: No training data found")
        except Exception as e:
            status_text.empty()
            st.error(f"Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# Main content area
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results", "ℹ️ About"])

with tab1:
    st.markdown('<h2 class="sub-header">Upload Scanned Document</h2>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a scanned document image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload a scanned document with text"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Display uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)
        
        with col2:
            # Process button
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Import pipeline components
                    from src.pipeline import DocumentRestorationPipeline
                    from src.denoising import load_image
                    import cv2
                    import numpy as np
                    
                    status_text.text("Loading image...")
                    progress_bar.progress(0.1)
                    
                    # Initialize pipeline
                    status_text.text("Initializing pipeline...")
                    progress_bar.progress(0.2)
                    
                    pipeline = DocumentRestorationPipeline(
                        denoising_method=denoising_method,
                        use_deskewing=use_deskewing,
                        use_clahe=use_clahe
                    )
                    
                    # Process image
                    status_text.text(f"Applying {denoising_method} denoising...")
                    progress_bar.progress(0.4)
                    
                    results = pipeline.process_image(Path(tmp_path))
                    
                    status_text.text("Processing complete!")
                    progress_bar.progress(1.0)
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    st.session_state['processed'] = True
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Document processed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Processing failed: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    st.session_state['processed'] = False
        
        # Clean up temp file
        os.unlink(tmp_path)

with tab2:
    st.markdown('<h2 class="sub-header">Processing Results</h2>', unsafe_allow_html=True)
    
    if 'processed' in st.session_state and st.session_state['processed']:
        results = st.session_state['results']
        
        # Display images
        st.markdown("### 📸 Visual Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if results.get("original_image") is not None:
                st.image(results["original_image"], caption="Original", use_container_width=True)
        
        with col2:
            if results.get("denoised_image") is not None:
                st.image(results["denoised_image"], caption="Denoised", use_container_width=True)
        
        with col3:
            if results.get("deskewed_image") is not None:
                st.image(results["deskewed_image"], caption="Deskewed", use_container_width=True)
        
        with col4:
            if results.get("final_image") is not None:
                st.image(results["final_image"], caption="Final", use_container_width=True)
        
        # Display metrics
        st.markdown("### 📊 Performance Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Character Error Rate", f"{results.get('cer', 0):.1f}%")
        
        with metric_col2:
            if 'psnr' in results:
                st.metric("PSNR", f"{results['psnr']:.1f} dB")
        
        with metric_col3:
            if 'ssim' in results:
                st.metric("SSIM", f"{results['ssim']:.3f}")
        
        with metric_col4:
            if 'predicted_angle' in results:
                st.metric("Rotation Angle", f"{results['predicted_angle']:.1f}°")
        
        # Extracted text
        st.markdown("### 📝 Extracted Text")
        extracted_text = results.get("extracted_text", "")
        
        if extracted_text:
            # Text display in a code block for easy copying
            st.code(extracted_text, language=None)
            
            # Copy to clipboard functionality
            col1, col2 = st.columns([3, 1])
            with col1:
                st.download_button(
                    label="📥 Download Text",
                    data=extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                # Info about copying
                st.info("💡 Click text above to select, then Ctrl+C to copy")
        else:
            st.warning("No text was extracted from the document.")
        
        # Processing time
        st.markdown("### ⏱️ Processing Time")
        time_col1, time_col2, time_col3 = st.columns(3)
        
        with time_col1:
            st.metric("Denoising", f"{results.get('denoising_time', 0):.2f}s")
        
        with time_col2:
            if 'deskewing_time' in results:
                st.metric("Deskewing", f"{results['deskewing_time']:.2f}s")
        
        with time_col3:
            st.metric("Total", f"{results.get('total_time', 0):.2f}s")
    
    else:
        st.info("👈 Upload and process a document first to see results here.")

with tab3:
    st.markdown('<h2 class="sub-header">About This System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Research Basis
    This system is based on the research paper:
    **"Scanned Document Restoration for Improved Optical Character Recognition"**
    by Cosmas Kiptoo Sang, Kisii University, March 2026.
    
    ### 🔧 Technical Pipeline
    1. **Denoising**: Removes speckle noise using advanced algorithms
    2. **Deskewing**: Corrects document rotation using a CNN AI model
    3. **OCR**: Extracts text using Tesseract with improved accuracy
    
    ### 📈 Performance
    - **Baseline accuracy**: 34.2% Character Error Rate (raw documents)
    - **After processing**: 8.6% Character Error Rate
    - **Improvement**: 25.6 percentage points reduction in errors
    
    ### 🖼️ Dataset
    - **600 scanned documents** with real-world degradation
    - **500 training images** with angle labels
    - **100 test images** for evaluation
    - Real speckle noise and rotation from institutional scanning
    
    ### 🚀 How to Use
    1. **Train the AI model** (sidebar button) - uses 500 labeled documents
    2. **Upload a scanned document** - PNG, JPG, TIFF, BMP formats
    3. **Process the document** - AI cleans and straightens the image
    4. **Copy extracted text** - Clean, corrected text ready to use
    
    ### 📚 Technology Stack
    - **Python 3.9+** with OpenCV, TensorFlow, Tesseract
    - **CNN Model**: 4-layer convolutional neural network
    - **Streamlit**: Web interface for easy interaction
    - **Non-Local Means**: Best-performing denoising algorithm
    
    ### 📄 License
    MIT License - Free for academic and commercial use
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    Document Restoration AI System • Based on research by Cosmas Kiptoo Sang • Kisii University 2026
    </div>
    """,
    unsafe_allow_html=True
)