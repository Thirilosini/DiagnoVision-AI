import os
import tempfile
from pathlib import Path
from PIL import Image as PILImage
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file if available
load_dotenv()

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

# Configure Streamlit page
st.set_page_config(
    page_title="DiagnoVision AI - Medical Image Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Medical Analysis Prompt Template
ANALYSIS_PROMPT = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. 
Analyze the provided medical image and structure your response thoroughly according to the following sections:

### 1. Image Type & Region
- **Imaging Modality**: Identify the modality (e.g., X-ray, MRI, CT scan, Ultrasound, Histopathology).
- **Anatomical Region & Positioning**: Specify the exact anatomical region, orientation, and patient positioning.
- **Technical Quality**: Evaluate image contrast, resolution, artifacts, and overall technical adequacy.

### 2. Key Findings
- **Primary Observations**: Document systematic observations across relevant anatomical structures.
- **Abnormalities & Pathology**: Describe detected anomalies with precision (location, margins, morphology).
- **Measurements & Densities**: Note estimated dimensions, radiodensity, echogenicity, or signal intensity where relevant.

### 3. Diagnostic Assessment
- **Primary Diagnosis**: State the most likely diagnosis along with an estimated confidence level.
- **Differential Diagnoses**: Provide a ranked list of differential diagnoses with likelihood levels.
- **Supporting Evidence**: Detail the specific visual findings supporting each diagnostic possibility.
- **Critical / Urgent Findings**: Highlight any acute conditions requiring emergent medical attention.

### 4. Patient-Friendly Explanation
- **Plain-Language Summary**: Translate the clinical findings into understandable, compassionate language.
- **Jargon Glossary**: Explain any complex medical terminology used.
- **Visual Analogies**: Use everyday analogies to help explain the anatomical context or finding.

### 5. Clinical Research & Context
- **Literature References**: Using DuckDuckGo search, identify relevant recent clinical literature or consensus guidelines.
- **Standard Care Pathways**: Summarize standard clinical management or follow-up protocols.
- **Key References**: Provide 2-3 cited references or authoritative medical guidelines.

Ensure the report is rigorous, organized, and clearly formatted with markdown headers and bullet points.
"""

def create_medical_agent(api_key: str, model_id: str = "gemini-2.0-flash") -> Agent:
    """Initializes and returns an Agno Agent configured with Google Gemini and search tools."""
    return Agent(
        model=Gemini(id=model_id, api_key=api_key),
        tools=[DuckDuckGoTools()],
        markdown=True
    )

def process_and_analyze_image(agent: Agent, image_bytes: bytes, file_suffix: str) -> str:
    """Resizes and processes the uploaded image, then runs the AI analysis."""
    temp_input_path = None
    temp_resized_path = None

    try:
        # Save uploaded bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_input:
            temp_input.write(image_bytes)
            temp_input_path = temp_input.name

        # Open and resize image while preserving aspect ratio
        with PILImage.open(temp_input_path) as img:
            # Convert RGBA/P to RGB if necessary for JPEG/PNG uniformity
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            width, height = img.size
            max_width = 800
            if width > max_width:
                aspect_ratio = width / height
                new_width = max_width
                new_height = max(1, int(new_width / aspect_ratio))
                resized_img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
            else:
                resized_img = img.copy()

            # Save resized image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_resized:
                temp_resized_path = temp_resized.name
                resized_img.save(temp_resized_path, format="PNG")

        # Create Agno Image and execute analysis
        agno_image = AgnoImage(filepath=temp_resized_path)
        response = agent.run(ANALYSIS_PROMPT, images=[agno_image])
        return response.content

    finally:
        # Securely cleanup temporary files
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass
        if temp_resized_path and os.path.exists(temp_resized_path):
            try:
                os.remove(temp_resized_path)
            except OSError:
                pass

# --- Streamlit UI Layout ---

st.title("🩺 DiagnoVision AI 🔬")
st.caption("AI-Powered Medical Image Analysis with Agno Agents, Google Gemini Vision & Clinical Web Research")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Handling (supports Streamlit Cloud Secrets, .env, and system environment)
    default_key = ""
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            default_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    if not default_key:
        default_key = os.getenv("GOOGLE_API_KEY", "")

    api_key = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Enter your Google AI Studio API key. You can also define it in .env or Streamlit Cloud Secrets."
    )
    
    # Model Selection
    model_choice = st.selectbox(
        "Gemini Vision Model",
        options=["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
        help="Select the Google Gemini model variant for analysis."
    )
    
    st.divider()
    st.header("📤 Upload Medical Image")
    uploaded_file = st.file_uploader(
        "Supported formats: DICOM previews, X-Ray, CT, MRI (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    
    analyze_button = st.button("🚀 Analyze Image", type="primary", use_container_width=True)

# Main Application Area
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.subheader("🖼️ Image Preview")
    if uploaded_file is not None:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "MIME type": uploaded_file.type
        }
        st.json(file_details)
    else:
        st.info("👈 Please upload an image from the sidebar to inspect and analyze.")

with col_right:
    st.subheader("📋 Analysis & Diagnostic Findings")
    
    if uploaded_file is not None and analyze_button:
        if not api_key:
            st.error("⚠️ Please provide a valid Google Gemini API Key in the sidebar or `.env` file to proceed.")
        else:
            with st.spinner("🔬 Examining modality, identifying key anatomical findings, and querying clinical research..."):
                try:
                    os.environ["GOOGLE_API_KEY"] = api_key
                    agent = create_medical_agent(api_key=api_key, model_id=model_choice)
                    file_suffix = Path(uploaded_file.name).suffix or ".png"
                    
                    report_content = process_and_analyze_image(
                        agent=agent,
                        image_bytes=uploaded_file.getvalue(),
                        file_suffix=file_suffix
                    )
                    
                    st.session_state["latest_report"] = report_content
                    st.session_state["analyzed_filename"] = uploaded_file.name
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    
    # Display cached report if available
    if "latest_report" in st.session_state:
        st.markdown(st.session_state["latest_report"])
        
        # Download report button
        st.download_button(
            label="📥 Download Diagnostic Report (.md)",
            data=st.session_state["latest_report"],
            file_name=f"medical_report_{st.session_state.get('analyzed_filename', 'analysis')}.md",
            mime="text/markdown"
        )
    elif uploaded_file is not None and not analyze_button:
        st.write("Click **Analyze Image** in the sidebar when ready.")

# Clinical Disclaimer Footer
st.divider()
st.warning(
    "⚠️ **Disclaimer for Clinical Safety**: This AI application is designed strictly for research, informational, "
    "and educational purposes. It does **not** constitute official medical advice, definitive clinical diagnosis, "
    "or treatment planning. All findings must be independently validated by a licensed physician, radiologist, or medical professional."
)
