import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import time

# Force CPU usage (Streamlit Cloud has no GPU)
DEVICE = "cpu"

# Page config
st.set_page_config(
    page_title="User Stories AI",
    page_icon="📋",
    layout="centered"
)

# Title
st.title("📋 User Stories Generator")
st.markdown("Convert user requirements into User Stories & Module Breakdowns")

# Check if files exist
st.sidebar.header("📁 File Check")
required_files = [
    "adapter_config.json",
    "adapter_model.safetensors", 
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json"
]

missing_files = []
for file in required_files:
    if os.path.exists(file):
        st.sidebar.success(f"✓ {file}")
    else:
        st.sidebar.error(f"✗ {file}")
        missing_files.append(file)

if missing_files:
    st.error(f"Missing files: {', '.join(missing_files)}")
    st.info("Upload these 6 files from your trained_model folder:")
    for f in missing_files:
        st.code(f)
    st.stop()

# Load model with caching
@st.cache_resource(show_spinner=False)
def load_model_cpu():
    """Load model on CPU only"""
    try:
        st.sidebar.info("Loading model (this may take 1-2 minutes)...")
        
        # Load base model on CPU
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter (using direct PEFT loading)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, "./")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("./")
        
        st.sidebar.success("✅ Model loaded!")
        return model, tokenizer
        
    except Exception as e:
        st.sidebar.error(f"Load error: {str(e)[:100]}...")
        return None, None

# Main input
st.subheader("Enter User Requirement")
user_input = st.text_area(
    "",
    height=100,
    placeholder="Example: As a user, I want to track my daily water intake..."
)

# Generate button
if st.button("Generate", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter a requirement")
    else:
        with st.spinner("Loading model..."):
            model, tokenizer = load_model_cpu()
        
        if model and tokenizer:
            with st.spinner("Generating..."):
                try:
                    # Create prompt
                    prompt = f"Convert to User Stories and Module Breakdown:\n\n{user_input}"
                    
                    # Tokenize
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                    
                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=400,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9
                        )
                    
                    # Decode
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Display
                    st.subheader("Generated Output:")
                    st.write(result)
                    
                    # Download
                    st.download_button(
                        "Download",
                        result,
                        file_name="output.txt"
                    )
                    
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")
        else:
            st.error("Failed to load model")

# Footer
st.markdown("---")
st.caption("FLAN-T5 + LoRA Fine-tuned • Running on CPU")
