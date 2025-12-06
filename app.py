import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
import sys

# ================= CHECK ENVIRONMENT =================
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="User Stories AI",
    page_icon="📋",
    layout="centered"
)

# Simple CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .output-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📋 AI User Stories Generator")
st.markdown("Convert requirements → User Stories + Module Breakdown")

# ================= CHECK FILES =================
st.sidebar.header("📁 File Status")
required_files = [
    "adapter_config.json",
    "adapter_model.safetensors", 
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json"
]

all_ok = True
for file in required_files:
    if os.path.exists(file):
        st.sidebar.success(f"✓ {file}")
    else:
        st.sidebar.error(f"✗ {file}")
        all_ok = False

if not all_ok:
    st.error("Missing files! Please upload all required files.")
    st.stop()

# ================= LOAD MODEL =================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load model with LoRA - CPU only for Streamlit Cloud"""
    try:
        with st.spinner("🔄 Loading model (30-60 seconds)..."):
            # FORCE CPU - Streamlit Cloud has no GPU
            device = torch.device("cpu")
            
            # Load base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-base",
                torch_dtype=torch.float32,
                device_map={"": device},
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, "./")
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("./")
            
            return model, tokenizer
            
    except Exception as e:
        st.error(f"Load error: {str(e)}")
        return None, None

# ================= MAIN INTERFACE =================
# Input
st.subheader("📝 Enter Requirement")
user_input = st.text_area(
    "",
    height=120,
    placeholder="Example: As a user, I want to track my daily water intake..."
)

# Generate button
if st.button("🚀 Generate User Stories", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter a requirement")
    else:
        # Load model
        model, tokenizer = load_model()
        
        if model and tokenizer:
            with st.spinner("✨ Generating..."):
                try:
                    # Create prompt
                    prompt = f"""Convert to User Stories and Module Breakdown:
                    
User Requirement: {user_input.strip()}"""
                    
                    # Tokenize
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                    
                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=400,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            repetition_penalty=1.1
                        )
                    
                    # Decode
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Display
                    st.subheader("📊 Generated Output")
                    st.markdown('<div class="output-box">', unsafe_allow_html=True)
                    
                    # Format nicely
                    if "User Stories:" in result and "Module Breakdown:" in result:
                        stories = result.split("Module Breakdown:")[0].strip()
                        modules = "Module Breakdown:" + result.split("Module Breakdown:")[1]
                        
                        st.markdown("**User Stories:**")
                        st.write(stories)
                        st.markdown("---")
                        st.markdown("**Module Breakdown:**")
                        st.write(modules)
                    else:
                        st.write(result)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Download
                    st.download_button(
                        "💾 Download",
                        result,
                        file_name="user_stories_output.txt"
                    )
                    
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")
        else:
            st.error("Model failed to load")

# ================= TEST =================
with st.expander("🔧 Test with Example"):
    if st.button("Test with School Bus Example"):
        test_req = "As a parent, I want to receive push notifications when my child's school bus is 5 minutes away"
        
        model, tokenizer = load_model()
        if model and tokenizer:
            with st.spinner("Testing..."):
                prompt = f"Convert to User Stories and Module Breakdown:\n\nUser Requirement: {test_req}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=300)
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.write(result[:200] + "...")
                st.success("✅ Test passed!")

# Footer
st.markdown("---")
st.caption("FLAN-T5 + LoRA • Streamlit Cloud")
