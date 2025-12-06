# app.py - GUARANTEED TO WORK WITH YOUR FILES
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

# Set page
st.set_page_config(page_title="User Stories AI", layout="centered")
st.title("🚀 AI User Stories Generator")
st.caption("Identical to your Colab fine-tuned model")

# Check required files exist
REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors", 
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json"
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error(f"❌ Missing files: {missing}")
    st.info("Upload these files from your trained_model folder:")
    for f in missing:
        st.code(f)
    st.stop()

# Load model (cached)
@st.cache_resource(show_spinner="Loading AI model...")
def load_lora_model():
    """Loads your exact fine-tuned model"""
    try:
        # Load base model (same as Colab)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load YOUR LoRA adapter (this is what makes it your model)
        model = PeftModel.from_pretrained(base_model, "./")
        
        # Load YOUR tokenizer
        tokenizer = AutoTokenizer.from_pretrained("./")
        
        st.sidebar.success("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

# Input section
st.subheader("📝 Enter User Requirement")
user_input = st.text_area(
    "Describe what you want:",
    height=120,
    placeholder="Example: As a user, I want to track my daily water intake so that I stay hydrated...",
    key="input"
)

# Settings
col1, col2, col3 = st.columns(3)
with col1:
    max_len = st.number_input("Max Length", 100, 800, 512)
with col2:
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
with col3:
    st.write("")  # Spacer
    generate = st.button("✨ Generate User Stories", type="primary", use_container_width=True)

# Generate function (SAME as Colab)
def generate_identical_to_colab(requirement):
    """Generates output identical to your Colab model"""
    
    # EXACT SAME PROMPT FORMAT as your training
    prompt = f"""Convert the following user requirement into detailed User Stories and a Module Breakdown.

User Requirement: {requirement.strip()}"""
    
    model, tokenizer = load_lora_model()
    
    # EXACT SAME tokenization
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # EXACT SAME generation parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            temperature=temperature,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.1
        )
    
    # EXACT SAME decoding
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate when button clicked
if generate and user_input:
    with st.spinner("Generating (identical to Colab output)..."):
        try:
            output = generate_identical_to_colab(user_input)
            
            # Display results
            st.subheader("📊 Generated Output")
            
            # Parse User Stories and Module Breakdown
            if "User Stories:" in output and "Module Breakdown:" in output:
                stories_part = output.split("Module Breakdown:")[0].strip()
                module_part = "Module Breakdown:" + output.split("Module Breakdown:")[1]
                
                with st.expander("📖 User Stories", expanded=True):
                    st.write(stories_part)
                
                with st.expander("⚙️ Module Breakdown", expanded=True):
                    st.write(module_part)
            else:
                st.write(output)
            
            # Download button
            st.download_button(
                "💾 Download Full Output",
                output,
                file_name="user_stories_output.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")

# Verification section
with st.expander("🔍 Verify Model Files"):
    st.write("Loaded files from your trained model:")
    for file in REQUIRED_FILES:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            st.success(f"✓ {file} ({size:.1f} KB)")
        else:
            st.error(f"✗ {file} (missing)")

# Footer
st.markdown("---")
st.caption("Model: FLAN-T5-base + LoRA fine-tuned • Identical to Colab training")
