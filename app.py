import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import os

# Set page config FIRST
st.set_page_config(
    page_title="User Stories Generator",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better appearance
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("📋 AI User Stories Generator")
st.markdown("Generate detailed User Stories and Module Breakdowns from requirements")

# Sidebar with settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model loading option
    use_cpu = st.checkbox("Use CPU (slower but works everywhere)", value=False)
    
    # Generation parameters
    st.subheader("Generation Parameters")
    max_length = st.slider("Output Length", 200, 800, 512)
    temperature = st.slider("Creativity", 0.1, 1.0, 0.7)
    
    # Info section
    st.markdown("---")
    st.info("""
    **Model Info:**
    - Base: FLAN-T5-base
    - Fine-tuned with LoRA
    - Trained on user requirements
    """)
    
    # File check
    st.subheader("📁 File Status")
    required_files = {
        "adapter_config.json": "LoRA configuration",
        "adapter_model.safetensors": "LoRA weights",
        "tokenizer_config.json": "Tokenizer settings",
        "spiece.model": "Tokenizer model"
    }
    
    for file, desc in required_files.items():
        if os.path.exists(file):
            st.success(f"✓ {file}")
        else:
            st.error(f"✗ {file}")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 Input Requirements")
    
    # Example selector
    example = st.selectbox(
        "Choose an example or write your own:",
        ["Write your own requirement",
         "School bus notifications for parents",
         "Freelancer invoicing app",
         "Plant care assistant with photo scan",
         "Expense splitting app for groups"]
    )
    
    # Set example text
    examples = {
        "School bus notifications for parents": "As a parent, I want to receive push notifications when my child's school bus is 5 minutes away so that I can be ready at the stop.",
        "Freelancer invoicing app": "As a freelancer, I want to create and send professional invoices directly from the app with automatic tax calculation and payment reminders so that I get paid faster and look more professional.",
        "Plant care assistant with photo scan": "Take photo of houseplant and get instant care tips, watering reminders, and disease detection.",
        "Expense splitting app for groups": "Create group expense splitting where one person pays and the app automatically calculates who owes what and sends payment requests."
    }
    
    # Text input area
    user_input = st.text_area(
        "Enter your user requirement:",
        height=150,
        value=examples.get(example, ""),
        placeholder="Example: As a user, I want to track my daily habits...",
        key="user_input"
    )

with col2:
    st.subheader("🚀 Generate")
    
    # Generate button
    generate_btn = st.button(
        "Generate User Stories & Modules",
        type="primary",
        use_container_width=True,
        disabled=not user_input.strip()
    )
    
    # Quick tips
    with st.expander("💡 Tips for best results"):
        st.markdown("""
        1. Start with "As a [user], I want [feature] so that [benefit]"
        2. Be specific about the functionality
        3. Mention the target user role
        4. Include the expected benefit
        """)

# Model loading function with error handling
@st.cache_resource(show_spinner=False)
def load_model(use_cpu=False):
    """Load the fine-tuned model with LoRA"""
    try:
        # Show loading message
        with st.spinner("🔄 Loading AI model (this may take 30-60 seconds)..."):
            
            # Set device
            device = "cpu" if use_cpu else "auto"
            
            # Load base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-base",
                device_map=device,
                torch_dtype=torch.float16 if not use_cpu else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, "./")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("./")
            
            return model, tokenizer
            
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None

# Generation function
def generate_output(requirement, model, tokenizer):
    """Generate user stories and module breakdown"""
    # Create prompt (same format as training)
    prompt = f"""Convert the following user requirement into detailed User Stories and a Module Breakdown.

User Requirement: {requirement.strip()}"""
    
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.92,
                repetition_penalty=1.1,
                num_return_sequences=1
            )
        
        # Decode
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        raise Exception(f"Generation error: {str(e)}")

# Handle generation when button is clicked
if generate_btn and user_input.strip():
    try:
        # Load model
        model, tokenizer = load_model(use_cpu)
        
        if model and tokenizer:
            # Generate output
            with st.spinner("✨ Generating User Stories..."):
                output = generate_output(user_input, model, tokenizer)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Generated Output")
            
            # Success message
            st.markdown('<div class="success-box">✅ Successfully generated!</div>', unsafe_allow_html=True)
            
            # Display in a nice format
            if "User Stories:" in output and "Module Breakdown:" in output:
                # Split the output
                parts = output.split("Module Breakdown:")
                stories = parts[0].strip()
                modules = "Module Breakdown:" + parts[1] if len(parts) > 1 else ""
                
                # Display User Stories
                with st.expander("📖 **User Stories**", expanded=True):
                    st.markdown(stories)
                
                # Display Module Breakdown
                with st.expander("⚙️ **Module Breakdown**", expanded=True):
                    st.markdown(modules)
            else:
                # Display as-is
                st.markdown(output)
            
            # Download button
            st.download_button(
                label="💾 Download Output",
                data=output,
                file_name="user_stories_output.txt",
                mime="text/plain",
                key="download_btn"
            )
            
        else:
            st.error("Failed to load model. Please check your files.")
            
    except Exception as e:
        st.error(f"❌ Error during generation: {str(e)}")
        st.info("Try enabling 'Use CPU' in the sidebar if you're having memory issues.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Powered by FLAN-T5 fine-tuned with LoRA • <a href="#" target="_blank">View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

# Run a quick test if in development mode
if st.sidebar.button("🔧 Run Quick Test"):
    with st.sidebar:
        st.info("Testing model loading...")
        model, tokenizer = load_model(use_cpu)
        if model:
            st.success("✅ Model loaded successfully!")
            st.caption(f"Device: {model.device}")
        else:
            st.error("❌ Model loading failed")
