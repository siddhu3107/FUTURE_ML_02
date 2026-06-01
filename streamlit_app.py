import streamlit as st
import joblib
import pandas as pd
from preprocess import TextPreprocessor

# Page Config
st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="centered")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .priority-high { color: #dc3545; font-weight: bold; }
    .priority-medium { color: #ffc107; font-weight: bold; }
    .priority-low { color: #28a745; font-weight: bold; }
    .priority-critical { color: #8b0000; font-weight: bold; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_models():
    type_model = joblib.load('type_model.joblib')
    priority_model = joblib.load('priority_model.joblib')
    preprocessor = TextPreprocessor()
    return type_model, priority_model, preprocessor

try:
    type_model, priority_model, preprocessor = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure you have run train.py first.")
    st.stop()

# Header
st.title("🎫 Support Ticket Classifier")
st.markdown("Enter a customer support ticket description below to automatically categorize it and assign a priority level.")

# Input Section
with st.container():
    ticket_text = st.text_area("Ticket Description", placeholder="e.g. My order #12345 hasn't arrived yet and I've been waiting for two weeks...", height=150)
    product_name = st.text_input("Product Name (Optional)", placeholder="e.g. Laptop, iPhone 13")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit = st.button("Classify Ticket")

# Logic
if submit:
    if not ticket_text.strip():
        st.warning("Please enter a ticket description.")
    else:
        with st.spinner("Analyzing ticket..."):
            # Clean and Predict
            cleaned_text = [preprocessor.clean_text(ticket_text, product_name if product_name else None)]
            
            t_type = type_model.predict(cleaned_text)[0]
            priority = priority_model.predict(cleaned_text)[0]
            
            # Display Results
            st.markdown("### 📊 Classification Results")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                <div class="result-card">
                    <h4>Ticket Category</h4>
                    <p style="font-size: 1.2em; color: #007bff;">{t_type}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                p_class = f"priority-{priority.lower()}"
                st.markdown(f"""
                <div class="result-card">
                    <h4>Assigned Priority</h4>
                    <p class="{p_class}" style="font-size: 1.2em;">{priority}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("Ticket processed successfully!")

# Sidebar Info
st.sidebar.title("About")
st.sidebar.info("""
This system uses a **Random Forest** model trained on NLTK-preprocessed support ticket data.
- **Model Accuracy**: ~23%
- **NLP Engine**: NLTK (Lemmatization + Stopword removal)
""")
