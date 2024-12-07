import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient
import time
from datetime import datetime
import altair as alt

class BaseTicketClassifier:
    def __init__(self, api_key):
        self.client = InferenceClient(api_key=api_key)
        self.categories = [
            "Hardware", "HR Support", "Access", "Miscellaneous",
            "Storage", "Purchase", "Internal Project", "Administrative rights"
        ]
    
    def create_prompt(self, ticket_text):
        return f"""You are an expert IT support ticket classifier. Your task is to classify the following support ticket into exactly one of these categories:

        Categories and their definitions:
        - Hardware: Issues with physical devices, computers, printers, or equipment
        - HR Support: Human resources related requests, employee matters
        - Access: Login issues, permissions, account access, passwords
        - Miscellaneous: General inquiries or requests that don't fit other categories
        - Storage: Data storage, disk space, file storage related issues
        - Purchase: Procurement requests, buying equipment or software
        - Internal Project: Project-related tasks and updates
        - Administrative rights: Requests for admin privileges or system permissions

        Support Ticket: "{ticket_text}"

        Instructions:
        1. Read the ticket carefully
        2. Match the main issue with the category definitions above
        3. Respond with only the category name, nothing else
        4. If a ticket could fit multiple categories, choose the most specific one
        5. Focus on the primary issue, not secondary mentions

        Category:"""

class LlamaClassifier(BaseTicketClassifier):
    def predict(self, ticket):
        messages = [{"role": "user", "content": self.create_prompt(ticket)}]
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",  # Updated model
                messages=messages,
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Llama Error: {e}")
            return "Error"

class MixtralClassifier(BaseTicketClassifier):  # Renamed from MistralClassifier
    def predict(self, ticket):
        messages = [{"role": "user", "content": self.create_prompt(ticket)}]
        try:
            response = self.client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Updated model
                messages=messages,
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Mixtral Error: {e}")
            return "Error"

def load_and_cache_data():
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(
            "https://raw.githubusercontent.com/gungwah/email-ticket-classification-auto-response/refs/heads/main/all_tickets_processed_improved_v3.csv"
        )
    return st.session_state.df

def plot_category_distribution(df):
    category_counts = df['Topic_group'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    chart = alt.Chart(category_counts).mark_bar().encode(
        x='Category',
        y='Count',
        color=alt.value('#1f77b4')
    ).properties(
        title='Distribution of Ticket Categories'
    )
    
    st.altair_chart(chart, use_container_width=True)

def apply_theme(theme):
    if theme == "Light":
        st.markdown(
            """
            <style>
            body { background-color: #ffffff; color: #000000; }
            .stSidebar { background-color: #f8f9fa; color: #000000; }
            </style>
            """, unsafe_allow_html=True)
    elif theme == "Dark":
        st.markdown(
            """
            <style>
            body { background-color: #121212; color: #ffffff; }
            .stSidebar { background-color: #1e1e1e; color: #ffffff; }
            </style>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="IT Ticket Classifier", layout="wide")
    
    theme_choice = st.sidebar.radio("Select Theme:", ["Light", "Dark"], index=0)
    apply_theme(theme_choice)
    
    st.image(
        "https://your-banner-image-link-here",  # Replace with the proper banner URL
        use_column_width=True
    )
    st.title("🎫 IT Support Ticket Classifier")
    st.markdown("""
    Welcome to the **IT Support Ticket Classifier**!  
    This application leverages **LLMs** (Llama-3.2-3B and Mixtral-8x7B) to classify IT support tickets into predefined categories.  
    Analyze ticket data, compare model performance, and explore insights easily.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        st.header("📊 Model Selection")
        selected_models = st.multiselect(
            "Choose models to compare:",
            ["Llama-3.2-3B", "Mixtral-8x7B"],
            default=["Llama-3.2-3B"]
        )
        st.markdown("---")
        st.write("**Developed by Team JAK**")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "🎫 Single Ticket",
            "📊 Batch Analysis",
            "📂 Dataset Overview"
        ]
    )

    # Single Ticket Analysis
    with tab1:
        st.header("Single Ticket Classification")
        input_text = st.text_area(
            "Enter support ticket text 📝:",
            height=100,
            placeholder="Type or paste the support ticket here...",
            help="Input the text of the IT support ticket for classification."
        )
        submit_button = st.button("🎯 Classify Ticket")
        if submit_button:
            st.success(f"Prediction logic to be added!")

if __name__ == "__main__":
    main()
