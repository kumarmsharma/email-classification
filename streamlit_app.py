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
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Llama Error: {e}")
            return "Error"

class MixtralClassifier(BaseTicketClassifier):
    def predict(self, ticket):
        messages = [{"role": "user", "content": self.create_prompt(ticket)}]
        try:
            response = self.client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=messages,
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Mixtral Error: {e}")
            return "Error"

def load_and_cache_data():
    """Load and cache the dataset."""
    if 'df' not in st.session_state:
        try:
            st.session_state.df = pd.read_csv(
                "https://raw.githubusercontent.com/gungwah/email-ticket-classification-auto-response/refs/heads/main/all_tickets_processed_improved_v3.csv"
            )
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return None
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
    if theme == "Dark":
        st.markdown("""
        <style>
            body {
                background-color: #121212;
                color: #FFFFFF;
            }
            .stApp {
                background-color: #121212;
                color: #FFFFFF;
            }
        </style>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="IT Ticket Classifier", layout="wide")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        theme = st.radio("Select Theme:", ["Light", "Dark"], index=0)
        apply_theme(theme)
        
        st.markdown("---")
        st.header("📊 Model Selection")
        api_key = st.secrets['HFkey']
        selected_models = st.multiselect(
            "Choose models to compare:",
            ["Llama-3.2-3B", "Mixtral-8x7B"],
            default=["Llama-3.2-3B"]
        )
        st.markdown("---")
        uploaded_file = st.file_uploader("Upload a CSV for Custom Analysis", type=["csv"])

        st.write("**Developed by Team JAK**")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎫 Single Ticket", "📊 Batch Analysis", "📂 Dataset Overview"])

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
        if submit_button and api_key and input_text:
            with st.spinner("Classifying ticket..."):
                for model in selected_models:
                    st.subheader(f"{model} Classification")
                    if model == "Llama-3.2-3B":
                        classifier = LlamaClassifier(api_key)
                    else:
                        classifier = MixtralClassifier(api_key)
                    
                    prediction = classifier.predict(input_text)
                    st.success(f"Predicted Category: {prediction}")
                    st.markdown("---")
    
    # Batch Analysis
    with tab2:
        st.header("Batch Analysis")
        df = load_and_cache_data()
        if df is None:
            st.stop()
        
        num_samples = st.slider("Number of tickets to analyze:", 1, 20, 5)
        if st.button("📊 Start Batch Analysis"):
            sample_df = df.head(num_samples)
            results = []
            
            for idx, row in sample_df.iterrows():
                ticket_text = row['Document']
                actual_category = row['Topic_group']
                result = {"Ticket": ticket_text[:100] + "...", "Actual": actual_category}
                
                for model in selected_models:
                    if model == "Llama-3.2-3B":
                        classifier = LlamaClassifier(api_key)
                    else:
                        classifier = MixtralClassifier(api_key)
                    
                    result[model] = classifier.predict(ticket_text)
                
                results.append(result)
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

    # Dataset Overview
    with tab3:
        st.header("Dataset Overview")
        df = load_and_cache_data()
        if df is None:
            st.stop()
        
        st.metric("Total Tickets", len(df))
        st.metric("Categories", df['Topic_group'].nunique())
        st.metric("Most Common Category", df['Topic_group'].mode()[0])
        st.subheader("Category Distribution")
        plot_category_distribution(df)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p>Developed by <strong>Team JAK</strong> | Powered by <strong>Streamlit</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
