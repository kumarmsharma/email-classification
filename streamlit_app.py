import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient
import time
from datetime import datetime
import altair as alt

# Apply the theme dynamically
def apply_theme(theme):
    if theme == "Dark":
        dark_css = """
        <style>
        body {
            background-color: #181818;
            color: #ffffff;
        }
        .stSidebar {
            background-color: #2c2c2c;
        }
        .stButton button {
            background-color: #333333;
            color: #ffffff;
        }
        .stTextArea textarea {
            background-color: #333333;
            color: #ffffff;
        }
        .stDataFrame {
            background-color: #222222;
            color: #ffffff;
        }
        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)
    elif theme == "Light":
        light_css = """
        <style>
        body {
            background-color: #ffffff;
            color: #000000;
        }
        .stSidebar {
            background-color: #f9f9f9;
        }
        .stButton button {
            background-color: #e6e6e6;
            color: #000000;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            color: #000000;
        }
        .stDataFrame {
            background-color: #f9f9f9;
            color: #000000;
        }
        </style>
        """
        st.markdown(light_css, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="IT Ticket Classifier", layout="wide")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        theme = st.radio("Select Theme:", ["Light", "Dark"], index=0)
        apply_theme(theme)  # Apply theme dynamically

        st.markdown("---")
        st.header("📊 Model Selection")
        api_key = st.secrets['HFkey']
        selected_models = st.multiselect(
            "Choose models to compare:",
            ["Llama-3.2-3B", "Mixtral-8x7B"],
            default=["Llama-3.2-3B"]
        )
        st.markdown("---")
        st.write("**Developed by Team JAK**")

    # Main content tabs
    st.title("🎫 IT Support Ticket Classifier")
    st.markdown("""
    Welcome to the **IT Support Ticket Classifier**!  
    This application leverages **LLMs** (Llama-3.2-3B and Mixtral-8x7B) to classify IT support tickets into predefined categories.  
    Analyze ticket data, compare model performance, and explore insights easily.
    """)

    tab1, tab2, tab3 = st.tabs(
        ["🎫 Single Ticket", "📊 Batch Analysis", "📂 Dataset Overview"]
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
        num_samples = st.slider("Number of tickets to analyze:", 1, 20, 5)
        if st.button("📊 Start Batch Analysis") and api_key:
            sample_df = df.head(num_samples)
            results = []

            for idx, row in enumerate(sample_df.iterrows()):
                ticket_text = row[1]['Document']
                actual_category = row[1]['Topic_group']
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
