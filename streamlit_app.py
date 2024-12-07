import streamlit as st

def main():
    st.set_page_config(page_title="IT Ticket Classifier", layout="wide")

    # Sidebar for Theme Switching and Configurations
    with st.sidebar:
        st.header("Configuration")
        theme_choice = st.radio("Select Theme:", ["Light", "Dark"], index=0)
        api_key = st.secrets['HFkey']
        st.markdown("---")
        st.header("Model Selection")
        selected_models = st.multiselect(
            "Choose models to compare:",
            ["Llama-3.2-3B", "Mixtral-8x7B"],
            default=["Llama-3.2-3B"]
        )
        uploaded_file = st.file_uploader("Upload a CSV for Custom Analysis", type="csv")

    # Apply Theme
    theme_styles = {
        "Light": {"background": "#FFFFFF", "text": "#000000", "sidebar_bg": "#F8F9FA", "sidebar_text": "#000000"},
        "Dark": {"background": "#1E1E1E", "text": "#FFFFFF", "sidebar_bg": "#F8F9FA", "sidebar_text": "#000000"}
    }
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme_styles[theme_choice]["background"]};
            color: {theme_styles[theme_choice]["text"]};
        }}
        .css-1d391kg {{
            background-color: {theme_styles[theme_choice]["sidebar_bg"]};
            color: {theme_styles[theme_choice]["sidebar_text"]};
        }}
        .css-1d391kg h2 {{
            color: {theme_styles[theme_choice]["sidebar_text"]};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.image(
        "https://media.assettype.com/analyticsinsight%2Fimport%2Fwp-content%2Fuploads%2F2020%2F08%2FIT-TICKET-CLASSIFICATION.jpg?w=1024&auto=format%2Ccompress&fit=max",
        use_column_width=True
    )
    st.title("🎫 IT Support Ticket Classifier")
    st.markdown("""
    Welcome to the **IT Support Ticket Classifier**!
    This application leverages **LLMs** to classify IT support tickets into predefined categories.
    Analyze ticket data, compare model performance, and explore insights easily.
    """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center;'>
            <p>Developed by <strong>Team JAK</strong> | Powered by <strong>Streamlit</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
