import streamlit as st
import pickle
import streamlit.components.v1 as components

# Set the page configuration
st.set_page_config(page_title="Spam Detector", page_icon="🕵️‍♀️", layout="wide", initial_sidebar_state="expanded")

# Load the model and vectorizer
try:
    model = pickle.load(open('spam.pkl', 'rb'))
    cv = pickle.load(open('vec.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Model or vectorizer file not found. Please upload 'spam.pkl' and 'vec.pkl'. Error: {str(e)}")
    st.stop()  # Stop execution if files are not found
except Exception as e:
    st.error(f"An error occurred while loading the model or vectorizer: {str(e)}")
    st.stop()  # Stop execution if there is an error with loading

# Custom HTML meta tags for social media sharing
meta_tags = """
<meta property="og:title" content="MailGuard AI: Spam Email Classifier" />
<meta property="og:description" content="Classify your emails as Spam or Not Spam using AI. Protect your inbox from unwanted emails." />
<meta property="og:url" content="https://mailguard.streamlit.app/" />
"""
components.html(meta_tags, height=0)

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 12px 24px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .email-input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #f8f8f8;
        font-size: 16px;
        width: 100%;
    }
    .header-image {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .main-container {
        background-color: #f1f1f1;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("About MailGuard AI")
st.sidebar.markdown("""
MailGuard AI is a powerful machine learning application designed to help you:
- **Identify Spam Emails**.
- **Filter out unwanted content**.
- **Protect your inbox** from phishing and junk emails.
""")
st.sidebar.info("🔒 **Privacy**: Your data is not stored or shared.")

# Main function
def main():
    st.title("📧 MailGuard AI - Spam Email Classifier")
    
    st.markdown("""
        Welcome to **MailGuard AI**, a tool that uses **Machine Learning** to classify emails as either **Spam** or **Not Spam**.
    """)
    
    # Header image
    st.image("https://via.placeholder.com/1000x300?text=MailGuard+AI", use_column_width=True, caption="Spam Email Classification")

    # Main content area
    with st.container():
        st.markdown("### 🔍 Email Classification")
        user_input = st.text_area("✏️ Enter the email content below:", height=250, key="email_input", label_visibility="collapsed", 
                                  placeholder="Paste or type the email content here...", max_chars=3000)

        # Display a better-looking classify button
        classify_button = st.button("🚀 Classify Email", use_container_width=True)

        if classify_button:
            if user_input.strip():
                try:
                    # Prepare input for model
                    data = [user_input]
                    vec = cv.transform(data).toarray()
                    
                    # Predict the class
                    result = model.predict(vec)
                    confidence = model.predict_proba(vec).max() * 100
                    
                    # Display results in a result box
                    with st.container():
                        st.markdown("### 📤 Email Content:")
                        st.markdown(f"```\n{user_input}\n```")

                        st.markdown("### 📊 Classification Result:")

                        result_box_class = "result-box"
                        if result[0] == 0:
                            st.success(f"✅ This is a **Not Spam** email ({confidence:.2f}% confidence).", icon="✅")
                        else:
                            st.error(f"🚫 This is a **Spam** email ({confidence:.2f}% confidence).", icon="🚫")
                    
                except Exception as e:
                    st.error(f"An error occurred while classifying the email: {str(e)}")
            else:
                st.warning("⚠️ Please enter some email content to classify.")
    
    # Footer section
    st.markdown("---")
    st.markdown("Developed with ❤️ using **Streamlit**")

# Run the main function
if __name__ == "__main__":
    main()

