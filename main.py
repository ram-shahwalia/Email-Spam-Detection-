
import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl")

st.title("📧 Email Spam Detection")
st.write("Check whether an email is Spam or Not Spam using Machine Learning.")

if not os.path.exists("spam_model.pkl"):
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()

model = load_model()

email_text = st.text_area(
    "Paste your complete email message below:",
    height=220,
    placeholder="Enter your email message here..."
)

if st.button("Check Email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter an email message first.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        confidence = max(probabilities) * 100

        if prediction == "Spam":
            st.error("🚨 This email is SPAM!")
        else:
            st.success("✅ This email is NOT SPAM!")

        st.write(f"Prediction Confidence: {confidence:.2f}%")

st.caption("Powered by Python, TF-IDF and Multinomial Naive Bayes")
