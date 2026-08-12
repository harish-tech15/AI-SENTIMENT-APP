import streamlit as st
from transformers import pipeline

# Webpage Title & Icon
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖")

st.title("🤖 AI Text Sentiment Analyzer")
st.write("Keezha text type pannunga, AI adhu Positive-a illai Negative-a nu sollum!")

# AI Model-a load pannudhu (Hugging Face)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# User input vangura Text Box
user_input = st.text_area("Enter your sentence:", "I am so happy to learn AI today!")

# Button click panna AI analyze pannum
if st.button("Analyze"):
    if user_input.strip() != "":
        with st.spinner("AI thinking..."):
            result = classifier(user_input)[0]
            label = result['label']
            score = round(result['score'] * 100, 2)

            if label == "POSITIVE":
                st.success(f"**Result:** Positive 😊 (Accuracy: {score}%)")
            else:
                st.error(f"**Result:** Negative 😔 (Accuracy: {score}%)")
    else:
        st.warning("Eadhavadhu text type pannunga!")
