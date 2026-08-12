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
import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="AI Sentiment & Visual Insights",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visual AI Sentiment & Emotion Analyzer")
st.write("Text enter pannunga — AI sentiment analyze panni interactive charts kaattum!")

# 2. Model Loading
@st.cache_resource
def load_model():
    # Return sentiment pipeline with all scores
    return pipeline("sentiment-analysis", return_all_scores=True)

classifier = load_model()

# 3. User Input Area
user_input = st.text_area(
    "Enter your feedback / review / paragraph:", 
    "I absolutely love this product! It works so smoothly and saves me a lot of time, though it was a bit expensive.",
    height=120
)

if st.button("Analyze & Generate Visuals", type="primary"):
    if user_input.strip() != "":
        with st.spinner("AI analyzing and generating charts..."):
            
            # Model Output
            results = classifier(user_input)[0]
            
            # Extract scores
            scores = {res['label']: round(res['score'] * 100, 2) for res in results}
            pos_score = scores.get('POSITIVE', 0)
            neg_score = scores.get('NEGATIVE', 0)
            
            dominant_sentiment = "POSITIVE" if pos_score > neg_score else "NEGATIVE"

            st.markdown("---")
            
            # --- FEATURE 1: Metric KPI Cards ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Sentiment", dominant_sentiment, delta="😊 Positive" if dominant_sentiment == "POSITIVE" else "😔 Negative")
            col2.metric("Positive Score", f"{pos_score}%")
            col3.metric("Negative Score", f"{neg_score}%")

            st.markdown("---")

            # --- FEATURE 2: Gauge Meter & Bar Chart (2 Column Layout) ---
            left_col, right_col = st.columns(2)

            with left_col:
                st.subheader("🎯 Sentiment Confidence Gauge")
                # Plotly Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pos_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Positivity Meter (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ecc71" if pos_score > 50 else "#e74c3c"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffcccc"},
                            {'range': [50, 100], 'color': "#ccffcc"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with right_col:
                st.subheader("📊 Score Comparison")
                # Plotly Donut Chart
                fig_bar = px.pie(
                    values=[pos_score, neg_score], 
                    names=['Positive', 'Negative'],
                    color=['Positive', 'Negative'],
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                    hole=0.4
                )
                fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)

            # --- FEATURE 3: Word Cloud Visual ---
            st.markdown("---")
            st.subheader("☁️ Text Word Cloud")
            
            wordcloud = WordCloud(
                width=800, 
                height=300, 
                background_color='white' if st.get_option("theme.base") == "light" else '#0e1117',
                colormap='viridis'
            ).generate(user_input)

            fig_wc, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)

    else:
        st.warning("Eadhavadhu text enter pannunga!")
