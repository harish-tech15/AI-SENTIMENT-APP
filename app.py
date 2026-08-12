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

# 2. Model Loading (Updated for Latest Transformers)
@st.cache_resource
def load_model():
    # 'top_k=None' is the modern way to get all label scores
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", top_k=None)

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
            results = classifier(user_input)[0]  # list of dicts [{'label': 'POSITIVE', 'score': 0.99}, ...]
            
            # Extract scores safely
            scores = {}
            for res in results:
                label_name = res['label'].upper()
                scores[label_name] = round(res['score'] * 100, 2)
            
            pos_score = scores.get('POSITIVE', 0.0)
            neg_score = scores.get('NEGATIVE', 0.0)
            
            dominant_sentiment = "POSITIVE" if pos_score >= neg_score else "NEGATIVE"

            st.markdown("---")
            
            # --- FEATURE 1: Metric KPI Cards ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Sentiment", dominant_sentiment, delta="😊 Positive" if dominant_sentiment == "POSITIVE" else "😔 Negative")
            col2.metric("Positive Score", f"{pos_score}%")
            col3.metric("Negative Score", f"{neg_score}%")

            st.markdown("---")

            # --- FEATURE 2: Gauge Meter & Donut Chart ---
            left_col, right_col = st.columns(2)

            with left_col:
                st.subheader("🎯 Sentiment Confidence Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pos_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Positivity Meter (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ecc71" if pos_score >= 50 else "#e74c3c"},
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
                fig_pie = px.pie(
                    values=[pos_score, neg_score], 
                    names=['Positive', 'Negative'],
                    color=['Positive', 'Negative'],
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                    hole=0.4
                )
                fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- FEATURE 3: Word Cloud Visual ---
            st.markdown("---")
            st.subheader("☁️ Text Word Cloud")
            
            wordcloud = WordCloud(
                width=800, 
                height=300, 
                background_color='white',
                colormap='viridis'
            ).generate(user_input)

            fig_wc, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)

    else:
        st.warning("Eadhavadhu text enter pannunga!")
