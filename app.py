import streamlit as st
import pandas as pd
import requests
import openai

# Set your OpenAI API key from Streamlit secrets
try:
    openai.api_key = st.secrets["openai_api_key"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

API_URL = "https://sentiment-api-1081516136341.us-central1.run.app/predict"

st.title("🌍 FC25 Multilingual Sentiment Dashboard")
st.write("Upload a CSV with columns: **comment**, **topic**, **region**")

# Single upload
file = st.file_uploader("Upload CSV", type="csv")
if file is not None:
    df = pd.read_csv(file)
    required = {"comment", "topic", "region"}
    if not required.issubset(df.columns):
        st.error(f"CSV must contain columns: {', '.join(required)}")
    else:
        st.success("File uploaded! Predicting sentiment...")

        # Call sentiment API
        payload = {
            "comments": df["comment"].fillna("").tolist(),
            "threshold": 0.65
        }
        try:
            res = requests.post(API_URL, json=payload)
            res.raise_for_status()
            preds = res.json()
            sent_df = pd.DataFrame(preds)[["sentiment", "confidence"]]
            sent_df = sent_df.rename(columns={"sentiment": "predicted_sentiment"})
            
            # Merge predictions back
            df_with_sentiment = pd.concat([df.reset_index(drop=True), sent_df], axis=1)

            # Download button
            st.subheader("Download predictions")
            st.download_button(
                "Download CSV with Sentiments",
                df_with_sentiment.to_csv(index=False),
                file_name="predicted_sentiment.csv"
            )

            # Analysis report
            st.header("📊 Sentiment Analysis Report")
            for topic in df_with_sentiment["topic"].unique():
                st.markdown(f"### 🗂️ Topic: {topic}")
                tdf = df_with_sentiment[df_with_sentiment["topic"] == topic]
                for region in tdf["region"].unique():
                    rdf = tdf[tdf["region"] == region]
                    st.markdown(f"**🌍 Region: {region}**")
                    
                    total = len(rdf)
                    pos = rdf[rdf["predicted_sentiment"] == "positive"]
                    neg = rdf[rdf["predicted_sentiment"] == "negative"]
                    pos_count, neg_count = len(pos), len(neg)
                    
                    if total:
                        pos_pct = pos_count / total * 100
                        neg_pct = neg_count / total * 100
                        st.write(
                            f"**Sentiment Breakdown:** "
                            f"Positive: {pos_count} ({pos_pct:.2f}%) | "
                            f"Negative: {neg_count} ({neg_pct:.2f}%)"
                        )
                    
                    st.write("**✅ Positive Examples:**")
                    for c in pos["comment"].dropna().head(5):
                        st.write(f"- {c}")
                    
                    st.write("**⚠️ Negative Examples:**")
                    for c in neg["comment"].dropna().head(5):
                        st.write(f"- {c}")

        except Exception as e:
            st.error(f"API error: {e}")
