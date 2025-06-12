import streamlit as st
import pandas as pd
import requests

API_URL = "https://sentiment-api-1081516136341.us-central1.run.app/predict"

st.title("🌍 FC25 Multilingual Sentiment Dashboard")
st.write("This tool analyzes translated feedback and generates topic/region insights.")

# STEP 1: Upload comments + topic CSV
st.header("Step 1: Upload Translated Comments")
file1 = st.file_uploader("Upload CSV with 'comment' and 'topic' columns", type="csv", key="upload1")

if file1 is not None:
    df1 = pd.read_csv(file1)
    if not {"comment", "topic"}.issubset(df1.columns):
        st.error("CSV must contain 'comment' and 'topic' columns.")
    else:
        st.success("File uploaded! Now predicting sentiment...")

        payload = {
            "comments": df1["comment"].fillna("").tolist(),
            "threshold": 0.65
        }
        try:
            res = requests.post(API_URL, json=payload)
            res.raise_for_status()
            predictions = res.json()
            sentiment_df = pd.DataFrame(predictions)
            df_with_sentiment = pd.concat([df1.reset_index(drop=True), sentiment_df[["sentiment", "confidence"]]], axis=1)

            st.subheader("Predicted Sentiments")
            st.dataframe(df_with_sentiment.head(10))
            st.download_button("Download CSV with Sentiments", df_with_sentiment.to_csv(index=False), file_name="predicted_sentiment.csv")
        except Exception as e:
            st.error(f"API error: {e}")

# STEP 2: Upload final CSV (region, topic, sentiment, comment)
st.header("Step 2: Upload Final CSV for Analysis")
file2 = st.file_uploader("Upload CSV with 'region', 'topic', 'sentiment', 'comment'", type="csv", key="upload2")

if file2 is not None:
    df2 = pd.read_csv(file2)
    required_cols = {"region", "topic", "sentiment", "comment"}
    if not required_cols.issubset(df2.columns):
        st.error(f"CSV must include: {', '.join(required_cols)}")
    else:
        st.success("Generating styled sentiment report...")

        topics = df2["topic"].unique()
        for topic in topics:
            st.markdown(f"### 🗂️ Topic: {topic}")
            topic_df = df2[df2["topic"] == topic]

            for region in topic_df["region"].unique():
                region_df = topic_df[topic_df["region"] == region]
                st.markdown(f"**🌍 Region: {region}**")

                total = len(region_df)
                pos = region_df[region_df["sentiment"] == "positive"]
                neg = region_df[region_df["sentiment"] == "negative"]
                pos_count = len(pos)
                neg_count = len(neg)

                if total > 0:
                    pos_pct = (pos_count / total) * 100
                    neg_pct = (neg_count / total) * 100
                    st.write(f"**Sentiment Breakdown:** Positive: {pos_count} ({pos_pct:.2f}%) | Negative: {neg_count} ({neg_pct:.2f}%)")

                st.write("**✅ Positive Sentiment Analysis:**")
                st.write("\n\n".join(pos["comment"].dropna().head(5)))

                st.write("**⚠️ Negative Sentiment Analysis:**")
                st.write("\n\n".join(neg["comment"].dropna().head(5)))
