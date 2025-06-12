import streamlit as st
import pandas as pd
import requests
from openai import OpenAI
import textwrap

# Load OpenAI client from secrets
client = OpenAI(api_key=st.secrets["openai_api_key"])

API_URL = "https://sentiment-api-1081516136341.us-central1.run.app/predict"

st.title("\U0001F30D FC25 Multilingual Sentiment Dashboard")
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
            df_with_sentiment = pd.concat([
                df1.reset_index(drop=True), 
                sentiment_df[["sentiment", "confidence"]]
            ], axis=1)
            df_with_sentiment.rename(columns={"sentiment": "predicted_sentiment"}, inplace=True)

            st.subheader("Predicted Sentiments")
            st.dataframe(df_with_sentiment.head(10))

            st.download_button(
                "Download CSV with Sentiments",
                df_with_sentiment.to_csv(index=False),
                file_name="predicted_sentiment.csv"
            )
        except Exception as e:
            st.error(f"API error: {e}")


# STEP 2: Upload final CSV (region, topic, sentiment, comment)
st.header("Step 2: Upload Final CSV for Analysis")
file2 = st.file_uploader("Upload CSV with 'region', 'topic', 'predicted_sentiment', 'comment'", type="csv", key="upload2")

if file2 is not None:
    df = pd.read_csv(file2)
    required_cols = {'comment', 'topic', 'region', 'predicted_sentiment'}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must include: {', '.join(required_cols)}")
    else:
        df['comment_id'] = range(1, len(df) + 1)
        df_filtered = df[df['predicted_sentiment'].isin(['positive', 'negative'])].copy()

        def generate_summary(comments_with_ids):
            formatted = [f"(ID {cid}) {text}" for cid, text in comments_with_ids]
            prompt = f"""
Summarize the following FC25 feedback by referencing comment IDs.
Only include points backed by at least 5 comments. Use the format:
- Positive point: ... (e.g. see IDs 12, 31, 54, ...)
- Negative point: ... (e.g. see IDs 8, 19, 27, ...)
Do not quote the comments, synthesize insights.

Comments:
{chr(10).join(formatted[:50])}
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You summarize customer feedback with traceable comment ID references."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return textwrap.fill(response.choices[0].message.content.strip(), 100)
            except Exception as e:
                return f"Summary error: {e}"

        topics = df_filtered['topic'].unique()
        appendix = []

        for topic in topics:
            st.markdown(f"### 📂 Topic: {topic}")
            topic_df = df_filtered[df_filtered['topic'] == topic]
            summary_data = []

            for region in topic_df['region'].unique():
                regional = topic_df[topic_df['region'] == region]
                pos = regional[regional['predicted_sentiment'] == 'positive']
                neg = regional[regional['predicted_sentiment'] == 'negative']

                pos_ids = pos[['comment_id', 'comment']].values.tolist()
                neg_ids = neg[['comment_id', 'comment']].values.tolist()

                st.markdown(f"**🌍 Region: {region}**")
                st.write(f"Sentiment Breakdown: {len(pos)} positive, {len(neg)} negative")

                if len(pos_ids) >= 5:
                    st.markdown("**✅ Positive Summary:**")
                    st.write(generate_summary(pos_ids))
                if len(neg_ids) >= 5:
                    st.markdown("**⚠️ Negative Summary:**")
                    st.write(generate_summary(neg_ids))

                appendix.append(regional)

        appendix_df = pd.concat(appendix)
        st.download_button("Download Appendix CSV", appendix_df.to_csv(index=False), file_name="comment_appendix.csv")
