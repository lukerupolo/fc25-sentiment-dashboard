import streamlit as st
import pandas as pd
import requests
import openai
import os

# Load OpenAI key from secrets
openai.api_key = st.secrets["openai_api_key"]

API_URL = "https://sentiment-api-1081516136341.us-central1.run.app/predict"

st.set_page_config(page_title="FC25 Sentiment Dashboard", layout="wide")
st.title("🌍 FC25 Multilingual Sentiment Dashboard")
st.write("This tool analyzes translated feedback and generates topic/region insights with full attribution.")

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

# --- OpenAI Summary Function ---
def generate_summary_with_ids(comments_with_ids, region, sentiment_type):
    if len(comments_with_ids) < 5:
        return "❌ Not enough comments for a confident summary."

    formatted = [f"(ID {cid}) {comment}" for cid, comment in comments_with_ids[:50]]
    input_text = "\n---\n".join(formatted)

    tone = "excitement" if sentiment_type == "positive" else "frustration"
    prompt = f"""
Summarize the following {sentiment_type} customer feedback for FC25 from the {region} region.
Each comment has an ID. Focus on recurring themes that appear in at least 5 comments. 
For each point, write: 
- \"[{tone}] is being generated around [theme] (e.g., see IDs 12, 35, 56, ...)\"
Do not quote comments. Focus on synthesis and decision-useful insights.

Comments:
{input_text}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize FC25 feedback with data-backed clarity for marketing and product teams."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating summary: {str(e)}"

# STEP 2: Upload final CSV for region/topic analysis
st.header("Step 2: Upload Final CSV for Detailed Attribution Analysis")
file2 = st.file_uploader("Upload CSV with 'region', 'topic', 'sentiment', 'comment'", type="csv", key="upload2")

if file2:
    df2 = pd.read_csv(file2)
    required = {"region", "topic", "sentiment", "comment"}
    if not required.issubset(df2.columns):
        st.error(f"CSV must include columns: {', '.join(required)}")
    else:
        df2["comment_id"] = range(1, len(df2) + 1)
        appendix_rows = []
        st.success("📊 Generating summaries and appendix...")

        for topic in df2["topic"].unique():
            st.markdown(f"## 📂 Topic: {topic}")
            topic_df = df2[df2["topic"] == topic]

            for region in topic_df["region"].unique():
                st.markdown(f"### 🌍 Region: {region}")
                region_df = topic_df[topic_df["region"] == region]

                pos_comments = region_df[region_df["sentiment"] == "positive"]["comment_id"].astype(str) + ", " + region_df[region_df["sentiment"] == "positive"]["comment"]
                neg_comments = region_df[region_df["sentiment"] == "negative"]["comment_id"].astype(str) + ", " + region_df[region_df["sentiment"] == "negative"]["comment"]

                pos_tuples = region_df[region_df["sentiment"] == "positive"][["comment_id", "comment"]].values.tolist()
                neg_tuples = region_df[region_df["sentiment"] == "negative"][["comment_id", "comment"]].values.tolist()

                if pos_tuples:
                    st.markdown(f"**✅ Positive Sentiment ({len(pos_tuples)} comments):**")
                    st.markdown(generate_summary_with_ids(pos_tuples, region, "positive"))

                if neg_tuples:
                    st.markdown(f"**⚠️ Negative Sentiment ({len(neg_tuples)} comments):**")
                    st.markdown(generate_summary_with_ids(neg_tuples, region, "negative"))

                appendix_rows.extend(region_df.to_dict(orient="records"))

        appendix_df = pd.DataFrame(appendix_rows)
        st.download_button("📥 Download Appendix", appendix_df.to_csv(index=False), file_name="appendix.csv")
