import streamlit as st
import pandas as pd
import requests
import openai
import os
from typing import List, Tuple

# Helper summary functions
# Paste or customize these implementations as needed:

def generate_region_summary_name(comments: List[str], region: str) -> str:
    """
    Create a concise summary title for a region based on its comments.
    """
    return f"Key themes in {region}"


def generate_actionable_summary_openai(comment_tuples: List[Tuple[int, str]]) -> str:
    """
    Generate an actionable summary from a list of (comment_id, comment) tuples using OpenAI.
    """
    # Format comments with IDs
    texts = "
".join(f"[{cid}] {text}" for cid, text in comment_tuples)
    prompt = (
        "You are an expert analyst. Given the following feedback comments labeled by ID:
"
        + texts + "

"
        "Please provide a concise, actionable summary that references the comment IDs in brackets."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize user feedback into actionable insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except openai.error.AuthenticationError:
        st.error("OpenAI authentication failed. Please check your API key in Streamlit secrets.")
        return ""
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "" from Streamlit secrets from Streamlit secrets
try:
    openai.api_key = st.secrets["openai_api_key"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

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

        # User-set threshold
        threshold = st.slider("Set sentiment confidence threshold", 0.0, 1.0, 0.65)
        payload = {"comments": df1["comment"].fillna("").tolist(), "threshold": threshold}

        try:
            res = requests.post(API_URL, json=payload)
            res.raise_for_status()
            predictions = res.json()
            if not isinstance(predictions, list) or 'sentiment' not in predictions[0]:
                st.error("Unexpected response format from API.")
                st.stop()

            sentiment_df = pd.DataFrame(predictions)
            df_with_sentiment = pd.concat([
                df1.reset_index(drop=True), sentiment_df[["sentiment", "confidence"]]
            ], axis=1)
            df_with_sentiment.insert(0, 'comment_id', range(1, len(df_with_sentiment) + 1))

            st.subheader("Predicted Sentiments")
            st.dataframe(df_with_sentiment.head(10))
            st.download_button(
                "Download CSV with Sentiments",
                df_with_sentiment.to_csv(index=False),
                file_name="predicted_sentiment.csv"
            )
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
        except Exception as e:
            st.error(f"Error parsing API response: {e}")

# STEP 2: Upload final CSV (region, topic, sentiment, comment)
st.header("Step 2: Upload Final CSV for Analysis")
file2 = st.file_uploader(
    "Upload CSV with 'region', 'topic', 'sentiment', 'comment', optional 'comment_id'",
    type="csv", key="upload2"
)

if file2 is not None:
    df = pd.read_csv(file2)
    required_cols = ['region', 'topic', 'sentiment', 'comment']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Missing required columns: {required_cols}")
    else:
        st.success("File loaded. Generating detailed regional summary tables...")

        # Clean and assign IDs
        for col in required_cols:
            df[col] = df[col].astype(str).fillna('').str.strip()
        if 'comment_id' not in df.columns:
            df.insert(0, 'comment_id', range(1, len(df) + 1))
        else:
            df['comment_id'] = df['comment_id'].astype(int)

        # Filter sentiments and set cluster
        df_filtered = df[df['sentiment'].isin(['positive', 'negative', 'neutral'])].copy()
        df_filtered['cluster'] = df_filtered['region'].fillna('Unknown')

        # Table styling
        css = """
        <style>
            table { border-collapse: collapse; width: 100%; text-align: left; }
            th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

        all_tables = []
        for topic in sorted(df_filtered['topic'].unique()):
            st.markdown("---")
            st.markdown(f"### 🗂️ Topic: {topic}")
            topic_df = df_filtered[df_filtered['topic'] == topic]

            # Overall metrics
            pos_total = topic_df[topic_df['sentiment']=='positive'].shape[0]
            neg_total = topic_df[topic_df['sentiment']=='negative'].shape[0]
            total = pos_total + neg_total
            fav_pct = (pos_total / total * 100) if total > 0 else 0
            st.write(
                f"**Overall Topic Summary:** {fav_pct:.2f}% favourable (n={total}) | "
                f"Positive: {pos_total} | Negative: {neg_total}"
            )

            # Table rows
            rows = []
            for region in sorted(topic_df['region'].unique()):
                r_df = topic_df[topic_df['region'] == region]
                r_pos = r_df[r_df['sentiment']=='positive']
                r_neg = r_df[r_df['sentiment']=='negative']
                r_tot = len(r_pos) + len(r_neg)
                if r_tot == 0:
                    continue
                r_pos_pct = len(r_pos)/r_tot*100
                r_neg_pct = len(r_neg)/r_tot*100

                # Summaries
                pos_comm = [(int(cid), txt) for cid, txt in zip(r_pos['comment_id'], r_pos['comment'])]
                neg_comm = [(int(cid), txt) for cid, txt in zip(r_neg['comment_id'], r_neg['comment'])]
                pos_summary = generate_actionable_summary_openai(pos_comm) if pos_comm else ''
                neg_summary = generate_actionable_summary_openai(neg_comm) if neg_comm else ''

                row = {
                    'Region': region,
                    'Sentiment Breakdown': f"Positive: {len(r_pos)} ({r_pos_pct:.2f}%) | Negative: {len(r_neg)} ({r_neg_pct:.2f}%)",
                    'Positive Analysis': f"({len(r_pos)}) {pos_summary}",
                    'Negative Analysis': f"({len(r_neg)}) {neg_summary}"
                }
                rows.append(row)

                # For CSV
                if pos_summary:
                    all_tables.append({**row, 'Topic': topic, 'Sentiment Type': 'Positive', 'Analysis Details': pos_summary})
                if neg_summary:
                    all_tables.append({**row, 'Topic': topic, 'Sentiment Type': 'Negative', 'Analysis Details': neg_summary})

            if rows:
                st.table(pd.DataFrame(rows))

        # Download combined
        if all_tables:
            combined_df = pd.DataFrame(all_tables)
            st.download_button(
                "Download Summary Tables as CSV",
                combined_df.to_csv(index=False),
                file_name="region_sentiment_tables.csv"
            )

        # Appendix
        appendix = df.sort_values(by=['topic', 'sentiment', 'region']).reset_index(drop=True)
        st.download_button(
            "Download Appendix CSV",
            appendix.to_csv(index=False),
            file_name="comment_region_appendix.csv"
        )
