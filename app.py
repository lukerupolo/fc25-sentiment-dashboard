import streamlit as st
import pandas as pd
import textwrap
import os
from openai import OpenAI
from io import StringIO

# --- Set up OpenAI client ---
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# --- Functions ---
def generate_actionable_summary_openai(comments_with_ids, chars_per_line=100):
    if not comments_with_ids:
        return "No comments in this group."

    formatted_comments = [f"(ID {cid}) {comment}" for cid, comment in comments_with_ids]
    combined_text = "\n---\n".join(formatted_comments[:50])
    if len(formatted_comments) > 50:
        combined_text += "\n..."

    if not client:
        return "(OpenAI not configured)"

    prompt = f"""Summarize the following customer feedback comments (each with a unique ID).
Focus on specific, granular insights relevant to FC25 gameplay, features, or issues.
Reference the comment IDs from the provided list.

Comments:
{combined_text}

Summary:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant summarizing customer feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        summary = f"Error generating summary: {e}"

    return summary

def generate_region_summary_name(comments_list, region_name):
    if not client or not comments_list:
        return f"Summary for {region_name}"

    sample_comments = comments_list[:7]
    combined_sample = "\n---\n".join(sample_comments)

    prompt = f"""Based on the following customer feedback from {region_name}, provide a short descriptive name:

{combined_sample}

Region Summary Name:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You name summaries of customer feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30
        )
        name = response.choices[0].message.content.strip().strip('"').strip("'")
    except Exception as e:
        name = f"Summary for {region_name}"

    return name

# --- Streamlit UI ---
st.title("📊 FC25 Regional Sentiment Summary Generator")

uploaded_file = st.file_uploader("Upload CSV with 'comment', 'topic', 'region', 'predicted_sentiment' columns", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['comment', 'topic', 'region', 'predicted_sentiment']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Required: {required_cols}")
    else:
        df['comment'] = df['comment'].astype(str).str.strip()
        df['comment_id'] = range(1, len(df) + 1)

        summary_output = []

        for topic in df['topic'].unique():
            topic_df = df[df['topic'] == topic]
            st.header(f"📝 Topic: {topic}")

            for region in topic_df['region'].unique():
                region_df = topic_df[topic_df['region'] == region]

                pos_comments = region_df[region_df['predicted_sentiment'] == 'positive'][['comment_id', 'comment']].values.tolist()
                neg_comments = region_df[region_df['predicted_sentiment'] == 'negative'][['comment_id', 'comment']].values.tolist()

                if not pos_comments and not neg_comments:
                    continue

                st.subheader(f"📍 Region: {region}")

                if pos_comments:
                    name = generate_region_summary_name([c for _, c in pos_comments], region)
                    summary = generate_actionable_summary_openai(pos_comments)
                    st.markdown(f"**Positive Summary Name:** {name}")
                    st.markdown(summary)
                    summary_output.append((topic, region, "positive", name, summary))

                if neg_comments:
                    name = generate_region_summary_name([c for _, c in neg_comments], region)
                    summary = generate_actionable_summary_openai(neg_comments)
                    st.markdown(f"**Negative Summary Name:** {name}")
                    st.markdown(summary)
                    summary_output.append((topic, region, "negative", name, summary))

        # --- Download option for summaries ---
        if summary_output:
            summary_df = pd.DataFrame(summary_output, columns=['Topic', 'Region', 'Sentiment', 'Summary Name', 'Summary'])
            csv = summary_df.to_csv(index=False)
            st.download_button("📥 Download Summary Table", csv, "summary_table.csv")

        # --- Download option for appendix ---
        appendix_csv = df.to_csv(index=False)
        st.download_button("📥 Download Appendix (All Comments)", appendix_csv, "appendix_all_comments.csv")
