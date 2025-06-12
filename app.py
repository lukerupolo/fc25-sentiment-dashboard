import streamlit as st
import pandas as pd
import os
import sys
import textwrap
from openai import OpenAI
from io import StringIO

# Initialize OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.set_page_config(page_title="FC25 Sentiment Dashboard", layout="wide")
st.title("📊 FC25 Sentiment Analysis Dashboard")

st.markdown("Upload a CSV containing the following columns: `comment`, `topic`, `region`, `predicted_sentiment`.")

# Upload file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ["comment", "topic", "region", "predicted_sentiment"]

    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing one or more required columns: {required_cols}")
    else:
        for col in required_cols:
            df[col] = df[col].astype(str).fillna('').str.strip()

        df['comment_id'] = range(1, len(df) + 1)
        df['cluster'] = df['region'].fillna('Unknown')

        # Function to generate region summary name
        def generate_region_summary_name(comments_list, region_name):
            if not comments_list:
                return f"Summary for {region_name}"

            sample_comments = comments_list[:7]
            combined_sample = "\n---\n".join(sample_comments)

            prompt = f"""Based on the following customer feedback comments for FC25 from the {region_name} region, provide a very short, descriptive name (5-8 words max) focused on the most important issues or praise.

Comments:
{combined_sample}

Region Summary Name:"""

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that names summaries of customer feedback by region."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=30,
                    temperature=0.5
                )
                return response.choices[0].message.content.strip().strip('"')
            except:
                return f"Summary for {region_name}"

        # Function to generate summary with comment IDs
        def generate_actionable_summary_openai(comments_with_ids):
            if not comments_with_ids:
                return "No comments in this group."

            formatted_comments = [f"(ID {cid}) {comment}" for cid, comment in comments_with_ids]
            combined_text = "\n---\n".join(formatted_comments[:50])

            prompt = f"""Summarize the following customer feedback comments (each with a unique ID).
Focus on specific gameplay details, bugs, features, or praise.
Use this structure:
"Excitement is being generated around [issue/topic] (e.g. see IDs 4, 18, 22...)"
or
"Frustration is being generated around [issue/topic] (e.g. see IDs 4, 18, 22...)"

Comments:
{combined_text}

Summary:"""

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes customer feedback, focusing on specific details and referencing comment IDs."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Error: {e}"

        st.success("✅ File loaded. Now displaying detailed summaries below.")

        filtered_df = df[df['predicted_sentiment'].isin(['positive', 'negative'])]

        for topic in filtered_df['topic'].unique():
            st.markdown(f"### 🧠 Topic: {topic}")
            topic_df = filtered_df[filtered_df['topic'] == topic]
            for region in sorted(topic_df['region'].unique()):
                region_df = topic_df[topic_df['region'] == region]
                st.markdown(f"**🌍 Region: {region}**")

                for sentiment in ['positive', 'negative']:
                    subset = region_df[region_df['predicted_sentiment'] == sentiment]
                    if len(subset) >= 5:
                        comments = subset[['comment_id', 'comment']].values.tolist()
                        name = generate_region_summary_name([c for _, c in comments], region)
                        summary = generate_actionable_summary_openai(comments)
                        st.markdown(f"**Sentiment:** {sentiment.capitalize()}  ")
                        st.markdown(f"**Summary Name:** {name}  ")
                        st.markdown(f"**Summary:**\n\n{summary}")
                    else:
                        st.markdown(f"*Not enough {sentiment} comments to summarize.*")

        # Save Appendix
        appendix_filename = "comment_region_appendix.csv"
        df.to_csv(appendix_filename, index=False)
        st.download_button(
            label="📥 Download Appendix CSV",
            data=open(appendix_filename, "rb").read(),
            file_name=appendix_filename,
            mime="text/csv"
        )
