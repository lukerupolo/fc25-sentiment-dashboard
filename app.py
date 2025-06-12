import streamlit as st
import pandas as pd
import requests
import openai
import os

# Set your OpenAI API key from Streamlit secrets
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

        threshold = st.slider("Set sentiment confidence threshold", 0.0, 1.0, 0.65)
        payload = {
            "comments": df1["comment"].fillna("").tolist(),
            "threshold": threshold
        }
        try:
            res = requests.post(API_URL, json=payload)
            res.raise_for_status()
            predictions = res.json()
            if not isinstance(predictions, list) or 'sentiment' not in predictions[0]:
                st.error("Unexpected response format from API.")
                st.stop()

            sentiment_df = pd.DataFrame(predictions)
            df_with_sentiment = pd.concat([
                df1.reset_index(drop=True),
                sentiment_df[["sentiment", "confidence"]]
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
file2 = st.file_uploader("Upload CSV with 'region', 'topic', 'sentiment', 'comment', optional 'comment_id'", type="csv", key="upload2")

if file2 is not None:
    df = pd.read_csv(file2)
    required_cols = ['region', 'topic', 'sentiment', 'comment']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Missing required columns: {required_cols}")
    else:
        st.success("File loaded. Generating detailed regional summary tables...")

        # Ensure required columns are clean strings
        for col in required_cols:
            df[col] = df[col].astype(str).fillna('').str.strip()
        # Assign unique IDs if not present
        if 'comment_id' not in df.columns:
            df.insert(0, 'comment_id', range(1, len(df) + 1))
        else:
            df['comment_id'] = df['comment_id'].astype(int)

        # Filter out only positive, negative, neutral for summary tables stage
        df_filtered = df[df['sentiment'].isin(['positive', 'negative', 'neutral'])].copy()

        # Use 'region' as cluster
        df_filtered['cluster'] = df_filtered['region'].fillna('Unknown')

        # CSS for left-aligned, top-aligned tables
        css_style = """
        <style>
            table { border-collapse: collapse; width: 100%; text-align: left; }
            th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
        """
        st.markdown(css_style, unsafe_allow_html=True)

        all_tables_data = []
        # Iterate topics
        for topic in sorted(df_filtered['topic'].unique()):
            # Separator and header
            st.markdown("---")
            st.markdown(f"### 🗂️ Topic: {topic}")
            topic_df = df_filtered[df_filtered['topic'] == topic]

            # Prepare table_data with headers
            pos_count = topic_df[topic_df['sentiment']=='positive'].shape[0]
            neg_count = topic_df[topic_df['sentiment']=='negative'].shape[0]
            total = pos_count + neg_count
            fav_pct = (pos_count/total)*100 if total>0 else 0
            table_data = []
            # Overall summary row
            table_data.append([
                "Overall Topic Summary",
                f"{fav_pct:.2f}% (n={total})",
                f"Positive: {pos_count}",
                f"Negative: {neg_count}"
            ])
            # Column headers row
            table_data.append(["Region", "% favourable sentiment (Sample Size)", "Positive Sentiment Analysis", "Negative Sentiment Analysis"])

            # Process each region
            for region in sorted(topic_df['region'].unique()):
                region_df = topic_df[topic_df['region'] == region]
                region_pos = region_df[region_df['sentiment']=='positive']
                region_neg = region_df[region_df['sentiment']=='negative']
                region_tot = len(region_pos) + len(region_neg)
                if region_tot == 0:
                    continue
                region_pos_pct = (len(region_pos)/region_tot)*100
                region_neg_pct = (len(region_neg)/region_tot)*100

                # Generate summaries via OpenAI
                pos_comments = list(zip(region_pos['comment_id'], region_pos['comment']))
                neg_comments = list(zip(region_neg['comment_id'], region_neg['comment']))
                # Naming and summary
                pos_name = generate_region_summary_name(
                    [c for _, c in pos_comments], region
                ) if pos_comments else ''
                pos_summary = generate_actionable_summary_openai(pos_comments) if pos_comments else ''
                neg_name = generate_region_summary_name(
                    [c for _, c in neg_comments], region
                ) if neg_comments else ''
                neg_summary = generate_actionable_summary_openai(neg_comments) if neg_comments else ''

                # Build row with escaped newline
                row = [
                    region,
                    f"Positive: {len(region_pos)} ({region_pos_pct:.2f}%) | Negative: {len(region_neg)} ({region_neg_pct:.2f}%)",
                    f"({len(region_pos)}) {pos_name}\n{pos_summary}",
                    f"({len(region_neg)}) {neg_name}\n{neg_summary}"
                ]
                table_data.append(row)

                # Collect for combined CSV
                # Melt into two rows per region (positive & negative)
                if pos_summary:
                    all_tables_data.append({
                        'Topic': topic,
                        'Region': region,
                        'Sentiment Breakdown': f"Positive: {len(region_pos)} ({region_pos_pct:.2f}%)",
                        'Analysis Details': pos_summary,
                        'Sentiment Type': 'Positive'
                    })
                if neg_summary:
                    all_tables_data.append({
                        'Topic': topic,
                        'Region': region,
                        'Sentiment Breakdown': f"Negative: {len(region_neg)} ({region_neg_pct:.2f}%)",
                        'Analysis Details': neg_summary,
                        'Sentiment Type': 'Negative'
                    })

            # Display HTML table
            if len(table_data) > 2:
                df_table = pd.DataFrame(table_data[1:], columns=table_data[0])
                html = df_table.to_html(index=False).replace("\\n", "<br/>")
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.warning(f"No positive or negative data to display for topic '{topic}'.")

        # Download buttons
        if all_tables_data:
            combined_df = pd.DataFrame(all_tables_data)
            st.download_button(
                "Download Summary Tables as CSV",
                combined_df.to_csv(index=False),
                file_name="region_sentiment_tables.csv"
            )

        # Appendix CSV
        appendix = df.sort_values(by=['topic','sentiment','region']).reset_index(drop=True)
        st.download_button(
            "Download Appendix (comment_id, comment, topic, region, sentiment, cluster)",
            appendix.to_csv(index=False),
            file_name="comment_region_appendix.csv"
        )
