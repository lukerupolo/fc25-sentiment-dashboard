import streamlit as st
import pandas as pd
import openai
import io

# Set up OpenAI key
try:
    openai.api_key = st.secrets["openai_api_key"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets.")
    st.stop()

def generate_region_summary_name(comments, region):
    prompt = f"""
    Provide a concise title (max 8 words) summarizing the following user sentiments from {region}:
    {comments[:10]}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=50
    )
    return response.choices[0].message['content'].strip()

def generate_actionable_summary_openai(comment_tuples):
    formatted = "\n".join([f"[{cid}] {txt}" for cid, txt in comment_tuples])
    prompt = f"""
    Analyze these user comments and provide a 2-3 sentence actionable summary with specific issues or strengths. Reference IDs:

    {formatted}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=250
    )
    return response.choices[0].message['content'].strip()

st.title("🌍 FC25 Regional Sentiment Summary Generator")
st.write("Upload a sentiment-labeled CSV to generate topic/region breakdowns and summaries.")

uploaded_file = st.file_uploader("Upload CSV with 'comment', 'topic', 'region', 'sentiment' columns", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['comment', 'topic', 'region', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns: {required_cols}")
            st.stop()

        for col in required_cols:
            df[col] = df[col].astype(str).fillna('').str.strip()

        df['comment_id'] = range(1, len(df) + 1)
        df['cluster'] = df['region'].fillna('Unknown')

        filtered_df = df[df['sentiment'].isin(['positive', 'negative'])].copy()

        all_tables = []
        st.markdown("<style>table {text-align: left;} th, td {text-align: left !important;}</style>", unsafe_allow_html=True)

        for topic in filtered_df['topic'].unique():
            topic_df = filtered_df[filtered_df['topic'] == topic]
            st.markdown(f"## 🗂️ Topic: {topic}")

            pos_count = topic_df[topic_df['sentiment'] == 'positive'].shape[0]
            neg_count = topic_df[topic_df['sentiment'] == 'negative'].shape[0]
            total = pos_count + neg_count
            pos_percent = (pos_count / total) * 100 if total > 0 else 0

            summary_row = [f"Overall Topic Summary", f"{pos_percent:.2f}% (n={total})", f"Positive: {pos_count}", f"Negative: {neg_count}"]
            table_rows = [
                ["Region", "Sentiment Breakdown", "Positive Sentiment Analysis", "Negative Sentiment Analysis"]
            ]

            for region in sorted(topic_df['region'].unique()):
                region_df = topic_df[topic_df['region'] == region]
                pos_region = region_df[region_df['sentiment'] == 'positive']
                neg_region = region_df[region_df['sentiment'] == 'negative']
                row = [region, "", "", ""]

                if not pos_region.empty:
                    pos_comments = pos_region[['comment_id', 'comment']].values.tolist()
                    pos_title = generate_region_summary_name([c for _, c in pos_comments], region)
                    pos_summary = generate_actionable_summary_openai(pos_comments)
                    pos_pct = (len(pos_comments) / len(region_df)) * 100
                    row[1] += f"Positive: {len(pos_comments)} ({pos_pct:.2f}%)"
                    row[2] = f"({len(pos_comments)})\n{pos_title}\n{pos_summary}"

                if not neg_region.empty:
                    neg_comments = neg_region[['comment_id', 'comment']].values.tolist()
                    neg_title = generate_region_summary_name([c for _, c in neg_comments], region)
                    neg_summary = generate_actionable_summary_openai(neg_comments)
                    neg_pct = (len(neg_comments) / len(region_df)) * 100
                    if row[1]:
                        row[1] += " | "
                    row[1] += f"Negative: {len(neg_comments)} ({neg_pct:.2f}%)"
                    row[3] = f"({len(neg_comments)})\n{neg_title}\n{neg_summary}"

                if len(region_df) > 0:
                    table_rows.append(row)

            summary_table = pd.DataFrame([summary_row], columns=["Region", "Sentiment Breakdown", "Positive Sentiment Analysis", "Negative Sentiment Analysis"])
            st.dataframe(summary_table)

            if len(table_rows) > 1:
                table_df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
                table_df['Topic'] = topic
                st.dataframe(table_df)

                # Append for CSV
                melted_pos = table_df[['Topic', 'Region', 'Sentiment Breakdown', 'Positive Sentiment Analysis']].copy()
                melted_pos.rename(columns={'Positive Sentiment Analysis': 'Analysis Details'}, inplace=True)
                melted_pos['Sentiment Type'] = 'Positive'

                melted_neg = table_df[['Topic', 'Region', 'Sentiment Breakdown', 'Negative Sentiment Analysis']].copy()
                melted_neg.rename(columns={'Negative Sentiment Analysis': 'Analysis Details'}, inplace=True)
                melted_neg['Sentiment Type'] = 'Negative'

                combined = pd.concat([melted_pos, melted_neg])
                combined = combined[combined['Analysis Details'] != ""]

                summary_row_df = pd.DataFrame({
                    'Topic': [topic],
                    'Region': ['Overall Topic Summary'],
                    'Sentiment Breakdown': [summary_row[1]],
                    'Analysis Details': [f"{summary_row[2]} | {summary_row[3]}"],
                    'Sentiment Type': ['Overall']
                })

                all_tables.append(pd.concat([summary_row_df, combined], ignore_index=True))

        if all_tables:
            final_table = pd.concat(all_tables, ignore_index=True)
            csv = final_table.to_csv(index=False).encode('utf-8')
            st.download_button("Download Summary CSV", data=csv, file_name="region_sentiment_tables.csv", mime='text/csv')

        appendix_df = df.sort_values(by=['topic', 'sentiment', 'region'])
        appendix_csv = appendix_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Appendix CSV", data=appendix_csv, file_name="comment_region_appendix.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Unexpected error: {e}")
