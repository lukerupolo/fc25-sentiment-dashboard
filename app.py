
!pip install openai -q
# %%
import pandas as pd
from google.colab import files
from IPython.display import display, HTML
import sys  # Flush output
import textwrap  # Wrap long summaries
import os
os.environ["OPENAI_API_KEY"] = "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"


# --- Install required libraries (run if needed) ---
# !pip install scikit-learn sentence-transformers -q # No longer needed for region analysis
# !pip install openai -q # Keep if using OpenAI

# ✅ Define the OpenAI summarization function globally
from openai import OpenAI
# Check if API key is set before initializing the client
if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print("✅ OpenAI client initialized.")
else:
    client = None
    print("⚠️ OPENAI_API_KEY not found or is empty. OpenAI functionality will be skipped.")
sys.stdout.flush()

# Added comment IDs to the comments list for reference in summaries
def generate_actionable_summary_openai(comments_with_ids, chars_per_line=100):
    """Generates a summary using OpenAI, referencing comment IDs."""
    import textwrap

    if not comments_with_ids:
        return "No comments in this group."

    # Format comments with IDs for the prompt
    formatted_comments = [f"(ID {cid}) {comment}" for cid, comment in comments_with_ids]

    # Use up to 50 comments for summary generation
    combined_text = "\n---\n".join(formatted_comments[:50])
    if len(formatted_comments) > 50:
        combined_text += "\n..."

    summary = ""

    # Use OpenAI only if client is initialized
    if client:
        try:
            # Updated prompt for more granularity and ID reference
            prompt = f"""Summarize the following customer feedback comments (each with a unique ID).
Focus on specific, granular insights relevant to FC25 gameplay, features, or issues (e.g., specific tactics, bugs, feature requests).
Keep the summary concise but detailed enough to be actionable for decision makers.
Reference the comment IDs from the provided list when discussing a specific poin. Use the structure --? If its a positive sentiment
("[Excitement/choose any synonym to excitement ] is being generated around [insert sub-topic/s found in the comments evaluated but focus on most frequent] (e.g see IDs 12, 35)...")

If its a negative sentiment
("[negative sentiment/choose any synonym to negative sentiment] is being generated around [insert sub-topic/s found in the comments evaluated but focus on most frequent] (e.g see IDs 12, 35)...").
Do not directly quote the comments but synthesize the feedback.

Comments:
{combined_text}

Summary:"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # You might consider 'gpt-4o-mini' for potentially better summarization if needed
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes customer feedback, focusing on specific details and referencing comment IDs."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200 # Increased max tokens for potentially longer, more detailed summaries
            )

            summary = response.choices[0].message.content.strip()

        except Exception as e:
            summary = f"Error generating summary: {e}\n(Using placeholder summary)"
            print(f"\n❌ Error calling OpenAI API for summary: {e}")
            sys.stdout.flush()
    else:
         summary = "(OpenAI not configured, cannot generate summary)"


    # Fallback if API returns empty, fails early, or client not configured
    if not summary or summary.lower() == "summary:" or "(OpenAI not configured" in summary: # Handle cases where model just returns the prompt header or client wasn't configured
        summary = f"Summary (Placeholder):"
        if comments_with_ids:
             # Add snippets from comments with IDs if no summary is generated
             snippet_count = 0
             for cid, comment in comments_with_ids:
                  if snippet_count < 3:
                       summary += f"\n- (ID {cid}) {comment[:chars_per_line]}..."
                       snippet_count += 1
                  else:
                       break
             if len(comments_with_ids) > 3:
                 summary += "\n..."
        else:
            summary += "\nNo comments available."

    # No fixed max_lines for wrapping, let the summary expand
    wrapped_summary_lines = []
    for line in summary.split('\n'):
        # Only wrap lines that contain actual text and are not placeholder bullet points or ellipsis
        if line.strip() and not line.startswith(("- (ID ","...")):
             wrapped_summary_lines.extend(
                 textwrap.wrap(line, width=chars_per_line, break_long_words=False, replace_whitespace=False)
             )
        else:
             # Keep empty lines or placeholder lines as they are
             wrapped_summary_lines.append(line)


    # Join lines without truncating, adding ellipsis only if the original OpenAI summary ended abruptly
    # Check if the original summary from OpenAI ended with a complete sentence/thought
    original_summary_ended_cleanly = summary.endswith(('.', '!', '?')) if client and not summary.startswith("(OpenAI not configured") and not "Error generating summary" in summary else True

    final_summary = "\n".join(wrapped_summary_lines)

    # Add ellipsis only if the original generated summary (before wrapping) didn't end cleanly and we used OpenAI
    if client and not summary.startswith("(OpenAI not configured") and not "Error generating summary" in summary and not original_summary_ended_cleanly:
         final_summary += "..."


    return final_summary


# New function to generate region summary names using OpenAI
def generate_region_summary_name(comments_list, region_name):
    """Generates a short, descriptive name for a region's summary using OpenAI."""
    if not comments_list:
        return f"Summary for {region_name}"

    name = f"Summary for {region_name}" # Default fallback

    # Use OpenAI only if client is initialized
    if client:
        # Take a sample of comments to summarize for the name
        sample_comments = comments_list[:7] # Increased sample size slightly
        combined_sample = "\n---\n".join(sample_comments)
        if len(comments_list) > 7:
             combined_sample += "\n..."

        try:
            # Updated prompt for more granular name
            prompt = f"""Based on the following customer feedback comments for the game FC25 from the {region_name} region, provide a very short, descriptive name for this group of feedback (5-8 words max) as it relates to the key granular issues or positive points in this region's comments. Focus on insights useful to decision makers.

Comments:
{combined_sample}

Region Summary Name:"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # Can try other models like 'gpt-4o-mini' if available and suitable
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that names summaries of customer feedback by region, focusing on key granular points."},
                    {"role": "user", "content": prompt} # CORRECTED LINE: Changed "role_content" to "role": "user", "content"
                ],
                max_tokens=30, # Slightly increased max tokens for name length
                temperature=0.5 # Keep temperature lower for more consistent names
            )

            name = response.choices[0].message.content.strip()
            # Clean up potential quotation marks or extra characters from the model output
            name = name.strip('"').strip("'").strip()

        except Exception as e:
            print(f"\n❌ Error generating region summary name: {e}")
            sys.stdout.flush()
            name = f"Error Naming Summary for {region_name}" # Indicate an error


        # Ensure the name is not just the prompt header
        if name.lower() == "region summary name:":
             name = f"Unnamed Summary for {region_name}"
             if comments_list:
                  name += f": {comments_list[0][:20]}..."


    return name if name else f"Unnamed Summary for {region_name}" # Ensure fallback if client is None or name is empty


# --- Load models ---
# Removed SentenceTransformer and KMeans imports as they are no longer used for region analysis.
# try:
#     from sentence_transformers import SentenceTransformer
#     from sklearn.cluster import KMeans
#     from sklearn.exceptions import ConvergenceWarning
#     import warnings

#     warnings.filterwarnings("ignore", category=ConvergenceWarning)
#     warnings.filterwarnings("ignore", message="Number of distinct clusters .* is less than 2")

#     model_st = SentenceTransformer('all-MiniLM-L6-v2')
#     print("\n✅ SentenceTransformer model loaded.")
#     sys.stdout.flush()
# except Exception as e:
#     print(f"\n❌ Failed to load SentenceTransformer model: {e}")
#     model_st = None
#     sys.stdout.flush()


# --- Begin Workflow ---
print("📁 Upload your sentiment-labeled CSV (must include 'comment', 'topic', 'region', 'predicted_sentiment')...")
sys.stdout.flush()
uploaded = files.upload()

if uploaded:
    uploaded_filename = next(iter(uploaded))
    try:
        df = pd.read_csv(uploaded_filename)
        print(f"✅ File '{uploaded_filename}' loaded.")
        sys.stdout.flush()

        required_cols = ['comment', 'topic', 'region', 'predicted_sentiment']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ Missing required columns: {required_cols}")
            sys.stdout.flush()
        else:
            # Ensure required columns are strings to avoid errors during processing
            for col in required_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('').str.strip()

            # --- Add Unique Comment ID ---
            df['comment_id'] = range(1, len(df) + 1) # Assign a unique ID to each comment


            # Filter out 'uncertain' sentiment early for the tables section
            df_filtered_sentiment = df[df['predicted_sentiment'].isin(['positive', 'negative', 'neutral'])].copy()


            # --- Granular Analysis ---
            # We no longer need appendix_parts in the same way,
            # as the 'cluster' column will be the 'region' column directly.
            print("\n--- Starting Granular Analysis (Region-based) ---")
            sys.stdout.flush()

            # Assign the 'region' column as the 'cluster' column for consistency in downstream processing
            # Fill missing region values with 'Unknown'
            df['cluster'] = df['region'].fillna('Unknown')
            clustered_df = df.copy() # Now clustered_df is essentially df with 'region' as 'cluster'


            # --- Generate Tables ---
            print("\n📊 Generating Summary Tables by Region...")
            sys.stdout.flush()
            # Only consider positive and negative sentiment for summary tables
            tables_df = clustered_df[clustered_df['predicted_sentiment'].isin(['positive', 'negative'])]

            all_tables_data = [] # List to collect data for the combined CSV

            # Add CSS for left alignment
            css_style = """
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                    text-align: left; /* Default left alignment */
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    vertical-align: top; /* Align content to the top */
                    text-align: left; /* Ensure cells are left-aligned */
                }
                th {
                    background-color: #f2f2f2;
                }
                /* Optional: More specific targeting if needed */
                /* td { text-align: left !important; } */
            </style>
            """
            # Display the CSS once
            display(HTML(css_style))


            # Ensure there are topics to process for tables
            if tables_df['topic'].nunique() == 0:
                 print("⚠️ No topics with positive or negative sentiment found to generate tables.")
            else:
                for topic in tables_df['topic'].unique():
                    print(f"\n--- Table for Topic: {topic} ---")
                    topic_df = tables_df[tables_df['topic'] == topic].copy()

                    pos_count = topic_df[topic_df['predicted_sentiment'] == 'positive'].shape[0]
                    neg_count = topic_df[topic_df['predicted_sentiment'] == 'negative'].shape[0]
                    total = pos_count + neg_count
                    pos_percent = (pos_count / total) * 100 if total > 0 else 0

                    # --- Modified Table Headers ---
                    table_data = [
                        ["Overall Topic Summary", f"{pos_percent:.2f}% (n={total})", f"Positive: {pos_count}", f"Negative: {neg_count}"],
                        ["Region", "Sentiment Breakdown", "Positive Sentiment Analysis", "Negative Sentiment Analysis"] # Changed headers
                    ]

                    # Get unique regions within this topic, sorted alphabetically for consistent output
                    regions_in_topic = sorted(topic_df['region'].unique())


                    # Process each region within the topic
                    for region_name in regions_in_topic:
                        region_df = topic_df[topic_df['region'] == region_name].copy()

                        region_pos_count = region_df[region_df['predicted_sentiment'] == 'positive'].shape[0]
                        region_neg_count = region_df[region_df['predicted_sentiment'] == 'negative'].shape[0]
                        region_total = region_pos_count + region_neg_count
                        region_pos_percent = (region_pos_count / region_total) * 100 if region_total > 0 else 0

                        row = [region_name, "", "", ""] # Start row with region name

                        # Positive Analysis for Region
                        if region_pos_count > 0:
                             # Pass list of (comment_id, comment) tuples
                             pos_comments_region = region_df[region_df['predicted_sentiment'] == 'positive'][['comment_id', 'comment']].values.tolist()
                             # Use the modified naming function for regional summaries
                             pos_summary_name = generate_region_summary_name([c for _, c in pos_comments_region], region_name) # Pass just comments for naming
                             pos_summary = generate_actionable_summary_openai(pos_comments_region) # Pass (id, comment) for summary
                             row[2] = f"({region_pos_count})\n{pos_summary_name}\n{pos_summary}"
                             row[1] += f"Positive: {region_pos_count} ({region_pos_percent:.2f}%)" # Add sentiment breakdown to col 1

                        # Negative Analysis for Region
                        if region_neg_count > 0:
                            # Pass list of (comment_id, comment) tuples
                            neg_comments_region = region_df[region_df['predicted_sentiment'] == 'negative'][['comment_id', 'comment']].values.tolist()
                            # Use the modified naming function for regional summaries
                            neg_summary_name = generate_region_summary_name([c for _, c in neg_comments_region], region_name) # Pass just comments for naming
                            neg_summary = generate_actionable_summary_openai(neg_comments_region) # Pass (id, comment) for summary
                            row[3] = f"({region_neg_count})\n{neg_summary_name}\n{neg_summary}"
                            if row[1]: # If positive count was added, add a separator
                                row[1] += " | "
                            # Calculate negative percentage based on region total, handling potential division by zero if region_total is 0 (though checked by if region_total > 0)
                            region_neg_percent = (region_neg_count / region_total) * 100 if region_total > 0 else 0
                            row[1] += f"Negative: {region_neg_count} ({region_neg_percent:.2f}%)" # Add sentiment breakdown


                        # Add the row to the table data if it contains any relevant information (i.e., if there were comments for this region)
                        if region_total > 0:
                             table_data.append(row)
                        # Else: If region_total is 0, this region has no positive or negative comments for this topic, so no row is added for it in the table.
                        # This prevents rows for regions with only 'neutral' or 'uncertain' sentiment in this topic from appearing in the P/N summary table.


                    # Create and display the table for this topic
                    if len(table_data) > 1: # Only display if there are rows beyond the header
                        table_df = pd.DataFrame(table_data, columns=['Category/Cluster', '% favourable sentiment (Sample Size)', 'Positive Details', 'Negative Details'])
                        # Rename headers for clarity
                        table_df.columns = ["Region", "Sentiment Breakdown", "Positive Sentiment Analysis", "Negative Sentiment Analysis"]
                        # Replace newline characters with HTML breaks for display
                        display(HTML(table_df.to_html(index=False).replace("\\n", "<br/>")))

                        # --- Collect data for the combined CSV ---
                        # Add 'Topic' column to this table data
                        table_df['Topic'] = topic
                        # Melt the DataFrame to a longer format suitable for CSV download
                        # Keep Topic, Region, Sentiment Breakdown
                        # Melt Positive and Negative Analysis columns into rows
                        melted_pos = table_df[['Topic', 'Region', 'Sentiment Breakdown', 'Positive Sentiment Analysis']].copy()
                        melted_pos.rename(columns={'Positive Sentiment Analysis': 'Analysis Details'}, inplace=True)
                        melted_pos['Sentiment Type'] = 'Positive'

                        melted_neg = table_df[['Topic', 'Region', 'Sentiment Breakdown', 'Negative Sentiment Analysis']].copy()
                        melted_neg.rename(columns={'Negative Sentiment Analysis': 'Analysis Details'}, inplace=True)
                        melted_neg['Sentiment Type'] = 'Negative'

                        # Combine melted dataframes, filtering out rows with no analysis details
                        combined_melted = pd.concat([melted_pos, melted_neg])
                        combined_melted = combined_melted[combined_melted['Analysis Details'] != ""]

                        # Add the overall summary row data (converted to a similar format)
                        overall_summary_row = pd.DataFrame({
                            'Topic': [topic],
                            'Region': ['Overall Topic Summary'],
                            'Sentiment Breakdown': [table_data[0][1]], # % favourable sentiment
                            'Analysis Details': [f"Positive: {table_data[0][2].split(': ')[1]} | Negative: {table_data[0][3].split(': ')[1]}"],
                            'Sentiment Type': ['Overall']
                        })
                        combined_melted = pd.concat([overall_summary_row, combined_melted], ignore_index=True)

                        all_tables_data.append(combined_melted)


                    elif total > 0:
                        # If total > 0 but no region rows were added (unlikely if regions_in_topic is not empty and total > 0)
                         print(f"⚠️ Could not generate detailed region table for topic '{topic}'. Check data or region column.")
                         # Display basic summary if detailed table fails
                         basic_summary_table = pd.DataFrame([table_data[0]], columns=['Overall Topic Summary', '% favourable sentiment (Sample Size)', 'Positive: #', 'Negative: #']) # Use original summary headers
                         # Update the first row with correct data structure if using original headers
                         basic_summary_table.iloc[0] = [f"Overall Topic Summary for {topic}", f"{pos_percent:.2f}% (n={total})", f"Positive: {pos_count}", f"Negative: {neg_count}"]

                         display(HTML(basic_summary_table.to_html(index=False).replace("\\n", "<br/>")))

            # --- Save Combined Tables to CSV ---
            if all_tables_data:
                 combined_tables_df = pd.concat(all_tables_data, ignore_index=True)
                 tables_csv_filename = "region_sentiment_tables.csv"
                 combined_tables_df.to_csv(tables_csv_filename, index=False)
                 print(f"\n✅ Combined tables saved as '{tables_csv_filename}'")
                 files.download(tables_csv_filename)
            else:
                 print("\n⚠️ No data generated for tables CSV.")


            # --- Save Appendix ---
            # Ensure 'cluster' column exists (it should now be a copy of 'region')
            if 'cluster' not in clustered_df.columns:
                clustered_df['cluster'] = clustered_df['region'].fillna('Unknown')


            # Sort by topic, sentiment, and *region* (since region is the cluster) for the appendix
            appendix_sorted = clustered_df.sort_values(by=['topic', 'predicted_sentiment', 'region']).reset_index(drop=True)
            appendix_filename = "comment_region_appendix.csv" # Renamed appendix file

            # Ensure column order for clarity in the output file - include comment_id
            output_cols = ['comment_id', 'comment', 'topic', 'region', 'predicted_sentiment', 'cluster']
            # Add other columns from original df if they exist and aren't already in output_cols
            for col in df.columns:
                if col not in output_cols:
                    output_cols.append(col)

            # Reindex to ensure columns are in desired order, handling cases where some cols might not exist
            final_output_df = appendix_sorted.reindex(columns=[col for col in output_cols if col in appendix_sorted.columns])


            final_output_df.to_csv(appendix_filename, index=False)
            print(f"\n✅ Appendix saved as '{appendix_filename}'")
            display(final_output_df.head(20)) # Display preview of the final saved data
            files.download(appendix_filename)

    except Exception as e:
        print(f"❌ An unexpected error occurred during processing: {e}")
        sys.stdout.flush()
else:
    print("❌ No file uploaded.")
    sys.stdout.flush()
