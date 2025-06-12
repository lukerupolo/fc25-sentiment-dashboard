import streamlit as st
import pandas as pd
import requests
import openai
from bs4 import BeautifulSoup

# ─── CONFIG ────────────────────────────────────────────────────────
API_URL = "https://sentiment-api-1081516136341.us-central1.run.app/predict"  # now handles both topic + sentiment
WEEKLY_CONTEXT_URL = "https://www.ea.com/games/ea-sports-fc/fc-25/news"

# ─── 1) OPENAI KEY ────────────────────────────────────────────────
try:
    openai.api_key = st.secrets["openai_api_key"]
except KeyError:
    st.error("OpenAI API key not found. Please add it to .streamlit/secrets.toml")
    st.stop()

# ─── 2) SCRAPE WEEKLY CONTEXT ──────────────────────────────────────
@st.cache_data(ttl=60*60*24*7)
def fetch_weekly_context(url: str) -> str:
    """Fetch and return concatenated article text from the EA FC25 news page."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Assume each article summary is in <p> or <h2> tags within a container
        texts = []
        for tag in soup.select(".c-content-listing__body p, .c-content-listing__body h2"):  # adjust selectors as needed
            texts.append(tag.get_text(strip=True))
        return "\n".join(texts)
    except Exception as e:
        st.warning(f"Could not fetch weekly context: {e}")
        return ""

weekly_context = fetch_weekly_context(WEEKLY_CONTEXT_URL)

# ─── 3) SESSION STATE ─────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.topics = []
    st.session_state.files = []
    st.session_state.regions = {}


def next_step():
    st.session_state.step += 1

st.title("🌍 FC25 Multilingual Sentiment Dashboard")

# ─── 4) TOPIC COLLECTION ─────────────────────────────────────────
if st.session_state.step == 0:
    st.write("### Step 1: Enter your 4 topics")
    st.write("These will be used for topic classification alongside weekly context.")
    if st.button("Start"):
        next_step()

elif 1 <= st.session_state.step <= 4:
    idx = st.session_state.step
    topic = st.text_input(f"Topic #{idx} name:", key=f"topic_{idx}")
    if st.button("Save Topic", key=f"save_topic_{idx}"):
        if topic.strip():
            st.session_state.topics.append(topic.strip())
            next_step()
        else:
            st.error("Please provide a topic name.")

# ─── 5) REGION FILE UPLOAD ────────────────────────────────────────
elif st.session_state.step == 5:
    st.write("### Step 2: Upload region CSVs")
    st.write("Select one or more CSVs containing comment data (text column auto-detected).")
    files = st.file_uploader("Select CSVs", type="csv", accept_multiple_files=True)
    if files:
        st.session_state.files = files
        next_step()

# ─── 6) REGION TAGGING ─────────────────────────────────────────────
elif st.session_state.step == 6:
    st.write("### Step 3: Tag each file's region")
    all_tagged = True
    for f in st.session_state.files:
        region = st.text_input(f"Region for '{f.name}':", key=f"region_{f.name}")
        if region.strip():
            st.session_state.regions[f.name] = region.strip()
        else:
            all_tagged = False
    if all_tagged and st.button("Combine & Analyze"):
        next_step()

# ─── 7) COMBINE & CLASSIFY ─────────────────────────────────────────
else:
    # auto-detect text column
    def find_text_column(df: pd.DataFrame) -> str | None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        best, best_score = None, -1.0
        for c in obj_cols:
            s = df[c].dropna().astype(str)
            if s.empty: continue
            avg_len = s.map(len).mean()
            space_ratio = s.map(lambda v: ' ' in v).mean()
            score = avg_len * space_ratio
            if score > best_score:
                best, best_score = c, score
        return best

    combined, skipped = [], []
    for f in st.session_state.files:
        if f.size == 0:
            skipped.append(f.name)
            continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            skipped.append(f.name)
            continue
        text_col = find_text_column(df)
        if not text_col:
            st.error(f"No text column found in '{f.name}'")
            st.stop()
        df = df.rename(columns={text_col: 'comment'})
        df['region'] = st.session_state.regions[f.name]
        combined.append(df[['comment','region']])
    if skipped:
        st.warning(f"Skipped files: {', '.join(skipped)}")
    if not combined:
        st.error("No valid data to combine.")
        st.stop()

    master_df = pd.concat(combined, ignore_index=True)
    st.subheader("✅ Combined Comments")
    st.dataframe(master_df.head())
    st.download_button("Download combined CSV",
                       master_df.to_csv(index=False),
                       file_name="combined_comments_all_regions.csv")

    # Prepare payload: comments, topics, weekly_context
    payload = {
        "comments": master_df['comment'].fillna("").tolist(),
        "threshold": 0.65,
        "topics": st.session_state.topics,
        "context": weekly_context
    }

    st.write("### Running combined topic & sentiment classification…")
    try:
        res = requests.post(API_URL, json=payload)
        res.raise_for_status()
        out_df = pd.DataFrame(res.json())
        result = pd.concat([master_df.reset_index(drop=True), out_df], axis=1)

        st.subheader("🔍 Results Preview")
        st.dataframe(result.head())
        st.download_button("Download final classified CSV",
                           result.to_csv(index=False),
                           file_name="classified_comments_all_regions.csv")
    except Exception as e:
        st.error(f"API error: {e}")
