import streamlit as st
import pandas as pd
import requests
import openai
from io import BytesIO

# Attempt to import BeautifulSoup; if unavailable, scraping is disabled
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

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
    if BeautifulSoup is None:
        st.warning("beautifulsoup4 not installed; skipping scraping.")
        return ""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        texts = [p.get_text(strip=True) for p in soup.select(
            ".c-content-listing__body p, .c-content-listing__body h2")]
        return "\n".join(texts)
    except Exception as e:
        st.warning(f"Could not fetch weekly context: {e}")
        return ""

weekly_context = fetch_weekly_context(WEEKLY_CONTEXT_URL)

# ─── 3) SESSION STATE ─────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.topics = []    # list of 4 topics
    st.session_state.files = []     # uploaded region files
    st.session_state.regions = {}   # filename -> region

def next_step():
    st.session_state.step += 1

st.title("🌍 FC25 Multilingual Sentiment Dashboard")

# ─── 4) STEP 0: COLLECT 4 TOPICS ──────────────────────────────────
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
            st.error("Please provide a non-empty topic.")

# ─── 5) STEP 5: UPLOAD REGION FILES ───────────────────────────────
elif st.session_state.step == 5:
    st.write("### Step 2: Upload region CSVs")
    st.write("Select one or more CSV files containing comment data.")
    files = st.file_uploader("Select CSVs", type="csv", accept_multiple_files=True)
    if files:
        st.session_state.files = files
        next_step()

# ─── 6) STEP 6: TAG REGIONS ───────────────────────────────────────
elif st.session_state.step == 6:
    st.write("### Step 3: Tag each file with its region")
    all_tagged = True
    for f in st.session_state.files:
        region = st.text_input(f"Region for '{f.name}':", key=f"region_{f.name}")
        if region and region.strip():
            st.session_state.regions[f.name] = region.strip()
        else:
            all_tagged = False
    if all_tagged and st.button("Combine & Analyze"):
        next_step()

# ─── 7) STEP 7+: COMBINE & CLASSIFY ───────────────────────────────
else:
    def find_text_column(df: pd.DataFrame) -> str | None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        best, best_score = None, -1.0
        for c in obj_cols:
            s = df[c].dropna().astype(str)
            if s.empty:
                continue
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

        df = None
        # 1) Try default CSV
        try:
            df = pd.read_csv(f)
        except Exception:
            # 2) Try delimiter sniffing
            try:
                f.seek(0)
                df = pd.read_csv(f, sep=None, engine='python')
            except Exception:
                # 3) Try Excel
                try:
                    f.seek(0)
                    df = pd.read_excel(f)
                except Exception:
                    skipped.append(f.name)
                    continue

        if df is None or df.empty:
            skipped.append(f.name)
            continue

        text_col = find_text_column(df)
        if not text_col:
            st.error(f"No text-like column found in '{f.name}'.")
            st.stop()

        df = df.rename(columns={text_col: 'comment'})
        df['region'] = st.session_state.regions[f.name]
        combined.append(df[['comment','region']])

    if skipped:
        st.warning(f"Skipped files (unreadable or empty): {', '.join(skipped)}")
    if not combined:
        st.error("No valid data to combine. Please upload non-empty CSVs.")
        st.stop()

    master_df = pd.concat(combined, ignore_index=True)
    st.subheader("✅ Combined Comments")
    st.dataframe(master_df.head())
    st.download_button(
        "Download combined CSV",
        master_df.to_csv(index=False),
        file_name="combined_comments_all_regions.csv"
    )

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
        st.download_button(
            "Download final classified CSV",
            result.to_csv(index=False),
            file_name="classified_comments_all_regions.csv"
        )
    except Exception as e:
        st.error(f"API error: {e}")
