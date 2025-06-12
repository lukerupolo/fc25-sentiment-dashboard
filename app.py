import streamlit as st
import pandas as pd
import requests
import openai
from bs4 import BeautifulSoup

# ─── 1) OPENAI KEY ────────────────────────────────────────────────
try:
    openai.api_key = st.secrets["openai_api_key"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

API_URL = "https://sentiment-api-1081516136341.us-central1.run.app/predict"

# ─── 2) SESSION STATE SETUP ───────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.topics = []       # Will hold the 4 topic names
    st.session_state.contexts = {}     # topic → context blob or list or scraped text
    st.session_state.files = []        # uploaded CSV files
    st.session_state.regions = {}      # filename → region tag

def next_step():
    st.session_state.step += 1

st.title("🌍 FC25 Multilingual Sentiment Dashboard")

# ─── 3) STEPS 0–4: ASK FOR 4 TOPIC NAMES ──────────────────────────
if st.session_state.step == 0:
    st.write("### Step 1: Define your 4 topics")
    st.write("I'll ask you for exactly four topics you care about.")
    if st.button("Start"):
        next_step()

elif 1 <= st.session_state.step <= 4:
    idx = st.session_state.step
    topic = st.text_input(f"Topic #{idx} name:", key=f"topic_in_{idx}")
    if st.button("Save topic", key=f"save_topic_{idx}"):
        if topic.strip():
            st.session_state.topics.append(topic.strip())
            next_step()
        else:
            st.error("Please enter a topic name.")

# ─── 4) STEPS 5–8: GATHER CONTEXT FOR EACH TOPIC ──────────────────
elif 5 <= st.session_state.step <= 8:
    ti = st.session_state.step - 4
    topic = st.session_state.topics[ti-1]
    st.write(f"### Step 2.{ti}: Context for **{topic}**")
    st.write("You can either upload a file, paste update-URL, or provide free text.")

    # Option A: upload CSV for structured context
    f = st.file_uploader(
        f"(Optional) Upload CSV context for '{topic}' (e.g. players list)",
        type="csv",
        key=f"ctx_file_{ti}"
    )
    if f:
        try:
            df = pd.read_csv(f)
            st.session_state.contexts[topic] = df.to_dict("records")
            st.success("File context saved.")
        except Exception:
            st.error("Failed to read CSV; please check format.")

    # Option B: scrape from website
    url = st.text_input(
        f"(Optional) Or paste a URL to scrape updates for '{topic}'", key=f"ctx_url_{ti}"
    )
    if url:
        try:
            resp = requests.get(url, timeout=5)
            soup = BeautifulSoup(resp.text, 'html.parser')
            # extract paragraphs
            texts = [p.get_text(strip=True) for p in soup.find_all('p')]
            joined = "\n".join(texts)
            st.session_state.contexts[topic] = joined
            st.success("Scraped website content.")
        except Exception:
            st.error("Failed to fetch or parse URL.")

    # Option C: free-text context
    extra = st.text_area(
        f"(Optional) Free-text context for '{topic}'", key=f"ctx_txt_{ti}"
    )
    if extra.strip():
        st.session_state.contexts[topic] = extra

    if st.button("Next", key=f"next_ctx_{ti}"):
        next_step()

# ─── 5) STEP 9: UPLOAD REGION FILES ───────────────────────────────
elif st.session_state.step == 9:
    st.write("### Step 3: Upload all region CSVs")
    st.write("Select one or more CSV files containing your comments.")
    files = st.file_uploader(
        "Select CSVs", type="csv", accept_multiple_files=True, key="upload_files"
    )
    if files:
        st.session_state.files = files
        next_step()

# ─── 6) STEP 10: ASSIGN REGION NAMES ─────────────────────────────
elif st.session_state.step == 10:
    st.write("### Step 4: Tag each file with its region")
    all_tagged = True
    for f in st.session_state.files:
        region = st.text_input(
            f"Region for '{f.name}'", key=f"region_{f.name}"
        ).strip()
        if region:
            st.session_state.regions[f.name] = region
        else:
            all_tagged = False
    if all_tagged and st.button("Combine & Analyze"):
        next_step()

# ─── 7) STEP 11+: COMBINE, CLASSIFY & REPORT ───────────────────────
else:
    def find_text_column(df):
        # consider object columns
        candidates = [c for c in df.columns if df[c].dtype == object]
        # score by avg length and proportion with spaces
        scores = {}
        for c in candidates:
            series = df[c].dropna().astype(str)
            if len(series) == 0:
                continue
            avg_len = series.map(len).mean()
            space_pct = series.map(lambda s: s.count(' ') >= 1).mean()
            scores[c] = (avg_len, space_pct)
        if not scores:
            return None
        # pick highest avg_len * space_pct
        col = max(scores, key=lambda c: scores[c][0] * scores[c][1])
        return col

    combined = []
    skipped = []
    for f in st.session_state.files:
        if f.size == 0:
            skipped.append(f.name)
            continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            skipped.append(f.name)
            continue
        # detect best text column
        text_col = find_text_column(df)
        if not text_col:
            st.error(f"No text-like column found in '{f.name}'.")
            st.stop()
        df = df.rename(columns={text_col: 'comment'})
        df['region'] = st.session_state.regions[f.name]
        combined.append(df[['comment','region']])

    if skipped:
        st.warning(f"Skipped empty/unreadable files: {', '.join(skipped)}")
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

    payload = {
        "comments": master_df['comment'].fillna("").tolist(),
        "threshold": 0.65,
        "topics": st.session_state.topics,
        "contexts": st.session_state.contexts
    }

    st.write("### Running sentiment & topic classification…")
    try:
        res = requests.post(API_URL, json=payload)
        res.raise_for_status()
        out = pd.DataFrame(res.json())
        result = pd.concat([master_df.reset_index(drop=True), out], axis=1)
        st.subheader("🔍 Results Preview")
        st.dataframe(result.head())
        st.download_button(
            "Download final classified CSV",
            result.to_csv(index=False),
            file_name="classified_comments_all_regions.csv"
        )
    except Exception as e:
        st.error(f"API error: {e}")
