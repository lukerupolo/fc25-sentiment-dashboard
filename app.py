import streamlit as st
import pandas as pd
import requests
import openai

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
    st.session_state.contexts = {}     # topic → context blob or list
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

    # Example: for "Ultimate TOTS", upload full CSV of players+ratings
    if topic.lower() == "ultimate tots":
        f = st.file_uploader(
            f"Upload your full TOTS players+ratings CSV", type="csv", key=f"ctx_tots_{ti}"
        )
        if f:
            df = pd.read_csv(f)
            st.session_state.contexts[topic] = df.to_dict("records")
            st.success("TOTS list saved.")
            if st.button("Next"):
                next_step()

    # Example: for any topic containing "gameplay", paste patch notes
    elif "gameplay" in topic.lower():
        notes = st.text_area(
            "Paste any new gameplay patch notes or bullet-points", key=f"ctx_game_{ti}"
        )
        if notes.strip():
            st.session_state.contexts[topic] = notes
            if st.button("Next"):
                next_step()

    # Fallback: generic free-text context
    else:
        extra = st.text_area(
            f"Any extra context for '{topic}'? (optional)", key=f"ctx_free_{ti}"
        )
        if extra is not None:
            st.session_state.contexts[topic] = extra
            if st.button("Next"):
                next_step()

# ─── 5) STEP 9: UPLOAD REGION FILES ───────────────────────────────
elif st.session_state.step == 9:
    st.write("### Step 3: Upload all region CSVs")
    st.write("Each file should have a `comment` column.")
    files = st.file_uploader(
        "Select one or more CSVs", type="csv", accept_multiple_files=True, key="upload_files"
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
            f"Region for `{f.name}`", key=f"region_{f.name}"
        ).strip()
        if region:
            st.session_state.regions[f.name] = region
        else:
            all_tagged = False

    if all_tagged and st.button("Combine & Analyze"):
        next_step()

# ─── 7) STEP 11+: COMBINE, CLASSIFY & REPORT ───────────────────────
else:
    # Combine CSVs with error handling
    combined = []
    skipped = []
    for f in st.session_state.files:
        # Skip truly empty files
        if f.size == 0:
            skipped.append(f.name)
            continue
        # Try reading; catch empty-data errors
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            skipped.append(f.name)
            continue
        # Normalize column name
        if "comment" in df.columns:
            df = df.rename(columns={"comment": "comment"})
        if "comment" not in df.columns:
            st.error(f"`{f.name}` is missing a 'comment' column.")
            st.stop()
        df["region"] = st.session_state.regions[f.name]
        combined.append(df[["comment", "region"]])

    if skipped:
        st.warning(f"Skipped empty or unreadable files: {', '.join(skipped)}")
    if not combined:
        st.error("No valid CSV data to combine. Please upload non-empty CSVs.")
        st.stop()

    master_df = pd.concat(combined, ignore_index=True)

    st.subheader("✅ Combined Comments")
    st.dataframe(master_df.head())

    st.download_button(
        "Download combined CSV",
        master_df.to_csv(index=False),
        file_name="combined_comments_all_regions.csv"
    )

    # Prepare payload including topics and contexts
    payload = {
        "comments": master_df["comment"].fillna("").tolist(),
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
