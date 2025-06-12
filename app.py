import streamlit as st
import pandas as pd
import requests
import openai

# ─── 1) OPENAI KEY ─────────────────────────────────────────────
try:
    openai.api_key = st.secrets["openai_api_key"]
except KeyError:
    st.error("Add your OpenAI key to .streamlit/secrets.toml")
    st.stop()

API_URL = "https://sentiment-api-1081516136341.us-central1.run.app/predict"

# ─── 2) STATE SETUP ─────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.topics = []           # ["Ultimate TOTS", ...]
    st.session_state.contexts = {}         # {"Ultimate TOTS": <csv or text>, ...}
    st.session_state.files = []            # uploaded files
    st.session_state.regions = {}          # {filename: region_tag, ...}

def next_step():
    st.session_state.step += 1

st.title("🌍 FC25 Multilingual Sentiment Dashboard")

# ─── 3) TOPIC NAMES (steps 0→4) ─────────────────────────────────
if st.session_state.step == 0:
    st.write("**👋 Let’s set up your 4 topics first.**")
    if st.button("Start"):
        next_step()

elif 1 <= st.session_state.step <= 4:
    idx = st.session_state.step
    t = st.text_input(f"Name of Topic #{idx}:", key=f"topic_in_{idx}")
    if st.button("Save Topic", key=f"save_topic_{idx}"):
        if not t.strip():
            st.error("Please enter a topic name.")
        else:
            st.session_state.topics.append(t.strip())
            next_step()

# ─── 4) TOPIC CONTEXT (steps 5→8) ───────────────────────────────
elif 5 <= st.session_state.step <= 8:
    ti = st.session_state.step - 4
    topic = st.session_state.topics[ti-1]
    st.write(f"### Context for **{topic}**")

    # Example: full TOTS list as CSV
    if topic.lower() == "ultimate tots":
        f = st.file_uploader(
            f"Upload your full TOTS players+ratings CSV",
            type="csv", key=f"ctx_tots_{ti}"
        )
        if f:
            df = pd.read_csv(f)
            st.session_state.contexts[topic] = df.to_dict("records")
            st.success("TOTS list saved.")
            if st.button("Next"):
                next_step()

    # Example: gameplay patch-notes text
    elif "gameplay" in topic.lower():
        notes = st.text_area(
            "Paste any new gameplay patch notes or bullet-points",
            key=f"ctx_game_{ti}"
        )
        if notes.strip():
            st.session_state.contexts[topic] = notes
            if st.button("Next"):
                next_step()

    # Fallback: free text
    else:
        extra = st.text_area(
            f"Any extra context for '{topic}'? (optional)",
            key=f"ctx_free_{ti}"
        )
        if extra is not None:
            st.session_state.contexts[topic] = extra
            if st.button("Next"):
                next_step()

# ─── 5) UPLOAD REGION FILES (step 9) ─────────────────────────────
elif st.session_state.step == 9:
    st.write("**Now upload all the region CSVs** (each should have a column `Comment` or `comment`).")
    files = st.file_uploader(
        "Select multiple CSVs",
        type="csv",
        accept_multiple_files=True,
        key="files_step"
    )
    if files:
        st.session_state.files = files
        next_step()

# ─── 6) ASSIGN REGION TAGS (step 10) ─────────────────────────────
elif st.session_state.step == 10:
    st.write("**Tell me which region each file represents.**")
    all_tagged = True
    for f in st.session_state.files:
        tag = st.text_input(
            f"Region name for `{f.name}`", key=f"reg_{f.name}"
        )
        if tag.strip():
            st.session_state.regions[f.name] = tag.strip()
        else:
            all_tagged = False
    if all_tagged and st.button("Combine & Go"):
        next_step()

# ─── 7) COMBINE + CLASSIFY (step 11) ──────────────────────────────
else:
    # 7a) Combine
    combined = []
    for f in st.session_state.files:
        df = pd.read_csv(f)
        # unify column name
        if "comment" in df.columns:
            df = df.rename(columns={"comment": "Comment"})
        if "Comment" not in df.columns:
            st.error(f"`{f.name}` has no 'Comment' column.")
            st.stop()
        df["region"] = st.session_state.regions[f.name]
        combined.append(df[["Comment", "region"]])
    master_df = pd.concat(combined, ignore_index=True)

    st.subheader("✅ Combined Data")
    st.dataframe(master_df.head())

    st.download_button(
        "Download combined CSV",
        master_df.to_csv(index=False),
        file_name="combined_comments_all_regions.csv"
    )

    # 7b) Call your sentiment+topic API
    payload = {
        "comments": master_df["Comment"].tolist(),
        "threshold": 0.65,
        "topics": st.session_state.topics,
        "contexts": st.session_state.contexts
    }
    st.write("Running sentiment & topic classification…")
    try:
        res = requests.post(API_URL, json=payload)
        res.raise_for_status()
        out = pd.DataFrame(res.json())
        result = pd.concat([master_df.reset_index(drop=True), out], axis=1)

        st.subheader("🔍 Results")
        st.dataframe(result.head())

        st.download_button(
            "Download final classified CSV",
            result.to_csv(index=False),
            file_name="classified_comments_all_regions.csv"
        )

    except Exception as e:
        st.error(f"Failed to classify: {e}")
