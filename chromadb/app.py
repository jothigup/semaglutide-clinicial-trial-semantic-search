from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Clinical Trial Search",
    page_icon="🔎",
    layout="wide",
)

MODEL_NAMES = {
    "General": "sentence-transformers/all-MiniLM-L6-v2",
    "Biomedical": "pritamdeka/S-PubMedBert-MS-MARCO",
}


@st.cache_data(max_entries=3)
def load_data(csv_bytes):
    df = pd.read_csv(BytesIO(csv_bytes))
    df.columns = df.columns.str.strip()

    required = [
        "NCT Number",
        "Study Title",
        "Brief Summary",
        "Start Date",
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

    # Extract the year only from the actual Start Date column.
    year_text = df["Start Date"].astype("string").str.extract(
        r"^\s*(\d{4})(?:-\d{2})?(?:-\d{2})?\s*$",
        expand=False,
    )
    df["start_year"] = pd.to_numeric(
        year_text, errors="coerce"
    ).astype("Int64")

    text_columns = [
        column
        for column in [
            "Study Title",
            "Conditions",
            "Interventions",
            "Brief Summary",
            "Primary Outcome Measures",
        ]
        if column in df.columns
    ]

    df["search_text"] = df[text_columns].fillna("").apply(
        lambda row: "\n\n".join(
            f"{column}: {str(row[column]).strip()}"
            for column in text_columns
            if str(row[column]).strip()
        ),
        axis=1,
    )

    df = df.loc[df["search_text"].str.strip().ne("")].reset_index(drop=True)

    if df.empty:
        raise ValueError("The CSV contains no searchable study text.")

    return df


@st.cache_resource
def load_model(model_name):
    # A lock protects each shared model during concurrent searches.
    return SentenceTransformer(model_name, device="cpu"), Lock()


@st.cache_data(max_entries=4, show_spinner=False)
def create_embeddings(texts, model_name):
    model, lock = load_model(model_name)

    with lock:
        return model.encode(
            list(texts),
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def best_match(query, texts, candidate_indices, model_name):
    embeddings = create_embeddings(texts, model_name)
    model, lock = load_model(model_name)

    with lock:
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

    # Normalized dot products equal cosine similarity.
    scores = embeddings[candidate_indices] @ query_embedding
    best_position = int(np.argmax(scores))

    return (
        int(candidate_indices[best_position]),
        float(scores[best_position]),
    )


def text_value(row, column, default="Not provided"):
    value = row.get(column)
    if pd.isna(value) or not str(value).strip():
        return default
    return str(value)


st.title("Clinical Trial Search")
st.write(
    "Find the most relevant study for your query using "
    "general and biomedical embedding models."
)

uploaded_file = st.file_uploader(
    "Upload your clinical trial CSV",
    type=["csv"],
)

default_csv = Path(__file__).resolve().parent / "ctg-studies(1).csv"

if uploaded_file is not None:
    csv_bytes = uploaded_file.getvalue()
elif default_csv.is_file():
    csv_bytes = default_csv.read_bytes()
    st.caption(f"Using {default_csv.name}")
else:
    st.info("Upload your CSV above to begin.")
    st.stop()

try:
    df = load_data(csv_bytes)
except Exception as error:
    st.error(f"Could not load the CSV: {error}")
    st.stop()

available_years = sorted(
    df["start_year"].dropna().astype(int).unique().tolist()
)

st.caption(f"{len(df):,} studies available.")

with st.form("search_form"):
    query = st.text_input(
        "What kind of study are you looking for?",
        placeholder="Example: semaglutide for weight loss in adults",
    )

    selected_year = st.selectbox(
        "Filter by trial start year",
        ["All years"] + [str(year) for year in available_years],
    )

    submitted = st.form_submit_button("Find most relevant study")

if submitted:
    if not query.strip():
        st.warning("Enter a search query.")
        st.stop()

    if selected_year == "All years":
        candidate_indices = np.arange(len(df))
    else:
        mask = (
            df["start_year"]
            .eq(int(selected_year))
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        candidate_indices = np.flatnonzero(mask)

    if len(candidate_indices) == 0:
        st.warning("No studies match that start year.")
        st.stop()

    texts = tuple(df["search_text"].tolist())

    st.caption(
        "Showing one best match per model. "
        "Both models may select the same study."
    )

    for label, model_name in MODEL_NAMES.items():
        st.subheader(f"{label} model")

        try:
            with st.spinner(
                f"Searching with the {label.lower()} model... "
                "The first search takes longer."
            ):
                row_index, score = best_match(
                    query.strip(),
                    texts,
                    candidate_indices,
                    model_name,
                )
        except Exception as error:
            st.error(f"Could not run the {label.lower()} model: {error}")
            continue

        study = df.iloc[row_index]

        st.write(f"**{text_value(study, 'Study Title')}**")
        st.write(
            f"NCT ID: {text_value(study, 'NCT Number')}  \n"
            f"Start date: {text_value(study, 'Start Date')}  \n"
            f"Status: {text_value(study, 'Study Status')}"
        )

        st.caption(
            f"Cosine similarity: {score:.3f}. "
            "This measures text similarity, not confidence "
            "or study quality."
        )

        st.write("**Brief summary**")
        st.write(text_value(study, "Brief Summary"))

        url = text_value(study, "Study URL", default="")
        if url.startswith("https://clinicaltrials.gov/"):
            st.link_button("View study on ClinicalTrials.gov", url)

        with st.expander("Additional study details"):
            for column in [
                "Conditions",
                "Interventions",
                "Primary Outcome Measures",
                "Study Type",
                "Primary Completion Date",
                "Completion Date",
            ]:
                st.write(f"**{column}**")
                st.write(text_value(study, column))

        st.divider()

st.caption(
    "Year filtering uses Start Date. Studies without a start date "
    "appear only under All years. Long records may be truncated "
    "to each model's input limit when creating embeddings."
)