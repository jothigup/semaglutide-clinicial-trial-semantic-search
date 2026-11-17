# Semaglutide Clinical Trial Semantic Search

Search clinical trial records using natural language and compare the most relevant result from a general-purpose embedding model and a biomedical embedding model.

The repository includes a CSV containing **837 study records**, a **Streamlit web app**, and a **Jupyter notebook with a Gradio interface**. Both interfaces return **one best-matching study per model**, with an optional filter for the trial's start year.

## Features

- Search trial descriptions by meaning using pretrained sentence embeddings.
- Compare results from two models:

  | Model | Hugging Face identifier |
  | --- | --- |
  | General-purpose | `sentence-transformers/all-MiniLM-L6-v2` |
  | Biomedical | `pritamdeka/S-PubMedBert-MS-MARCO` |

- Filter results by the year in the CSV's `Start Date` column.
- Upload a compatible CSV through the Streamlit app.
- View study titles, summaries, dates, and additional details in Streamlit, with links to ClinicalTrials.gov when supplied.
- View cosine similarity scores in Streamlit.

The two models run independently and may return the same study. Their results are displayed separately; the project does not combine them into a single ranking.

## Repository structure

```text
.
├── README.md
├── chromadb/
│   ├── app.py                         # Streamlit app; uses NumPy similarity search
│   ├── ctg-studies.csv                 # Included clinical trial records
│   └── requirements.txt               # App dependencies
└── notebooks/
    └── search_query_updated.ipynb      # ChromaDB search with a Gradio interface
```

Despite its folder name, the Streamlit app does not use ChromaDB. It compares normalized embeddings directly with NumPy. The notebook uses separate in-memory ChromaDB collections for the two models.

## Run the Streamlit app locally

### 1. Get the repository

```bash
git clone https://github.com/jothigup/semaglutide-clinicial-trial-semantic-similarity.git
cd semaglutide-clinicial-trial-semantic-similarity
```

### 2. Create and activate a Python environment

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies and start the app

Run these commands from the repository root:

```bash
python -m pip install -r chromadb/requirements.txt
python -m streamlit run chromadb/app.py
```

Open the local URL printed in the terminal.

### 4. Load the CSV and search

1. Use **Upload your clinical trial CSV** to select `chromadb/ctg-studies.csv` or your own compatible CSV.
2. Enter a query, such as `semaglutide for weight loss in adults`.
3. Select **All years** or a trial start year.
4. Click **Find most relevant study** to see one result from each model.

**Current filename mismatch:** `app.py` looks for `ctg-studies(1).csv` beside the app, while the included file is named `ctg-studies.csv`. Uploading the included CSV works without changing the code. To load it automatically, change the default path in `chromadb/app.py` to:

```python
default_csv = Path(__file__).resolve().parent / "ctg-studies.csv"
```

The first search downloads the pretrained models and creates embeddings for the dataset, so it takes longer and requires internet access. The app runs the models on the CPU and caches models and document embeddings for reuse.

## CSV format

Each row represents one clinical trial. These column headers are required:

| Column | Purpose |
| --- | --- |
| `NCT Number` | Trial identifier |
| `Study Title` | Searchable trial title |
| `Brief Summary` | Searchable trial description |
| `Start Date` | Source of the optional year filter |

The following columns are optional:

| Column | Use |
| --- | --- |
| `Conditions` | Search text and study details |
| `Interventions` | Search text and study details |
| `Primary Outcome Measures` | Search text and study details |
| `Study URL` | Link to the trial record |
| `Study Status` | Displayed in Streamlit results |
| `Study Type` | Additional study details |
| `Primary Completion Date` | Additional study details |
| `Completion Date` | Additional study details |

The included CSV also contains `Study Results`, which the current code does not use for embedding or display.

Search text combines the title, conditions, interventions, brief summary, and primary outcome measures, when those fields are available. Surrounding whitespace is removed from column headers; names are otherwise case-sensitive.

For year filtering, use `YYYY`, `YYYY-MM`, or `YYYY-MM-DD` values in `Start Date`. Missing or unrecognized dates are included under **All years** and excluded from a specific year filter. Years are never inferred from the summary or other text.

## Run the notebook

From the repository root, with the virtual environment activated:

```bash
python -m pip install jupyterlab
cd notebooks
jupyter lab search_query_updated.ipynb
```

Run the cells from top to bottom. The first cell installs the notebook dependencies: `pandas`, `sentence-transformers`, `chromadb`, and `gradio`. Restart the kernel if installation requires it, then continue running the cells.

The code sets:

```python
CSV_PATH = Path("../chromadb/ctg-studies.csv")
```

This path is correct when the notebook runs from the `notebooks` directory. Edit it if you move the notebook or use another dataset. Some introductory text and the missing-file error still refer to `ctg-studies(1).csv`; the `CSV_PATH` assignment determines which file is loaded.

The notebook downloads both models, builds two in-memory ChromaDB collections using cosine distance, and launches a Gradio interface. It returns one study per model, including its identifier, title, URL, start year, and combined text. The index is not saved to disk and must be rebuilt in a new session.

## Deploy on Streamlit Community Cloud

Use these repository paths when configuring the app:

| Setting | Value |
| --- | --- |
| Repository | `jothigup/semaglutide-clinicial-trial-semantic-similarity` |
| Branch | `main` |
| Main file path | `chromadb/app.py` |
| Dependency file | `chromadb/requirements.txt` |

Keep `requirements.txt` beside `app.py`. Once the app starts, upload the CSV through the interface. If you want the included CSV to load automatically, apply the default-path correction above and commit it before deployment.

The deployment environment needs internet access to download model weights and enough memory to load both models and the study embeddings. Initial startup and the first search can take longer than later searches.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Install `sentence-transformers` in the environment running the app. Locally, use `python -m pip install -r chromadb/requirements.txt`. For deployment, confirm that the requirements file is committed beside the app and rebuild or restart after updating dependencies. |
| The app asks for a CSV even though one is included | Upload `chromadb/ctg-studies.csv`, or correct the default filename in `app.py` as described above. |
| `Missing CSV columns` | Check the four required headers in the CSV format section. |
| The notebook cannot find the CSV | Run it from `notebooks`, or update `CSV_PATH`. |
| A model fails to load or the first search is slow | Check internet access, available memory, and the error shown by the app. Model downloads and initial embedding creation take time. |
| A study does not appear under a specific year | Check its `Start Date`; missing or unsupported date formats only appear under **All years**. |

## Interpreting results

- The models retrieve similar text; they are not fine-tuned or trained on this CSV by the project.
- Each model returns the highest-ranked eligible record even if its relevance is weak. There is no minimum similarity threshold or relevance evaluation included.
- Cosine similarity measures text similarity, not confidence, study quality, treatment efficacy, or patient eligibility. Scores from the two models are not calibrated against each other.
- Long trial records may be truncated to each model's input limit when creating embeddings, although results display source text beyond that limit.
- Year filtering refers to trial start year. It does not measure changes in treatment outcomes over time.
- The app searches the supplied CSV snapshot; it does not fetch live updates from ClinicalTrials.gov.

This project supports exploration of clinical trial records. Review the original study record before drawing conclusions or making clinical decisions.
