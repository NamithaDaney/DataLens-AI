"""
DataLens AI · Vehicle Anomaly Detection
Upload a CSV / Excel file -> automatic cleaning, EDA, autoencoder training
-> downloadable anomaly report.

Run:  streamlit run app.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

from anomaly_pipeline import PipelineConfig, list_excel_sheets, load_dataset, run_pipeline

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "vehicles_sample.csv"
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# Copies of every result are also written on the machine running the app.
# On Google Colab that is /content/anomaly_outputs (erased when the runtime recycles).
OUT_DIR = Path("/content/anomaly_outputs") if Path("/content").is_dir() else Path("anomaly_outputs")

st.set_page_config(
    page_title="DataLens AI · Anomaly Detection",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.1rem;
           background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);}
    .hero h1 {margin: 0; font-size: 2rem; color: #ffffff !important;}
    .hero p {margin: .35rem 0 0; color: #ffffff; opacity: .92; font-size: 1.02rem;}
    .step-row {display: flex; gap: .55rem; flex-wrap: wrap; margin: .1rem 0 1.1rem;}
    .step {background: rgba(99,102,241,.10); border: 1px solid rgba(99,102,241,.35);
           padding: .35rem .85rem; border-radius: 999px; font-size: .88rem;}
    div[data-testid="stMetric"] {background: rgba(99,102,241,.07);
           border: 1px solid rgba(99,102,241,.25); padding: .8rem 1rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🔎 DataLens AI — Anomaly Detection</h1>
      <p>Upload a vehicle dataset. The pipeline cleans it, explores it, trains an autoencoder
         and hands you a downloadable list of suspicious rows — no code needed.</p>
    </div>
    <div class="step-row">
      <span class="step">1 · Upload</span><span class="step">2 · Auto-clean</span>
      <span class="step">3 · EDA</span><span class="step">4 · Train</span>
      <span class="step">5 · Download anomalies</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Sidebar: settings
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pipeline settings")
    st.caption("Defaults work well. Change them only if you need to.")

    threshold_pct = st.slider(
        "Anomaly sensitivity (threshold percentile)",
        min_value=90.0, max_value=99.9, value=95.0, step=0.1,
        help="A row is flagged when its reconstruction error is above this percentile of the "
             "training errors. 95 flags roughly the top 5% most unusual rows; "
             "raise it to flag fewer rows.",
    )

    with st.expander("Training"):
        epochs = st.slider("Max epochs", 5, 200, 50, 5)
        batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128, 256], value=32)
        patience = st.slider("Early-stopping patience", 2, 20, 5)
        test_size = st.slider("Held-out test share", 0.10, 0.40, 0.20, 0.05)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

    with st.expander("Data handling"):
        score_all_rows = st.checkbox(
            "Score every row (recommended)", value=True,
            help="Off = report only the held-out test rows, as in the original notebook.",
        )
        dayfirst = st.checkbox("Dates are day-first (DD/MM/YYYY)", value=False)

    st.divider()
    if importlib.util.find_spec("tensorflow") is not None:
        st.success("TensorFlow detected: Keras autoencoder")
    else:
        st.warning("TensorFlow not found: a scikit-learn autoencoder will be used instead.")

    if st.button("🗑️ Clear results"):
        for key in ("result", "source_key", "use_sample", "downloads", "saved_files"):
            st.session_state.pop(key, None)
        st.rerun()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes, filename: str, sheet):
    return load_dataset(file_bytes, filename, sheet)


@st.cache_data(show_spinner=False)
def cached_sheets(file_bytes: bytes, filename: str):
    return list_excel_sheets(file_bytes, filename)


def save_outputs(result) -> dict:
    """Write the four deliverables to OUT_DIR and return {label: path}."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(result.filename).stem
    paths = {
        "Excel report": OUT_DIR / f"{stem}_anomaly_report.xlsx",
        "Anomalies CSV": OUT_DIR / f"{stem}_anomalies.csv",
        "Scored dataset CSV": OUT_DIR / f"{stem}_scored.csv",
        "Model artifacts ZIP": OUT_DIR / f"{stem}_model_artifacts.zip",
    }
    paths["Excel report"].write_bytes(result.excel_report)
    paths["Model artifacts ZIP"].write_bytes(result.artifacts_zip)
    result.anomalies.to_csv(paths["Anomalies CSV"], index=False, encoding="utf-8-sig")
    result.scored.to_csv(paths["Scored dataset CSV"], index=False, encoding="utf-8-sig")
    return {label: str(path) for label, path in paths.items()}


def csv_bytes(key: str, df: pd.DataFrame) -> bytes:
    """Build CSV once per result so reruns stay fast."""
    store = st.session_state.setdefault("downloads", {})
    if key not in store:
        store[key] = df.to_csv(index=False).encode("utf-8-sig")
    return store[key]


# ----------------------------------------------------------------------
# 1. Upload
# ----------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload your dataset (CSV, XLSX or XLS)",
    type=["csv", "tsv", "xlsx", "xls"],
    help="The first row must contain column names. Max size is set in .streamlit/config.toml.",
)

file_bytes, filename = None, None
if uploaded is not None:
    file_bytes, filename = uploaded.getvalue(), uploaded.name
    st.session_state.pop("use_sample", None)
elif st.session_state.get("use_sample") and SAMPLE_PATH.exists():
    file_bytes, filename = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name

if file_bytes is None:
    if "result" not in st.session_state:
        st.info("👆 Upload a file to begin, or try the built-in sample.")
        if SAMPLE_PATH.exists() and st.button("Try the sample vehicle dataset"):
            st.session_state["use_sample"] = True
            st.rerun()
        with st.expander("What does the pipeline do automatically?", expanded=True):
            st.markdown(
                "- **Loads & validates** the file (delimiter and encoding detection, Excel sheets)\n"
                "- **Cleans** text numbers such as `1197 CC`, `84 bhp`, `113 Nm`, `₹ 5.5 Lakh`, "
                "and converts power to kW and torque to Nm\n"
                "- **Detects dates / years** and turns them into model features\n"
                "- **Drops** empty, constant, duplicate and ID-like columns (nothing is deleted "
                "from your original data)\n"
                "- **Explores** the data: missing values, distributions, correlations, outliers\n"
                "- **Trains** an autoencoder on a training split and sets the anomaly threshold "
                "from training errors only\n"
                "- **Explains** each anomaly with the columns that contribute most\n"
                "- **Exports** an Excel report, CSVs and the trained model"
            )
    else:
        file_bytes = None  # keep showing the previous result below
else:
    source_key = (filename, len(file_bytes))
    if st.session_state.get("source_key") != source_key:
        st.session_state["source_key"] = source_key
        st.session_state.pop("result", None)
        st.session_state.pop("downloads", None)

sheet = None
if file_bytes is not None:
    sheets = cached_sheets(file_bytes, filename)
    if len(sheets) > 1:
        sheet = st.selectbox("Excel sheet to analyse", sheets)

    try:
        preview = cached_load(file_bytes, filename, sheet)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read this file: {exc}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(preview):,}")
    c2.metric("Columns", f"{preview.shape[1]:,}")
    c3.metric("Missing cells", f"{preview.isna().mean().mean() * 100:.1f}%")
    with st.expander("Preview (first 10 rows)", expanded="result" not in st.session_state):
        st.dataframe(preview.head(10), hide_index=True)

    # ------------------------------------------------------------------
    # 2. Run
    # ------------------------------------------------------------------
    if st.button("🚀 Run full pipeline", type="primary"):
        config = PipelineConfig(
            test_size=float(test_size),
            random_state=int(seed),
            threshold_percentile=float(threshold_pct),
            epochs=int(epochs),
            batch_size=int(batch_size),
            patience=int(patience),
            score_all_rows=bool(score_all_rows),
            dayfirst=bool(dayfirst),
        )
        bar = st.progress(0.0, text="Starting…")

        def on_progress(fraction: float, message: str):
            bar.progress(fraction, text=message)

        try:
            st.session_state["result"] = run_pipeline(
                file_bytes, filename, config, sheet_name=sheet, progress=on_progress
            )
            st.session_state.pop("downloads", None)
            try:
                st.session_state["saved_files"] = save_outputs(st.session_state["result"])
            except OSError:
                st.session_state["saved_files"] = {}
        except ValueError as exc:
            bar.empty()
            st.error(f"⚠️ {exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.error("Something unexpected went wrong while running the pipeline.")
            st.exception(exc)
            st.stop()
        bar.empty()


# ----------------------------------------------------------------------
# 3. Results
# ----------------------------------------------------------------------
result = st.session_state.get("result")
if result is not None:
    summary = result.summary
    stem = Path(result.filename).stem

    st.divider()
    for note in result.warnings:
        st.warning(note)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Rows scored", f"{summary['Rows scored']:,}")
    k2.metric("Model features", f"{summary['Model inputs after encoding']:,}")
    k3.metric("Potential anomalies", f"{summary['Potential anomalies']:,}")
    k4.metric("Flag rate", f"{summary['Potential anomaly rate (%)']}%")
    k5.metric("Runtime", f"{summary['Runtime (seconds)']}s")

    st.success(
        f"Pipeline finished: **{summary['Potential anomalies']:,}** potential anomalies found "
        f"in **{summary['Rows scored']:,}** rows. Open the **Downloads** tab to get the report."
    )

    tab_anom, tab_eda, tab_prep, tab_model, tab_dl = st.tabs(
        ["🚨 Anomalies", "📊 EDA", "🧹 Data preparation", "🧠 Model", "📦 Downloads"]
    )

    # ---------------- Anomalies ----------------
    with tab_anom:
        anomalies = result.anomalies
        if anomalies.empty:
            st.info("No rows exceeded the anomaly threshold. Try a lower sensitivity percentile.")
        else:
            f1, f2 = st.columns([2, 1])
            max_severity = float(anomalies["anomaly_severity"].max())
            min_severity = f1.slider(
                "Minimum severity (× threshold)",
                1.0, max(1.01, round(max_severity, 2)), 1.0, 0.05,
                help="Severity = reconstruction error ÷ anomaly threshold. "
                     "2.0 means the row is twice as unusual as the cut-off.",
            )
            splits = sorted(anomalies["data_split"].unique())
            chosen_splits = f2.multiselect("Data split", splits, default=splits)

            view = anomalies[
                (anomalies["anomaly_severity"] >= min_severity)
                & (anomalies["data_split"].isin(chosen_splits))
            ]
            st.caption(f"Showing {min(len(view), 1000):,} of {len(view):,} matching anomalies "
                       f"({len(anomalies):,} in total). Downloads contain every row.")
            st.dataframe(
                view.head(1000),
                hide_index=True,
                column_config={
                    "reconstruction_error": st.column_config.NumberColumn(format="%.5f"),
                    "anomaly_severity": st.column_config.NumberColumn("severity (×)", format="%.2f"),
                },
            )
            with st.expander("How to read this table"):
                st.markdown(
                    "- **reconstruction_error**: how badly the autoencoder rebuilt the row. "
                    "Unusual rows are rebuilt poorly.\n"
                    "- **severity (×)**: error divided by the threshold. Higher = more unusual.\n"
                    "- **top_contributing_features**: the columns responsible for most of the error, "
                    "with their share. Start your investigation there.\n"
                    "- **row_number**: position of the row in your uploaded file (first data row = 1).\n"
                    "- **data_split**: whether the row was in the training or held-out test set."
                )
            st.caption("Rows flagged here are *potential* anomalies: unusual compared with the rest "
                       "of your data, not necessarily wrong. Review them before acting.")

    # ---------------- EDA ----------------
    with tab_eda:
        figures = result.figures
        for name in ("Missing values", "Numeric distributions", "Correlation heatmap",
                     "Categorical breakdown"):
            if name in figures:
                st.subheader(name)
                st.pyplot(figures[name])
        if "Missing values" not in figures:
            st.success("No missing values in the uploaded data.")
        if "iqr_outliers" in result.tables:
            with st.expander("Classic IQR outliers per numeric column"):
                st.dataframe(result.tables["iqr_outliers"], hide_index=True)
        if "numeric_summary" in result.tables:
            with st.expander("Numeric summary statistics"):
                st.dataframe(result.tables["numeric_summary"], hide_index=True)
        if "categorical_summary" in result.tables:
            with st.expander("Categorical summary"):
                st.dataframe(result.tables["categorical_summary"], hide_index=True)
        with st.expander("Column profile (types, missing values, unique counts)"):
            st.dataframe(result.tables["profile"], hide_index=True)

    # ---------------- Data preparation ----------------
    with tab_prep:
        plan = result.tables["column_plan"]
        counts = plan["role"].value_counts()
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Numeric features", int(counts.get("numeric", 0)))
        p2.metric("Categorical features", int(counts.get("categorical", 0)))
        p3.metric("Date columns", int(counts.get("date", 0)))
        p4.metric("Excluded columns", int(counts.get("excluded", 0)))

        st.subheader("What the pipeline did")
        st.dataframe(result.tables["pipeline_log"], hide_index=True)

        st.subheader("Column plan")
        st.caption("Excluded columns stay in your downloads. They are only left out of model training.")
        role_filter = st.multiselect(
            "Show roles", ["numeric", "categorical", "date", "excluded"],
            default=["numeric", "categorical", "date", "excluded"],
        )
        st.dataframe(plan[plan["role"].isin(role_filter)], hide_index=True)

        conv = result.tables["conversion_log"]
        if not conv.empty:
            st.subheader("Text columns converted to numbers")
            st.dataframe(conv, hide_index=True)
        review = result.tables["review_report"]
        if not review.empty:
            st.subheader("Values that could not be parsed")
            st.caption("These are kept as-is in your data; for modelling they are treated as missing.")
            st.dataframe(review.head(500), hide_index=True)
        temporal = result.tables["temporal_report"]
        if not temporal.empty:
            st.subheader("Date / year columns detected")
            st.dataframe(temporal, hide_index=True)
        near = result.tables["near_duplicate_columns"]
        if not near.empty:
            st.subheader("Near-identical column pairs")
            st.dataframe(near, hide_index=True)

    # ---------------- Model ----------------
    with tab_model:
        m1, m2 = st.columns(2)
        with m1:
            st.pyplot(result.figures["Training loss"])
        with m2:
            st.pyplot(result.figures["Error distribution"])
        if "Anomaly drivers" in result.figures:
            d1, d2 = st.columns([3, 2])
            with d1:
                st.pyplot(result.figures["Anomaly drivers"])
            with d2:
                drivers = result.tables["anomaly_drivers"].copy()
                drivers["share"] = (drivers["share"] * 100).round(1).astype(str) + "%"
                st.dataframe(drivers, hide_index=True)
        st.subheader("Run summary")
        st.dataframe(
            pd.DataFrame({"Metric": list(summary.keys()), "Value": [str(v) for v in summary.values()]}),
            hide_index=True,
        )

    # ---------------- Downloads ----------------
    with tab_dl:
        st.subheader("Download your results")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Excel report** (recommended)")
            st.caption("Sheets: Summary, Anomalies, All_Rows_Scored, Column_Plan, "
                       "Conversion_Log, Review_Items, Pipeline_Log.")
            st.download_button(
                "⬇️ Download Excel report", data=result.excel_report,
                file_name=f"{stem}_anomaly_report.xlsx", mime=EXCEL_MIME, type="primary",
            )
            st.markdown("**Anomalies only (CSV)**")
            st.download_button(
                "⬇️ Download anomalies CSV", data=csv_bytes("anomalies", result.anomalies),
                file_name=f"{stem}_anomalies.csv", mime="text/csv",
            )
        with d2:
            st.markdown("**All rows with scores (CSV)**")
            st.caption("Your original columns plus reconstruction error, severity and status.")
            st.download_button(
                "⬇️ Download scored dataset CSV", data=csv_bytes("scored", result.scored),
                file_name=f"{stem}_scored.csv", mime="text/csv",
            )
            st.markdown("**Trained model bundle (ZIP)**")
            st.caption("Autoencoder, fitted preprocessor and metadata (threshold, features).")
            st.download_button(
                "⬇️ Download model artifacts", data=result.artifacts_zip,
                file_name=f"{stem}_model_artifacts.zip", mime="application/zip",
            )
        saved = st.session_state.get("saved_files") or {}
        if saved:
            st.info(
                f"Copies of these files were also saved on the machine running the app, in "
                f"`{OUT_DIR}`. On Google Colab that folder is erased when the runtime recycles, "
                "so download what you need now."
            )
        if len(result.scored) > 1_000_000:
            st.info("The Excel report omits the All_Rows_Scored sheet because it exceeds "
                    "Excel's row limit. Use the CSV instead.")

st.caption("DataLens AI · autoencoder-based anomaly detection · results are indicators to review, "
           "not verdicts.")
