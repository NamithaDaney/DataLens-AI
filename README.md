# DataLens AI · Vehicle Anomaly Detection (Streamlit app)

Upload a CSV or Excel file, click **Run full pipeline**, download the anomaly report.

```
Upload → load & validate → profile → clean numeric text → detect dates
       → choose features → EDA → preprocess → train autoencoder
       → score every row → explain → Excel / CSV / model download
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Click **Try the sample vehicle dataset** for a quick demo.

### Run it in Google Colab (as a website)
Upload **`DataLens_AI_Web_App_Colab.ipynb`** to https://colab.research.google.com, choose **Runtime ▸ Run all**,
and open the public link printed by the last cell. Upload your file from your computer inside the app and download
the results to your computer. Copies are also saved in `/content/anomaly_outputs` (erased when the runtime recycles).

`Vehicle_Anomaly_Detection_Colab.ipynb` is an alternative that shows the interface inside the notebook itself
(no website, no tunnel).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (upload, settings, results tabs, downloads) |
| `anomaly_pipeline.py` | The whole pipeline; no Streamlit dependency, usable from a script |
| `sample_data/vehicles_sample.csv` | Synthetic, deliberately messy demo data with injected anomalies |
| `.streamlit/config.toml` | Upload limit (500 MB) and theme |

Use the engine without the UI:

```python
from anomaly_pipeline import run_pipeline, PipelineConfig
result = run_pipeline(open("cars.xlsx", "rb").read(), "cars.xlsx", PipelineConfig(threshold_percentile=97))
result.anomalies.to_csv("anomalies.csv", index=False)
```

## What the output contains

* **Anomalies**: flagged rows with all your original columns plus `reconstruction_error`,
  `anomaly_severity` (error ÷ threshold), `top_contributing_features` (the columns that
  drive the error, with their share), `data_split`, and `row_number`.
* **Excel report**: Summary, Anomalies, All_Rows_Scored, Column_Plan, Conversion_Log,
  Review_Items, Pipeline_Log.
* **Model bundle (zip)**: autoencoder, fitted preprocessor, metadata (threshold, features).

## Settings (sidebar)

* **Sensitivity**: threshold percentile of the *training* reconstruction errors (default 95).
* **Score every row** (default on): the notebook only scored the 20% held-out rows; the app
  scores all rows so you get a verdict for your whole file. Turn it off to reproduce the
  notebook's test-only behaviour.
* Epochs, batch size, early-stopping patience, test share, random seed, day-first dates.

## Changes from the notebook

Same logic and defaults, plus:

1. **Bug fix**: in `convert_numeric_text_columns`, a variable from an earlier loop (`position`)
   leaked into the output list, so an unparseable value could be replaced by the *last row's*
   value. Now each row keeps its own original value.
2. **Bug fix**: dead code after a `return` in `parse_alternate_measurement` removed;
   `84bhp@3750rpm` style strings are now recognised.
3. **Bug fix**: pandas' delimiter sniffer treated the decimal point as a separator on
   single-column files; it now chooses only among `, ; tab |`.
4. **Adaptive layer sizes**: 128 → 64 → 32 for wide data (as in the notebook), scaled down for
   narrow data so the bottleneck stays smaller than the input.
5. **Dates** are expanded into year / month / day-of-week features instead of being one-hot
   encoded; year-only text becomes a number.
6. **Identifier-like text** (almost every value unique, e.g. registration numbers) and
   high-cardinality text are excluded from training but kept in the output.
7. **Explanations** per anomaly, bool columns handled, Excel-safe export, friendly errors.

If TensorFlow is not installed the app falls back to a scikit-learn neural network and says so
in the UI and in the report's Summary sheet.

## Things worth knowing

* An autoencoder flags rows that are *unusual*, not rows that are *wrong*. Review them.
* The model is trained on the file you upload (unsupervised). If that file is mostly bad data,
  bad data looks normal.
* Rare-but-valid values (e.g. the only 2025 listing in a 2023–2024 file) can rank high.
  `top_contributing_features` makes these easy to spot.
* The numeric-text parser is tuned for Indian vehicle listings (₹, lakh/crore, bhp/PS/kW,
  Nm/kgm). Other units are left as text and treated as categorical or missing.
