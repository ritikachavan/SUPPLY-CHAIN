# Supply Chain Delay Risk Predictor

A student-scale data-science project that predicts whether an order will be
delivered **later than promised** using the *DataCo Smart Supply Chain for Big
Data Analysis* dataset, and surfaces the operational **bottleneck segments**
that drive delays.

## Contents

- `data/` – raw dataset (downloaded) and processed modelling table
- `src/` – data loading, preprocessing, EDA and model training modules
- `dashboard.py` – Streamlit KPI dashboard with a risk-score lookup
- `run_pipeline.py` – one command that runs the whole pipeline
- `outputs/` – charts (PNG), metrics, trained model artifacts
- `requirements.txt` – pinned dependencies

## Approach

1. **Data** – load the DataCo dataset (180k orders, 53 columns) with Pandas.
2. **Cleaning** – drop identifiers, PII/free text, leaky post-hoc columns;
   engineer date features; impute missing values; one-hot encode categoricals.
3. **Target** – `late_delivery = 1` when
   `Days for shipping (real) > Days for shipment (scheduled)`.
4. **EDA** – late-delivery rate by shipping mode, region/market and product
   category; identify the top-3 bottleneck segments; save bar charts.
5. **Modelling** – stratified 80/20 split, Logistic Regression vs Random
   Forest, full metric comparison + feature importance.
6. **Dashboard** – Streamlit app with KPIs, segment charts, feature
   importance and a shipping mode → region → category risk lookup.

## Results

_(Filled in after running the full pipeline — see "Results" below.)_

## How to run

```bash
pip install -r requirements.txt
python src/download_data.py     # fetch the dataset (Kaggle mirror)
python run_pipeline.py          # preprocess + EDA + train + evaluate
streamlit run dashboard.py      # launch the KPI dashboard
```

## Dataset

**DataCo Smart Supply Chain for Big Data Analysis**
(Kaggle): <https://www.kaggle.com/datasets/shashankthecookie/dataco-smart-supply-chain-for-big-data-analysis>

The download script uses a public mirror of the same CSV so the project runs
without Kaggle API credentials.
