"""
Central configuration for the Supply Chain Delay Risk Predictor.

All paths, column names, and constants live here so that the rest of the
code base never hard-codes them.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DIR / "DataCoSupplyChainDataset.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "modeling_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = OUTPUT_DIR / "model.joblib"
META_PATH = OUTPUT_DIR / "preprocess_meta.json"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
BOTTLENECK_PATH = OUTPUT_DIR / "bottlenecks.csv"

# --------------------------------------------------------------------------- #
# Dataset sources
# --------------------------------------------------------------------------- #
# Official Kaggle dataset (requires `kaggle` CLI credentials to download):
#   https://www.kaggle.com/datasets/shashankthecookie/dataco-smart-supply-chain-for-big-data-analysis
# Public mirror used by src/download_data.py so the project runs without a
# Kaggle API token:
MIRROR_URL = (
    "https://huggingface.co/idoyaaran/dataco-supply-chain-model/"
    "resolve/main/DataCoSupplyChainDataset.csv"
)
CSV_ENCODING = "latin-1"  # the raw file contains accented characters

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2          # 80 / 20 stratified split
TARGET_COL = "late_delivery"
MIN_ORDERS_FOR_SEGMENT = 200  # ignore tiny segments in bottleneck analysis

# --------------------------------------------------------------------------- #
# Column names (kept explicit — the raw CSV uses human-readable headers)
# --------------------------------------------------------------------------- #
SHIPPING_MODE_COL = "Shipping Mode"
REGION_COL = "Order Region"
MARKET_COL = "Market"
CATEGORY_COL = "Category Name"

REAL_DAYS_COL = "Days for shipping (real)"
SCHEDULED_DAYS_COL = "Days for shipment (scheduled)"

# Categorical features that are one-hot encoded before modelling.
CATEGORICAL_FEATURES = [
    "Type",                    # payment type of the order
    "Customer Segment",        # Consumer / Corporate / Home Office
    "Department Name",
    MARKET_COL,                # geographic market (5 values)
    REGION_COL,                # sub-region within a country (23 values)
    CATEGORY_COL,              # product category (50 values)
    SHIPPING_MODE_COL,         # Standard / Second / First / Same Day
]

# Numeric features kept for modelling (all known at order-placement time).
NUMERIC_FEATURES = [
    SCHEDULED_DAYS_COL,
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Quantity",
    "Order Item Profit Ratio",
    "Order Item Total",
    "Sales",
    "Order Profit Per Order",
    "Product Price",
]

# Columns that are dropped: identifiers, free text / PII, near-constant fields,
# post-hoc (leaky) fields, and raw dates replaced by engineered features.
DROP_COLUMNS = [
    # --- identifiers ---
    "Category Id", "Customer Id", "Department Id", "Order Customer Id",
    "Order Id", "Order Item Cardprod Id", "Order Item Id",
    "Product Card Id", "Product Category Id",
    # --- PII / free text / high cardinality ---
    "Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
    "Customer Street", "Customer Zipcode", "Customer City", "Customer State",
    "Customer Country", "Order City", "Order State", "Order Zipcode",
    "Order Country", "Product Name", "Product Image", "Product Description",
    "Product Status",
    # --- leakage / target components (cannot be known before delivery) ---
    REAL_DAYS_COL,            # defines the target -> must not be a feature
    "Delivery Status",        # human label derived from the target
    "Late_delivery_risk",     # dataset's own target (we rebuild our own)
    "shipping date (DateOrders)",  # only known once the order ships
    # --- misc / redundant ---
    "Order Status",           # order-lifecycle status, unrelated to lateness
    "Latitude", "Longitude",  # redundant with Market / Order Region
    "order date (DateOrders)",  # consumed by date feature engineering
]

# Columns required for the target / EDA to be computable.
REQUIRED_COLUMNS = [
    REAL_DAYS_COL, SCHEDULED_DAYS_COL,
    SHIPPING_MODE_COL, REGION_COL, MARKET_COL, CATEGORY_COL,
    TARGET_COL,
]
