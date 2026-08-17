"""Central configuration: paths, column groups and constants"""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

TRAIN_FILE = DATA_RAW / "training_merged.csv"        # X_train + y_train merged on id
COMP_FILE = DATA_RAW / "test_set_values_x_test.csv"  # competition set (no labels)

# --- Target ------------------------------------------------------------------
TARGET = "status_group"

# Integer codes used internally. 0 = failure, 1 = healthy, 2 = at-risk.
LABEL_TO_CODE = {
    "non functional": 0,
    "functional": 1,
    "functional needs repair": 2,
}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --- Column groups -----------------------------------------------------------
# Columns dropped outright: single-valued, free-text ids, or duplicated
# hierarchy levels kept only in their most useful granularity.
DROP_COLS = [
    "id",              # identifier
    "recorded_by",     # single constant value -> no signal
    "wpt_name",        # ~37k unique free-text names (handled via hashing instead)
    "num_private",     # 98.7% zeros, undocumented -> no signal
    "date_recorded",   # replaced by engineered year_recorded / years_in_operation
    # Redundant hierarchy duplicates (we keep one level of each family):
    "payment_type",            # duplicate of `payment`
    "quantity_group",          # duplicate of `quantity`
    "source_type",             # coarser duplicate of `source`
    "waterpoint_type_group",   # coarser duplicate of `waterpoint_type`
    "extraction_type_group",   # middle level of extraction hierarchy
    "management_group",         # coarser duplicate of `management`
    "quality_group",           # coarser duplicate of `water_quality`
]

# High-cardinality categoricals imputed with the literal token "other".
FILL_OTHER = ["scheme_management", "installer", "subvillage",
              "public_meeting", "permit", "funder"]

# Numeric columns whose 0 means "missing" and are imputed with the train median.
ZERO_AS_MISSING_MEDIAN = ["amount_tsh", "construction_year"]

# Same, but missingness itself correlates with the target -> keep a flag.
ZERO_AS_MISSING_FLAGGED = ["gps_height", "population"]


# Very high cardinality -> feature hashing into a fixed number of buckets.
HASH_COLS = ["subvillage"]
HASH_BUCKETS = 128

# High cardinality -> frequency encoding (map category -> its train frequency).
FREQ_COLS = ["ward", "lga", "funder", "installer"]

# Low cardinality -> one-hot encoding.
ONEHOT_COLS = [
    "basin", "region", "scheme_management",
    "extraction_type", "extraction_type_class",
    "management", "payment", "water_quality",
    "quantity", "source", "source_class", "waterpoint_type",
]

# Boolean-like columns to map to {0,1}.
BOOL_COLS = ["public_meeting", "permit"]
