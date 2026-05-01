"""config.py - configuration file"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = "datasets/dataset_main.csv"
OUTPUT_DIR = "outputs"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Train / validation / test split ──────────────────────────────────────────
VALID_TEST_FRAC = 0.30   # 30 % held out for valid + test
TEST_FRAC_OF_VALID_TEST = 0.50   # half of that 30 % becomes test

# ── Preprocessing ─────────────────────────────────────────────────────────────
COLS_TO_SCALE = ["displacement", "cylinder", "airbags", "make"]
COLS_TO_DROP_OHE = ["transmission_type", "steering_type"]
OHE_COLUMNS = [
    "max_torque", "area_cluster", "engine_type",
    "max_power", "rear_brakes_type", "model",
    "segment", "fuel_type",
]

# ── Decision Tree ─────────────────────────────────────────────────────────────
DT_MAX_DEPTH = 10
DT_CRITERION = "gini"

# ── Random Forest ─────────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = 12
RF_CRITERION = "gini"
RF_MAX_FEATURES = "log2"
RF_MIN_SAMPLES_LEAF = 1
RF_MIN_SAMPLES_SPLIT = 5

# ── Neural Network ────────────────────────────────────────────────────────────
NN_EPOCHS = 20
NN_TEST_SIZE = 0.20
NN_ACTIVATION = "relu"
NN_OUTPUT_ACTIVATION = "sigmoid"
NN_LOSS = "binary_crossentropy"
NN_OPTIMIZER = "adam"
NN_THRESHOLD = 0.55