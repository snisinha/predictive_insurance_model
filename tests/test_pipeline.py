"""tests/test_pipeline.py - run tests on the full pipeline"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data_loader import load_data
from preprocessing import encode_and_scale, split_data
from models import logistic_regression, decision_tree


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Synthetic dataframe that mirrors the real dataset's structure.
    is_claim is exactly 50/50 so the balancer never runs out of rows to sample.
    """
    np.random.seed(42)
    n = 600
    # Force balanced target so _balance() always has enough rows in each class
    is_claim = np.array([0, 1] * (n // 2))
    np.random.shuffle(is_claim)
    return pd.DataFrame({
        "policy_id":                        range(n),
        "policy_tenure":                    np.random.uniform(0, 5, n),
        "age_of_car":                       np.random.uniform(0, 10, n),
        "age_of_policyholder":              np.random.randint(18, 70, n),
        "area_cluster":                     np.random.choice(["C1", "C2", "C3"], n),
        "population_density":               np.random.randint(100, 5000, n),
        "make":                             np.random.randint(1, 10, n),
        "segment":                          np.random.choice(["A", "B", "C"], n),
        "model":                            np.random.choice(["M1", "M2", "M3"], n),
        "fuel_type":                        np.random.choice(["Petrol", "Diesel"], n),
        "max_torque":                       np.random.choice(["90Nm@3500rpm", "110Nm@4000rpm"], n),
        "max_power":                        np.random.choice(["80bhp@6000rpm", "100bhp@5500rpm"], n),
        "engine_type":                      np.random.choice(["F8D Petrol", "K Series"], n),
        "airbags":                          np.random.randint(0, 6, n),
        "is_esc":                           np.random.choice(["Yes", "No"], n),
        "is_adjustable_steering":           np.random.choice(["Yes", "No"], n),
        "is_tpms":                          np.random.choice(["Yes", "No"], n),
        "is_parking_sensors":               np.random.choice(["Yes", "No"], n),
        "is_parking_camera":                np.random.choice(["Yes", "No"], n),
        "rear_brakes_type":                 np.random.choice(["Drum", "Disc"], n),
        "displacement":                     np.random.randint(800, 2000, n),
        "cylinder":                         np.random.randint(3, 6, n),
        "transmission_type":                np.random.choice(["Manual", "Automatic"], n),
        "gear_box":                         np.random.randint(4, 6, n),
        "steering_type":                    np.random.choice(["Power", "Manual"], n),
        "turning_radius":                   np.random.uniform(4, 6, n),
        "length":                           np.random.randint(3000, 5000, n),
        "width":                            np.random.randint(1500, 2000, n),
        "height":                           np.random.randint(1400, 1800, n),
        "gross_weight":                     np.random.randint(1000, 2000, n),
        "is_front_fog_lights":              np.random.choice(["Yes", "No"], n),
        "is_rear_window_wiper":             np.random.choice(["Yes", "No"], n),
        "is_rear_window_washer":            np.random.choice(["Yes", "No"], n),
        "is_rear_window_defogger":          np.random.choice(["Yes", "No"], n),
        "is_brake_assist":                  np.random.choice(["Yes", "No"], n),
        "is_power_door_locks":              np.random.choice(["Yes", "No"], n),
        "is_central_locking":               np.random.choice(["Yes", "No"], n),
        "is_power_steering":                np.random.choice(["Yes", "No"], n),
        "is_driver_seat_height_adjustable": np.random.choice(["Yes", "No"], n),
        "is_day_night_rear_view_mirror":    np.random.choice(["Yes", "No"], n),
        "is_ecw":                           np.random.choice(["Yes", "No"], n),
        "is_speed_alert":                   np.random.choice(["Yes", "No"], n),
        "ncap_rating":                      np.random.randint(0, 6, n),
        "is_claim":                         is_claim,
    })


@pytest.fixture
def encoded_df(raw_df):
    return encode_and_scale(raw_df)


# ── data_loader ───────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_drops_nulls(self, tmp_path, raw_df):
        """Rows with nulls should be removed after loading."""
        raw_df.loc[0, "age_of_car"] = None
        csv = tmp_path / "data.csv"
        raw_df.to_csv(csv, index=False)

        df = load_data(str(csv))
        assert df.isnull().sum().sum() == 0

    def test_returns_dataframe(self, tmp_path, raw_df):
        csv = tmp_path / "data.csv"
        raw_df.to_csv(csv, index=False)
        df = load_data(str(csv))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ── preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_encode_drops_policy_id(self, encoded_df):
        assert "policy_id" not in encoded_df.columns

    def test_encode_no_object_columns(self, encoded_df):
        obj_cols = encoded_df.select_dtypes(include="object").columns.tolist()
        assert obj_cols == [], f"Object columns remain: {obj_cols}"

    def test_encode_target_intact(self, encoded_df):
        assert "is_claim" in encoded_df.columns
        assert set(encoded_df["is_claim"].unique()).issubset({0, 1})

    def test_scaled_cols_in_range(self, encoded_df):
        for col in config.COLS_TO_SCALE:
            if col in encoded_df.columns:
                assert encoded_df[col].min() >= 0.0, f"{col} below 0"
                assert encoded_df[col].max() <= 1.0, f"{col} above 1"

    def test_split_shapes_consistent(self, encoded_df):
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(encoded_df)
        assert X_train.shape[1] == X_valid.shape[1] == X_test.shape[1]
        assert len(X_train) == len(y_train)
        assert len(X_valid) == len(y_valid)
        assert len(X_test)  == len(y_test)

    def test_split_both_classes_in_train(self, encoded_df):
        X_train, _, _, y_train, _, _ = split_data(encoded_df)
        counts = pd.Series(y_train).value_counts()
        assert len(counts) == 2
        assert counts.min() > 0


# ── models ────────────────────────────────────────────────────────────────────

class TestModels:
    @pytest.fixture
    def split(self, encoded_df):
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(encoded_df)
        return X_train, X_valid, y_train, y_valid

    def test_logistic_regression_predict_shape(self, split):
        X_train, X_valid, y_train, _ = split
        model = logistic_regression.train(X_train, y_train)
        y_pred, y_score = logistic_regression.predict(model, X_valid)
        assert len(y_pred)  == len(X_valid)
        assert len(y_score) == len(X_valid)

    def test_logistic_regression_binary_output(self, split):
        X_train, X_valid, y_train, _ = split
        model = logistic_regression.train(X_train, y_train)
        y_pred, _ = logistic_regression.predict(model, X_valid)
        assert set(y_pred).issubset({0, 1})

    def test_decision_tree_predict_shape(self, split):
        X_train, X_valid, y_train, _ = split
        model = decision_tree.train(X_train, y_train)
        y_pred, y_score = decision_tree.predict(model, X_valid)
        assert len(y_pred)  == len(X_valid)
        assert len(y_score) == len(X_valid)

    def test_decision_tree_scores_are_probabilities(self, split):
        X_train, X_valid, y_train, _ = split
        model = decision_tree.train(X_train, y_train)
        _, y_score = decision_tree.predict(model, X_valid)
        assert y_score.min() >= 0.0
        assert y_score.max() <= 1.0