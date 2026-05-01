"""main.py - entry point; runs the full pipeline
Usage:
python main.py               runfull pipeline (EDA + all models)
python main.py --eda-only    stop after EDA
python main.py --skip-eda    skip EDA, run models only
python main.py --predict     save NN test predictions to CSV"""

import argparse
import os
import time

import config
import data_loader
import eda
import preprocessing
import evaluation
from models import logistic_regression, decision_tree, random_forest, neural_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insurance claim prediction pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--eda-only",  action="store_true", help="Run EDA then exit")
    group.add_argument("--skip-eda",  action="store_true", help="Skip EDA, run models only")
    parser.add_argument("--predict",  action="store_true", help="Save NN test predictions to CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n 1. Loading data")
    df = data_loader.load_data(config.DATA_PATH)

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    if not args.skip_eda:
        print("\n 2. Exploratory Data Analysis")
        eda.run_eda(df, output_dir=os.path.join(config.OUTPUT_DIR, "eda"))

    if args.eda_only:
        print("\nEDA-only mode: exiting.")
        return

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    print("\n 3. Preprocessing")
    df_enc = preprocessing.encode_and_scale(df)
    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocessing.split_data(df_enc)

    # ── 4. Train & evaluate sklearn models ───────────────────────────────────
    results = {}
    model_dir = os.path.join(config.OUTPUT_DIR, "models")

    sklearn_models = [
        ("Logistic Regression", logistic_regression),
        ("Decision Tree",       decision_tree),
        ("Random Forest",       random_forest),
    ]

    for name, module in sklearn_models:
        print(f"\n Training: {name}")
        t0 = time.time()
        model = module.train(X_train, y_train)
        print(f"  Fit time: {time.time() - t0:.1f}s")

        y_pred, y_score = module.predict(model, X_valid)
        train_acc = (model.predict(X_train) == y_train).mean() * 100

        metrics = evaluation.evaluate_model(
            name, y_valid, y_pred, y_score,
            output_dir=model_dir,
        )
        metrics["train_acc"] = train_acc
        results[name] = metrics

    # ── 5. Train & evaluate neural network ───────────────────────────────────
    print("\n Training: Neural Network")
    X_nn_train, X_nn_test, y_nn_train, y_nn_test = preprocessing.split_data_nn(df_enc)

    t0 = time.time()
    nn_model = neural_network.train(X_nn_train, y_nn_train)
    print(f"  Fit time: {time.time() - t0:.1f}s")

    y_pred_nn, y_score_nn = neural_network.predict(nn_model, X_nn_test)
    _, train_acc_nn = nn_model.evaluate(X_nn_train, y_nn_train, verbose=0)

    metrics_nn = evaluation.evaluate_model(
        "Neural Network", y_nn_test, y_pred_nn, y_score_nn,
        output_dir=model_dir,
    )
    metrics_nn["train_acc"] = train_acc_nn * 100
    results["Neural Network"] = metrics_nn

    # ── 6. Compare all models ─────────────────────────────────────────────────
    print("\n 6. Model Comparison")
    evaluation.compare_models(results, output_dir=model_dir)

    # ── 7. Save NN predictions to CSV (optional) ──────────────────────────────
    if args.predict:
        print("\n 7. Saving final predictions")
        neural_network.predict_to_csv(
            model=nn_model,
            X_test=X_nn_test,
            y_test=y_nn_test,
            output_dir=model_dir,
        )

    print("\n✓ Pipeline complete. Outputs written to:", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()