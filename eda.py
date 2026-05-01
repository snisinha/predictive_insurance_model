"""eda.py - exploratory data analysis: statistics and visualisations"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def _save(fig_name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, fig_name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def chi_squared_ncap(df: pd.DataFrame) -> None:
    """Chi-squared test: is_claim vs ncap_rating."""
    table = pd.crosstab(df["is_claim"], df["ncap_rating"])
    chi2, p, dof, _ = chi2_contingency(table)
    print(f"Chi-squared statistic: {chi2:.4f}")
    print(f"p-value:               {p:.4f}")
    print(f"Degrees of freedom:    {dof}")


def plot_target_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart of is_claim counts (class imbalance)."""
    plt.figure(figsize=(6, 4))
    order = sorted(df["is_claim"].dropna().unique())
    ax = sns.countplot(data=df, x="is_claim", order=order, color="steelblue", alpha=0.88)
    plt.xlabel("is_claim (0 = no claim, 1 = claim)")
    plt.ylabel("Count")
    plt.title("Distribution of target class (is_claim)")
    total = len(df)
    for p in ax.patches:
        h = p.get_height()
        if h <= 0:
            continue
        ax.annotate(
            f"{int(h)}\n({100 * h / total:.2f}%)",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    _save("target_distribution_is_claim.png", output_dir)


def plot_categorical_vs_claim(df: pd.DataFrame, output_dir: str) -> None:
    """Bar plots for each categorical feature vs is_claim."""
    categorical_cols = [
        "ncap_rating", "is_driver_seat_height_adjustable", "is_central_locking",
        "make", "is_parking_sensors", "segment", "population_density",
        "airbags", "width", "turning_radius", "length", "is_speed_alert",
        "engine_type", "is_front_fog_lights", "rear_brakes_type",
    ]
    for col in categorical_cols:
        sns.countplot(data=df, x=col, hue="is_claim")
        plt.title(f"{col} impact on claims")
        plt.xlabel(col)
        plt.xticks(rotation=90)
        plt.tight_layout()
        _save(f"countplot_{col}.png", output_dir)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Pearson correlation heatmap (numeric columns only)."""
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    _save("correlation_heatmap.png", output_dir)


def plot_pairplots(df: pd.DataFrame, output_dir: str) -> None:
    """Pair plots for selected feature groups."""
    cols1 = ["policy_tenure", "age_of_car", "age_of_policyholder",
             "area_cluster", "population_density", "is_claim"]
    sns.pairplot(df[cols1], hue="is_claim", diag_kind="auto")
    plt.suptitle("Pairplot — policy & demographic features", y=1.02)
    _save("pairplot_policy.png", output_dir)

    cols2 = ["make", "segment", "model", "is_claim", "age_of_car",
             "turning_radius", "length", "width", "height", "gross_weight"]
    sns.pairplot(df[cols2], hue="is_claim")
    plt.suptitle("Pairplot — vehicle features", y=1.02)
    _save("pairplot_vehicle.png", output_dir)


def pivot_make_ncap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: claim % by make × ncap_rating."""
    def pct(s):
        return (s.sum() / len(s)) * 100

    return pd.pivot_table(
        df, values="is_claim", index="make",
        columns="ncap_rating", aggfunc=pct,
    )


def run_eda(df: pd.DataFrame, output_dir: str) -> None:
    """Run the full EDA pipeline."""
    print("\n── Descriptive statistics ──────────────────────────────────────")
    print(df.describe())

    print("\n── Chi-squared: is_claim vs ncap_rating ────────────────────────")
    chi_squared_ncap(df)

    print("\n── Claim % by make ─────────────────────────────────────────────")
    def claim_pct(s):
        return s.mean() * 100
    print(df.groupby("make")["is_claim"].apply(claim_pct))

    print("\n── Pivot: make × ncap_rating ───────────────────────────────────")
    print(pivot_make_ncap(df))

    print("\n── Target class distribution (is_claim) ────────────────────────")
    vc = df["is_claim"].value_counts().sort_index()
    print(vc.to_string())
    print("\nPercent:")
    print((100 * vc / len(df)).round(3).to_string())

    print("\n── Generating plots … ──────────────────────────────────────────")
    plot_target_distribution(df, output_dir)
    plot_categorical_vs_claim(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_pairplots(df, output_dir)
    print("EDA complete.")