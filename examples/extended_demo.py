"""
Extended Anonymizer — Demo
==========================
Builds a small synthetic dataset, trains a classifier, then runs
the full three-stage privacy pipeline:

    k-Anonymity  →  l-Diversity  →  Differential Privacy (Laplace)

No external datasets are downloaded; everything is generated in-memory.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from apt.utils.datasets import ArrayDataset
from apt.anonymization.extended_anonymizer import ExtendedAnonymizer


def build_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a small tabular dataset with numeric and categorical columns."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age":       rng.randint(18, 70, size=n).astype(float),
        "income":    rng.randint(20000, 120000, size=n).astype(float),
        "zipcode":   rng.choice(["10001", "10002", "10003", "10004"], size=n),
        "education": rng.choice(["HS", "BSc", "MSc", "PhD"], size=n),
        "disease":   rng.choice(["flu", "cold", "none", "covid"], size=n),
    })
    return df


def main():
    # ------------------------------------------------------------------ #
    # 1. Prepare data                                                     #
    # ------------------------------------------------------------------ #
    print("Building synthetic dataset …")
    df = build_synthetic_data(n=500, seed=42)

    feature_cols = ["age", "income", "zipcode", "education"]
    sensitive_col = "disease"
    X = df[feature_cols].copy()
    y = df[sensitive_col].copy()

    # Encode categoricals for the tree (label-encode for simplicity)
    X_encoded = X.copy()
    for col in ["zipcode", "education"]:
        X_encoded[col] = X_encoded[col].astype("category").cat.codes.astype(float)

    # Train a simple classifier (the "original model")
    clf = DecisionTreeClassifier(random_state=0)
    y_encoded = y.astype("category").cat.codes.values
    clf.fit(X_encoded.values, y_encoded)
    predictions = clf.predict(X_encoded.values)

    # Build the ArrayDataset expected by Anonymize
    # We attach the sensitive column back so l-diversity can use it
    X_full = X_encoded.copy()
    X_full[sensitive_col] = y_encoded.astype(float)
    features = list(X_full.columns)

    dataset = ArrayDataset(
        X_full.values,
        predictions,
        features_names=features,
    )

    # ------------------------------------------------------------------ #
    # 2. Run extended pipeline                                            #
    # ------------------------------------------------------------------ #
    quasi_ids = ["age", "income", "zipcode", "education"]
    numerical_qis = ["age", "income"]

    ext = ExtendedAnonymizer(
        k=5,
        quasi_identifiers=quasi_ids,
        sensitive_attribute=sensitive_col,
        l=2,
        epsilon=0.5,
        numerical_columns=numerical_qis,
        sensitivity=1.0,
        random_seed=99,
    )

    result = ext.fit_transform(dataset)

    # ------------------------------------------------------------------ #
    # 3. Reports                                                          #
    # ------------------------------------------------------------------ #
    print("\n---- l-Diversity Audit Report (first 10 rows) ----")
    audit = ext.get_audit_report()
    print(audit.head(10).to_string(index=False))

    print("\n---- Privacy Budget Report ----")
    budget = ext.get_privacy_budget_report()
    if budget:
        for key, val in budget.items():
            print(f"  {key}: {val}")

    print("\n---- Result Preview (first 5 rows) ----")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
