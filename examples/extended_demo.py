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


def build_synthetic_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create a small tabular dataset with numeric and categorical columns.

    The sensitive attribute ('disease') is deliberately spread across all
    combinations of features so that l-diversity can find diverse groups
    after k-anonymization.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age":       rng.randint(18, 70, size=n).astype(float),
        "income":    rng.randint(20000, 120000, size=n).astype(float),
        "zipcode":   rng.choice([0.0, 1.0, 2.0, 3.0], size=n),
        "education": rng.choice([0.0, 1.0, 2.0, 3.0], size=n),
        "disease":   rng.choice(["flu", "cold", "none", "covid"], size=n),
    })
    return df


def main():
    # ------------------------------------------------------------------ #
    # 1. Prepare data                                                     #
    # ------------------------------------------------------------------ #
    print("Building synthetic dataset …")
    df = build_synthetic_data(n=1000, seed=42)

    feature_cols = ["age", "income", "zipcode", "education"]
    sensitive_col = "disease"

    # Train a simple classifier on the features (the "original model")
    X_train = df[feature_cols].values
    # Use a numeric target for the tree (label-encode disease)
    disease_categories = pd.Categorical(df[sensitive_col])
    y_train = disease_categories.codes.astype(int)
    disease_mapping = dict(enumerate(disease_categories.categories))

    clf = DecisionTreeClassifier(random_state=0, max_depth=6)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_train)

    # Build the ArrayDataset expected by Anonymize.
    # The sensitive column is stored as its numeric code so the array
    # stays fully numeric (Anonymize rejects mixed-type arrays).
    # We'll map it back to strings after k-anonymization.
    X_full = df[feature_cols].copy()
    X_full[sensitive_col] = disease_categories.codes.astype(float)
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
        k=10,
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
