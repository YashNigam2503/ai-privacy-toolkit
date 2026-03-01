"""
Extended Anonymizer
===================
A lightweight pipeline that chains the original model-guided
k-anonymization with l-diversity enforcement and differential-privacy
noise injection.

The original ``Anonymize`` class is used **as-is** — its output is fed
into the two new privacy layers defined in ``privacy_guard.py``.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from apt.anonymization.anonymizer import Anonymize
from apt.anonymization.privacy_guard import LDiversityEnforcer, DifferentialPrivacyLayer
from apt.utils.datasets import ArrayDataset


class ExtendedAnonymizer:
    """Three-stage privacy pipeline: k-Anonymity → l-Diversity → DP noise.

    Parameters
    ----------
    k : int
        k-anonymity parameter (forwarded to ``Anonymize``).
    quasi_identifiers : list
        Feature names (str) or indices (int) for both anonymization and
        downstream privacy layers.
    sensitive_attribute : str
        Column name of the sensitive attribute used for l-diversity.
    l : int, default 2
        Minimum distinct sensitive values per equivalence class.
    epsilon : float, default 1.0
        Differential-privacy budget **per numerical column**.
    numerical_columns : list of str or None
        Numerical QI columns to receive Laplace noise.  If ``None`` no
        noise is added.
    sensitivity : float, default 1.0
        Global L1-sensitivity for the Laplace mechanism.
    random_seed : int or None
        Seed for reproducible DP noise.
    categorical_features : list or None
        Forwarded to ``Anonymize``.
    quasi_identifier_slices : list or None
        Forwarded to ``Anonymize``.
    is_regression : bool, default False
        Forwarded to ``Anonymize``.
    train_only_QI : bool, default False
        Forwarded to ``Anonymize``.
    """

    def __init__(
        self,
        k: int,
        quasi_identifiers: list,
        sensitive_attribute: str,
        l: int = 2,
        epsilon: float = 1.0,
        numerical_columns: Optional[List[str]] = None,
        sensitivity: float = 1.0,
        random_seed: Optional[int] = None,
        categorical_features: Optional[list] = None,
        quasi_identifier_slices: Optional[list] = None,
        is_regression: bool = False,
        train_only_QI: bool = False,
    ):
        # ---- original anonymizer --------------------------------- #
        self._anonymizer = Anonymize(
            k=k,
            quasi_identifiers=list(quasi_identifiers),
            quasi_identifer_slices=quasi_identifier_slices,
            categorical_features=categorical_features,
            is_regression=is_regression,
            train_only_QI=train_only_QI,
        )

        # ---- l-diversity ----------------------------------------- #
        self._l_enforcer = LDiversityEnforcer(
            quasi_identifiers=[
                qi for qi in quasi_identifiers if isinstance(qi, str)
            ],
            sensitive_attribute=sensitive_attribute,
            l=l,
        )

        # ---- differential privacy -------------------------------- #
        self._dp_layer: Optional[DifferentialPrivacyLayer] = None
        if numerical_columns:
            self._dp_layer = DifferentialPrivacyLayer(
                epsilon=epsilon,
                numerical_columns=numerical_columns,
                sensitivity=sensitivity,
                random_seed=random_seed,
            )

        self._qi_names = quasi_identifiers
        self._sensitive = sensitive_attribute

    # -------------------------------------------------------------- #
    def fit_transform(self, dataset: ArrayDataset) -> pd.DataFrame:
        """Run the full three-stage privacy pipeline.

        Parameters
        ----------
        dataset : ArrayDataset
            Wrapper around training data + original model predictions.

        Returns
        -------
        pd.DataFrame
            The privacy-enhanced dataset.
        """
        # ---- Stage 1: k-anonymization (original script) ---------- #
        print("=" * 60)
        print("Stage 1 / 3 — k-Anonymization (original Anonymize class)")
        print("=" * 60)
        X_k = self._anonymizer.anonymize(dataset)

        # Ensures pandas DataFrame for downstream layers
        if isinstance(X_k, np.ndarray):
            if dataset.features_names is not None:
                columns = dataset.features_names
            else:
                columns = [str(i) for i in range(X_k.shape[1])]
            X_k = pd.DataFrame(X_k, columns=columns)

        print(f"[k-Anon] Output shape: {X_k.shape}")

        # ---- Stage 2: l-diversity -------------------------------- #
        print()
        print("=" * 60)
        print("Stage 2 / 3 — l-Diversity enforcement")
        print("=" * 60)

        self._l_enforcer.fit_analyze(X_k)
        X_l = self._l_enforcer.enforce(X_k)

        # ---- Stage 3: differential privacy ----------------------- #
        print()
        print("=" * 60)
        print("Stage 3 / 3 — Differential Privacy (Laplace noise)")
        print("=" * 60)

        if self._dp_layer is not None:
            X_dp = self._dp_layer.apply_noise(X_l)
            budget = self._dp_layer.privacy_budget_report()
            print(f"[DP] Total privacy budget (ε): {budget['total_epsilon']}")
        else:
            X_dp = X_l
            print("[DP] No numerical columns specified — skipping noise.")

        print()
        print("=" * 60)
        print("Pipeline complete.")
        print(f"Final dataset shape: {X_dp.shape}")
        print("=" * 60)
        return X_dp

    # ------ convenience accessors --------------------------------- #
    def get_audit_report(self) -> pd.DataFrame:
        """Return the l-diversity audit report."""
        return self._l_enforcer.get_audit_report()

    def get_privacy_budget_report(self) -> Optional[dict]:
        """Return the DP budget report, or *None* if DP was skipped."""
        if self._dp_layer is not None:
            return self._dp_layer.privacy_budget_report()
        return None
