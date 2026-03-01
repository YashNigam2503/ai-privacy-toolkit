"""
Privacy Guard Module
====================
Two post-processing privacy layers that sit on top of the existing
model-guided k-anonymization provided by `Anonymize`.

Classes
-------
LDiversityEnforcer  – ensures l-diversity within each equivalence class.
DifferentialPrivacyLayer – adds calibrated Laplace noise to numerical QIs.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  L-Diversity Enforcer                                               #
# ------------------------------------------------------------------ #
class LDiversityEnforcer:
    """Check and enforce l-diversity on a k-anonymized pandas DataFrame.

    Groups are formed by the quasi-identifier columns.  A group is
    *l-diverse* when it contains at least `l` distinct values of the
    sensitive attribute.  Groups that fail the check are suppressed
    (removed) from the dataset.

    Parameters
    ----------
    quasi_identifiers : list of str
        Column names used as quasi-identifiers.
    sensitive_attribute : str
        Column name of the sensitive attribute.
    l : int, default 2
        Minimum number of distinct sensitive values per group.
    """

    def __init__(
        self,
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        l: int = 2,
    ):
        if l < 2:
            raise ValueError("l must be at least 2")
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attribute = sensitive_attribute
        self.l = l
        self._audit: Optional[pd.DataFrame] = None

    # ---- analysis ------------------------------------------------ #
    def fit_analyze(self, X: pd.DataFrame) -> None:
        """Compute per-group diversity statistics and store them for audit.

        Parameters
        ----------
        X : pd.DataFrame
            The k-anonymized dataset (must contain QI + sensitive columns).
        """
        grouped = X.groupby(self.quasi_identifiers, sort=False)
        audit = grouped[self.sensitive_attribute].agg(
            group_size="count",
            distinct_sensitive="nunique",
        ).reset_index()
        audit["is_l_diverse"] = audit["distinct_sensitive"] >= self.l
        self._audit = audit

        total = len(audit)
        passing = int(audit["is_l_diverse"].sum())
        failing = total - passing
        print(f"[L-Diversity] Total groups: {total}")
        print(f"[L-Diversity] Groups passing (>= {self.l} distinct): {passing}")
        print(f"[L-Diversity] Groups to suppress: {failing}")

    # ---- enforcement --------------------------------------------- #
    def enforce(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *X* with non-l-diverse groups removed.

        If `fit_analyze` has not been called yet it is called automatically.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Filtered dataset containing only l-diverse groups.
        """
        if self._audit is None:
            self.fit_analyze(X)

        # keep only group keys that pass
        passing = self._audit.loc[
            self._audit["is_l_diverse"], self.quasi_identifiers
        ]
        merged = X.merge(passing, on=self.quasi_identifiers, how="inner")

        removed = len(X) - len(merged)
        print(f"[L-Diversity] Rows before enforcement: {len(X)}")
        print(f"[L-Diversity] Rows after enforcement:  {len(merged)}")
        print(f"[L-Diversity] Rows suppressed:          {removed}")
        return merged.reset_index(drop=True)

    # ---- reporting ----------------------------------------------- #
    def get_audit_report(self) -> pd.DataFrame:
        """Return a DataFrame summarising group sizes and diversity counts.

        Columns: quasi-identifier columns, `group_size`,
        `distinct_sensitive`, `is_l_diverse`.
        """
        if self._audit is None:
            raise RuntimeError(
                "No audit data available. Call fit_analyze() first."
            )
        return self._audit.copy()


# ------------------------------------------------------------------ #
#  Differential Privacy Layer (Laplace mechanism)                     #
# ------------------------------------------------------------------ #
class DifferentialPrivacyLayer:
    """Add Laplace noise to numerical quasi-identifiers.

    Uses the Laplace mechanism where ``noise_scale = sensitivity / epsilon``.
    After adding noise the values are optionally clipped to the original
    column min/max so that the output stays within a plausible range.

    Parameters
    ----------
    epsilon : float
        Privacy budget *per column*.
    numerical_columns : list of str
        Column names to which noise will be added.
    sensitivity : float, default 1.0
        Global sensitivity (L1) used for every column unless overridden.
    random_seed : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        epsilon: float,
        numerical_columns: List[str],
        sensitivity: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.numerical_columns = numerical_columns
        self.sensitivity = sensitivity
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)

    # ---- internal ------------------------------------------------ #
    def _get_sensitivity(self, col: str) -> float:
        """Return the sensitivity for *col*.

        Currently returns the global sensitivity for every column.
        Override this method in a subclass for per-column sensitivity.
        """
        return self.sensitivity

    # ---- noise application --------------------------------------- #
    def apply_noise(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *X* with Laplace noise added to numerical columns.

        Values are clipped to the original [min, max] range of each column.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with noisy numerical QI columns.
        """
        X_out = X.copy()
        print(f"[DP] Epsilon per column: {self.epsilon}")

        for col in self.numerical_columns:
            if col not in X_out.columns:
                print(f"[DP] WARNING – column '{col}' not found, skipping.")
                continue

            sens = self._get_sensitivity(col)
            scale = sens / self.epsilon

            original_min = X_out[col].min()
            original_max = X_out[col].max()

            noise = self._rng.laplace(loc=0.0, scale=scale, size=len(X_out))
            X_out[col] = X_out[col].astype(float) + noise

            # clip to original range
            X_out[col] = X_out[col].clip(lower=original_min, upper=original_max)

            print(f"[DP] Column '{col}': noise_scale={scale:.4f}, "
                  f"range=[{original_min}, {original_max}]")

        return X_out

    # ---- reporting ----------------------------------------------- #
    def privacy_budget_report(self) -> dict:
        """Return a summary of the privacy budget.

        Returns
        -------
        dict
            Keys: ``epsilon_per_column``, ``total_epsilon``, ``noise_scales``.
        """
        noise_scales = {}
        for col in self.numerical_columns:
            sens = self._get_sensitivity(col)
            noise_scales[col] = sens / self.epsilon

        return {
            "epsilon_per_column": self.epsilon,
            "total_epsilon": self.epsilon * len(self.numerical_columns),
            "noise_scales": noise_scales,
        }
