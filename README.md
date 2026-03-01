# AI Privacy Toolkit — Extended Anonymization Pipeline

## Overview

This repository is a fork of the [IBM AI Privacy Toolkit](https://github.com/IBM/ai-privacy-toolkit). The original toolkit provides model-guided k-anonymization for ML training data. The `Anonymize` class trains a decision tree on the original model's predictions, using `min_samples_leaf = k` to ensure each leaf contains at least k records. Records within a leaf have their quasi-identifier values replaced with representative values, making them indistinguishable from one another.

This extension adds two post-processing privacy layers on top of the existing k-anonymization output, without modifying the original codebase.

## Added Features

### l-Diversity Enforcement

**File:** `apt/anonymization/privacy_guard.py`  
**Class:** `LDiversityEnforcer`

k-Anonymity alone does not prevent homogeneity attacks — if all records in a group share the same sensitive attribute value, an attacker can infer it regardless of anonymization. l-Diversity addresses this by requiring each equivalence class (group of records sharing quasi-identifier values) to contain at least `l` distinct sensitive attribute values.

The enforcer groups the anonymized dataset by quasi-identifiers using pandas `groupby`, counts unique sensitive values per group via `nunique`, and suppresses (removes) any group that does not meet the threshold. An audit report is available summarising group sizes and diversity counts.

### Differential Privacy (Laplace Mechanism)

**File:** `apt/anonymization/privacy_guard.py`  
**Class:** `DifferentialPrivacyLayer`

After l-diversity enforcement, calibrated Laplace noise is added to numerical quasi-identifiers. The noise scale for each column is calculated as `sensitivity / epsilon`, where epsilon is the privacy budget per column. Values are clipped to the original column range after noise addition. The total privacy budget across all columns is `epsilon × number of numerical columns`. A budget report is available listing the epsilon and noise scale per column.

## Pipeline Execution Flow

**File:** `apt/anonymization/extended_anonymizer.py`  
**Class:** `ExtendedAnonymizer`

The `fit_transform(dataset)` method runs three stages in sequence:

1. **k-Anonymization** — `Anonymize.anonymize(dataset)` produces the k-anonymous dataset.
2. **l-Diversity** — `LDiversityEnforcer.fit_analyze()` computes group statistics, then `LDiversityEnforcer.enforce()` removes non-compliant groups.
3. **Differential Privacy** — `DifferentialPrivacyLayer.apply_noise()` adds Laplace noise to numerical quasi-identifiers.

Each stage prints monitoring output: group counts, suppressed rows, epsilon values, and noise scales.

## Running Instructions

```bash
# Create and activate a conda environment with Python 3.8
conda create -n ai_privacy python=3.8 -y
conda activate ai_privacy

# Install dependencies
pip install -r requirements.txt

# Set PYTHONPATH so the apt package is importable
set PYTHONPATH=.

# Run the demo
python examples/extended_demo.py
```

The demo generates a synthetic dataset, trains a classifier, and runs the full three-stage pipeline. It prints the l-diversity audit report and the differential privacy budget report at the end.

## Summary

This extension layers l-diversity enforcement and Laplace-mechanism differential privacy on top of the existing k-anonymization provided by the IBM AI Privacy Toolkit. All additions are contained in three new files and do not modify the original toolkit code. The implementation uses only numpy, pandas, and scikit-learn.
