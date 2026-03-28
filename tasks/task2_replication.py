"""
Task 2: Experiment Replication — MEDIUM
----------------------------------------
Agent receives a methods section describing a logistic regression experiment
on a tabular CSV dataset. The agent must write and run code to replicate
the reported AUC-ROC and F1 score.

Key challenge: the dataset has class imbalance. A naive model trained without
stratified splitting will score ~0.71 AUC — outside the pass threshold.
The agent must notice the imbalance and handle it correctly.

The dataset is generated procedurally each episode with random seed,
varying feature names and imbalance ratio to prevent memorisation.
"""
from __future__ import annotations

import os
import tempfile
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tasks.base import BaseTask


class ReplicationTask(BaseTask):
    task_id    = "task2_replication"
    task_name  = "Experiment Replication"
    difficulty = "medium"
    max_steps  = 20

    def generate_episode(self) -> dict:
        rng    = self.rng
        np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

        # --- Procedurally vary the domain ---
        domains = [
            {
                "name":     "patient readmission",
                "features": ["age", "num_prior_admissions", "los_days",
                              "num_medications", "comorbidity_score"],
                "target":   "readmitted_30d",
            },
            {
                "name":     "loan default",
                "features": ["credit_score", "debt_to_income", "loan_amount",
                              "employment_years", "num_open_accounts"],
                "target":   "defaulted",
            },
            {
                "name":     "equipment failure",
                "features": ["operating_hours", "temperature_avg", "vibration_rms",
                              "pressure_delta", "maintenance_lag_days"],
                "target":   "failure_within_30d",
            },
        ]
        domain = rng.choice(domains)

        n_samples      = 400
        imbalance_rate = rng.choice([0.18, 0.20, 0.22])   # ~20% positive class
        n_pos          = int(n_samples * imbalance_rate)
        n_neg          = n_samples - n_pos

        # Generate correlated features
        n_feat = len(domain["features"])
        X_neg  = np_rng.randn(n_neg, n_feat)
        X_pos  = np_rng.randn(n_pos, n_feat) + np_rng.uniform(0.4, 0.7, n_feat)

        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * n_neg + [1] * n_pos)

        # Shuffle
        idx = np_rng.permutation(n_samples)
        X, y = X[idx], y[idx]

        df = pd.DataFrame(X, columns=domain["features"])
        df[domain["target"]] = y

        # Save dataset to temp file
        tmp_dir      = tempfile.mkdtemp(prefix="rig_task2_")
        dataset_path = os.path.join(tmp_dir, "dataset.csv")
        df.to_csv(dataset_path, index=False)

        # --- Compute ground truth (stratified split, seed=42) ---
        X_arr = df[domain["features"]].values
        y_arr = df[domain["target"]].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=0.25, random_state=42, stratify=y_arr
        )
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        gt_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
        gt_f1  = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

        # --- Build methods section ---
        feature_list = ", ".join(f"`{f}`" for f in domain["features"])
        paper_text = textwrap.dedent(f"""
            STUDY: Predictive modelling for {domain['name']}

            METHODS SECTION (to replicate)
            --------------------------------
            Dataset: {n_samples} samples with {n_feat} features ({feature_list})
            and binary target `{domain['target']}`.

            Preprocessing: Standardise all features using StandardScaler
            (fit on training set, transform both sets).

            Model: Logistic Regression with class_weight='balanced', random_state=42,
            max_iter=1000.

            Split: 75% train / 25% test using train_test_split with
            random_state=42 and stratify=y.

            Metrics to report:
              - AUC-ROC (roc_auc_score on test set, using predicted probabilities)
              - F1-score (f1_score on test set, using predicted class labels)

            The dataset is available at path: DATASET_PATH
            (use read_dataset action to inspect it, or reference DATASET_PATH in code)

            REPORTED RESULTS (from original paper):
              AUC-ROC : {gt_auc}
              F1-score: {gt_f1}

            Your task: reproduce these results. Submit with submit_results action.
            Tolerance: AUC within ±0.03, F1 within ±0.03 for full credit.
        """).strip()

        ground_truth = {
            "auc":          gt_auc,
            "f1":           gt_f1,
            "target_col":   domain["target"],
            "feature_cols": domain["features"],
            "domain":       domain["name"],
            "n_samples":    n_samples,
            "imbalance_rate": imbalance_rate,
        }

        return {
            "paper_text":     paper_text,
            "paper_sections": {"methods": paper_text},
            "dataset_path":   dataset_path,
            "ground_truth":   ground_truth,
        }

    def _action_schema(self) -> dict:
        return {
            "read_dataset":    {"rows": "int (optional, default 20)"},
            "execute_code":    {"code": "str — Python code; DATASET_PATH constant available"},
            "submit_results":  {
                "results_payload": {
                    "auc":            "float",
                    "f1":             "float",
                    "interpretation": "str — brief explanation of results",
                }
            },
        }
