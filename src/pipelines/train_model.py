import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.pipelines.pipeline import Pipeline
from src.utils.utils import (
    read_config,
    read_data,
    write_data,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrainModelPipeline(Pipeline):
    """End-to-end training pipeline."""

    def __init__(self, run_name: Optional[str] = None) -> None:
        """Initialise configuration and placeholders.

        Args:
            run_name: Optional MLflow run name. If not provided, a default is used.

        Raises:
            ValueError: If the configuration file cannot be loaded.
        """
        logger.info("Initializing the TrainModelPipeline.")
        self.config = read_config()
        if not self.config:
            raise ValueError("Configuration file is empty or not found.")

        self.model_name = str(self.config.get("model_name", "xgboost")).lower()
        raw_params = self.config.get("model_params", {}) or {}
        if (
            isinstance(raw_params, dict)
            and self.model_name in raw_params
            and isinstance(raw_params[self.model_name], dict)
        ):
            self.model_params = raw_params[self.model_name]
        else:
            self.model_params = raw_params
        self.run_name = run_name or "Default_Run_Name"
        self.model: Optional[BaseEstimator] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_cols_: Optional[pd.Index] = None

    def run(self) -> None:
        """Execute training and log metrics, model, and plots to MLflow."""
        logger.info("Starting the training pipeline.")

        tracking_uri = os.getenv("LOCAL_MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

        # Load data
        df = read_data(
            table_name="OA_DATASET_PREPARED",
            schema_obj=None,  # Ignore schema validation for now
            database="TEST_DS_DATABASE",
            schema="PUBLIC",
        )
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split config (defaults if not present)
        test_size = float(self.config.get("test_size", 0.2))
        random_state = int(self.config.get("random_state", 42))
        stratify = y if bool(self.config.get("stratify", True)) else None

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Add scaling and normalization
        X_train, X_test = self.scaler(X_train, X_test)

        # Class imbalance handling via data resampling
        # Alternatively use class_weight in model params
        # X_train, y_train = self.resample(X_train, y_train)

        # Train & evaluate
        self.train(X_train, y_train)
        train_accuracy = self.evaluate(X_train, y_train)
        test_accuracy = self.evaluate(X_test, y_test)

        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_param("model_name", self.model_name)
            if self.model_params:
                mlflow.log_params(self.model_params)
            mlflow.log_metric("train_accuracy", float(train_accuracy))
            mlflow.log_metric("test_accuracy", float(test_accuracy))

            # Classification report on test set
            self._log_classification_report(X_test, y_test)

            # ROC on the test set
            self._log_roc_curve(X_test, y_test)
            
            # SHAP analysis
            self._log_shap_analysis(X_test)

            # Log model
            if self.model is not None:
                mlflow.sklearn.log_model(self.model, artifact_path=self.model_name)

        # Persist predictions for the full dataset to match schema
        # Update so that this will instead read from original dataset + predictions
        df_out = df.copy()
        if self.model is not None:
            X_full = self._prepare_features(X)
            pred_scores = self._predict_scores(X_full)
            df_out["PREDICTION"] = pd.Series(pred_scores, index=X_full.index)
            df_out["PREDICTION_CATEGORY"] = self._bin_predictions(df_out["PREDICTION"])

            write_data(
                df_out,
                table_name="OA_DATASET_OUTPUT",
                schema_obj=None,  # ignore schema validation for now
                database="TEST_DS_DATABASE",
                schema="PUBLIC",
            )

    def scaler(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler.

        Args:
            X_train: Training feature matrix.
            X_test: Testing feature matrix.

        Returns:
            Tuple of scaled (X_train, X_test).
        """
        logger.info("Scaling features using StandardScaler.")
        scaler = StandardScaler()
        num_cols = X_train.select_dtypes(include="number").columns
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train[num_cols]),
            columns=num_cols,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test[num_cols]),
            columns=num_cols,
            index=X_test.index,
        )
        self.scaler_ = scaler
        self.feature_cols_ = num_cols
        return X_train_scaled, X_test_scaled

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align prediction features with the fitted scaler/feature set."""
        if self.scaler_ is None or self.feature_cols_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        X_num = X[self.feature_cols_]
        X_scaled = pd.DataFrame(
            self.scaler_.transform(X_num),
            columns=self.feature_cols_,
            index=X.index,
        )
        return X_scaled

    def _predict_scores(self, features: pd.DataFrame) -> pd.Series:
        """Generate prediction scores for downstream binning."""
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)
            return pd.Series(proba[:, 1], index=features.index)
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(features)
            if hasattr(scores, "ndim") and scores.ndim > 1:
                return pd.Series(scores[:, 1], index=features.index)
            return pd.Series(scores, index=features.index)
        preds = self.model.predict(features)
        return pd.Series(preds, index=features.index)

    def _bin_predictions(self, scores: pd.Series) -> pd.Series:
        """Bin prediction scores into Low/Medium/High with safe fallbacks."""
        scores = pd.Series(scores, index=scores.index)
        unique_count = scores.nunique(dropna=True)
        if unique_count <= 1:
            return pd.Series(["Medium"] * len(scores), index=scores.index)
        try:
            cats = pd.qcut(scores, q=3, duplicates="drop")
            n_bins = len(cats.cat.categories)
            if n_bins == 1:
                labels = ["Medium"]
            elif n_bins == 2:
                labels = ["Low", "High"]
            else:
                labels = ["Low", "Medium", "High"]
            return pd.qcut(scores, q=3, labels=labels, duplicates="drop")
        except ValueError:
            labels = ["Low", "High"]
            return pd.cut(scores, bins=2, labels=labels, include_lowest=True)

    def resample(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE.

        Args:
            X_train: Training feature matrix.
            y_train: Training target series.

        Returns:
            Tuple of resampled (X_train, y_train).
        """
        logger.info("Applying SMOTE to handle class imbalance.")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Fit the configured classifier.

        Args:
            features: Feature matrix (no target column).
            target: Target series aligned with `features`.
        """
        logger.info("Training the model.")
        self.model = self._build_model()
        self.model.fit(features, target)

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> float:
        """Compute accuracy on the provided data.

        Args:
            features: Feature matrix for evaluation.
            target: True labels for evaluation.

        Returns:
            Accuracy as a float in [0, 1].

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        logger.info("Evaluating the model.")
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        y_pred = self.model.predict(features)
        return float(accuracy_score(target, y_pred))

    def _build_model(self) -> BaseEstimator:
        """Construct the model based on configuration.

        Returns:
            An untrained sklearn-compatible estimator.

        Raises:
            ValueError: If an unsupported model_name is provided.

        """
        model_registry = {
            "logistic_regression": LogisticRegression,
            "lr": LogisticRegression,
            "xgboost": xgb.XGBClassifier,
            "xgb": xgb.XGBClassifier,
            "random_forest": RandomForestClassifier,
            "rf": RandomForestClassifier,
        }
        model_cls = model_registry.get(self.model_name)
        if model_cls is None:
            raise ValueError(f"Unsupported model_name '{self.model_name}'.")
        return model_cls(**self.model_params)

    def _log_classification_report(
        self, features: pd.DataFrame, target: pd.Series
    ) -> None:
        logger.info("Logging the classification report.")
        if self.model is None:
            logger.warning(
                "Model not available; skipping classification report logging."
            )
            return

        y_pred = self.model.predict(features)
        report = classification_report(target, y_pred)

        # extract metrics from classification report and log to MLflow
        report_dict = classification_report(target, y_pred, output_dict=True)
        for key, metrics in report_dict.items():
            if key not in ["accuracy", "macro avg", "weighted avg"]:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"Class {key}_{metric_name}", float(value))

        # Save report to a text file
        reports_dir = Path(os.getenv("LOCAL_REPORTS_PATH", "outputs/reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # Log to MLflow under a reports/ folder
        mlflow.log_artifact(str(report_path), artifact_path=report_path.parent.name)

        logger.info(
            "Classification report saved locally to %s and logged to MLflow.",
            report_path,
        )

    def _log_roc_curve(self, features: pd.DataFrame, target: pd.Series) -> None:
        logger.info("Logging the ROC curve.")
        if self.model is None:
            logger.warning("Model not available; skipping ROC logging.")
            return

        if hasattr(self.model, "predict_proba"):
            y_pred_prob = pd.Series(self.model.predict_proba(features)[:, 1])
        elif hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(features)
            if hasattr(raw_scores, "ndim") and raw_scores.ndim > 1:
                y_pred_prob = pd.Series(raw_scores[:, 1])
            else:
                y_pred_prob = pd.Series(raw_scores)
        else:
            logger.warning("Model does not expose predict_proba or decision_function.")
            return

        y_binary = (target == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()

        # Resolve local plots directory from env (with fallback)
        plots_dir = Path(os.getenv("LOCAL_PLOTS_PATH", "outputs/plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        roc_path = plots_dir / "roc_curve.png"

        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()

        # Log to MLflow under a plots/ folder
        mlflow.log_artifact(str(roc_path), artifact_path=roc_path.parent.name)

        logger.info("ROC curve saved locally to %s and logged to MLflow.", roc_path)
        
    def _log_shap_analysis(self, X_test: pd.DataFrame) -> None:
        """Calculate SHAP values and log their shape."""
        logger.info("Starting SHAP analysis.")
        
        if self.model is None:
            logger.warning("Model not available; skipping SHAP analysis.")
            return

        # TreeExplainer: exact SHAP for tree models (XGBoost, RF, LightGBM)
        explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_test)

        # Log base rate for reference
        base_prob = 1 / (1 + np.exp(-explainer.expected_value))
        logger.info(f"SHAP base probability: {base_prob:.1%}")

        # Mean |SHAP| = average impact magnitude across all samples
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create sorted DataFrame
        feature_importance = pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)
        
        # Log top 3 features
        top3 = feature_importance.head(3)["feature"].tolist()
        logger.info(f"Top SHAP features: {', '.join(top3)}")
        
        plots_dir = Path(os.getenv("LOCAL_PLOTS_PATH", "outputs/plots"))
        shap_dir = plots_dir / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature importance CSV
        csv_path = shap_dir / "shap_feature_importance.csv"
        feature_importance.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path), artifact_path="shap")

        # Bar plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        bar_path = shap_dir / "shap_bar.png"
        plt.savefig(bar_path, bbox_inches="tight", dpi=150)
        plt.close()
        mlflow.log_artifact(str(bar_path), artifact_path="shap")

        # Beeswarm plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        summary_path = shap_dir / "shap_beeswarm.png"
        plt.savefig(summary_path, bbox_inches="tight", dpi=150)
        plt.close()
        mlflow.log_artifact(str(summary_path), artifact_path="shap")

        # Cohort analysis: mean SHAP per prediction bucket
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_test)[:, 1]

            # Split into terciles
            bucket_labels = pd.qcut(proba, q=3, labels=["low", "mid", "high"])

            # Calculate mean SHAP (directional) per bucket
            shap_df = pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)
            shap_df["bucket"] = np.array(bucket_labels, dtype=str)

            cohort_shap = shap_df.groupby("bucket", observed=True).mean().T
            cohort_shap.index.name = "feature"
            cohort_shap = cohort_shap.reset_index()

            # Reorder columns (only include those that exist)
            cols = ["feature"] + [c for c in ["low", "mid", "high"] if c in cohort_shap.columns]
            cohort_shap = cohort_shap[cols]

            # Sort by absolute difference between high and low
            if "high" in cohort_shap.columns and "low" in cohort_shap.columns:
                cohort_shap["spread"] = cohort_shap["high"] - cohort_shap["low"]
                cohort_shap = cohort_shap.sort_values("spread", ascending=False)
                cohort_shap = cohort_shap.drop(columns=["spread"])

            cohort_path = shap_dir / "shap_cohort_analysis.csv"
            cohort_shap.to_csv(cohort_path, index=False)
            mlflow.log_artifact(str(cohort_path), artifact_path="shap")

            # SHAP grouped bar plot by cohort
            cohort_explanations = {}
            for bucket_name in ["Low", "Mid", "High"]:
                mask = shap_df["bucket"] == bucket_name.lower()
                bucket_shap = shap_values[mask.values]
                cohort_explanations[bucket_name] = shap.Explanation(
                    values=bucket_shap,
                    base_values=explainer.expected_value,
                    data=X_test[mask].values,
                    feature_names=X_test.columns.tolist(),
                )

            plt.figure()
            shap.plots.bar(cohort_explanations, show=False)
            cohort_chart_path = shap_dir / "shap_cohort_chart.png"
            plt.savefig(cohort_chart_path, bbox_inches="tight", dpi=150)
            plt.close()
            mlflow.log_artifact(str(cohort_chart_path), artifact_path="shap")

            logger.info("SHAP cohort analysis saved.")

        # Waterfall plots for low, mid, high predictions
        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_test.values,
            feature_names=X_test.columns.tolist(),
        )

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_test)[:, 1]
            samples = {
                "low": int(np.argmin(proba)),
                "mid": int(np.argsort(proba)[len(proba) // 2]),
                "high": int(np.argmax(proba)),
            }

            for label, idx in samples.items():
                plt.figure()
                shap.waterfall_plot(explanation[idx], show=False)
                waterfall_path = shap_dir / f"shap_waterfall_{label}.png"
                plt.savefig(waterfall_path, bbox_inches="tight", dpi=150)
                plt.close()
                mlflow.log_artifact(str(waterfall_path), artifact_path="shap")

        logger.info("SHAP analysis complete. Artifacts logged to MLflow.")


