import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
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
        # Class imbalance handling
        X_train, y_train = self.resample(X_train, y_train)

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

            # Log model
            if self.model is not None:
                mlflow.sklearn.log_model(self.model, artifact_path=self.model_name)

            # Persist predictions for the full dataset to match schema
            # Update so that this will instead read from original dataset + predictions
            df_out = df.copy()
            if self.model is not None:
                X_full = self._prepare_features(X)
                df_out["PREDICTION"] = self.model.predict(X_full)

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
        # logger.info("Classification Report:\n%s", report)

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
