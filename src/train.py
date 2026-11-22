"""Model Training, Hyperparameter Tuning, and Experiment Tracking for SuperKart."""

import json
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

TARGET_COL = "Product_Store_Sales_Total"

NUMERIC_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
    "Store_Age",
]

CATEGORICAL_FEATURES = [
    "Product_Sugar_Content",
    "Product_Type",
    "Product_Category_Prefix",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression evaluation metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def train_and_tune(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    model_dir: str = "models",
) -> None:
    """Train, tune, log to MLflow, evaluate, and save the best SuperKart model."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].values
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    candidate_models = {
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [10, 15, None],
                "regressor__min_samples_split": [2, 5],
            },
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "regressor__n_estimators": [100, 200],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_depth": [3, 5],
            },
        ),
        "XGBoost": (
            XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=-1),
            {
                "regressor__n_estimators": [100, 200],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_depth": [3, 5, 7],
            },
        ),
    }

    mlflow.set_experiment("SuperKart_Sales_Forecasting")

    best_overall_model = None
    best_overall_name = None
    best_overall_rmse = float("inf")
    results_summary = {}

    for model_name, (estimator, param_grid) in candidate_models.items():
        with mlflow.start_run(run_name=f"{model_name}_GridSearch"):
            pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", estimator),
                ]
            )

            grid_search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=3,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train, y_train)

            best_est = grid_search.best_estimator_
            train_metrics = evaluate_predictions(y_train, best_est.predict(X_train))
            test_metrics = evaluate_predictions(y_test, best_est.predict(X_test))

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(
                {
                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "train_r2": train_metrics["r2"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_r2": test_metrics["r2"],
                }
            )
            mlflow.sklearn.log_model(best_est, artifact_path=f"model_{model_name}", serialization_format="cloudpickle")

            results_summary[model_name] = {
                "best_params": grid_search.best_params_,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
            print(f"[{model_name}] Test RMSE: {test_metrics['rmse']} | Test R2: {test_metrics['r2']}")

            if test_metrics["rmse"] < best_overall_rmse:
                best_overall_rmse = test_metrics["rmse"]
                best_overall_model = best_est
                best_overall_name = model_name

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.joblib")
    metrics_path = os.path.join(model_dir, "metrics.json")

    joblib.dump(best_overall_model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_overall_name,
                "best_test_rmse": best_overall_rmse,
                "all_models": results_summary,
            },
            f,
            indent=2,
        )

    print(f"\n[BEST MODEL] {best_overall_name} (Test RMSE: {best_overall_rmse})")
    print(f"[SAVED] Model -> {model_path}")
    print(f"[SAVED] Metrics -> {metrics_path}")


if __name__ == "__main__":
    train_and_tune()
