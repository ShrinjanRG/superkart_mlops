# SuperKart Sales Forecasting — MLOps Pipeline

Automated end-to-end Machine Learning & MLOps pipeline for forecasting product-store sales revenue (`Product_Store_Sales_Total`).

## Repository Structure
```text
superkart_mlops/
├── .github/
│   └── workflows/
│       └── pipeline.yml        # Automated CI/CD GitHub Actions workflow
├── data/
│   ├── SuperKart.csv           # Raw dataset (8,763 rows x 12 columns)
│   └── processed/              # Generated train.csv & test.csv splits
├── models/
│   ├── best_model.joblib       # Trained & tuned best regression pipeline
│   └── metrics.json            # Evaluation metrics across tuned models
├── src/
│   ├── data_registration.py    # Schema validation & dataset summary
│   ├── data_prep.py            # Cleaning, feature engineering & 80/20 split
│   └── train.py                # GridSearchCV tuning + MLflow experiment tracking
├── app.py                      # Streamlit web application
└── requirements.txt            # Python dependencies
```

## Quick Start
1. **Validate Data**: `python src/data_registration.py`
2. **Prepare Splits**: `python src/data_prep.py`
3. **Train & Tune**: `python src/train.py`
4. **Launch App**: `streamlit run app.py`
