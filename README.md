# A2: Predicting Car Price

**Student ID:** st127262
**Course:** AT82.03 Machine Learning

This project continues from A1, using the same cleaned Car Price dataset,
and implements a Linear Regression model from scratch (with regularization,
gradient descent variants, Xavier initialization, and momentum), compares
144 configurations via 5-fold cross-validation tracked with MLflow, and
deploys a web app with both the old (A1) and new (A2) models.

## Contents
├── Ass2.ipynb # main notebook: implementation, experiment, report
├── experiment_results.csv # checkpointed results of all 144 configurations
├── mlflow.db # MLflow tracking database (SQLite)
├── app/
│ └── code/
│ ├── app.py # Dash web app (old model page + new model page)
│ ├── model_class.py # the from-scratch LinearRegression class (Task 1)
│ ├── requirements.txt
│ ├── Dockerfile
│ └── model/ # saved models used by the app
│ ├── car_price_model.pkl # A1 Random Forest
│ ├── best_model.pkl # A2 best from-scratch model
│ ├── scaler.pkl
│ ├── poly.pkl
│ ├── feature_names.pkl
│ ├── model_columns.pkl
│ ├── defaults.pkl
│ └── best_config.pkl
└── README.md

## Task 1 - Implementation

`model_class.py` extends the `LinearRegression` class from
`03 - Regularization.ipynb` with:
- `r2()` - R² score
- Xavier weight initialization (`init_method='xavier'`), alongside `'zeros'`
- Momentum (`use_momentum`, `momentum`)
- `plot_feature_importance()` - bar chart of `|coefficient|` per feature

## Task 2 - Experiment

`Ass2.ipynb` compares, with 5-fold cross-validation:
- Regularization: normal, lasso, ridge, polynomial
- Gradient descent method: stochastic, mini-batch, batch
- Initialization: zeros, xavier
- Momentum: with / without
- Learning rate: 0.01, 0.001, 0.0001

144 configurations total, tracked with MLflow (`mlflow.db`). The best
configuration found was **Lasso regression, mini-batch gradient descent,
zeros initialization, learning rate 0.01, no momentum**, reaching a test
R² of 0.7777 (log-price scale) / 0.5811 (price scale).

To view the MLflow run history:
```bash
pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
then open http://127.0.0.1:5000.

## Task 3 - Deployment

The `app/` folder contains a Dash web application with two pages:
- **Old model** (`/`) - the A1 Random Forest model
- **New model** (`/new`) - the A2 from-scratch Lasso model

### Run locally
```bash
cd app/code
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:8000

### Run with Docker
```bash
cd app/code
docker build -t car-price-app .
docker run -p 8000:8000 car-price-app
```

### Live deployment
Deployed at: https://web-st127262.ml.brain.cs.ait.ac.th/

## Dataset

Car Dekho used car listings dataset (same as A1): `name`, `year`,
`selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`,
`owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`. Prices are
in Indian Rupees (INR).