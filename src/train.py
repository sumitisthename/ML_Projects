import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from mlflow.models import infer_signature
import mlflow
from urllib.parse import urlparse

# Set MLflow tracking credentials with new token
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/sumitisthename/MachineLearningPipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sumitisthename'
# Replace with your new token
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'a9243f39c64d2d6812f764347c1e2d10485836b8'

def hyperparameter_tuning(X_train, y_train, param_grid):
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

def train(data_path, model_path, params):
    data = pd.read_csv(data_path)
    
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=params["random_state"]
        )
        
        # Hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [params["n_estimators"]],
            'max_depth': [params["max_depth"]],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_
        
        # Evaluation
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        
        # Log metrics and parameters
        mlflow.log_metric("accuracy", accuracy)
        
        # Log parameters
        mlflow.log_params({
            "n_estimators": best_model.n_estimators,
            "max_depth": best_model.max_depth,
            "min_samples_split": best_model.min_samples_split,
            "min_samples_leaf": best_model.min_samples_leaf,
            "random_state": params["random_state"]
        })
        
        # Log evaluation metrics as text files
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        mlflow.log_text(cr, "classification_report.txt")
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        
        # Log artifacts
        try:
            mlflow.log_artifact("params.yaml")
        except Exception as e:
            print(f"Warning: Could not log params.yaml artifact: {e}")
        
        # Infer signature and log model (without model registration for DagsHub compatibility)
        try:
            signature = infer_signature(X_test, y_test)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
            print("Model logged to MLflow successfully")
        except Exception as e:
            print(f"Warning: Could not log model to MLflow: {e}")
            # Fallback: just log the model without signature
            try:
                mlflow.sklearn.log_model(best_model, "model")
                print("Model logged to MLflow without signature")
            except Exception as e2:
                print(f"Error: Could not log model at all: {e2}")
        
        # Save model locally as backup
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"Model saved locally to {model_path}")
        
        # Log additional hyperparameter tuning results
        mlflow.log_params({
            "best_score": grid_search.best_score_,
            "cv_folds": 3
        })
        
        # Log best parameters from grid search
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param_name}", param_value)

if __name__ == "__main__":
    config = yaml.safe_load(open("params.yaml"))
    train_config = config["train"]
    train(
        data_path=train_config["input"],
        model_path=train_config["model"],
        params=train_config["parameters"]
    )