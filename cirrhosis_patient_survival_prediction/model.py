import argparse
import os
import joblib
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from fire import Fire


class My_Classifier_Model:
    """
    Wraps any sklearn pipeline for training, saving, loading, and predicting.
    The pipeline should include preprocessing + model.
    """
    
    logfile = "./data/log_file.log"
    results_file = "./data/results.csv"
    classifier_file = "./model/pipeline.pkl"
    

    def __init__(self, pipeline=None):
        """
        pipeline: any sklearn Pipeline object (preprocessing + estimator)
        """
        self.pipeline = pipeline
        
        os.makedirs("./model", exist_ok=True)
        os.makedirs("./data", exist_ok=True)

        logging.basicConfig(
            filename="./data/log_file.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def _build_default_pipeline(self, dataset_path):
        """
        Automatically detect numerical and categorical columns
        from dataset and build pipeline.
        """

        data = pd.read_csv(dataset_path, index_col=0)
        
        data = data.drop(['Status'], axis=1)

        categorical_cols = data.select_dtypes(include=["str"]).columns.tolist()
        numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="constant"), numerical_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), categorical_cols)
            ]
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                random_state=0,
                eval_metric="logloss"
            ))
        ])

        return pipeline

    def train(self, dataset):
        """
        Train pipeline on dataset and save artifacts
        """
        try: 
            data = pd.read_csv(dataset, index_col=0)

            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(data["Status"])
            X = data.drop(["Status"], axis=1)

            # Train pipeline
            if self.pipeline is None:
                self.pipeline = self._build_default_pipeline(dataset)
            self.pipeline.fit(X, y)

            # Save artifacts
            joblib.dump(self.pipeline, self.classifier_file)
            logging.info(f"Trained model on {dataset}")
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise e
        

    def predict(self, dataset):
        """
        Predict on a test dataset and save predictions to submission_probs.csv
        """
        try:
            pipeline = joblib.load(self.classifier_file)
            X_test = pd.read_csv(
                dataset, index_col=0
            )
            
            X_test_ids = X_test.index

            test_preds_proba = pipeline.predict_proba(X_test)
            
            class_names = [f"Status_{i}" for i in range(test_preds_proba.shape[1])]
            output = pd.DataFrame(test_preds_proba, columns=class_names)
            
            output.insert(0, "id", X_test_ids)
            output.to_csv(self.results_file, index=False)
            logging.info(f"Predicted on {dataset} and saved results to {self.results_file}")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e
        





if __name__ == "__main__":
    Fire(My_Classifier_Model)