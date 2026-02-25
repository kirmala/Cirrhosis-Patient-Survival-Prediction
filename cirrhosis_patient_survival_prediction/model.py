import os
import joblib
import logging
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from fire import Fire
from clearml import Task, OutputModel


class My_Classifier_Model:
    """
    Wraps any sklearn pipeline for training, saving, loading, and predicting.
    The pipeline should include preprocessing + model.
    """

    logfile = "./data/log_file.log"
    results_file = "./data/results.csv"
    classifier_file = "./model/pipeline.pkl"

    default_hyperparameters = {
        "n_estimators": 850,
        "learning_rate": 0.03655427593513622,
        "max_depth": 4,
        "subsample": 0.7683370898124168,
        "colsample_bytree": 0.5523898120612046,
    }

    def __init__(self, hyperparameters=None):
        """
        pipeline: any sklearn Pipeline object (preprocessing + estimator)
        """

        if hyperparameters is None:
            hyperparameters = self.default_hyperparameters

        self.hyperparameters = hyperparameters

        os.makedirs("./model", exist_ok=True)
        os.makedirs("./data", exist_ok=True)

        logging.basicConfig(
            filename="./data/log_file.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def train(self, dataset):
        try:
            data = pd.read_csv(dataset, index_col=0)

            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(data["Status"])
            X = data.drop(["Status"], axis=1)

            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            categorical_cols = X.select_dtypes(include=["str"]).columns.tolist()
            numerical_cols = X.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="constant"), numerical_cols),
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="constant")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        categorical_cols,
                    ),
                ]
            )


            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        XGBClassifier(
                            random_state=42,
                            **self.hyperparameters,
                            eval_metric="logloss",
                        ),
                    ),
                ]
            )

            pipeline.fit(X_train, y_train)

            logging.info(f"Model trained with hyperparameters: {self.hyperparameters}")

            joblib.dump(pipeline, self.classifier_file)


        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise e

    def predict(self, dataset):
        """
        Predict on a test dataset and save predictions to submission_probs.csv
        """
        try:
            pipeline = joblib.load(self.classifier_file)
            X_test = pd.read_csv(dataset, index_col=0)

            X_test_ids = X_test.index

            test_preds_proba = pipeline.predict_proba(X_test)

            class_names = [f"Status_{i}" for i in range(test_preds_proba.shape[1])]
            output = pd.DataFrame(test_preds_proba, columns=class_names)

            output.insert(0, "id", X_test_ids)
            output.to_csv(self.results_file, index=False)
            logging.info(
                f"Predicted on {dataset} and saved results to {self.results_file}"
            )
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e


if __name__ == "__main__":
    Fire(My_Classifier_Model)
