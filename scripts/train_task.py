import argparse
import joblib
import pandas as pd
from clearml import Task, OutputModel, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from cirrhosis_patient_survival_prediction.model import My_Classifier_Model  # adjust import


def main(dataset_path):

    # 1️⃣ Initialize ClearML Task
    task = Task.init(
        project_name="Cirrosis Patient Survival Prediction",
        task_name="clear ml training task",
        task_type=Task.TaskTypes.training
    )

    logger = task.get_logger()
    
    # dataset = Dataset.create(
    #     dataset_name="Train Dataset",
    #     dataset_project="My ML Project",
    # )

    # dataset.add_files(dataset_path)
    # dataset.upload()
    # dataset.finalize()

    # 2️⃣ Define default hyperparameters
    params = My_Classifier_Model.default_hyperparameters

    # 3️⃣ Allow ClearML UI / HPO to override parameters
    params = task.connect(params)

    # 4️⃣ Load dataset
    data = pd.read_csv(dataset_path, index_col=0)

    y = data["Status"]
    X = data.drop(["Status"], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5️⃣ Train model (pure ML code)
    model = My_Classifier_Model(hyperparameters=params)
    model.train(dataset_path)

    # 6️⃣ Evaluate validation performance
    pipeline = joblib.load(model.classifier_file)
    preds = pipeline.predict_proba(X_valid)
    score = log_loss(y_valid, preds)

    logger.report_scalar("log_loss", "validation", score, iteration=0)

    print(f"Validation log_loss: {score}")

    # # 7️⃣ Upload best model artifact
    # output_model = OutputModel(task=task)
    
    # output_model.update_weights(
    #     weights_filename=model.classifier_file,
    # )

    # print("Model uploaded to ClearML.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    main(args.dataset)