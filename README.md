# Cirrhosis Patient Survival Prediction

# Author

**Name:** Малахов Кирилл Антонович
**Group:** 972401


# How to run

## Clone Repository

```bash
git clone https://github.com/kirmala/Cirrhosis-Patient-Survival-Prediction.git
cd Cirrhosis-Patient-Survival-Prediction
```
## Without Docker

### Install Dependencies (Poetry)

Install Poetry if needed:

```bash
pip install poetry
```

Install project dependencies:

```bash
poetry install
```


### Train Model

```bash
poetry run python3 cirrhosis_patient_survival_prediction/model.py train --dataset example_train.csv
```

### Evaluate Model

```bash
poetry run python3 cirrhosis_patient_survival_prediction/model.py predict --dataset example_test.csv 
```
view results at ./data/results.csv

---


## Using Docker

## Build Image

```bash
docker build -t cirrhosis-ml .
```

## Train Model

```bash
docker run cirrhosis-ml train --dataset example_train.csv
```

## Evaluate Model

```bash
docker run -v "./data:/app/data" cirrhosis-ml predict --dataset example_test.csv
```
view results at ./data/results.csv

---


# Resources utilized

- Python
- Scikit-learn
- XGBoost
- ClearML
- Optuna
- Poetry
- Docker