from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.parameters import (
    UniformIntegerParameterRange,
    UniformParameterRange,
)


def main():
    task = Task.init(
        project_name="Cirrosis Patient Survival Prediction",
        task_name="HPO Controller",
        task_type=Task.TaskTypes.optimizer,
    )

    base_task_id = "e6975e6495f64153b7a9f64552e02a2d"

    search_space = [
        UniformIntegerParameterRange("n_estimators", 300, 1500),
        UniformParameterRange("learning_rate", 0.005, 0.1),
        UniformIntegerParameterRange("max_depth", 3, 10),
        UniformParameterRange("subsample", 0.5, 1.0),
        UniformParameterRange("colsample_bytree", 0.5, 1.0),
        UniformParameterRange("gamma", 0, 5),
    ]

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=search_space,
        objective_metric_title="log_loss",
        objective_metric_series="validation",
        objective_metric_sign="min",
        optimizer_class=OptimizerOptuna,
        # This dictionary is passed to the OptimizerOptuna constructor
        max_iteration_per_job = 30,
        max_number_of_concurrent_tasks=2,
        execution_queue="default",
        total_max_jobs=30,
    )

    optimizer.start()
    optimizer.wait()
    optimizer.stop()


if __name__ == "__main__":
    main()
