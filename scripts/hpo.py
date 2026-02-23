from clearml import OutputModel, Task
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
    
    args = {
    'target_project': 'Cirrosis Patient Survival Prediction',
    'target_task_name': 'clear ml training task'
    }
    task.connect(args)  # This makes it editable in the UI

    # Use the values from the editable UI dictionary
    base_task = Task.get_task(
        project_name=args['target_project'], 
        task_name=args['target_task_name']
    )
    print(base_task)

    search_space = [
        UniformIntegerParameterRange("n_estimators", 300, 1500),
        UniformParameterRange("learning_rate", 0.005, 0.1),
        UniformIntegerParameterRange("max_depth", 3, 10),
        UniformParameterRange("subsample", 0.5, 1.0),
        UniformParameterRange("colsample_bytree", 0.5, 1.0),
    ]

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task.id,
        hyper_parameters=search_space,
        objective_metric_title="log_loss",
        objective_metric_series="validation",
        objective_metric_sign="min",
        optimizer_class=OptimizerOptuna,
        # This dictionary is passed to the OptimizerOptuna constructor
        max_number_of_concurrent_tasks=4,
        execution_queue="default",
        total_max_jobs=50,
        max_iteration_per_job=1,  # Each job runs the training task once with a different set of hyperparameters
    )

    optimizer.start()
    optimizer.wait()
    top_tasks = optimizer.get_top_experiments(top_k=1)
    if top_tasks:
        best_task = top_tasks[0]
        print(f"Best Task ID: {best_task.id}")

        # 3. Log Best Task Weights to the Controller
        # Get the output model from the best training task
        best_model = best_task.get_models_associated_with_task()[-1]

        # Create an OutputModel for the Controller to "inherit" the best weights
        output_model = OutputModel(task=task, name="Best HPO Model")
        output_model.update_weights(registered_uri=best_model.url)

        # 4. Log Training Dataset Information
        # Use task.set_user_properties or artifacts to link the dataset path/ID
        task.set_user_properties(
            best_task_id=best_task.id,
            dataset_used=best_task.get_parameters().get("Args/dataset", "unknown"),
        )

        best_params = best_task.get_parameters()

        # 2. Log them to the Controller's UI (Configuration Tab)
        # This makes it easy to see exactly what won without clicking into sub-tasks
        task.connect_configuration(best_params, name="Best_Hyperparameters")

    optimizer.stop()


if __name__ == "__main__":
    main()
