from wake_word_data_prep_classes import (
    BasicFileOperations,
    TrainTestSplit,
    PreciseModelingOperations,
)
import os
from dialog_handler import DialogHandler

train_test_split_instance = TrainTestSplit()
precise_modeling_operations_instance = PreciseModelingOperations()
basic_file_operations_instance = BasicFileOperations()

dialog_name = "base_model_menu_dialog"
dialog_handler_instance = DialogHandler("dialog.json", dialog_name)


# Get base model

# TODO: MAJOR BUG! the training accuraries are not corret, it takes the last one however sb is saving the best epoch! Need to get this info somehow
# TODO: results of models should contain the epochs,


def move_random_user_collection(source_directory):
    random_source_directory = source_directory + "random/"
    random_destination_directory = random_source_directory + "user_collected/"

    if not os.path.exists(random_destination_directory):
        files = basic_file_operations_instance.get_files(random_source_directory)
        basic_file_operations_instance.copy_directory(
            files, random_source_directory, random_destination_directory
        )
        basic_file_operations_instance.delete_files_in_directory(
            files, random_source_directory
        )

    return random_destination_directory


def split_data_and_get_best_model_flow(
    source_directory,
    random_split_directories,
    even_odd_split_directories,
    three_four_split_directories,
    root_model_name,
):
    print(
        dialog_handler_instance.render_template(
            "inform-splitting_data", source_directory=source_directory
        )
    )
    model_names = train_test_split_instance.experimental_splits(
        source_directory,
        random_split_directories,
        even_odd_split_directories,
        three_four_split_directories,
        root_model_name,
    )

    print(dialog_handler_instance.render_template("inform-experiment_train_test_split"))
    precise_modeling_operations_instance.run_experimental_training(model_names)
    (
        average_val_acc,
        standard_deviation_val_acc,
        average_acc,
        standard_deviation_acc,
    ) = precise_modeling_operations_instance.get_models_analytics()
    # TODO: change to f1 score
    experimental_average_accuracy_dialog = dialog_handler_instance.render_template(
        "inform-accuracy",
        average_val_acc=average_val_acc,
        standard_deviation_val_acc=standard_deviation_val_acc,
        average_acc=average_acc,
        standard_deviation_acc=standard_deviation_acc,
    )
    print(experimental_average_accuracy_dialog)

    (
        selected_model_name,
        selected_model_results,
    ) = precise_modeling_operations_instance.pick_best_model()
    experimental_best_model = dialog_handler_instance.render_template(
        "inform-best_model",
        selected_model_name=selected_model_name,
        selected_model_results=selected_model_results,
    )
    print(experimental_best_model)

    return (
        selected_model_name,
        selected_model_results,
        experimental_average_accuracy_dialog,
    )


def train_model_flow(wakeword_model_name, epochs=None):
    # TODO: Can epochs still be set to None and still work?
    # TODO: keep f1 results of model
    print(
        dialog_handler_instance.render_template(
            "inform-training_start", wakeword_model_name=wakeword_model_name
        )
    )
    training_run = precise_modeling_operations_instance.run_precise_train(
        wakeword_model_name, epochs
    )
    print(
        dialog_handler_instance.render_template(
            "inform-training_complete", wakeword_model_name=wakeword_model_name
        )
    )

    precise_modeling_operations_instance.get_last_epoch_model_info(
        wakeword_model_name, training_run
    )

    return precise_modeling_operations_instance.models[wakeword_model_name]


def incremental_training_flow(
    random_user_recordings_directory, selected_model_name, epochs=None
):
    # TODO: add CLI elements into class?
    print(
        dialog_handler_instance.render_template(
            "inform-incremental_training_start",
            random_user_recordings_directory=random_user_recordings_directory,
        )
    )
    precise_modeling_operations_instance.incremental_training(
        selected_model_name, random_user_recordings_directory
    )
    print(
        dialog_handler_instance.render_template(
            "inform-incremental_training_complete",
            random_user_recordings_directory=random_user_recordings_directory,
        )
    )

    train_test_split_instance.split_incremental_results(selected_model_name)

    precise_modeling_operations_instance.delete_generated_directories(
        selected_model_name
    )
    precise_modeling_operations_instance.delete_model(selected_model_name + "_tmp_copy")

    model_info = train_model_flow(selected_model_name, epochs)

    return model_info


def get_base_model_flow(
    source_directory,
    random_split_directories,
    even_odd_split_directories,
    three_four_split_directories,
    root_model_name,
    wakeword_model_name,
):
    random_user_recordings_directory = move_random_user_collection(source_directory)

    (
        selected_model_name,
        selected_model_results,
        experimental_average_accuracy_dialog,
    ) = split_data_and_get_best_model_flow(
        source_directory,
        random_split_directories,
        even_odd_split_directories,
        three_four_split_directories,
        root_model_name,
    )

    precise_modeling_operations_instance.delete_experiment_directories(
        selected_model_name
    )

    base_model_info = incremental_training_flow(
        random_user_recordings_directory, selected_model_name, epochs="50"
    )

    precise_modeling_operations_instance.delete_experiment_models(selected_model_name)
    precise_modeling_operations_instance.rename_model(
        selected_model_name, wakeword_model_name
    )
    basic_file_operations_instance.rename_directory(
        "out/" + selected_model_name, "out/" + wakeword_model_name
    )
    print(
        dialog_handler_instance.render_template(
            "inform-changed_model_name",
            selected_model_name=selected_model_name,
            wakeword_model_name=wakeword_model_name,
        )
    )

    print(dialog_handler_instance.render_template("inform-base_model_accuracies"))
    print(experimental_average_accuracy_dialog)
    # TODO: results should include the following information: model_name, number of epochs, model_accuracies, number of files (in wake-word, not-wake-word)
    print(
        dialog_handler_instance.render_template(
            "inform-original_best_base_model_results",
            selected_model_results=selected_model_results,
        )
    )
    print(
        f"Current base model with incremental training on your random audio recordings: {base_model_info}"
    )
    print("Not bad when you consider the average accuracies of the first base model...")

    print(
        dialog_handler_instance.render_template(
            "inform-continue", wakeword_model_name=wakeword_model_name
        )
    )

    return base_model_info

    # test stuff


# 1. get_base_model_flow configuration
"""
source_directory = 'flow_test_delete_after'

random_split_directories = [
    '/wake-word/',
    '/not-wake-word/background/'
]

even_odd_split_directories = [
    '/wake-word/variations/'
]

three_four_split_directories = [
    '/not-wake-word/parts/'
]

root_model_name = 'experiment'

destination_directory = 'file_split_test/'
"""

# train_test_split_instance.experimental_splits(source_directory, random_split_directories, even_odd_split_directories, three_four_split_directories, root_model_name)

# train model and incremental training flow configuration
"""
wakeword_model_name = 'test_wakeword_model_delete_after'
noise_destination_directory = wakeword_model_name + '/random/non-utterances/pdsounds_march2009/'
model_info = train_model_flow(wakeword_model_name, epochs)
print(model_info)
model_info = incremental_training_flow(noise_destination_directory, wakeword_model_name)
print(model_info)"""
