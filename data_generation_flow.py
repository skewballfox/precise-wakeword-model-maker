from wake_word_data_prep_classes import (
    BasicFileOperations,
    PreciseModelingOperations,
    GaussianNoiseHandler,
)
from get_base_model_flow import incremental_training_flow

basic_file_operations_instance = BasicFileOperations()
precise_modeling_operations_instance = PreciseModelingOperations()
gaussian_handler_instance = GaussianNoiseHandler()

def data_generation_flow(wakeword_model_name, source_directories, destination_directories, directories_to_gauss, noise_directory, base_model_info=None):
    if noise_directory is None:
        noise_directory = str(
            input(
                f"Enter the relative noise files directory path (ie {wakeword_model_name}/pdsounds_march2009/mp3/):"
            )
        )
    noise_destination_directory = (
        "out/" + wakeword_model_name + "/random/non-utterances/pdsounds_march2009/"
    )
    basic_file_operations_instance.convert_mp3s_in_directory_to_wavs(
        noise_directory, noise_destination_directory
    )

    print("Adding background noise...")
    precise_modeling_operations_instance.add_background_noise(
        wakeword_model_name, noise_destination_directory
    )
    precise_modeling_operations_instance.move_noise_directories(
        wakeword_model_name, source_directories, destination_directories
    )

    print("Adding Gaussian noise...")
    gaussian_handler_instance.add_gaussian_noise_to_directories(
        wakeword_model_name, directories_to_gauss
    )

    generated_model_info = incremental_training_flow(
        noise_destination_directory, wakeword_model_name, epochs="30"
    )

    generated_model_info = incremental_training_flow(noise_destination_directory, wakeword_model_name, epochs='30')

    if base_model_info:
        print(f'original base model info:\n{wakeword_model_name}: {base_model_info[0]}\ndataset size: {base_model_info[2]}')
    print(f'data generated model info:\n{wakeword_model_name}: {generated_model_info[0]}\ndataset size: {generated_model_info[2]}')
    print('Done!')
    return generated_model_info

# test stuff
#TODO: move to test file

# Config for testing

# wakeword_model_name = 'test_wakeword_model_delete_after'

# noise_directory = 'flow_test_delete_after/pdsounds_march2009/mp3/'


'''

with open('data_prep_user_configuration.json', 'r') as file:
    configuration_dictionary = json.load(file)

directories_to_process = configuration_dictionary['extra_audio_directories_to_process']
max_files_from_directory = configuration_dictionary['max_files_from_directory']
max_files_per_destination_directory = configuration_dictionary['max_files_per_destination_directory']


for directory in directories_to_process:
    print(directory)
"""

"""
source_directories = [
    'background_noise/wake-word/',
    'background_noise/wake-word/variations/',
    'background_noise/test/wake-word/',
    'background_noise/test/wake-word/variations/'
    ]

destination_directories = [
    'wake-word/background_noise/',
    'wake-word/background_noise/variations/',
    'test/wake-word/background_noise/',
    'test/wake-word/background_noise/variations/'
    ]

directories_to_gauss = [
    'wake-word/',
    'wake-word/variations/',
    'not-wake-word/background/',
    'not-wake-word/parts/',
    'test/wake-word/',
    'test/wake-word/variations/',
    'test/not-wake-word/background/',
    'test/not-wake-word/parts/'
    ]


data_generation_flow(wakeword_model_name, source_directories, destination_directories, directories_to_gauss, noise_directory)
"""
# basic_file_operations_instance.convert_mp3s_in_directory_to_wavs(noise_directory, noise_directory + 'wav/')
# noise_directory = noise_directory + 'wav/'
# precise_modeling_operations_instance.add_background_noise(wakeword_model_name, noise_directory)
# precise_modeling_operations_instance.move_noise_directories(wakeword_model_name, source_directories, destination_directories)
# gaussian_handler_instance.add_gaussian_noise_to_directories(wakeword_model_name, directories_to_gauss)
