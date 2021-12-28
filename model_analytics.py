import json

# Open the json file and load the content
with open('training_run.json', 'r') as file:
    training_run_json = json.load(file)

# string matches for the lines we want to scrape
def scrape_epoch_training_results(training_run_json):
    training_complete_string = '[==============================]'
    epoch_number = 'Epoch'
    #TODO: add in the category counts and return them
    category_counts = 'Data:'

    epochs = [line for line in training_run_json if epoch_number in line]

    epoch_training_results = [line.split(' - ')[2:] for line in training_run_json if training_complete_string in line]
    return zip(epoch_training_results, epochs)



def transform_epoch_number_and_accuracy(training_run_json):
    epochs_accuracy = {}
    epochs_training_results = scrape_epoch_training_results(training_run_json)
    for epoch_result, epoch_number in epochs_training_results:
        epoch_number = int(epoch_number.strip('Epoch ').rsplit('/', 1)[0])
        loss = float(epoch_result[0].strip('loss: '))
        acc = float(epoch_result[1].strip('acc: '))
        val_loss = float(epoch_result[2].strip('val_loss: '))
        val_acc = float(epoch_result[3].strip('val_acc: '))

        accuracy_dict = {
            'epoch': epoch_number,
            'loss': loss,
            'acc': acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'acc_difference': abs(acc - val_acc)
            }

        epochs_accuracy[epoch_number] = accuracy_dict
    return epochs_accuracy


model_name = 'experiment_model_1'

def get_maximum_accuracy(training_run_json):
    epochs_accuracy = transform_epoch_number_and_accuracy(training_run_json)
    minimum_loss_accuracy = min(epochs_accuracy.values(), key=lambda x: x['loss'])
    minimum_loss_val_accuracy = min(epochs_accuracy.values(), key=lambda x: x['val_loss'])

    return (minimum_loss_accuracy, minimum_loss_val_accuracy)

minimum_loss_accuracy, minimum_loss_val_accuracy = get_maximum_accuracy(training_run_json)
#print(minimum_loss_accuracy)
#print(minimum_loss_val_accuracy)

models = {}

def add_model(model_name, training_run_json):
    minimum_loss_accuracy, minimum_loss_val_accuracy = get_maximum_accuracy(training_run_json)
    models = {}
    models[model_name] = [
    {
        'minimum_loss_accuracy': minimum_loss_accuracy
        },
    {
       'minimum_loss_val_accuracy': minimum_loss_val_accuracy
        }
        ]
    return models

models = add_model(model_name, training_run_json)

models['experiment_model_2'] = [
    {
        'minimum_loss_accuracy': {
            'epoch': 15,
            'loss': 0.11,
            'acc': 0.9,
            'val_loss': 0.12,
            'val_acc': 0.89,
            'acc_difference': abs(0.9 - 0.89)
        },
    },
    {
        'minimum_loss_val_accuracy': {
            'epoch': 16,
            'loss': 0.12,
            'acc': 0.89,
            'val_loss': 0.15,
            'val_acc': 0.62,
            'acc_difference': abs(0.89 - 0.91)
            }
        }
        
]


#print(models)



# This is the part that calculates the smallest minimum_loss and minimum_loss_val_acc for all the models

def get_model_accuracies(models):
    acc_for_min_loss_models = [item[1][0]['minimum_loss_accuracy']['acc'] for item in models.items()]
    val_acc_for_min_val_loss_models = [item[1][1]['minimum_loss_val_accuracy']['val_acc'] for item in models.items()]
    return (acc_for_min_loss_models, val_acc_for_min_val_loss_models)

def get_max_accuracies_over_all_models(models):
    acc_for_min_loss_models, val_acc_for_min_val_loss_models = get_model_accuracies(models)
    max_acc_for_min_loss_model = max(acc_for_min_loss_models)
    max_val_acc_for_min_val_loss_model = max(val_acc_for_min_val_loss_models)
    return (max_acc_for_min_loss_model, max_val_acc_for_min_val_loss_model)

#print(get_max_accuracies_over_all_models(models))

import statistics

def get_model_analytics(models):
    acc_for_min_loss_models, val_acc_for_min_val_loss_models = get_model_accuracies(models)
    return (statistics.mean(acc_for_min_loss_models), statistics.stdev(acc_for_min_loss_models) , statistics.mean(val_acc_for_min_val_loss_models), statistics.stdev(val_acc_for_min_val_loss_models))

#print(get_model_analytics(models))

def get_best_models(models):
    max_acc_for_min_loss_model, max_val_acc_for_min_val_loss_model = get_max_accuracies_over_all_models(models)
    best_models = []
    max_acc_for_min_loss_model_dict = {}
    max_val_acc_for_min_val_loss_model_dict = {}
    for item in models.items():
        if item[1][0]['minimum_loss_accuracy']['acc'] == max_acc_for_min_loss_model:
            max_acc_for_min_loss_model_dict[item[0]] = item[1][0]
            best_models.append(max_acc_for_min_loss_model_dict)
        if item[1][1]['minimum_loss_val_accuracy']['val_acc'] == max_val_acc_for_min_val_loss_model:
            max_val_acc_for_min_val_loss_model_dict[item[0]] = item[1][1]
            best_models.append(max_val_acc_for_min_val_loss_model_dict)
    return best_models

best_models = get_best_models(models)

# This is used when the training is run for -sb with the loss being the measure (default)
def get_best_max_acc_for_min_loss_model(best_models):
    max_acc_for_min_loss_model_dict = best_models[0]
    return max_acc_for_min_loss_model_dict

# This is used when the training is run for -sb with the val_loss being the measure (important to optimize when the test set is big enough)
def get_best_max_val_acc_for_min_val_loss_model(best_models):
    max_val_acc_for_min_val_loss_model_dict = best_models[1]
    return max_val_acc_for_min_val_loss_model_dict