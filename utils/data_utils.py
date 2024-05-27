import os

def create_experiment_folder(config):
    # Create the experiment folder in results directory
    experiment_folder = os.path.join('results', config['experiment_name'])
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    # create a subdirectory called models
    models_folder = os.path.join(experiment_folder, 'models')
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # create a subdirectory called plots
    plots_folder = os.path.join(experiment_folder, 'plots')
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    return experiment_folder











