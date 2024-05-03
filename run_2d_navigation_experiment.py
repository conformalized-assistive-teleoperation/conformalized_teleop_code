import pdb
import sys
import yaml, os, pickle
import numpy as np
from utils.data_utils import create_experiment_folder
from utils.pinball import qr_criterion, ensemble_criterion
from utils.remapping_dataset import Nav2DRemappedDataset as RemappedDataset
from utils.nav2d_utils.train_teleop_model import train_telop_controller_with_quantiles, train_ensemble_teleop_controller
from torch.utils.data import DataLoader

if __name__ == '__main__':
    yaml_filename = sys.argv[1]

    print('yaml_filename: ', yaml_filename)

    # Load yaml file parameters
    with open(f'experiment_configs/{yaml_filename}.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # print the configuration
    print(config)

    # Create the experiment folder in results directory
    experiment_folder = create_experiment_folder(config)

    # load training data
    experiment_name = config['experiment_name']
    uncertainty_context = config['uncertainty_context']
    task_domain = config['task_domain']
    training_data = f'data/training_data/{task_domain}_{uncertainty_context}/train_dataset.pkl'
    with open(training_data, 'rb') as f:
        train_dataset = pickle.load(f)

    # Train or load (if possible) a single quantile regressor for teleop if calibration type is acqr or qr
    if config['calibration_type'] == 'acqr' or config['calibration_type'] == 'qr':
        # check if trained model exists in the experiment folder, otherwise, train the model
        teleop_model_filepath = f'results/{experiment_name}/models/epoch_final.pt'
        from utils.remapping_net import FC_Action_Remapper_with_Quantiles as TeleopModel
        teleop_model_architecture = TeleopModel
        if os.path.exists(teleop_model_filepath) is False:
            # Create Dataloader from remapping dataset
            remapped_dataset = RemappedDataset(train_dataset)
            remapping_train_loader = DataLoader(remapped_dataset, batch_size=config['bsz'], shuffle=True)
            print("DONE LOADING DATASET for REMAPPING!")

            # train the model
            train_telop_controller_with_quantiles(teleop_model_architecture, remapping_train_loader, qr_criterion,
                                                  config['state_dim'], config['action_dim'], config['latent_dim'],
                                                  config['hidden_dim'], config['bsz'], config['lr'], config['epochs'],
                                                  experiment_name, to_save=True)
    elif config['calibration_type'] == 'ensemble':
        num_models_in_ensemble = 5 # 5 is a fixed number of models in the ensemble. This can be changed.
        teleop_model_filepath = f'results/{experiment_name}/models/remapping_{0}_epoch_final.pt' # check if the first model exists
        from utils.ensemble_net import Ensemble_Estimator as TeleopModel
        teleop_model_architecture = TeleopModel
        if os.path.exists(teleop_model_filepath) is False:
            # Create 5 (differently ordered) Dataloaders from remapping dataset
            remapped_dataset = RemappedDataset(train_dataset)
            train_loaders = []
            for remapping_model in range(num_models_in_ensemble):
                train_loader = DataLoader(remapped_dataset, batch_size=config['bsz'], shuffle=True)
                train_loaders.append(train_loader)

            # train the ensemble model
            train_ensemble_teleop_controller(teleop_model_architecture, train_loaders, ensemble_criterion,
                                                  config['state_dim'], config['action_dim'], config['latent_dim'],
                                                  config['hidden_dim'], config['bsz'], config['lr'], config['epochs'],
                                                  experiment_name, to_save=True)
        teleop_model_filepath = [f'results/{experiment_name}/models/remapping_{remapping_idx}_epoch_final.pt' for
                             remapping_idx in range(0, 5)]

    else:
        print("Base model type not supported")
        exit(1)

    # Calibrate
    calibration_type = config['calibration_type']
    if calibration_type == 'acqr':
        from utils.nav2d_utils.calibrate_teleop_model import calibrate_teleop_model_with_acqr as calibrate_teleop_model
    elif calibration_type == 'qr':
        from utils.nav2d_utils.calibrate_teleop_model import calibrate_teleop_model_with_qr as calibrate_teleop_model
    elif calibration_type == 'ensemble':
        from utils.nav2d_utils.calibrate_teleop_model import calibrate_teleop_model_with_ensemble as calibrate_teleop_model
    else:
        print("Calibration type not supported")
        exit(1)


    print("\n\nCalibrating on Alice")
    # load alice data
    indistr_alice_file = f'data/calibration_data/{task_domain}_{uncertainty_context}/calibration_alice_indistribution_traj_idx_to_data.pkl'
    with open(indistr_alice_file, 'rb') as f:
        indistr_alice_data = pickle.load(f)

    if calibration_type == 'acqr':
        beta_uncertainty_threshold = float(sys.argv[2])
        config['beta_uncertainty_threshold'] = beta_uncertainty_threshold
    calibrate_teleop_model(indistr_alice_data,
                            teleop_model_filepath,
                            teleop_model_architecture,
                            config,
                            experiment_name, 'alice')

    # Out of distribution trajectory user data is relevant if the uncertainty context is latent_prefs
    if uncertainty_context == 'latent_prefs':
        print("\n\nCalibrating on Bob")
        # load bob data
        indistr_bob_file = f'data/calibration_data/{task_domain}_{uncertainty_context}/calibration_bob_outdistribution_traj_idx_to_data.pkl'
        with open(indistr_bob_file, 'rb') as f:
            indistr_bob_data = pickle.load(f)

        if calibration_type == 'acqr':
            beta_uncertainty_threshold = float(sys.argv[3])
            config['beta_uncertainty_threshold'] = beta_uncertainty_threshold
        calibrate_teleop_model(indistr_bob_data,
                               teleop_model_filepath,
                               teleop_model_architecture,
                               config,
                               experiment_name, 'bob')











