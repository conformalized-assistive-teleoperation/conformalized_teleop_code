import copy
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
    # yaml_filename = 'grasp7dof/grasp7dof_latentprefs_acqr'
    yaml_list = ['grasp7dof/grasp7dof_latentprefs_acqr', 'grasp7dof/grasp7dof_latentprefs_ensemble',
                 'grasp7dof/grasp7dof_latentprefs_qr', 'grasp7dof/grasp7dof_controlprec_acqr',
                    'grasp7dof/grasp7dof_controlprec_ensemble', 'grasp7dof/grasp7dof_controlprec_qr',
                 'goal7dof/goal7dof_latentprefs_acqr', 'goal7dof/goal7dof_latentprefs_ensemble',
                    'goal7dof/goal7dof_latentprefs_qr', 'goal7dof/goal7dof_controlprec_acqr',
                    'goal7dof/goal7dof_controlprec_ensemble', 'goal7dof/goal7dof_controlprec_qr',


                 ]
    for yaml_filename in yaml_list:
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

        # plot the training data
        new_training_data = []
        for t in range(len(train_dataset)):
            # pdb.set_trace()
            datapt = train_dataset[t]
            robot_state_t = datapt[0][0:7]
            action_t = datapt[1][0:7]
            if np.any(abs(action_t - 2*np.pi) < 0.1):
                idx_wrap = np.where(abs(action_t - 2*np.pi) < 0.1)

                if action_t[idx_wrap] > 0:
                    action_t[idx_wrap] = -2*np.pi + action_t[idx_wrap]
                else:
                    action_t[idx_wrap] = 2*np.pi + action_t[idx_wrap]

                assert np.all(abs(action_t - 2*np.pi) > 0.1)

            new_datapt = copy.deepcopy(datapt)
            new_datapt[1][0:7] = action_t
            new_training_data.append(new_datapt)

        # save training data
        # new_training_data = np.array(new_training_data)
        with open(f'data/training_data/{task_domain}_{uncertainty_context}/fixed_wrapped_train_dataset.pkl', 'wb') as f:
            pickle.dump(new_training_data, f)

