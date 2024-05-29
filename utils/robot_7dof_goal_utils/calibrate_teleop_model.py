import pdb

import kinpy as kp
import torch
import numpy as np
from scipy import stats
import json


def decimal_ceil(a, precision=2):
    return np.round(a + 0.5 * 10**(-precision), precision)

def calibrate_teleop_model_with_acqr(calibration_traj_idx_to_data,
                                  remapping_model_filepath, remapping_model_architecture, config, experiment_name, calib_user,
                                     to_plot=False):

    state_dim, action_dim, latent_dim, hidden_dim = config['state_dim'], config['action_dim'], config['latent_dim'], config['hidden_dim']
    # load teleoperation controller model
    remapping_model = remapping_model_architecture(action_dim, state_dim, latent_dim, hidden_dim)
    keys_from_filepath_remapping = torch.load(remapping_model_filepath)
    remapping_model.load_state_dict(keys_from_filepath_remapping)
    remapping_model.eval()

    beta_uncertainty_threshold = config['beta_uncertainty_threshold'] # this computed after the fact
    gamma = config['acqr_gamma']
    desired_alpha = config['desired_alpha']

    # create lists to store the results
    mean_prediction_error_at_uncertain_states, mean_prediction_error_at_certain_states = [], []
    list_of_coverages = []
    list_of_interval_sizes = []
    individual_trial_results = {}

    # iterate through the calibration trials
    for goal_object in calibration_traj_idx_to_data:
        for traj_index in calibration_traj_idx_to_data[goal_object]:
            # print(f"Trajectory index: {traj_index}")
            calibration_traj_data = calibration_traj_idx_to_data[goal_object][traj_index]

            # initialize all acqr parameters
            all_s_i_lambda = []
            current_alpha = 0.1
            current_quantile = 1 - current_alpha
            current_multiplicative_factor = 1

            # initialize lists to store the results
            list_of_errors_over_time = []
            list_of_alphas_over_time = [current_alpha]
            list_of_factor_lambda_over_time = [current_multiplicative_factor]
            uncertainty_list = []

            # initialize dictionary to store the bounds for each dimension
            dimension_to_bounds = {}
            for dim in range(7):
                dimension_to_bounds[dim] = {}
                dimension_to_bounds[dim]["lb"] = []
                dimension_to_bounds[dim]["ub"] = []
                dimension_to_bounds[dim]["true"] = []
                dimension_to_bounds[dim]["lb_calib"] = []
                dimension_to_bounds[dim]["ub_calib"] = []


            for timestep in range(0, len(calibration_traj_data)):
                human_input_state, predicted_action, next_state_applying_action, rotated_z = calibration_traj_data[
                    timestep]
                state = torch.FloatTensor([human_input_state[0:7]])
                true_high_dim_action = np.array(predicted_action[0:7])
                # if np.any(abs(true_high_dim_action - 2 * np.pi) < 0.2):
                #     idx_wrap = np.where(abs(true_high_dim_action - 2 * np.pi) < 0.2)
                #
                #     if true_high_dim_action[idx_wrap] > 0:
                #         true_high_dim_action[idx_wrap] = -2 * np.pi + true_high_dim_action[idx_wrap]
                #     else:
                #         true_high_dim_action[idx_wrap] = 2 * np.pi + true_high_dim_action[idx_wrap]
                #
                #     assert np.all(abs(true_high_dim_action - 2 * np.pi) > 0.2)

                true_high_dim_action = torch.FloatTensor([true_high_dim_action])

                rotated_z = torch.FloatTensor([rotated_z])
                # we want dx, dy, not dz: drop last dimension (dz) of rotated_z
                low_dim_action = rotated_z[:, :-1]

                # get the predicted action with quantiles
                predicted_action_with_quantiles = remapping_model.forward(state, low_dim_action)

                # split into 3
                split_action = torch.split(predicted_action_with_quantiles, action_dim, dim=1)
                action_quant_low = split_action[0]
                action_med = split_action[1]
                action_quant_high = split_action[2]

                # compute the uncalibrated error bounds
                lb = (action_quant_low.detach().numpy() - true_high_dim_action.detach().numpy())[0]
                ub = (true_high_dim_action.detach().numpy() - action_quant_high.detach().numpy())[0]

                # compute the deltas
                delta_hi = np.maximum(action_quant_high.detach().numpy() - action_med.detach().numpy(),
                                      np.array([0.001 for _ in range(action_dim)]))[0]
                delta_lo = np.maximum(action_med.detach().numpy() - action_quant_low.detach().numpy(),
                                      np.array([0.001 for _ in range(action_dim)]))[0]

                # compute the calibrated bounds, given current multiplicative factor
                calibrated_lb = action_med.detach().numpy() - current_multiplicative_factor * delta_lo
                calibrated_ub = action_med.detach().numpy() + current_multiplicative_factor * delta_hi

                # check current prediction error, and current uncertainty
                true_y = true_high_dim_action.detach().numpy()[0]
                uncertainty_at_timestep = np.linalg.norm(calibrated_ub[0] - calibrated_lb[0])
                uncertainty_list.append(uncertainty_at_timestep)
                prediction_error = torch.norm(action_med - true_high_dim_action, p=2, dim=1).detach().numpy()[0]

                # check if err_t is 0 or 1
                error_at_timestep = max([0, 1 if any([true_y[i] < calibrated_lb[0][i] or true_y[i] > calibrated_ub[0][i]
                                                      for i in range(len(true_y))]) else 0])
                list_of_coverages.append(1 - error_at_timestep)

                # compute the average interval size for each dimension
                avg_interval_size_for_dimension = []
                for dim in range(7):
                    dimension_to_bounds[dim]["lb_calib"].append(calibrated_lb[0][dim])
                    dimension_to_bounds[dim]["ub_calib"].append(calibrated_ub[0][dim])
                    dimension_to_bounds[dim]["true"].append(true_y[dim])
                    dimension_to_bounds[dim]["ub"].append(action_quant_high.detach().numpy()[0][dim])
                    dimension_to_bounds[dim]["lb"].append(action_quant_low.detach().numpy()[0][dim])
                    avg_interval_size_for_dimension.append(calibrated_ub[0][dim] - calibrated_lb[0][dim])

                avg_interval_size_for_dimension = np.mean(avg_interval_size_for_dimension)
                list_of_interval_sizes.append(avg_interval_size_for_dimension)

                # find smallest lambda such that true_y is in the set (smallest expansion factor)
                diff_between_mean_pred_and_true_upper = true_y - action_med.detach().numpy()[0]
                expansion_needed_on_upper = diff_between_mean_pred_and_true_upper / delta_hi

                diff_between_mean_pred_and_true_lower = action_med.detach().numpy()[0] - true_y
                expansion_needed_on_lower = diff_between_mean_pred_and_true_lower / delta_lo
                min_lambda = decimal_ceil(max(max(expansion_needed_on_upper), max(expansion_needed_on_lower)))
                all_s_i_lambda.append(min_lambda)

                # compute quantile of all_s_i_lambda
                all_s_i_lambda = np.array(all_s_i_lambda)
                all_s_i_lambda = list(np.sort(all_s_i_lambda))
                index = int((current_quantile * (1 + (1 / len(all_s_i_lambda)))) * len(all_s_i_lambda)) # do finite sample correction
                index = min(index, len(all_s_i_lambda) - 1)
                current_quantile_s = all_s_i_lambda[index] # multiplicative expansion we will use in t+1

                # update alpha_t
                current_multiplicative_factor = current_quantile_s
                current_alpha = current_alpha + gamma * (desired_alpha - error_at_timestep)
                # we run clipped ACQR, where the Q(>1) is max nonconformity value seen thus far, Q(<0) is the minimum.
                current_quantile = np.clip(1 - current_alpha, 0, 1)

                list_of_alphas_over_time.append(current_alpha)
                list_of_factor_lambda_over_time.append(current_multiplicative_factor)
                list_of_errors_over_time.append(error_at_timestep)

                if uncertainty_at_timestep > beta_uncertainty_threshold:
                    mean_prediction_error_at_uncertain_states.append(prediction_error)
                else:
                    mean_prediction_error_at_certain_states.append(prediction_error)

            individual_trial_results[(goal_object, traj_index)] = {}
            individual_trial_results[(goal_object, traj_index)]["uncertainty_list"] = uncertainty_list
            individual_trial_results[(goal_object, traj_index)]["list_of_errors_over_time"] = list_of_errors_over_time
            individual_trial_results[(goal_object, traj_index)]["list_of_alphas_over_time"] = list_of_alphas_over_time
            individual_trial_results[(goal_object, traj_index)]["list_of_factor_lambda_over_time"] = list_of_factor_lambda_over_time
            individual_trial_results[(goal_object, traj_index)]["dimension_to_bounds"] = dimension_to_bounds


    print("Prediction error in uncertain states: ", np.mean(mean_prediction_error_at_uncertain_states))
    print("Prediction error in certain states: ", np.mean(mean_prediction_error_at_certain_states))
    print("Std dev in uncertain states: ", np.std(mean_prediction_error_at_uncertain_states))
    print("Std dev in certain states: ", np.std(mean_prediction_error_at_certain_states))

    print("T-test result: ",
          stats.ttest_ind(mean_prediction_error_at_uncertain_states, mean_prediction_error_at_certain_states))


    # print list_of_coverages
    print("Coverage: ", np.mean(list_of_coverages))
    print("Mean interval size: ", np.mean(list_of_interval_sizes))
    print("STD interval size: ", np.std(list_of_interval_sizes))

    # save the results to a json file in the experiment_name folder

    results = {}
    results['coverage'] = np.mean(list_of_coverages)
    results['mean_interval_size'] = np.mean(list_of_interval_sizes)
    results['std_interval_size'] = np.std(list_of_interval_sizes)
    results['mean_prediction_error_at_uncertain_states'] = np.mean(mean_prediction_error_at_uncertain_states)
    results['mean_prediction_error_at_certain_states'] = np.mean(mean_prediction_error_at_certain_states)
    results['std_prediction_error_at_uncertain_states'] = np.std(mean_prediction_error_at_uncertain_states)
    results['std_prediction_error_at_certain_states'] = np.std(mean_prediction_error_at_certain_states)
    results['ttest_result'] = stats.ttest_ind(mean_prediction_error_at_uncertain_states, mean_prediction_error_at_certain_states)
    # convert all values in results to strings
    results = {k: str(v) for k, v in results.items()}
    with open(f"results/{experiment_name}/acqr_calibration_results_{calib_user}.json", 'w') as f:
        json.dump(results, f)

    if to_plot:
        from utils.robot_7dof_goal_utils.plotting_calibration import plot_calibration_results
        plot_calibration_results(calibration_traj_idx_to_data,
                                 remapping_model_filepath, remapping_model_architecture, config,
                                 experiment_name, calib_user, list_of_coverages, list_of_interval_sizes,
                                 individual_trial_results)

    return

def calibrate_teleop_model_with_qr(calibration_traj_idx_to_data,
                                  remapping_model_filepath, remapping_model_architecture, config, experiment_name, calib_user,
                                   to_plot=False):

    state_dim, action_dim, latent_dim, hidden_dim = config['state_dim'], config['action_dim'], config['latent_dim'], config['hidden_dim']
    # load teleoperation controller model
    remapping_model = remapping_model_architecture(action_dim, state_dim, latent_dim, hidden_dim)
    keys_from_filepath_remapping = torch.load(remapping_model_filepath)
    remapping_model.load_state_dict(keys_from_filepath_remapping)
    remapping_model.eval()

    # create lists to store the results
    list_of_coverages = []
    list_of_interval_sizes = []

    # iterate through the calibration trials
    for goal_object in calibration_traj_idx_to_data:
        for traj_index in calibration_traj_idx_to_data[goal_object]:
            # print(f"Trajectory index: {traj_index}")
            calibration_traj_data = calibration_traj_idx_to_data[goal_object][traj_index]

            # initialize all qr parameters
            current_multiplicative_factor = 1

            # initialize lists to store the results
            uncertainty_list = []

            # initialize dictionary to store the bounds for each dimension
            dimension_to_bounds = {}
            for dim in range(7):
                dimension_to_bounds[dim] = {}
                dimension_to_bounds[dim]["lb"] = []
                dimension_to_bounds[dim]["ub"] = []
                dimension_to_bounds[dim]["true"] = []
                dimension_to_bounds[dim]["lb_calib"] = []
                dimension_to_bounds[dim]["ub_calib"] = []


            for timestep in range(0, len(calibration_traj_data)):
                human_input_state, predicted_action, next_state_applying_action, rotated_z = calibration_traj_data[
                    timestep]
                state = torch.FloatTensor([human_input_state[0:7]])
                true_high_dim_action = np.array(predicted_action[0:7])
                # if np.any(abs(true_high_dim_action - 2 * np.pi) < 0.2):
                #     idx_wrap = np.where(abs(true_high_dim_action - 2 * np.pi) < 0.2)
                #
                #     if true_high_dim_action[idx_wrap] > 0:
                #         true_high_dim_action[idx_wrap] = -2 * np.pi + true_high_dim_action[idx_wrap]
                #     else:
                #         true_high_dim_action[idx_wrap] = 2 * np.pi + true_high_dim_action[idx_wrap]
                #
                #     assert np.all(abs(true_high_dim_action - 2 * np.pi) > 0.2)

                true_high_dim_action = torch.FloatTensor([true_high_dim_action])
                rotated_z = torch.FloatTensor([rotated_z])
                # we want dx, dy, not dz: drop last dimension (dz) of rotated_z
                low_dim_action = rotated_z[:, :-1]

                # get the predicted action with quantiles
                predicted_action_with_quantiles = remapping_model.forward(state, low_dim_action)

                # split into 3
                split_action = torch.split(predicted_action_with_quantiles, action_dim, dim=1)
                action_quant_low = split_action[0]
                action_med = split_action[1]
                action_quant_high = split_action[2]

                # compute the error bounds
                lb = (action_quant_low.detach().numpy() - true_high_dim_action.detach().numpy())[0]
                ub = (true_high_dim_action.detach().numpy() - action_quant_high.detach().numpy())[0]

                # compute the deltas
                delta_hi = np.maximum(action_quant_high.detach().numpy() - action_med.detach().numpy(),
                                      np.array([0.001 for _ in range(action_dim)]))[0]
                delta_lo = np.maximum(action_med.detach().numpy() - action_quant_low.detach().numpy(),
                                      np.array([0.001 for _ in range(action_dim)]))[0]

                # compute the "calibrated" bounds (there is no calibration here, as current_multiplicative_factor=1)
                calibrated_lb = action_med.detach().numpy() - current_multiplicative_factor * delta_lo
                calibrated_ub = action_med.detach().numpy() + current_multiplicative_factor * delta_hi

                # check current prediction error, and current uncertainty
                true_y = true_high_dim_action.detach().numpy()[0]
                uncertainty_at_timestep = np.linalg.norm(calibrated_ub[0] - calibrated_lb[0])
                uncertainty_list.append(uncertainty_at_timestep)

                # check if err_t is 0 or 1
                error_at_timestep = max([0, 1 if any([true_y[i] < calibrated_lb[0][i] or true_y[i] > calibrated_ub[0][i] for i in range(len(true_y))]) else 0])
                list_of_coverages.append(1 - error_at_timestep)

                # compute the average interval size for each dimension
                avg_interval_size_for_dimension = []
                for dim in range(7):
                    avg_interval_size_for_dimension.append(calibrated_ub[0][dim] - calibrated_lb[0][dim])

                avg_interval_size_for_dimension = np.mean(avg_interval_size_for_dimension)
                list_of_interval_sizes.append(avg_interval_size_for_dimension)

    # print list_of_coverages
    print("Coverage: ", np.mean(list_of_coverages))
    print("Mean interval size: ", np.mean(list_of_interval_sizes))
    print("STD interval size: ", np.std(list_of_interval_sizes))

    # save the results to a json file in the experiment_name folder
    results = {}
    results['coverage'] = np.mean(list_of_coverages)
    results['mean_interval_size'] = np.mean(list_of_interval_sizes)
    results['std_interval_size'] = np.std(list_of_interval_sizes)
    # convert all values in results to strings
    results = {k: str(v) for k, v in results.items()}
    with open(f"results/{experiment_name}/qr_calibration_results_{calib_user}.json", 'w') as f:
        json.dump(results, f)

    return



def calibrate_teleop_model_with_ensemble(calibration_traj_idx_to_data,
                                  list_remapping_model_filepath, remapping_model_architecture, config, experiment_name, calib_user,
                                         to_plot=False):

    # load trained ensemble of remapping models
    list_of_remapping_models = []
    for i in range(5):
        remapping_model_filepath = list_remapping_model_filepath[i]
        remapping_model = remapping_model_architecture(config['action_dim'], config['state_dim'],
                                                       config['latent_dim'], config['hidden_dim'])
        keys_from_filepath_remapping = torch.load(remapping_model_filepath)
        remapping_model.load_state_dict(keys_from_filepath_remapping)
        remapping_model.eval()
        list_of_remapping_models.append(remapping_model)

    # create lists to store the results
    coverage_list = []
    list_of_interval_sizes = []

    for goal_object in calibration_traj_idx_to_data:
        for traj_index in calibration_traj_idx_to_data[goal_object]:
            # print(f"Trajectory index: {traj_index}")
            calibration_traj_data = calibration_traj_idx_to_data[goal_object][traj_index]
            uncertainty_list = []

            for timestep in range(0, len(calibration_traj_data)):
                human_input_state, predicted_action, _, rotated_z = calibration_traj_data[timestep]
                human_input_state = torch.FloatTensor([human_input_state[0:7]])
                # predicted_action = torch.FloatTensor([predicted_action[0:7]])
                rotated_z = torch.FloatTensor([rotated_z])
                # drop last dimension 2 of rotated_z
                rotated_z = rotated_z[:, :-1]

                state = human_input_state
                low_dim_action = rotated_z
                true_high_dim_action = np.array(predicted_action[0:7])
                # if np.any(abs(true_high_dim_action - 2 * np.pi) < 0.2):
                #     idx_wrap = np.where(abs(true_high_dim_action - 2 * np.pi) < 0.2)
                #
                #     if true_high_dim_action[idx_wrap] > 0:
                #         true_high_dim_action[idx_wrap] = -2 * np.pi + true_high_dim_action[idx_wrap]
                #     else:
                #         true_high_dim_action[idx_wrap] = 2 * np.pi + true_high_dim_action[idx_wrap]
                #
                #     assert np.all(abs(true_high_dim_action - 2 * np.pi) > 0.2)

                true_high_dim_action = torch.FloatTensor([true_high_dim_action])
                list_of_prediction_mean_vars = []
                list_of_means = []
                list_of_vars = []
                for i in range(5):
                    remapping_model = list_of_remapping_models[i]
                    predicted_action_with_quantiles = remapping_model.forward(state, low_dim_action)

                    # split into 3
                    split_action = torch.split(predicted_action_with_quantiles, config['action_dim'], dim=1)
                    # take the middle part
                    action_med = split_action[0].detach().numpy()[0]
                    action_var = split_action[1].detach().numpy()[0]

                    list_of_prediction_mean_vars.append((action_med, action_var))
                    list_of_means.append(action_med)
                    list_of_vars.append(action_var)

                    # print("action_med", action_med)

                # compute the combined variance and means of the gaussians
                combined_mean = np.mean(list_of_means, axis=0)
                combined_var = np.mean(
                    [list_of_means[i] ** 2 + list_of_vars[i] for i in range(len(list_of_means))]) - combined_mean ** 2

                # compute one std deviation interval
                interval_size = 1 * np.sqrt(combined_var)
                # print("Interval size: ", combined_var)
                # take average over dimensions
                list_of_interval_sizes.append(np.mean(interval_size))

                # check if 2 times the variance contains the true action
                covered = 0
                true_high_dim_action = true_high_dim_action.detach().numpy()[0]
                if np.all(combined_mean - 1 * np.sqrt(combined_var) <= true_high_dim_action) \
                        and np.all(true_high_dim_action <= combined_mean + 1 * np.sqrt(combined_var)):
                    # then we are certain
                    covered = 1
                    uncertainty_list.append(0)
                    coverage_list.append(1)
                else:
                    uncertainty_list.append(1)
                    coverage_list.append(0)


    # print the coverage
    print("Coverage: ", np.mean(coverage_list))
    print("Mean Interval Size: ", np.mean(list_of_interval_sizes))
    print("Std Interval Size: ", np.std(list_of_interval_sizes))

    # save the results to a json file in the experiment_name folder
    results = {}
    results['coverage'] = np.mean(coverage_list)
    results['mean_interval_size'] = np.mean(list_of_interval_sizes)
    results['std_interval_size'] = np.std(list_of_interval_sizes)
    # convert all values in results to strings
    results = {k: str(v) for k, v in results.items()}
    with open(f"results/{experiment_name}/ensemble_calibration_results_{calib_user}.json", 'w') as f:
        json.dump(results, f)

    return coverage_list



def calibrate_teleop_model_with_acqr_w_human_data(calibration_traj_idx_to_data,
                                  remapping_model_filepath, remapping_model_architecture, config, experiment_name, calib_user, task_color,
                                     to_plot=False):

    state_dim, action_dim, latent_dim, hidden_dim = config['state_dim'], config['action_dim'], config['latent_dim'], config['hidden_dim']
    # load teleoperation controller model
    remapping_model = remapping_model_architecture(action_dim, state_dim, latent_dim, hidden_dim)
    keys_from_filepath_remapping = torch.load(remapping_model_filepath)
    remapping_model.load_state_dict(keys_from_filepath_remapping)
    remapping_model.eval()

    beta_uncertainty_threshold = config['beta_uncertainty_threshold'] # this computed after the fact
    gamma = config['acqr_gamma']
    desired_alpha = config['desired_alpha']

    # create lists to store the results
    mean_prediction_error_at_uncertain_states, mean_prediction_error_at_certain_states = [], []
    list_of_coverages = []
    list_of_interval_sizes = []
    individual_trial_results = {}

    # iterate through the timesteps
    # print(f"Trajectory index: {traj_index}")
    calibration_traj_data = calibration_traj_idx_to_data

    # initialize all acqr parameters
    all_s_i_lambda = []
    current_alpha = 0.1
    current_quantile = 1 - current_alpha
    current_multiplicative_factor = 1

    # initialize lists to store the results
    list_of_errors_over_time = []
    list_of_alphas_over_time = [current_alpha]
    list_of_factor_lambda_over_time = [current_multiplicative_factor]
    uncertainty_list = []

    # initialize dictionary to store the bounds for each dimension
    dimension_to_bounds = {}
    for dim in range(7):
        dimension_to_bounds[dim] = {}
        dimension_to_bounds[dim]["lb"] = []
        dimension_to_bounds[dim]["ub"] = []
        dimension_to_bounds[dim]["true"] = []
        dimension_to_bounds[dim]["lb_calib"] = []
        dimension_to_bounds[dim]["ub_calib"] = []

    # resave
    # resave_file = []
    # import matplotlib.pyplot as plt
    # plt.figure()
    for timestep in range(0, len(calibration_traj_data)):
        human_input_state, predicted_action, rotated_z = calibration_traj_data[
            timestep]
        # pdb.set_trace()
        state = human_input_state
        true_high_dim_action = np.array(predicted_action[0:7])
        # if np.any(abs(true_high_dim_action - 2 * np.pi) < 0.2):
        #     idx_wrap = np.where(abs(true_high_dim_action - 2 * np.pi) < 0.2)
        #
        #     if true_high_dim_action[idx_wrap] > 0:
        #         true_high_dim_action[idx_wrap] = -2 * np.pi + true_high_dim_action[idx_wrap]
        #     else:
        #         true_high_dim_action[idx_wrap] = 2 * np.pi + true_high_dim_action[idx_wrap]
        #
        #     assert np.all(abs(true_high_dim_action - 2 * np.pi) > 0.2)

        # swap rotated_z
    #     Lx,Ly = rotated_z
    # #     Lx, Ly = -Ly, Lx
    # #     rotated_z = [Lx, Ly]
    # #     resave_file.append([state, true_high_dim_action, rotated_z])
    #     plt.scatter(Lx, Ly, color=task_color)
    # plt.show()
    # # save resave file
    # import pickle
    # with open(f"data/resave_{calib_user}_{task_color}.pkl", 'wb') as f:
    #     pickle.dump(resave_file, f)

    # if False:
        true_high_dim_action = torch.FloatTensor([true_high_dim_action])

        rotated_z = torch.FloatTensor([rotated_z])
        # we want dx, dy, not dz: drop last dimension (dz) of rotated_z
        low_dim_action = rotated_z

        # get the predicted action with quantiles
        predicted_action_with_quantiles = remapping_model.forward(state, low_dim_action)

        # split into 3
        split_action = torch.split(predicted_action_with_quantiles, action_dim, dim=1)
        action_quant_low = split_action[0]
        action_med = split_action[1]
        action_quant_high = split_action[2]

        # compute the uncalibrated error bounds
        lb = (action_quant_low.detach().numpy() - true_high_dim_action.detach().numpy())[0]
        ub = (true_high_dim_action.detach().numpy() - action_quant_high.detach().numpy())[0]

        # compute the deltas
        delta_hi = np.maximum(action_quant_high.detach().numpy() - action_med.detach().numpy(),
                              np.array([0.001 for _ in range(action_dim)]))[0]
        delta_lo = np.maximum(action_med.detach().numpy() - action_quant_low.detach().numpy(),
                              np.array([0.001 for _ in range(action_dim)]))[0]

        # compute the calibrated bounds, given current multiplicative factor
        calibrated_lb = action_med.detach().numpy() - current_multiplicative_factor * delta_lo
        calibrated_ub = action_med.detach().numpy() + current_multiplicative_factor * delta_hi

        # check current prediction error, and current uncertainty
        true_y = true_high_dim_action.detach().numpy()[0]
        uncertainty_at_timestep = np.linalg.norm(calibrated_ub[0] - calibrated_lb[0])
        uncertainty_list.append(uncertainty_at_timestep)
        prediction_error = torch.norm(action_med - true_high_dim_action, p=2, dim=1).detach().numpy()[0]

        # check if err_t is 0 or 1
        error_at_timestep = max([0, 1 if any([true_y[i] < calibrated_lb[0][i] or true_y[i] > calibrated_ub[0][i]
                                              for i in range(len(true_y))]) else 0])
        list_of_coverages.append(1 - error_at_timestep)

        # compute the average interval size for each dimension
        avg_interval_size_for_dimension = []
        for dim in range(7):
            dimension_to_bounds[dim]["lb_calib"].append(calibrated_lb[0][dim])
            dimension_to_bounds[dim]["ub_calib"].append(calibrated_ub[0][dim])
            dimension_to_bounds[dim]["true"].append(true_y[dim])
            dimension_to_bounds[dim]["ub"].append(action_quant_high.detach().numpy()[0][dim])
            dimension_to_bounds[dim]["lb"].append(action_quant_low.detach().numpy()[0][dim])
            avg_interval_size_for_dimension.append(calibrated_ub[0][dim] - calibrated_lb[0][dim])

        avg_interval_size_for_dimension = np.mean(avg_interval_size_for_dimension)
        list_of_interval_sizes.append(avg_interval_size_for_dimension)

        # find smallest lambda such that true_y is in the set (smallest expansion factor)
        diff_between_mean_pred_and_true_upper = true_y - action_med.detach().numpy()[0]
        expansion_needed_on_upper = diff_between_mean_pred_and_true_upper / delta_hi

        diff_between_mean_pred_and_true_lower = action_med.detach().numpy()[0] - true_y
        expansion_needed_on_lower = diff_between_mean_pred_and_true_lower / delta_lo
        min_lambda = decimal_ceil(max(max(expansion_needed_on_upper), max(expansion_needed_on_lower)))
        all_s_i_lambda.append(min_lambda)

        # compute quantile of all_s_i_lambda
        all_s_i_lambda = np.array(all_s_i_lambda)
        all_s_i_lambda = list(np.sort(all_s_i_lambda))
        index = int((current_quantile * (1 + (1 / len(all_s_i_lambda)))) * len(all_s_i_lambda)) # do finite sample correction
        index = min(index, len(all_s_i_lambda) - 1)
        current_quantile_s = all_s_i_lambda[index] # multiplicative expansion we will use in t+1

        # update alpha_t
        current_multiplicative_factor = current_quantile_s
        current_alpha = current_alpha + gamma * (desired_alpha - error_at_timestep)
        # we run clipped ACQR, where the Q(>1) is max nonconformity value seen thus far, Q(<0) is the minimum.
        current_quantile = np.clip(1 - current_alpha, 0, 1)

        list_of_alphas_over_time.append(current_alpha)
        list_of_factor_lambda_over_time.append(current_multiplicative_factor)
        list_of_errors_over_time.append(error_at_timestep)

        if uncertainty_at_timestep > beta_uncertainty_threshold:
            mean_prediction_error_at_uncertain_states.append(prediction_error)
        else:
            mean_prediction_error_at_certain_states.append(prediction_error)

    individual_trial_results['uncertainty_list'] = uncertainty_list
    individual_trial_results['list_of_errors_over_time'] = list_of_errors_over_time
    individual_trial_results['list_of_alphas_over_time'] = list_of_alphas_over_time
    individual_trial_results['list_of_factor_lambda_over_time'] = list_of_factor_lambda_over_time
    individual_trial_results['dimension_to_bounds'] = dimension_to_bounds

    print("Prediction error in uncertain states: ", np.mean(mean_prediction_error_at_uncertain_states))
    print("Prediction error in certain states: ", np.mean(mean_prediction_error_at_certain_states))
    print("Std dev in uncertain states: ", np.std(mean_prediction_error_at_uncertain_states))
    print("Std dev in certain states: ", np.std(mean_prediction_error_at_certain_states))

    print("T-test result: ",
          stats.ttest_ind(mean_prediction_error_at_uncertain_states, mean_prediction_error_at_certain_states))


    # print list_of_coverages
    print("Coverage: ", np.mean(list_of_coverages))
    print("Mean interval size: ", np.mean(list_of_interval_sizes))
    print("STD interval size: ", np.std(list_of_interval_sizes))

    # save the results to a json file in the experiment_name folder

    results = {}
    results['coverage'] = np.mean(list_of_coverages)
    results['mean_interval_size'] = np.mean(list_of_interval_sizes)
    results['std_interval_size'] = np.std(list_of_interval_sizes)
    results['mean_prediction_error_at_uncertain_states'] = np.mean(mean_prediction_error_at_uncertain_states)
    results['mean_prediction_error_at_certain_states'] = np.mean(mean_prediction_error_at_certain_states)
    results['std_prediction_error_at_uncertain_states'] = np.std(mean_prediction_error_at_uncertain_states)
    results['std_prediction_error_at_certain_states'] = np.std(mean_prediction_error_at_certain_states)
    results['ttest_result'] = stats.ttest_ind(mean_prediction_error_at_uncertain_states, mean_prediction_error_at_certain_states)
    # convert all values in results to strings
    results = {k: str(v) for k, v in results.items()}
    with open(f"results/{experiment_name}/acqr_calibration_results_{calib_user}_{task_color}.json", 'w') as f:
        json.dump(results, f)

    if to_plot:
        from utils.robot_7dof_goal_utils.plotting_calibration import plot_calibration_results_w_human_data
        plot_calibration_results_w_human_data(calibration_traj_idx_to_data,
                                 remapping_model_filepath, remapping_model_architecture, config,
                                 experiment_name, calib_user + task_color, list_of_coverages, list_of_interval_sizes,
                                 individual_trial_results)

    return

def calibrate_teleop_model_with_qr_w_human_data(calibration_traj_idx_to_data,
                                  remapping_model_filepath, remapping_model_architecture, config, experiment_name, calib_user, task_color,
                                   to_plot=False):

    state_dim, action_dim, latent_dim, hidden_dim = config['state_dim'], config['action_dim'], config['latent_dim'], config['hidden_dim']
    # load teleoperation controller model
    remapping_model = remapping_model_architecture(action_dim, state_dim, latent_dim, hidden_dim)
    keys_from_filepath_remapping = torch.load(remapping_model_filepath)
    remapping_model.load_state_dict(keys_from_filepath_remapping)
    remapping_model.eval()

    # create lists to store the results
    list_of_coverages = []
    list_of_interval_sizes = []

    # iterate through the calibration trials
    # print(f"Trajectory index: {traj_index}")
    calibration_traj_data = calibration_traj_idx_to_data

    # initialize all qr parameters
    current_multiplicative_factor = 1

    # initialize lists to store the results
    uncertainty_list = []

    # initialize dictionary to store the bounds for each dimension
    dimension_to_bounds = {}
    for dim in range(7):
        dimension_to_bounds[dim] = {}
        dimension_to_bounds[dim]["lb"] = []
        dimension_to_bounds[dim]["ub"] = []
        dimension_to_bounds[dim]["true"] = []
        dimension_to_bounds[dim]["lb_calib"] = []
        dimension_to_bounds[dim]["ub_calib"] = []


    for timestep in range(0, len(calibration_traj_data)):
        human_input_state, predicted_action, rotated_z = calibration_traj_data[
            timestep]
        state = human_input_state
        true_high_dim_action = np.array(predicted_action[0:7])
        # if np.any(abs(true_high_dim_action - 2 * np.pi) < 0.2):
        #     idx_wrap = np.where(abs(true_high_dim_action - 2 * np.pi) < 0.2)
        #
        #     if true_high_dim_action[idx_wrap] > 0:
        #         true_high_dim_action[idx_wrap] = -2 * np.pi + true_high_dim_action[idx_wrap]
        #     else:
        #         true_high_dim_action[idx_wrap] = 2 * np.pi + true_high_dim_action[idx_wrap]
        #
        #     assert np.all(abs(true_high_dim_action - 2 * np.pi) > 0.2)

        true_high_dim_action = torch.FloatTensor([true_high_dim_action])
        rotated_z = torch.FloatTensor([rotated_z])
        # we want dx, dy, not dz: drop last dimension (dz) of rotated_z
        low_dim_action = rotated_z

        # get the predicted action with quantiles
        predicted_action_with_quantiles = remapping_model.forward(state, low_dim_action)

        # split into 3
        split_action = torch.split(predicted_action_with_quantiles, action_dim, dim=1)
        action_quant_low = split_action[0]
        action_med = split_action[1]
        action_quant_high = split_action[2]

        # compute the error bounds
        lb = (action_quant_low.detach().numpy() - true_high_dim_action.detach().numpy())[0]
        ub = (true_high_dim_action.detach().numpy() - action_quant_high.detach().numpy())[0]

        # compute the deltas
        delta_hi = np.maximum(action_quant_high.detach().numpy() - action_med.detach().numpy(),
                              np.array([0.001 for _ in range(action_dim)]))[0]
        delta_lo = np.maximum(action_med.detach().numpy() - action_quant_low.detach().numpy(),
                              np.array([0.001 for _ in range(action_dim)]))[0]

        # compute the "calibrated" bounds (there is no calibration here, as current_multiplicative_factor=1)
        calibrated_lb = action_med.detach().numpy() - current_multiplicative_factor * delta_lo
        calibrated_ub = action_med.detach().numpy() + current_multiplicative_factor * delta_hi

        # check current prediction error, and current uncertainty
        true_y = true_high_dim_action.detach().numpy()[0]
        uncertainty_at_timestep = np.linalg.norm(calibrated_ub[0] - calibrated_lb[0])
        uncertainty_list.append(uncertainty_at_timestep)

        # check if err_t is 0 or 1
        error_at_timestep = max([0, 1 if any([true_y[i] < calibrated_lb[0][i] or true_y[i] > calibrated_ub[0][i] for i in range(len(true_y))]) else 0])
        list_of_coverages.append(1 - error_at_timestep)

        # compute the average interval size for each dimension
        avg_interval_size_for_dimension = []
        for dim in range(7):
            avg_interval_size_for_dimension.append(calibrated_ub[0][dim] - calibrated_lb[0][dim])

        avg_interval_size_for_dimension = np.mean(avg_interval_size_for_dimension)
        list_of_interval_sizes.append(avg_interval_size_for_dimension)

    # print list_of_coverages
    print("Coverage: ", np.mean(list_of_coverages))
    print("Mean interval size: ", np.mean(list_of_interval_sizes))
    print("STD interval size: ", np.std(list_of_interval_sizes))

    # save the results to a json file in the experiment_name folder
    results = {}
    results['coverage'] = np.mean(list_of_coverages)
    results['mean_interval_size'] = np.mean(list_of_interval_sizes)
    results['std_interval_size'] = np.std(list_of_interval_sizes)
    # convert all values in results to strings
    results = {k: str(v) for k, v in results.items()}
    with open(f"results/{experiment_name}/qr_calibration_results_{calib_user}_{task_color}.json", 'w') as f:
        json.dump(results, f)

    return


