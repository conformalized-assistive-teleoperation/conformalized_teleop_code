import matplotlib.pyplot as plt
import numpy as np
import os


def plot_calibration_results(calibration_traj_idx_to_data,
                             remapping_model_filepath, remapping_model_architecture, config,
                             experiment_name, calib_user, list_of_coverages, list_of_interval_sizes,
                             individual_trial_results):
    print("Plotting calibration results")

    beta_uncertainty_threshold = config['beta_uncertainty_threshold']

    # iterate through the calibration trials
    if config['uncertainty_context'] == 'latent_prefs':
        episodes = np.arange(1, 4)
    elif config['uncertainty_context'] == 'control_prec':
        episodes = np.arange(1, 7)
    else:
        raise NotImplementedError

    for ep_index in episodes:
        # print(f"Episode index: {ep_index}")
        for traj_index in calibration_traj_idx_to_data[ep_index]:
            # print(f"Trajectory index: {traj_index}")
            calibration_traj_data = calibration_traj_idx_to_data[ep_index][traj_index]

            uncertainty_list = individual_trial_results[(ep_index, traj_index)]['uncertainty_list']
            dimension_to_bounds = individual_trial_results[(ep_index, traj_index)]['dimension_to_bounds']
            list_of_errors_over_time = individual_trial_results[(ep_index, traj_index)]['list_of_errors_over_time']
            list_of_alphas_over_time = individual_trial_results[(ep_index, traj_index)]['list_of_alphas_over_time']
            list_of_factor_lambda_over_time = individual_trial_results[(ep_index, traj_index)]['list_of_factor_lambda_over_time']

            fig = plt.figure(figsize=(10, 20))
            ax1 = fig.add_subplot(5, 1, 1)
            ax2 = fig.add_subplot(5, 1, 2)
            ax3 = fig.add_subplot(5, 1, 3)
            ax4 = fig.add_subplot(5, 1, 4)
            ax5 = fig.add_subplot(5, 1, 5)

            for timestep in range(len(calibration_traj_data)):
                # plot the human input state
                human_input_state, true_high_dim_action, rotated_z = calibration_traj_data[timestep]
                uncertainty_at_timestep = uncertainty_list[timestep]
                if uncertainty_at_timestep > beta_uncertainty_threshold:
                    ax5.scatter(human_input_state[0], human_input_state[1], c='r', s=10)
                else:
                    ax5.scatter(human_input_state[0], human_input_state[1], c='b', s=6)


            save_user_name = str((ep_index, traj_index))

            # plot list_of_alphas_over_time, list_of_factor_lambda_over_time, list_of_errors_over_time in one plot
            ax5.set_xlim(-1, 25)
            ax5.set_ylim(-25, 25)

            ax1.plot(list_of_alphas_over_time, label="alpha", color="#C44FB1", linewidth=3)
            ax1.set_title("Alpha over time")
            ax2.plot(list_of_factor_lambda_over_time, label="lambda", color="#7C4FC4", linewidth=3)
            ax2.set_title("Lambda over time")
            # list_of_errors_over_time should be offset by 1 because is 1 timestep shorter than the rest
            ax3.plot(np.arange(1, len(list_of_errors_over_time) + 1), list_of_errors_over_time, label="error", color="red")
            ax3.set_title("Error over time")

            # plot uncertainty over time
            # plot scatter plot of uncertainty at each timestep, color red if above threshold, green if below
            for t in range(len(uncertainty_list)):
                if uncertainty_list[t] > beta_uncertainty_threshold:
                    ax4.scatter(t + 1, uncertainty_list[t], c='r', marker='o', s=25, alpha=0.4)
                else:
                    ax4.scatter(t + 1, uncertainty_list[t], c='g', marker='o', s=12, alpha=0.4)

            ax4.plot(np.arange(1, len(uncertainty_list) + 1), uncertainty_list, label="uncertainty", color="purple")
            ax4.set_title("Uncertainty over time")

            # set title to be trajectory number and preference index
            fig.suptitle(f"Trajectory {save_user_name}")

            # create new folder called f"user-{calib_user}_plots"
            user_folder = f"results/{experiment_name}/plots/user-{calib_user}_plots"
            # create if it doesn't exist
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)

            # savefig to exp_results/{experiment_name}/aci_over_time/{traj_index}_{preference_type}.png
            plt.savefig(f"results/{experiment_name}/plots/user-{calib_user}_plots/uncertainty_trajid-{save_user_name}.png")

            # plt.show()
            plt.close()

            for dim in range(2):
                plt.figure(figsize=(40, 20))
                plt.subplot(2, 1, 1)

                # plt.subplot(2, 1, 1)
                plt.plot(dimension_to_bounds[dim]["true"], label="true", c="black", linewidth=5)
                # plot bounds for lb and ub in dotted lines
                plt.plot(dimension_to_bounds[dim]["lb"], label="lb", c="red", linestyle="--")
                plt.plot(dimension_to_bounds[dim]["ub"], label="ub", c="red", linestyle="--")
                # fill between lb and ub
                plt.fill_between(range(len(dimension_to_bounds[dim]["lb"])), dimension_to_bounds[dim]["lb"],
                                 dimension_to_bounds[dim]["ub"], alpha=0.25, color="red")
                plt.title("Uncalibrated Intervals")
                plt.legend()
                plt.subplot(2, 1, 2)
                plt.plot(dimension_to_bounds[dim]["true"], label="true", c="black", linewidth=5)
                # plot bounds for lb and ub in dotted lines
                plt.plot(dimension_to_bounds[dim]["lb_calib"], label="lb_calib", c="green", linestyle="--")
                plt.plot(dimension_to_bounds[dim]["ub_calib"], label="ub_calib", c="green", linestyle="--")
                # fill between lb and ub
                plt.fill_between(range(len(dimension_to_bounds[dim]["lb_calib"])), dimension_to_bounds[dim]["lb_calib"],
                                 dimension_to_bounds[dim]["ub_calib"], alpha=0.25, color="green")
                plt.title(f"Calibrated Intervals for traj {save_user_name}")
                plt.legend()


                # make directory within plots with the trajectory number
                # create folder results/{experiment_name}/plots/interval_trajid{traj_index}
                new_folder = f"results/{experiment_name}/plots/user-{calib_user}_plots/interval_trajid{save_user_name}"
                # create if it doesn't exist

                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)

                plt.savefig(
                    f"results/{experiment_name}/plots/user-{calib_user}_plots/interval_trajid{save_user_name}/trajid{save_user_name}_dim-{dim}.png")
                plt.close()