import pdb

import matplotlib.pyplot as plt
import numpy as np
import os
import kinpy as kp


def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def convert_to_fk_dict(joint_angles_test):
    joint_names = ['gen3_joint_1', 'gen3_joint_2', 'gen3_joint_3', 'gen3_joint_4', 'gen3_joint_5',
                   'gen3_joint_6', 'gen3_joint_7', 'gen3_robotiq_85_left_knuckle_joint',
                   'gen3_robotiq_85_left_finger_tip_joint', 'gen3_robotiq_85_right_knuckle_joint',
                   'gen3_robotiq_85_right_finger_tip_joint', 'gen3_robotiq_85_left_inner_knuckle_joint',
                   'gen3_robotiq_85_right_inner_knuckle_joint']

    name_to_rad = {}
    for i in range(len(joint_names)):
        name_to_rad[joint_names[i]] = joint_angles_test[i]
    return name_to_rad

def plot_calibration_results(calibration_traj_idx_to_data,
                             remapping_model_filepath, remapping_model_architecture, config,
                             experiment_name, calib_user, list_of_coverages, list_of_interval_sizes,
                             individual_trial_results):
    print("Plotting calibration results")

    chain = kp.build_chain_from_urdf(open("utils/urdf/gen3_2f85.urdf").read())
    joint_names = chain.get_joint_parameter_names()
    joint_angle_keys = ['position_0', 'position_1', 'position_2',
                        'position_3', 'position_4', 'position_5', 'position_6', 'position_7',
                        'position_8', 'position_9', 'position_10', 'position_11', 'position_12']

    joint_vel_keys = ['velocity_0', 'velocity_1', 'velocity_2', 'velocity_3', 'velocity_4',
                      'velocity_5', 'velocity_6', 'velocity_7', 'velocity_8', 'velocity_9',
                      'velocity_10', 'velocity_11', 'velocity_12']
    gripper_angles = [0.0069869209377008,
                      0.0069869209377008,
                      -0.0069869209377008,
                      0.0069869209377008,
                      0.0069869209377008,
                      -0.0069869209377008]

    beta_uncertainty_threshold = config['beta_uncertainty_threshold']

    # iterate through the calibration trials
    if config['uncertainty_context'] == 'latent_prefs':
        episodes = np.arange(1, 4)
    elif config['uncertainty_context'] == 'control_prec':
        episodes = np.arange(1, 7)
    else:
        raise NotImplementedError

    # iterate through the calibration trials
    for goal_object in calibration_traj_idx_to_data:
        for traj_index in calibration_traj_idx_to_data[goal_object]:
            print(f"Trajectory index: {traj_index}")
            calibration_traj_data = calibration_traj_idx_to_data[goal_object][traj_index]

            uncertainty_list = individual_trial_results[(goal_object, traj_index)]['uncertainty_list']
            dimension_to_bounds = individual_trial_results[(goal_object, traj_index)]['dimension_to_bounds']
            list_of_errors_over_time = individual_trial_results[(goal_object, traj_index)]['list_of_errors_over_time']
            list_of_alphas_over_time = individual_trial_results[(goal_object, traj_index)]['list_of_alphas_over_time']
            list_of_factor_lambda_over_time = individual_trial_results[(goal_object, traj_index)]['list_of_factor_lambda_over_time']

            fig = plt.figure(figsize=(10, 20))
            ax1 = fig.add_subplot(5, 1, 1)
            ax2 = fig.add_subplot(5, 1, 2)
            ax3 = fig.add_subplot(5, 1, 3)
            ax4 = fig.add_subplot(5, 1, 4)
            ax5 = fig.add_subplot(5, 1, 5, projection='3d')

            for timestep in range(len(calibration_traj_data)):
                # plot the human input state
                human_input_state, predicted_action, next_state_applying_action, rotated_z = calibration_traj_data[
                    timestep]

                full_state = list(human_input_state) + gripper_angles
                pred_pos = chain.forward_kinematics(convert_to_fk_dict(full_state))['gen3_end_effector_link'].pos
                x, y, z = pred_pos[0], pred_pos[1], pred_pos[2]
                uncertainty_at_timestep = uncertainty_list[timestep]
                if uncertainty_at_timestep > beta_uncertainty_threshold:
                    ax5.scatter(x, y, z, c='#E07800', marker='o', s=25, alpha=0.8)
                else:
                    ax5.scatter(x, y, z, c='#5E5E5E', marker='o', s=12, alpha=0.4)

            save_user_name = str((goal_object, traj_index))

            # plot list_of_alphas_over_time, list_of_factor_lambda_over_time, list_of_errors_over_time in one plot
            ax1.plot(list_of_alphas_over_time, label="alpha", color="#C44FB1", linewidth=3)
            ax1.set_title("Alpha over time")
            # ax1.set_xlim(0, 1200)
            # ax1.set_ylim(0, 0.5)

            ax2.plot(list_of_factor_lambda_over_time, label="lambda", color="#7C4FC4", linewidth=3)
            ax2.set_title("Lambda over time")
            # ax2.set_xlim(0, 1200)
            # ax2.set_ylim(0, 30)
            # list_of_errors_over_time should be offset by 1 because is 1 timestep shorter than the rest
            ax3.plot(np.arange(1, len(list_of_errors_over_time) + 1), list_of_errors_over_time, label="error", color="red")
            ax3.set_title("Error over time")

            # plot uncertainty over time
            # plot scatter plot of uncertainty at each timestep, color red if above threshold, green if below
            for t in range(len(uncertainty_list)):
                if uncertainty_list[t] > beta_uncertainty_threshold:
                    ax4.scatter(t + 1, uncertainty_list[t], c='#E07800', marker='o', s=25, alpha=0.7)
                    # ax1.scatter(t, list_of_factor_lambda_over_time[t], c='#E07800', marker='o', s=100, alpha=0.7)
                    # ax2.scatter(t, list_of_factor_lambda_over_time[t], c='#E07800', marker='o', s=100, alpha=0.7)
                else:
                    ax4.scatter(t + 1, uncertainty_list[t], c='#5E5E5E', marker='o', s=12, alpha=0.5)

            ax4.plot(np.arange(1, len(uncertainty_list) + 1), uncertainty_list, label="uncertainty", color="purple")
            ax4.set_title("Uncertainty over time")

            Xc, Yc, Zc = data_for_cylinder_along_z(0.70980445 + 0.1, 0.25232062, 0.03, 0.3)
            ax5.plot_surface(Xc, Yc, Zc, color='cyan', alpha=0.3)
            Xc, Yc, Zc = data_for_cylinder_along_z(0.70980445 + 0.1, -0.24289359, 0.03, 0.3)
            ax5.plot_surface(Xc, Yc, Zc, color='purple', alpha=0.3)
            ax5.set_xlim3d(0.4, 0.85)
            ax5.set_ylim3d(-0.4, 0.2)
            ax5.set_zlim3d(0.0, 0.55)
            ax5.set_xlabel('x')
            ax5.set_ylabel('y')
            ax5.set_zlabel('z')

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

            for dim in range(config['action_dim']):
                plt.figure(figsize=(40, 20))
                plt.subplot(2, 1, 1)

                plt.plot(dimension_to_bounds[dim]["lb_calib"], label="lb_calib", c="#f4b400", linestyle="--")
                plt.plot(dimension_to_bounds[dim]["ub_calib"], label="ub_calib", c="#f4b400", linestyle="--")
                # fill between lb and ub
                plt.fill_between(range(len(dimension_to_bounds[dim]["lb_calib"])), dimension_to_bounds[dim]["lb_calib"],
                                 dimension_to_bounds[dim]["ub_calib"], alpha=0.25, color="#f4b400")

                plt.plot(dimension_to_bounds[dim]["true"], label="true", c="black", linewidth=5)
                # plot bounds for lb and ub in dotted lines
                plt.plot(dimension_to_bounds[dim]["lb"], label="lb", c="#ef38ac", linestyle="--")
                plt.plot(dimension_to_bounds[dim]["ub"], label="ub", c="#ef38ac", linestyle="--")
                # fill between lb and ub
                plt.fill_between(range(len(dimension_to_bounds[dim]["lb"])), dimension_to_bounds[dim]["lb"],
                                 dimension_to_bounds[dim]["ub"], alpha=1.0, color="#FEB9F8")
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



def plot_calibration_results_w_human_data(calibration_traj_idx_to_data,
                             remapping_model_filepath, remapping_model_architecture, config,
                             experiment_name, calib_user, list_of_coverages, list_of_interval_sizes,
                             individual_trial_results):
    print("Plotting calibration results")

    chain = kp.build_chain_from_urdf(open("utils/urdf/gen3_2f85.urdf").read())
    joint_names = chain.get_joint_parameter_names()
    joint_angle_keys = ['position_0', 'position_1', 'position_2',
                        'position_3', 'position_4', 'position_5', 'position_6', 'position_7',
                        'position_8', 'position_9', 'position_10', 'position_11', 'position_12']

    joint_vel_keys = ['velocity_0', 'velocity_1', 'velocity_2', 'velocity_3', 'velocity_4',
                      'velocity_5', 'velocity_6', 'velocity_7', 'velocity_8', 'velocity_9',
                      'velocity_10', 'velocity_11', 'velocity_12']
    gripper_angles = [0.0069869209377008,
                      0.0069869209377008,
                      -0.0069869209377008,
                      0.0069869209377008,
                      0.0069869209377008,
                      -0.0069869209377008]

    beta_uncertainty_threshold = config['beta_uncertainty_threshold']

    # iterate through the calibration trials
    if config['uncertainty_context'] == 'latent_prefs':
        episodes = np.arange(1, 4)
    elif config['uncertainty_context'] == 'control_prec':
        episodes = np.arange(1, 7)
    else:
        raise NotImplementedError

    # iterate through the calibration trials

    calibration_traj_data = calibration_traj_idx_to_data

    uncertainty_list = individual_trial_results['uncertainty_list']
    dimension_to_bounds = individual_trial_results['dimension_to_bounds']
    list_of_errors_over_time = individual_trial_results['list_of_errors_over_time']
    list_of_alphas_over_time = individual_trial_results['list_of_alphas_over_time']
    list_of_factor_lambda_over_time = individual_trial_results['list_of_factor_lambda_over_time']

    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5, projection='3d')

    for timestep in range(len(calibration_traj_data)):
        # plot the human input state
        human_input_state, predicted_action, rotated_z = calibration_traj_data[
            timestep]
        human_input_state = human_input_state[0].detach().numpy()
        full_state = list(human_input_state) + gripper_angles
        pred_pos = chain.forward_kinematics(convert_to_fk_dict(full_state))['gen3_end_effector_link'].pos
        x, y, z = pred_pos[0], pred_pos[1], pred_pos[2]
        uncertainty_at_timestep = uncertainty_list[timestep]
        if uncertainty_at_timestep > beta_uncertainty_threshold:
            ax5.scatter(x, y, z, c='#E07800', marker='o', s=25, alpha=0.8)
        else:
            ax5.scatter(x, y, z, c='#5E5E5E', marker='o', s=12, alpha=0.4)

    save_user_name = str(calib_user)

    # plot list_of_alphas_over_time, list_of_factor_lambda_over_time, list_of_errors_over_time in one plot
    ax1.plot(list_of_alphas_over_time, label="alpha", color="#C44FB1", linewidth=3)
    ax1.set_title("Alpha over time")
    # ax1.set_xlim(0, 1200)
    # ax1.set_ylim(0, 0.5)

    ax2.plot(list_of_factor_lambda_over_time, label="lambda", color="#7C4FC4", linewidth=3)
    ax2.set_title("Lambda over time")
    # ax2.set_xlim(0, 1200)
    # ax2.set_ylim(0, 30)
    # list_of_errors_over_time should be offset by 1 because is 1 timestep shorter than the rest
    ax3.plot(np.arange(1, len(list_of_errors_over_time) + 1), list_of_errors_over_time, label="error", color="red")
    ax3.set_title("Error over time")

    # plot uncertainty over time
    # plot scatter plot of uncertainty at each timestep, color red if above threshold, green if below
    for t in range(len(uncertainty_list)):
        if uncertainty_list[t] > beta_uncertainty_threshold:
            ax4.scatter(t + 1, uncertainty_list[t], c='#E07800', marker='o', s=30, alpha=0.7)
            # ax1.scatter(t, list_of_factor_lambda_over_time[t], c='#E07800', marker='o', s=100, alpha=0.7)
            # ax2.scatter(t, list_of_factor_lambda_over_time[t], c='#E07800', marker='o', s=100, alpha=0.7)
        else:
            ax4.scatter(t + 1, uncertainty_list[t], c='#5E5E5E', marker='o', s=30, alpha=0.5)

    ax4.plot(np.arange(1, len(uncertainty_list) + 1), uncertainty_list, label="uncertainty", color="black")
    ax4.set_title("Uncertainty over time")
    ax4.set_ylim(-0.01, 2.5)
    # ax4.set_ylim(-0.01, 1.6)

    Xc, Yc, Zc = data_for_cylinder_along_z(0.70980445 + 0.1, 0.25232062, 0.03, 0.3)
    ax5.plot_surface(Xc, Yc, Zc, color='cyan', alpha=0.3)
    Xc, Yc, Zc = data_for_cylinder_along_z(0.70980445 + 0.1, -0.24289359, 0.03, 0.3)
    ax5.plot_surface(Xc, Yc, Zc, color='purple', alpha=0.3)
    ax5.set_xlim3d(0.4, 0.85)
    ax5.set_ylim3d(-0.4, 0.2)
    ax5.set_zlim3d(0.0, 0.55)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')

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

    for dim in range(config['action_dim']):
        plt.figure(figsize=(40, 20))
        plt.subplot(2, 1, 1)

        plt.plot(dimension_to_bounds[dim]["lb_calib"], label="lb_calib", c="#f4b400", linestyle="--")
        plt.plot(dimension_to_bounds[dim]["ub_calib"], label="ub_calib", c="#f4b400", linestyle="--")
        # fill between lb and ub
        plt.fill_between(range(len(dimension_to_bounds[dim]["lb_calib"])), dimension_to_bounds[dim]["lb_calib"],
                         dimension_to_bounds[dim]["ub_calib"], alpha=0.25, color="#f4b400")

        plt.plot(dimension_to_bounds[dim]["true"], label="true", c="black", linewidth=5)
        # plot bounds for lb and ub in dotted lines
        plt.plot(dimension_to_bounds[dim]["lb"], label="lb", c="#ef38ac", linestyle="--")
        plt.plot(dimension_to_bounds[dim]["ub"], label="ub", c="#ef38ac", linestyle="--")
        # fill between lb and ub
        plt.fill_between(range(len(dimension_to_bounds[dim]["lb"])), dimension_to_bounds[dim]["lb"],
                         dimension_to_bounds[dim]["ub"], alpha=1.0, color="#FEB9F8")
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