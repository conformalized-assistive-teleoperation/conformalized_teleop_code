
# Run the 7dof cup grasp experiments
echo "\n\nRunning 7dof cup grasp experiments"
#python run_7dof_cup_grasp_experiment.py grasp7dof/grasp7dof_latentprefs_acqr 0.1 0.1 &
#python run_7dof_cup_grasp_experiment.py grasp7dof/grasp7dof_latentprefs_qr &
#python run_7dof_cup_grasp_experiment.py grasp7dof/grasp7dof_latentprefs_ensemble &
#
#python run_7dof_cup_grasp_experiment.py grasp7dof/grasp7dof_controlprec_acqr 0.11 &
#python run_7dof_cup_grasp_experiment.py grasp7dof/grasp7dof_controlprec_qr &
#python run_7dof_cup_grasp_experiment.py grasp7dof/grasp7dof_controlprec_ensemble

# Run the 7dof goal reaching experiments
echo "\n\nRunning 7dof goal reaching experiments"
python run_7dof_goal_reaching_experiment.py goal7dof/goal7dof_latentprefs_acqr 0.4 0.4
python run_7dof_goal_reaching_experiment.py goal7dof/goal7dof_latentprefs_qr
python run_7dof_goal_reaching_experiment.py goal7dof/goal7dof_latentprefs_ensemble
#
#python run_7dof_goal_reaching_experiment.py goal7dof/goal7dof_controlprec_acqr 1.2 &
#python run_7dof_goal_reaching_experiment.py goal7dof/goal7dof_controlprec_qr &
#python run_7dof_goal_reaching_experiment.py goal7dof/goal7dof_controlprec_ensemble


# Run the 2d navigation experiments
echo "\n\nRunning 2d navigation experiments"
#python run_2d_navigation_experiment.py 2dnav/2dnav_latentprefs_acqr 1.0 1.0
#python run_2d_navigation_experiment.py 2dnav/2dnav_latentprefs_qr
#python run_2d_navigation_experiment.py 2dnav/2dnav_latentprefs_ensemble

#python run_2d_navigation_experiment.py 2dnav/2dnav_controlprec_acqr 1.0
#python run_2d_navigation_experiment.py 2dnav/2dnav_controlprec_qr
#python run_2d_navigation_experiment.py 2dnav/2dnav_controlprec_ensemble


