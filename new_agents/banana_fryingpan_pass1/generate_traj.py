
from tcdm.planner.generate import motion_plan_one_obj

# TODO: load initial pose from saved trajectories

################ First trajectory (zero-indexed): moving banana to frying pan ################

obj_list = ['banana', 'fryingpan']
float_obj_idx = 0
# use global frame
obj_Xs = [[0.01895152-0.2, -0.01185687, 0.021, -3.05750797, 0.08599904, -1.99028331],
          [0.001307, 0.03177, -0.1743+0.2, -0.09513, -0.09165, -0.8584],
          ]
float_obj_target_X = [0.001307-0.03, 0.03177+0.05, -0.1743+0.2+0.04, 
                      -3.05750797, 0.08599904, -1.99028331+1]
ignore_collision_obj_idx_all = []
save_path = f'./new_agents/{obj_list[0]}_{obj_list[1]}_pass1/traj_0.npz'
visualize = True

# run
motion_plan_one_obj(obj_list=obj_list,
                    float_obj_idx=float_obj_idx,
                    obj_Xs=obj_Xs,
                    float_obj_target_X=float_obj_target_X,
                    save_path=save_path,
                    ignore_collision_obj_idx_all=ignore_collision_obj_idx_all,
                    visualize=visualize)

################ Second trajectory: moving frying pan up a bit ################

obj_list = ['banana', 'fryingpan']
float_obj_idx = 1
# use global frame
obj_Xs = [float_obj_target_X.copy(),
          obj_Xs[1]]
float_obj_target_X = obj_Xs[1].copy()
float_obj_target_X[2] += 0.2    # move upwards
float_obj_target_X[3] += 0.3    # pitch up a bit
float_obj_target_X[4] += 0.3 
ignore_collision_obj_idx_all = [0]  # ignore collision with banana
save_path = f'./new_agents/{obj_list[0]}_{obj_list[1]}_pass1/traj_1.npz'
visualize = True

# run
motion_plan_one_obj(obj_list=obj_list,
                    float_obj_idx=float_obj_idx,
                    obj_Xs=obj_Xs,
                    float_obj_target_X=float_obj_target_X,
                    save_path=save_path,
                    ignore_collision_obj_idx_all=ignore_collision_obj_idx_all,
                    visualize=visualize)
