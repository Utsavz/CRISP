import numpy as np
from mujoco_py.generated import const
import cv2
from gym.envs.robotics import rotations, robot_env, utils
from shapely.geometry import Polygon, Point, MultiPoint
import random
from mpi4py import MPI

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,image_obs=None, random_maze=False,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = 0.15#distance_threshold
        self.image_obs = 1
        self.count=0
        self.block_gripper = True
        self.target_in_the_air = False
        self.randomize = 0
        self.random_maze = random_maze
        self.index = 0
        self.flag = 0
        self.maze_array = None
        self.collision_with_initial_rope = 0
        self.fixed_goal = 1
        self.generate_random_maze = 0
        self.generate_three_room_maze = 0
        self.generate_four_room_maze = 0
        self.generate_five_room_maze = 0
        self.generate_six_room_maze = 0
        self.generate_eight_room_maze = 0
        self.generate_one_wall_maze = 0
        self.generate_two_wall_maze = 0
        curr_eps_num_wall_collisions = []
        self.rank_seed =  np.random.get_state()[1][0]
        # print('Data Loading...')
        data = np.load('./crisp/data/robotic_rope_dataset_simplified.npz')
        self.obs = data['obs'].copy()
        self.states = data['states'].copy()
        del data
        self.gates = [-1 for _ in range(8)]

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

        self.body_pos_initial_backup = self.sim.model.body_pos.copy()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info=None, reward_type='sparse'):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if reward_type == 'sparse':
            rew = -(d > self.distance_threshold).astype(np.float32)
            return rew
        else:
            return -d

    def set_subgoal(self, name, action):
        site_id = self.sim.model.site_name2id(name)
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        self.sim.model.site_pos[site_id] = action - sites_offset[0]
        self.sim.model.site_rgba[site_id][3] = 1

    def reset_subgoal(self, name, action):
        site_id = self.sim.model.site_name2id(name)
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        self.sim.model.site_pos[site_id] = action - sites_offset[0]
        self.sim.model.site_rgba[site_id][3] = 0


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action_rope(self.sim, action)

    def _get_obs(self):
        rope_size = 15
        rope_pos = np.zeros((rope_size, 2))
        for num in range(rope_size):
            curr_pos = self.sim.data.get_geom_xpos('G'+str(num))[:2].copy()
            rope_pos[num] = curr_pos
        rope_pos = rope_pos.ravel().copy()
        if not self.image_obs:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), maze
                ])
        else:
            obs = np.concatenate([
                rope_pos.ravel(),
                ])

        return {
            'observation': obs.copy(),
            'achieved_goal': rope_pos.copy(),
            'desired_goal': self.goal.copy(),
        }

    def map_cor(self, pos, X1=0.0,X2=30.0,Y1=0.,Y2=30.0,x1=1.05,x2=1.55,y1=0.48,y2=1.02):
        x = pos[0]
        y = pos[1]
        
        X = X1 + ( (x-x1) * (X2-X1) / (x2-x1) )
        Y = Y1 + ( (y-y1) * (Y2-Y1) / (y2-y1) )
        return(np.array([X,Y]))

    def get_image_obs(self, grip_pos, object_pos):
        import cv2
        blackBox = grip_pos.copy()
        redCircle = self.goal.copy()
        blueBox = object_pos.copy()
        height = 30
        width = 30
        imgGripper = np.zeros((height,width), np.uint8)
        imgWall = np.zeros((height,width), np.uint8)
        imgGoal = np.zeros((height,width), np.uint8)
        half_block_len = 1.2
        gripper_len = 1.2
        sphere_rad = 1.2
        
        if self.maze_array is not None:
            for i in range(self.maze_array.shape[0]):
                for j in range(self.maze_array.shape[1]):
                    if self.maze_array[i][j]:
                        imgWall[int(3*i):int(3*(i+1)),int(2.76*(j)):int(2.76*(j+1))] = 255 #maze[flag] = (5*(i+1)/2.0, 4.6*(j+1)/2.0)

        image = imgWall

        image = image/255.
        image = image.astype(np.uint8)
        obs = image.ravel().copy()
        return obs

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('table0')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.fixedcamid = 3
        self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.distance = 1.8
        self.viewer.cam.azimuth = 140.
        self.viewer.cam.elevation = -30.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        if self.goal.shape[0] == 3:
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self, seed=0):
        self.sim.set_state(self.initial_state)
        # Randomize the environment
        if self.randomize:
            if self.random_maze:
                self.randomize_environ(seed)
            else:
                # Randomize the walls to make it a dynamic environment
                self.randomize_environ_room_env()

        self.curr_eps_num_wall_collisions = []
        if self.has_object:
            object_xpos = self.get_object_pos()
            while self.check_overlap(object_xpos):
                object_xpos = self.get_object_pos()
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def setIndex(self, index):
        self.index = index
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim(index)
        # Set initial state
        env_state_current = self.sim.get_state().flatten().copy()
        curr_states_list = self.states#[index]
        random_pick = np.random.randint(len(curr_states_list))
        random_pick_1 = np.random.randint(len(curr_states_list[random_pick]))
        env_state_old = curr_states_list[random_pick][random_pick_1].copy()
        env_state_old[0] = env_state_current[0]
        self.sim.set_state_from_flattened(env_state_old)

        # Move end effector into position.
        gripper_target = np.array([1.13, 0.65, 0.5])
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        self.sim.step()


        # Set goal state
        self.goal = self._sample_goal(index).copy()
        self.curr_eps_num_wall_collisions = []
        obs = self._get_obs()
        return obs

    def getIndex(self):
        return self.index

    def get_object_pos(self):
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.05:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            # object_xpos = [1.2, .75]
        return object_xpos.copy()


    def _set_state(self, state, goal):
        env_state_current = self.sim.get_state().flatten().copy()
        env_state_old = state.copy()
        env_state_old[0] = env_state_current[0]
        self.sim.set_state_from_flattened(env_state_old)
        self.goal = goal

        # Simulation step
        for _ in range(10):
            self.sim.step()
            self.sim.forward()
        return True

    def _get_state(self):
        return self.sim.get_state().flatten().copy()

    def generate_maze(self, width=11, height=10, complexity=.2, density=.9, seed=0):
        """Generate a maze using a maze generation algorithm."""
        # Easy: complexity=.1, density=.2
        # Medium: complexity=.2, density=.3
        # Hard: complexity=.4, density=.4
        if self.generate_random_maze:
            ret_array = self.generate_maze_prim(width=width, height=height, complexity=complexity, density=density, seed=seed)
        elif self.generate_three_room_maze:
            ret_array = self.generate_maze_three_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
        elif self.generate_four_room_maze:
            ret_array = self.generate_maze_four_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
        elif self.generate_five_room_maze:
            ret_array = self.generate_maze_five_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
        elif self.generate_six_room_maze:
            ret_array = self.generate_maze_six_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
        elif self.generate_eight_room_maze:
            ret_array = self.generate_maze_eight_room(width=width, height=height, complexity=complexity, density=density, seed=seed)
        elif self.generate_one_wall_maze:
            ret_array = self.generate_maze_one_wall(width=width, height=height, complexity=complexity, density=density, seed=seed)
        elif self.generate_two_wall_maze:
            ret_array = self.generate_maze_two_wall(width=width, height=height, complexity=complexity, density=density, seed=seed)
        else:
            ret_array = None

        # Reset to old seed
        random.seed(self.rank_seed)
        return ret_array

    def generate_maze_prim(self, width=11, height=10, complexity=.3, density=.2, seed=0):
        """Generate a maze using a maze generation algorithm."""
        # Only odd shapes
        # complexity = density
        # Easy:  -.3 0.2
        # Hard: 0.6, 0.9
        random.seed(seed)
        shape = (height, width)#((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))//4 +1  # Size of components
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Number of components

        # assert False
        # Build actual maze
        Z = np.zeros((11,11), dtype=int)
        Z[-1:] = 1
        # Make aisles
        for i in range(density):
            cap_index = 10
            x, y = (random.randrange(0, shape[0]), random.randrange(0, shape[1]))  # Pick a random position
            while (Z[x, y] or (x+1>=shape[0] or Z[x+1, y]) or (x-1<0 or Z[x-1, y]) or (y+1>=shape[1] or Z[x, y+1]) or 
                (y-1<0 or Z[x, y-1]) or (x-1<0 or y-1<0 or Z[x-1, y-1]) or (x-1<0 or y+1>=shape[1] or Z[x-1, y+1]) or 
                (y-1<0 or x+1>=shape[0] or Z[x+1, y-1]) or (x+1>=shape[0] or y+1>=shape[1] or Z[x+1, y+1])):
                x, y = (random.randrange(0, shape[0]), random.randrange(0, shape[1]))  # Pick a random position
                cap_index -= 1
                if cap_index<0:
                    break
            if cap_index<0:
                continue
            Z[x, y] = 1
            for j in range(complexity):
                neighbours = []
                if y > 1:             neighbours.append(('u',x, y - 2))
                if y < shape[1] - 2:  neighbours.append(('d',x, y + 2))
                if x > 1:             neighbours.append(('l',x - 2, y))
                if x < shape[0] - 2:  neighbours.append(('r',x + 2, y))
                if len(neighbours):
                    direction, x_, y_ = neighbours[random.randrange(0, len(neighbours))]
                    if ((direction == 'u' and Z[x_, y_]==0 and Z[x_, y_+1]==0 and (y_-1<0 or Z[x_, y_-1]==0)) and 
                            ((x_-1<0 or Z[x_-1, y_]==0) and (x_+1>=shape[0] or Z[x_+1, y_]==0)) and
                            ((x_-1<0 or y_-1<0 or Z[x_-1, y_-1]==0) and (x_-1<0 or y_+1>=shape[1] or Z[x_-1, y_+1]==0)) and
                            ((x_+1>=shape[0] or y_-1<0 or Z[x_+1, y_-1]==0) and (x_+1>=shape[0] or y_+1>=shape[1] or Z[x_+1, y_+1]==0)) or
                        (direction == 'd' and Z[x_, y_]==0 and Z[x_, y_-1]==0 and (y_+1>=shape[1] or Z[x_, y_+1]==0)) and
                            ((x_-1<0 or Z[x_-1, y_]==0) and (x_+1>=shape[0] or Z[x_+1, y_]==0)) and
                            ((x_-1<0 or y_-1<0 or Z[x_-1, y_-1]==0) and (x_-1<0 or y_+1>=shape[1] or Z[x_-1, y_+1]==0)) and
                            ((x_+1>=shape[0] or y_-1<0 or Z[x_+1, y_-1]==0) and (x_+1>=shape[0] or y_+1>=shape[1] or Z[x_+1, y_+1]==0)) or
                        (direction == 'l' and Z[x_, y_]==0 and Z[x_+1, y_]==0 and (x_-1<0 or Z[x_-1, y_]==0)) and
                            ((y_-1<0 or Z[x_, y_-1]==0) and (y_+1>=shape[1] or Z[x_, y_+1]==0)) and
                            ((x_-1<0 or y_-1<0 or Z[x_-1, y_-1]==0) and (x_-1<0 or y_+1>=shape[1] or Z[x_-1, y_+1]==0)) and
                            ((x_+1>=shape[0] or y_-1<0 or Z[x_+1, y_-1]==0) and (x_+1>=shape[0] or y_+1>=shape[1] or Z[x_+1, y_+1]==0)) or
                        (direction == 'r' and Z[x_, y_]==0 and Z[x_-1, y_]==0 and (x_+1>=shape[0] or Z[x_+1, y_]==0)) and
                            ((y_-1<0 or Z[x_, y_-1]==0) and (y_+1>=shape[1] or Z[x_, y_+1]==0)) and
                            ((x_-1<0 or y_-1<0 or Z[x_-1, y_-1]==0) and (x_-1<0 or y_+1>=shape[1] or Z[x_-1, y_+1]==0)) and
                            ((x_+1>=shape[0] or y_-1<0 or Z[x_+1, y_-1]==0) and (x_+1>=shape[0] or y_+1>=shape[1] or Z[x_+1, y_+1]==0))):
                        Z[x_, y_] = 1
                        Z[x_ + (x - x_) // 2, y_ + (y - y_) // 2] = 1
                        x, y = x_, y_
        Z[1,3] = 0
        Z[9,9] = 0
        return Z#.astype(int).copy()


    def generate_maze_three_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
        """Generation a random two room env"""
        # Only odd shapes
        random.seed(seed)
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        height = 10
        width = 10
        # Fill borders
        # Z[0, :] = Z[-1, :] = 1
        # Z[:, 0] = Z[:, -1] = 1
        # Horizontal Wall
        x = random.randrange(1, height)
        Z[x,:] = 1
        # Vertical Wall
        y1 = -1
        y2 = -1
        x1 = -1
        x2 = -1
        y = random.randrange(2, width-1)
        while y == 3:
            y = random.randrange(2, width-1)
        Z[x+1:,y] = 1
        # Horizontal gates
        if y != 1:
            y1 = random.randrange(1, y)
            Z[x,y1] = 0
        if y != width-1:
            y2 = random.randrange(y+1, width)
            Z[x,y2] = 0
        self.gates[0] = 1.08+x*0.049
        self.gates[1] = 0.5+y1*0.05
        self.gates[2] = 1.08+x*0.049
        self.gates[3] = 0.5+y2*0.05
        # Vertical gates
        if x != 1:
            pass
        else:
            x1 = 1
        if x != height-1:
            x2 = random.randrange(x+1, height)
            Z[x2,y] = 0
        else:
            x2 = height-1
        self.gates[6] = 1.08+x2*0.049
        self.gates[7] = 0.5+y*0.05
        #Start
        Z[1,3] = 0
        # Goal
        Z[9,9] = 0
        return Z.astype(int).copy()


    def generate_maze_four_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
        """Generation a ransom four room env"""
        # Only odd shapes
        random.seed(seed)
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        height = 10
        width = 10
        # Fill borders
        # Z[0, :] = Z[-1, :] = 1
        # Z[:, 0] = Z[:, -1] = 1
        # Horizontal Wall
        x = random.randrange(2, height-1)
        Z[x,:] = 1
        # Vertical Wall
        y1 = -1
        y2 = -1
        x1 = -1
        x2 = -1
        y = random.randrange(2, width-1)
        while y == 3:
            y = random.randrange(2, width-1)
        Z[:,y] = 1
        # Horizontal gates
        if y != 1:
            y1 = random.randrange(1, y)
            Z[x,y1] = 0
        if y != width-1:
            y2 = random.randrange(y+1, width)
            Z[x,y2] = 0
        self.gates[0] = 1.08+x*0.049
        self.gates[1] = 0.5+y1*0.05
        self.gates[2] = 1.08+x*0.049
        self.gates[3] = 0.5+y2*0.05
        # Vertical gates
        if x != 1:
            x1 = random.randrange(1, x)
            Z[x1,y] = 0
        else:
            x1 = 1
        if x != height-1:
            x2 = random.randrange(x+1, height)
            Z[x2,y] = 0
        else:
            x2 = height-1
        self.gates[4] = 1.08+x1*0.049
        self.gates[5] = 0.5+y*0.05
        self.gates[6] = 1.08+x2*0.049
        self.gates[7] = 0.5+y*0.05
        #Start
        Z[1,3] = 0
        # Goal
        Z[9,9] = 0
        return Z.astype(int).copy()

    def generate_maze_five_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
        """Generation a ransom four room env"""
        # Only odd shapes
        random.seed(seed)
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        height = 10
        width = 10
        # Fill borders
        # Z[0, :] = Z[-1, :] = 1
        # Z[:, 0] = Z[:, -1] = 1
        # Horizontal Wall
        x = random.randrange(1, height)
        # Z[x,:] = 1
        # Vertical Wall
        y1 = -1
        y2 = -1
        x1 = -1
        x2 = -1
        y = random.randrange(2, width-1)
        while y == 3:
            y = random.randrange(2, width-1)
        Z[:,y] = 1
        Z[x,:y] = 1
        # Horizontal gates
        if y != 1:
            y1 = random.randrange(1, y)
            Z[x,y1] = 0
        if y != width-1:
            y2 = random.randrange(y+1, width)
            Z[x,y2] = 0
        self.gates[0] = 1.08+x*0.049
        self.gates[1] = 0.5+y1*0.05
        self.gates[2] = 1.08+x*0.049
        self.gates[3] = 0.5+y2*0.05
        # Vertical gates
        if x != 1:
            x1 = random.randrange(1, x)
            Z[x1,y:] = 1
        else:
            x1 = 1
        if x != height-1:
            x2 = random.randrange(x+1, height)
            Z[x2,y:] = 1
        else:
            x2 = height-1
        if y != width-1:
            y21 = random.randrange(y+1, width)
            Z[x1,y21] = 0
            y22 = random.randrange(y+1, width)
            Z[x2,y22] = 0

        if x1 < 3:
            Z[x1-1,y] = 0
        else:
            x1_gate = random.randrange(1, x1)
            Z[x1_gate,y] = 0
        if x2 - x1 == 2:
            Z[x2-1,y] = 0
        elif x2 - x1 > 2:
            x2_gate = random.randrange(x1+1, x2-1)
            Z[x2_gate,y] = 0
        if x2 == height-2:
            Z[x2+1,y] = 0
        elif x2 != height-1:
            x3_gate = random.randrange(x2+1, height-1)
            Z[x3_gate,y] = 0
        self.gates[4] = 1.08+x1*0.049
        self.gates[5] = 0.5+y*0.05
        self.gates[6] = 1.08+x2*0.049
        self.gates[7] = 0.5+y*0.05
        #Start
        Z[1,3] = 0
        # Goal
        Z[9,9] = 0
        return Z.astype(int).copy()

    def generate_maze_six_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
        """Generation a random six room env"""
        # Only odd shapes
        random.seed(seed)
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        height = 10
        width = 10
        # Fill borders
        # Z[0, :] = Z[-1, :] = 1
        # Z[:, 0] = Z[:, -1] = 1
        # Horizontal Wall
        x11 = random.randrange(2, height-2)
        if x11+2 == height-1:
            x22 = height-1
        else:
            x22 = random.randrange(x11+2, height-1)
        Z[x11,:] = 1
        Z[x22,:] = 1
        # Vertical Wall
        y1 = -1
        y2 = -1
        x1 = -1
        x2 = -1
        y = random.randrange(2, width-1)
        while y == 3:
            y = random.randrange(2, width-1)
        Z[:,y] = 1
        # Z[x,:y] = 1

        # Vertical gates
        x_min = min(x11, x22)
        x_max = max(x11, x22)
        if x_min != 1:
            x1_gate = random.randrange(1, x_min)
            Z[x1_gate,y] = 0
        else:
            Z[1,y] = 0
        if x_min +1 < x_max:
            x2_gate = random.randrange(x_min+1, x_max)
            Z[x2_gate,y] = 0
        else:
            Z[x_max,y] = 0
        if x_max +1 < height -1:
            x22_gate = random.randrange(x_max+1, height-1)
            Z[x22_gate,y] = 0
        else:
            Z[height-1,y] = 0

        # Horizontal gates
        y1 = random.randrange(1, y)
        Z[x11,y1] = 0
        y2 = random.randrange(y+1, width)
        Z[x11,y2] = 0

        y1 = random.randrange(1, y)
        Z[x22,y1] = 0
        y2 = random.randrange(y+1, width)
        Z[x22,y2] = 0
        Z[1,3] = 0
        # Goal
        Z[9,9] = 0
        return Z.astype(int).copy()


    def generate_maze_eight_room(self, width=11, height=10, complexity=.2, density=.2, seed=0):
        """Generation a random six room env"""
        # Only odd shapes
        random.seed(seed)
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        height = 10
        width = 10
        # Fill borders
        # Z[0, :] = Z[-1, :] = 1
        # Z[:, 0] = Z[:, -1] = 1
        # Horizontal Wall
        x11 = random.randrange(2, height-2)
        if x11+2 == height-1:
            x22 = height-1
        else:
            x22 = random.randrange(x11+2, height-1)
        Z[x11,:] = 1
        Z[x22,:] = 1
        # Vertical Wall
        y1 = -1
        y2 = -1
        x1 = -1
        x2 = -1
        y11 = random.randrange(2, width-2)
        if y11+2 == width-1:
            y22 = width-1
        else:
            y22 = random.randrange(y11+2, width-1)
        Z[:,y11] = 1
        Z[:,y22] = 1
        # Z[x,:y] = 1

        # Vertical gates
        x_min = min(x11, x22)
        x_max = max(x11, x22)
        if x_min != 1:
            x1_gate = random.randrange(1, x_min)
            Z[x1_gate,y11] = 0
        else:
            Z[1,y11] = 0
        if x_min +1 < x_max:
            x2_gate = random.randrange(x_min+1, x_max)
            Z[x2_gate,y11] = 0
        else:
            Z[x_max,y11] = 0
        if x_max +1 < height -1:
            x22_gate = random.randrange(x_max+1, height-1)
            Z[x22_gate,y11] = 0
        else:
            Z[height-1,y11] = 0

        if x_min != 1:
            x1_gate = random.randrange(1, x_min)
            Z[x1_gate,y22] = 0
        else:
            Z[1,y2] = 0
        if x_min +1 < x_max:
            x2_gate = random.randrange(x_min+1, x_max)
            Z[x2_gate,y22] = 0
        else:
            Z[x_max,y11] = 0
        if x_max +1 < height -1:
            x22_gate = random.randrange(x_max+1, height-1)
            Z[x22_gate,y22] = 0
        else:
            Z[height-1,y22] = 0

        # Horizontal gates

        y_min = min(y11, y22)
        y_max = max(y11, y22)
        if y_min != 1:
            y1_gate = random.randrange(1, y_min)
            Z[x11,y1_gate] = 0
        else:
            Z[x11,1] = 0
        if y_min +1 < y_max:
            y2_gate = random.randrange(y_min+1, y_max)
            Z[x11,y2_gate] = 0
        else:
            Z[x11,y_max] = 0
        if y_max +1 < width -1:
            y22_gate = random.randrange(y_max+1, width-1)
            Z[x11,y22_gate] = 0
        else:
            Z[x11,width-1] = 0

        if y_min != 1:
            y1_gate = random.randrange(1, y_min)
            Z[x22,y1_gate] = 0
        else:
            Z[x22,1] = 0
        if y_min +1 < y_max:
            y2_gate = random.randrange(y_min+1, y_max)
            Z[x22,y2_gate] = 0
        else:
            Z[x22,y_max] = 0
        if y_max +1 < width -1:
            y22_gate = random.randrange(y_max+1, width-1)
            Z[x22,y22_gate] = 0
        else:
            Z[x22,width-1] = 0

        #Start
        Z[1,3] = 0
        # Goal
        Z[9,9] = 0
        return Z.astype(int).copy()


    def generate_maze_one_wall(self, width=11, height=10, complexity=.2, density=.2, seed=0):
        """Generation a random one wall env"""
        # Only odd shapes
        random.seed(seed)
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        height = 10
        width = 10
        # Fill borders
        # Z[0, :] = Z[-1, :] = 1
        # Z[:, 0] = Z[:, -1] = 1
        y1 = -1
        y2 = -1
        x1 = -1
        x2 = -1
        # Horizontal Wall
        # x = random.randrange(1, height)
        x = int(height/2-1)
        Z[x,:] = 1
        # y1 = int(width/2)
        y1 = int(random.randrange(1, width))
        Z[x,y1] = 0
        Z[x,y1-1] = 0
        #Start
        Z[1,3] = 0
        # Goal
        Z[9,9] = 0
        return Z.astype(int).copy()


    def generate_maze_two_wall(self, width=11, height=10, complexity=.2, density=.2, seed=0):
        """Generation a random two wall env"""
        # Only odd shapes
        random.seed(seed)
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))  # Number of components
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # Size of components
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        height = 10
        width = 10
        y1 = -1
        y2 = -1
        x1 = -1
        x2 = -1
        # Fill borders
        # Z[0, :] = Z[-1, :] = 1
        # Z[:, 0] = Z[:, -1] = 1
        # Horizontal Wall
        x1 = int(height/2-2)
        x2 = int(height/2+1)
        Z[x1,:] = 1
        Z[x2,:] = 1
        y1 = int(random.randrange(1, width))
        y2 = int(random.randrange(1, width))
        Z[x1,y1] = 0
        Z[x1,y1-1] = 0
        Z[x2,y2] = 0
        Z[x2,y2-1] = 0
        #Start
        Z[1,3] = 0
        # Goal
        Z[9,9] = 0
        return Z.astype(int).copy()

    def randomize_environ(self, seed=0):
        # Prim's maze generation algorithm
        # self.index = 0
        # seed = self.index
        maze_array = self.generate_maze(seed=seed)
        self.maze_array = maze_array.copy()
        num = -1
        for i in range(len(self.sim.model.body_pos)):
            self.sim.model.body_pos[i] = self.body_pos_initial_backup[i]
        for col in range(10):
            for row in range(11):
                num += 1
                if maze_array[col][row] == 1:
                    self.sim.model.body_pos[num+35] = [1.08 + col*0.049, 0.5 + row*0.05, 0.42]

    def if_collision(self):
        for i in range(self.sim.data.ncon):
            flag_contact = 0
            contact = self.sim.data.contact[i]
            name1 = self.sim.model.geom_id2name(contact.geom1)
            name2 = self.sim.model.geom_id2name(contact.geom2)
            if name1 is None or name2 is None:
                break
            for num in range(3,6):
                if ("robot0:l_gripper_finger_link" == name1 and "G"+str(num) in name2) or ("robot0:r_gripper_finger_link" == name2 and "G"+str(num) in name1):
                    return 1
        return 0

    def collision_with_initial_rope(self):
        old_value = self.collision_with_initial_rope
        return old_value

    def num_walls_collision(self):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            name1 = self.sim.model.geom_id2name(contact.geom1)
            name2 = self.sim.model.geom_id2name(contact.geom2)
            if name1 is None or name2 is None:
                break
            if "robot0:l_gripper_finger_link" == name1 and "object" in name2:
                if name2 not in self.curr_eps_num_wall_collisions:
                    self.curr_eps_num_wall_collisions.append(name2)
            elif "robot0:l_gripper_finger_link" == name2 and "object" in name1:
                if name1 not in self.curr_eps_num_wall_collisions:
                    self.curr_eps_num_wall_collisions.append(name1)
            elif "robot0:r_gripper_finger_link" == name1 and "object" in name2:
                if name2 not in self.curr_eps_num_wall_collisions:
                    self.curr_eps_num_wall_collisions.append(name2)
            elif "robot0:r_gripper_finger_link" == name2 and "object" in name1:
                if name1 not in self.curr_eps_num_wall_collisions:
                    self.curr_eps_num_wall_collisions.append(name1)

        return len(self.curr_eps_num_wall_collisions)

    def _sample_goal(self, index = -1):
        if self.has_object:
            goal = self.get_goal_pos()
        else:
            goal = self.get_goal_pos(index)
        self.goal = goal.copy()
        return goal.copy()

    def get_possible_goals(self):
        max_u = [0.25, 0.27, 0.145]
        action_offset = [1.3, 0.75, 0.555]
        outer_array = np.arange(action_offset[0]-max_u[0], action_offset[0]+max_u[0]+(2*max_u[0])/(10), (2*max_u[0])/(9))
        inner_array = np.arange(action_offset[1]-max_u[1], action_offset[1]+max_u[1]+(2*max_u[1])/(11), (2*max_u[1])/(10))
        return outer_array, inner_array

    def get_goal_pos(self, index = -1):
        if index != -1:
            random.seed(index)

        goal_index = index#random.randrange(self.obs.shape[0])
        curr_obs_list = self.obs[goal_index]
        goal_obs = curr_obs_list[len(curr_obs_list)-1]['desired_goal'].copy()
        for num in range(5,20):
            self.set_subgoal('subgoal_goal_'+str(num), np.array([goal_obs[2*(num-5)], goal_obs[2*(num-5)+1], 0.42]))
        if index != -1:
            random.seed(self.rank_seed)
        return goal_obs.copy()

    def get_maze_array(self):
        return self.maze_array.copy()

    def get_maze_array_simple(self):
        maze = self.maze_array.copy()
        maze[:,-1] = 0
        maze[:,0] = 0
        maze[-1,:] = 0
        maze[0,:] = 0
        maze[-2,:] = 0
        return maze

    def check_overlap(self, point):
        xx = Point(point[:2])
        for polygon in self.polygons:
            if xx.within(polygon):
                return 1
        return 0

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def get_current_gripper_pos(self):
        return self.sim.data.get_site_xpos('robot0:grip').copy()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = np.array([1.13, 0.65, 0.44])
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_gripper_xpos = np.array([1.3, 0.75, 0.44])
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def capture(self,depth=False):
        if self.viewer == None:
            pass
        else:
            self.viewer.cam.fixedcamid = 3
            self.viewer.cam.type = const.CAMERA_FIXED
            width, height = 1920, 1080
            img = self._get_viewer().read_pixels(width, height, depth=depth)
            if depth:
                rgb_image = img[0][::-1]
                rgb_image = rgb_image[458:730,726:1193]
                rgb_image = cv2.resize(rgb_image, (50,50))
                depth_image = np.expand_dims(img[1][::-1],axis=2)
                depth_image = depth_image[458:730,726:1193]
                depth_image = cv2.resize(depth_image, (50,50))
                depth_image = np.reshape(depth_image, (50,50,1))
                rgbd_image = np.concatenate((rgb_image,depth_image),axis=2)
                return rgbd_image
            else:
                return img[::-1]

    def is_achievable(self, curr_pos, pred_pos):
        dist = np.linalg.norm(curr_pos - pred_pos)
        threshold = 0.35
        if dist < threshold:
            return 0
        else:
            return -1








