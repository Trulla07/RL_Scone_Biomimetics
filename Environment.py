import sys
import os
import math
import numpy as np
import gym
from abc import ABC, abstractmethod
from sconetools import sconepy
#import sconegym_example

def find_model_file( model_file ):
    this_dir, this_file = os.path.split(__file__)
    return os.path.join(this_dir, "data", model_file)

class SconeGym(gym.Env):
    """
    Main general purpose class that gives you a gym-ready sconepy interface
    It has to be inherited by the environments you actually want to use and 
    some methods have to be defined (see end of class). This class would probably
    be a good starting point for new environments.
    New environments also have to be registered in sconegym/__init__.py !
    """
    def __init__(self, model_file, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sconepy.set_log_level(3)
        self.model = sconepy.load_model('Scenario_withActuator_rl.scone')
        self.init_dof_pos = self.model.dof_position_array().copy() #init values of init file => values
        self.init_dof_vel = self.model.dof_velocity_array().copy() #init values of init file => velocities

        # Internal settings
        self.episode = 0
        self.total_reward = 0.0
        self.min_velocity = 1.0
        self.base_energy = 0.01
        self.init_dof_pos_std = 0.05
        self.init_dof_vel_std = 0.1
        self.init_load = 0
        self.init_activations_mean = 0.3
        self.init_activations_std = 0.1
        self.min_com_height = 0.8
        self.step_size = 0.01
        self.has_reset = False
        self.store_next = False
        self.use_delayed_sensors = True
        self.use_delayed_actuators = False
        self.bNewStep_r = False
        self.bNewStep_l = False
        self.bFlight_r = False
        self.bFlight_l = True
        self._find_head_body() #checks for torso
        self._setup_action_observation_spaces() #find all actuators in the model => here 20
        self.set_output_dir('DATE_TIME.' + self.model.name())

    def step(self, action):
        """
        takes an action and advances environment by 1 step.
        """
        if not self.has_reset:
            raise Exception('You have to call reset() once before step()')

        if self.use_delayed_actuators:
            self.model.set_delayed_actuator_inputs(action)
        else:
            self.model.set_actuator_inputs(action)
        
        self.time += self.step_size
        self.model.advance_simulation_to(self.time)
        reward = self._get_rew()
        obs = self._get_obs()
        done = self._get_done()
        self.total_reward += reward

        if done:
            if self.store_next:
                filename = f'{self.episode:05d}_{self.total_reward:.3f}'
                print("Results written to", self.output_dir + '/' + filename)
                self.model.write_results(self.output_dir, filename)
                self.store_next = False
            
                self.episode += 1

        return obs, reward, done, {}

    def reset(self, *args, **kwargs):
        """
        Reset and randomize the initial state.
        """
        self.model.reset()
        self.has_reset = True
        self.time = 0
        self.bFlight_l = False
        self.bFlight_r = False
        self.total_reward = 0.0

        # Check if data should be stored (slow)
        self.model.set_store_data(self.store_next)

        # Randomize initial pose
        dof_pos = self.init_dof_pos + np.random.normal(0, self.init_dof_pos_std, len(self.init_dof_pos))
        self.model.set_dof_positions(dof_pos)
        dof_vel = self.init_dof_vel + np.random.normal(0, self.init_dof_vel_std, len(self.init_dof_vel))
        self.model.set_dof_velocities(dof_vel)

        # Randomize initial muscle activations
        muscle_activations = np.clip(
            np.random.normal(
                self.init_activations_mean,
                self.init_activations_std,
                size=len(self.model.muscles(),)),
            0.01, 1.0)
        self.model.init_muscle_activations(muscle_activations)

        # Initialize state and equilibrate muscles
        self.model.init_state_from_dofs()

        if self.init_load > 0:
            self.model.adjust_state_for_load(self.init_load)

        grf_r = self.model.bodies()[4].contact_force().y
        grf_l = self.model.bodies()[7].contact_force().y

        if grf_r <20:
            self.bFlight_r = True
        if grf_l < 20:
            self.bFlight_l = True

        obs = self._get_obs()
        return obs

    def store_next_episode(self):
        self.store_next = True

    def set_output_dir(self, dir_name):
        self.output_dir = sconepy.replace_string_tags(dir_name)

    def manually_load_model(self):
        self.model = sconepy.load_model(self.model_file)
        self.model.set_store_data(True)

    def render(self, *args, **kwargs):
        """
        Not yet supported
        """
        return

    def model_energy(self):
        # Simple sum of squared muscle activation
        act = self.model.muscle_activation_array()
        e = np.mean(np.square(act))
        return self.base_energy + e

    def model_velocity(self):
        return self.model.com_vel().x

    def cost_of_transport(self):
        v = self.model_velocity()
        if v > self.min_velocity:
            return self.min_velocity + ( v - self.min_velocity ) / self.model_energy()
        else:
            return v 

    def _setup_action_observation_spaces(self):
        num_act = len(self.model.actuators())
        self.action_space = gym.spaces.Box(low=np.zeros(shape=(num_act,)), high=np.ones(shape=(num_act,)), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10000, high=10000, shape=self._get_obs().shape, dtype=np.float32)

    def _find_head_body(self):
        head_names = ['torso', 'head', 'lumbar']
        self.head_body = None
        for b in self.model.bodies():
            if b.name() in head_names:
                self.head_body = b
        if self.head_body is None:
            raise Exception('Could not find head body')

    # these all need to be defined by environments
    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _get_rew(self):
        pass

    @abstractmethod
    def _get_done(self):
        pass


class GaitGym(SconeGym):
    def __init__(self, model_file, *args, **kwargs):
        super().__init__(model_file, *args, **kwargs)

    def _get_obs(self):

        grf_r = self.model.bodies()[4].contact_force().y
        grf_r = np.expand_dims(grf_r,axis=0)
        grf_l = self.model.bodies()[7].contact_force().y
        grf_l = np.expand_dims(grf_l,axis=0)
        
        if self.use_delayed_sensors:
            return np.concatenate([
                self.model.delayed_muscle_fiber_length_array(),
                self.model.delayed_muscle_fiber_velocity_array(),
                self.model.delayed_muscle_force_array(),
                self.model.muscle_excitation_array(),
                self.head_body.orientation().array(), #4 Values => body orientation in quaternion
                self.head_body.ang_vel().array(), #3 values => velocity in x,y and z
                self.model.dof_position_array(), #9 values
                grf_r,
                grf_l
                ], dtype=np.float32).copy()
        else:
            #ham_r, bifemsh_r, glut_max_r, iliops_r, rf_r, vasti_r, gastroc_r, soleus_r, tib_r,
            #ham_l, bifemsh_l, glut_max_l, iliops_l, rf_l, vasti_l, gastroc_l, soleus_l, tib_l
            #not including the actuators


            observations = np.concatenate([
                self.model.muscle_fiber_length_array(), #18 values
                self.model.muscle_fiber_velocity_array(), #18 values
                self.model.muscle_force_array(), #18 values
                self.model.muscle_excitation_array(), #18 values
                self.head_body.orientation().array(), #4 Values => body orientation in quaternion
                self.head_body.ang_vel().array(), #3 values => velocity in x,y and z
                self.model.dof_position_array(), #9 values
                grf_r,
                grf_l
                ], dtype=np.float32).copy()
                    
            return observations

    def _get_rew(self):
        """
        Reward function.
        """
        reward = 0

        alive = 0.1

        grf_r = self.model.bodies()[4].contact_force().y
        grf_l = self.model.bodies()[7].contact_force().y

        if grf_r > 20 and self.bFlight_r and not self.bNewStep_r:
            self.bFlight_r = False
            self.bNewStep_r = True
            reward+=0.01

        if grf_r < 20 and not self.bFlight_r:
            self.bFlight_r = True
            self.bNewStep_r = False

        if grf_l > 20 and self.bFlight_l and not self.bNewStep_l:
            self.bFlight_l = False
            self.bNewStep_l = True
            reward+=0.01

        if grf_l < 20 and not self.bFlight_l:
            self.bFlight_l = True
            self.bNewStep_l = False

        effort = self.cost_of_transport() # effort related to velocity

        reward += effort+alive

        return reward

    def _get_done(self) -> bool:
        """
        The episode ends if the center of mass is below min_com_height.
        """
        if self.model.com_pos().y < self.min_com_height:
            return True
        return False

class GaitGym2D(GaitGym):
    def __init__(self, *args, **kwargs):
        super().__init__(find_model_file('Scenario_withoutActuator_rl.scone'), *args, **kwargs)
