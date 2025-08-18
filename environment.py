import matplotlib.pyplot as plt

from tudatpy import constants, numerical_simulation
from tudatpy.math import root_finders
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.util import result2array, split_history
from tudatpy.interface import spice

import numpy as np
import yaml

from utils import *

class Environment:
    def __init__(self, environment_config: dict):
        N = len(environment_config['bodies_to_propagate'])

        self.ti = environment_config.get('initial_time', 0)
        self.tf = environment_config.get('final_time', 86400)

        # CREATE ENVIRONMENT & ACCELERATION SETTINGS
        # Create Earth
        spice.load_standard_kernels()
        body_settings = environment_setup.get_default_body_settings(['Earth', 'Sun', 'Moon'], 'Earth', 'J2000')

        # Create satellite and secondary body(ies)
        bodies_to_propagate = []
        self.system_initial_state = []

        acceleration_settings = {}
        for body_name, body_params in environment_config['bodies_to_propagate'].items():
            body_settings.add_empty_settings(body_name)

            body_settings.get(body_name).constant_mass = body_params['mass']
            area = body_params['area']

            body_settings.get(body_name).aerodynamic_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
                area, [body_params['Cd'], 0, 0]
            )

            body_settings.get(body_name).radiation_pressure_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
                area, body_params['Cr'], {'Sun': ['Earth']}
            )

            # Define accelerations on bodies (could make this flexible by passing a dict[str, list] from config)
            acceleration_settings[body_name] = {
                'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(8,8), propagation_setup.acceleration.aerodynamic()],
                'Sun': [propagation_setup.acceleration.point_mass_gravity(), propagation_setup.acceleration.radiation_pressure()],
                'Moon': [propagation_setup.acceleration.point_mass_gravity()]
            }

            bodies_to_propagate.append(body_name)
            self.system_initial_state.extend(body_params['X0'])

        central_bodies = ['Earth'] * len(bodies_to_propagate)        # Define one central body per body to propagate
        bodies = environment_setup.create_system_of_bodies(body_settings) # Create environment

        # Create acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )
        
        # CREATE PROPAGATION SETTINGS
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step=10,
            coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
        )

        # Define termination settings
        time_termination = propagation_setup.propagator.time_termination(self.tf)

        # relative_distance_termination = propagation_setup.propagator.dependent_variable_termination(
        #     propagation_setup.dependent_variable.relative_distance('primary', 'secondary'),
        #     limit_value=500,
        #     use_as_lower_limit=True,
        #     terminate_exactly_on_final_condition=True,
        #     termination_root_finder_settings=root_finders.bisection(1e-14, 1e-14)
        # )
        
        termination_settings = propagation_setup.propagator.hybrid_termination(
            # [time_termination, relative_distance_termination],
            [time_termination],
            fulfill_single_condition=True
        )

        # Define relative state between bodies (for now only 2 objects)
        dependent_variables_to_save = [
            propagation_setup.dependent_variable.relative_distance('primary', 'secondary'),
            # propagation_setup.dependent_variable.relative_position('primary', 'secondary'),
            # propagation_setup.dependent_variable.relative_velocity('primary', 'secondary'),
        ]

        # Create propagation settings.
        propagator_settings = lambda system_initial_states: propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            system_initial_states,
            self.ti,
            integrator_settings,
            termination_settings,
            output_variables=dependent_variables_to_save
        )

        self.dynamics_simulator_function = lambda system_initial_states: numerical_simulation.create_dynamics_simulator(bodies, propagator_settings(system_initial_states))

        # Propagate the system from the initial state
        self.reset()


    def get_state(self, epoch: float):
        '''
        Returns:
          state: np.ndarray of shape (12,) representing the position and velocity of the primary body and its relative position and velocity from the secondary body.
        '''
        # Extract numerical solution for states and dependent variables
        state = self.primary_history[epoch][:6]
        state_relative = self.secondary_history[epoch][:6] - state

        return np.concatenate((state, state_relative))


    def reset(self):
        # Propagate the system from the initial state
        dynamics_simulator = self.dynamics_simulator_function(self.system_initial_state)

        # Extract the state history of the primary and secondary bodies
        propagation_results = dynamics_simulator.propagation_results
        
        self.primary_history = {}
        self.secondary_history = {}
        for epoch, system_state in propagation_results.state_history.items():
            self.primary_history[epoch] = system_state[:6]
            self.secondary_history[epoch] = system_state[6:]

        # Reset training variables
        self.current_epoch = self.ti
        self.maneuver_done = False
        self.done = False
        self.current_state = self.get_state(self.current_epoch)

        return self.current_state

        
    def step(self, action: np.ndarray, thrust_threshold: float = 0.1):
        '''
          action: np.ndarray of shape (3,) representing the impulsive maneuver in m/s. If norm(action) < threshold, interpret as no maneuver.
        '''

        COLLISION_THRESHOLD = 1e3  # meters

        if self.maneuver_done:
            raise Exception("Maneuver already performed. Reset environment.")
        
        # Check if action is below threshold
        if np.linalg.norm(action) < thrust_threshold:
            self.current_epoch += 10
            if self.current_epoch >= self.tf:
                self.done = True

            return self.get_state(self.current_epoch), 0, self.done, {}
        
        # Perform the impulsive maneuver
        self.maneuver_done = True

        # Update the state of the primary body
        pos = self.primary_history[self.current_epoch][:3]
        vel = self.primary_history[self.current_epoch][3:]
        vel += action

        # Repropagate the orbit with the new initial conditions
        system_state = np.concatenate((pos, vel, self.secondary_history[self.current_epoch]))
        dynamics_simulator = self.dynamics_simulator_function(system_state)

        # Compute miss distance
        relative_distance = list(dynamics_simulator.propagation_results.dependent_variable_history.values())
        miss_distance = min(relative_distance)

        # Compute difference from original orbit
        ...

        # Reward logic
        if miss_distance < COLLISION_THRESHOLD:
            reward = -1000 # Penalize collision
        else:
            reward = 1000 - 10 * np.linalg.norm(action) # Penalize large delta-v
            
        self.done = True
        return self.get_state(self.current_epoch), reward, self.done, {
            'miss_distance': miss_distance,
            'action': action,
            'epoch': self.current_epoch
        }



      



if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
      config = yaml.safe_load(file)

    env = Environment(config['environment'])

    # PLOT TRAJECTORIES
    pos_primary = np.array(list(env.primary_history.values()))[:,0:3]
    pos_secondary = np.array(list(env.secondary_history.values()))[:,0:3]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*pos_primary.T / 1e3, label='Satellite')
    ax.plot(*pos_secondary.T / 1e3, label='Debris')

    ax.plot(*pos_primary[0] / 1e3, 'bx', label='Satellite Start')
    ax.plot(*pos_secondary[0] / 1e3, 'rx', label='Debris Start')

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend()

    plot_earth(ax)

    rescale_plot(ax)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    fig.savefig('plots/trajectories.pdf')
    plt.close()
    
    # PLOT RELATIVE DISTANCE
    time = np.array(list(env.primary_history.keys()))
    rel_distance = np.linalg.norm(pos_primary - pos_secondary, axis=1)
    print(min(rel_distance))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(time, rel_distance / 1e3, label='Relative Distance')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Relative Distance [km]')
    ax.grid()

    fig.savefig('plots/rel_distance.pdf')
    plt.close()

# TODO:
# - Add termination conditions if relative position is below a certain threshold
# - Figure out a way to re-propagate the orbit with different initial conditions (agent knows full state history a-priori and can choose at what epoch to perform an impulsive-maneuver. If a CAM is performed, the new initial state have to be computed and the orbit must be re-propagated.).