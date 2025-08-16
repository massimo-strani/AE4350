import matplotlib.pyplot as plt

from tudatpy import constants, numerical_simulation
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.util import result2array
from tudatpy.interface import spice

import numpy as np
import yaml

from utils import *

class Environment:
    def __init__(self, environment_config: dict):
        self.num_bodies = len(environment_config['bodies_to_propagate'])

        # CREATE ENVIRONMENT & ACCELERATION SETTINGS
        # Create Earth
        spice.load_standard_kernels()
        body_settings = environment_setup.get_default_body_settings(['Earth'], 'Earth', 'J2000')

        EARTH_GM = body_settings.get('Earth').gravity_field_settings.gravitational_parameter

        # Create satellite and secondary body(ies)
        bodies_to_propagate = [''] * self.num_bodies
        system_initial_state = np.zeros((6 * self.num_bodies,))

        acceleration_settings = {}
        for i, (body_name, initial_states) in enumerate(environment_config['bodies_to_propagate'].items()):
            body_settings.add_empty_settings(body_name)

            # Use following lines for additional environment settings (e.g. SRP, drag) of type dict[str, dict]
            # for body_name, dict_of_settings in environment_settings.items():
            #   for setting_name in dict_of_settings.keys():
            #     setattr(body_settings.get(body_name), setting_name, dict_of_settings[setting_name])

            # Define accelerations on bodies (could make this flexible by passing a dict[str, list] from config)
            acceleration_settings[body_name] = {'Earth': [propagation_setup.acceleration.point_mass_gravity()]} 

            bodies_to_propagate[i] = body_name
            system_initial_state[6*i:6*(i+1)] = initial_states

        central_bodies = ['Earth'] * len(bodies_to_propagate)        # Define one central body per body to propagate
        bodies = environment_setup.create_system_of_bodies(body_settings) # Create environment

        # Create acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )
        
        # CREATE PROPAGATION SETTINGS
        # Create numerical integrator settings.
        step_size_control_settings = propagation_setup.integrator.step_size_control_custom_blockwise_scalar_tolerance(
            block_indices_function=propagation_setup.integrator.standard_cartesian_state_element_blocks,
            relative_error_tolerance=1e-12,
            absolute_error_tolerance=1e-12,
        )

        step_size_validation_settings = propagation_setup.integrator.step_size_validation(
            minimum_step=1e-3,
            maximum_step=1000,
        )


        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
            initial_time_step=10, 
            coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
            step_size_control_settings=step_size_control_settings,
            step_size_validation_settings=step_size_validation_settings,
        )

        # Define relative state between bodies (for now only 2 objects)
        dependent_variables_to_save = [
            propagation_setup.dependent_variable.relative_position('primary', 'debris1'),
            propagation_setup.dependent_variable.relative_velocity('primary', 'debris1'),
        ]

        # Define termination settings
        time_termination = propagation_setup.propagator.time_termination(environment_config.get('final_time', 86400))

        relative_distance_termination = propagation_setup.propagator.dependent_variable_termination(
            propagation_setup.dependent_variable.relative_distance('primary', 'debris1'),
            limit_value=1000.0,
            use_as_lower_limit=True,
        )
        
        termination_settings = propagation_setup.propagator.hybrid_termination(
            [time_termination, relative_distance_termination],
            fulfill_single_condition=True
        )

        # Create propagation settings.
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            system_initial_state,
            environment_config.get('initial_time', 0),
            integrator_settings,
            termination_settings,
            output_variables=dependent_variables_to_save,
        )

        self.dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)

        # Extract numerical solution for states and dependent variables
        propagation_results = self.dynamics_simulator.propagation_results

        self.state_history = propagation_results.state_history
        self.dependent_variables = propagation_results.dependent_variable_history
      



if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
      config = yaml.safe_load(file)

    env = Environment(config['environment'])

    # print(env.dynamics_simulator.propagation_termination_details.was_condition_met_when_stopping)
    state_history = result2array(env.state_history)

    # PLOT TRAJECTORIES
    primary_state = state_history[:,1:4]
    secondary_state = state_history[:,7:10]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*primary_state.T / 1e3, label='Satellite')
    ax.plot(*secondary_state.T / 1e3, label='Debris')

    ax.plot(*primary_state[0] / 1e3, 'bx', label='Satellite Start')
    ax.plot(*secondary_state[0] / 1e3, 'rx', label='Debris Start')

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
    dependent_variables = result2array(env.dependent_variables)

    time = dependent_variables[:,0]
    rel_distance = dependent_variables[:,1:4]
    rel_distance = np.linalg.norm(rel_distance, axis=1)

    print(rel_distance[0])

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