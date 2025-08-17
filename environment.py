import matplotlib.pyplot as plt

from tudatpy import constants, numerical_simulation
from tudatpy.math import root_finders
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.util import result2array
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

        EARTH_GM = body_settings.get('Earth').gravity_field_settings.gravitational_parameter

        # Create satellite and secondary body(ies)
        bodies_to_propagate = []
        system_initial_state = []

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
            system_initial_state.extend(body_params['X0'])

        central_bodies = ['Earth'] * len(bodies_to_propagate)        # Define one central body per body to propagate
        bodies = environment_setup.create_system_of_bodies(body_settings) # Create environment

        # Create acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )
        
        # CREATE PROPAGATION SETTINGS
        # Create numerical integrator settings.
        # step_size_control_settings = propagation_setup.integrator.step_size_control_custom_blockwise_scalar_tolerance(
        #     block_indices_function=propagation_setup.integrator.standard_cartesian_state_element_blocks,
        #     relative_error_tolerance=1e-12,
        #     absolute_error_tolerance=1e-12,
        # )

        # step_size_validation_settings = propagation_setup.integrator.step_size_validation(
        #     minimum_step=1e-3,
        #     maximum_step=1000,
        # )

        # integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        #     initial_time_step=10, 
        #     coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
        #     step_size_control_settings=step_size_control_settings,
        #     step_size_validation_settings=step_size_validation_settings,
        # )

        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step=10,
            coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
        )

        # Define relative state between bodies (for now only 2 objects)
        dependent_variables_to_save = [
            propagation_setup.dependent_variable.relative_distance('primary', 'debris1'),
            propagation_setup.dependent_variable.relative_position('primary', 'debris1'),
            propagation_setup.dependent_variable.relative_velocity('primary', 'debris1'),
        ]

        # Define termination settings
        time_termination = propagation_setup.propagator.time_termination(self.tf)

        # relative_distance_termination = propagation_setup.propagator.dependent_variable_termination(
        #     propagation_setup.dependent_variable.relative_distance('primary', 'debris1'),
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

        # Create propagation settings.
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            system_initial_state,
            self.ti,
            integrator_settings,
            termination_settings,
            output_variables=dependent_variables_to_save,
        )

        self.dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)

        # Extract numerical solution for states and dependent variables
        propagation_results = self.dynamics_simulator.propagation_results

        state_history = result2array(propagation_results.state_history)
        self.tout = state_history[:, 0]
        self.state_history = state_history[:, 1:]

        self.dependent_variables = result2array(propagation_results.dependent_variable_history)[:, 1:]

    def get_termination_details(self):
        if self.tout[-1] == self.tf:
            print('Time termination. NO COLLISION.')

            miss_distance_idx = np.argmin(self.dependent_variables[:, 0])
            print(f'Miss distance: {self.dependent_variables[miss_distance_idx, 0]} m')
            print(f'TCA: {self.tout[miss_distance_idx]} s')

        else:
            print('Relative distance termination. COLLISION POSSIBLE.')
            print(f'Miss distance: {self.dependent_variables[-1,0]} m')
            print(f'TCA: {self.tout[-1]} s')
        
        
      



if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
      config = yaml.safe_load(file)

    env = Environment(config['environment'])
    env.get_termination_details()

    # PLOT TRAJECTORIES
    pos_primary = env.state_history[:,0:3]
    pos_secondary = env.state_history[:,6:9]

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
    rel_distance = env.dependent_variables[:,0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(env.tout, rel_distance / 1e3, label='Relative Distance')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Relative Distance [km]')
    ax.grid()

    fig.savefig('plots/rel_distance.pdf')
    plt.close()

# TODO:
# - Add termination conditions if relative position is below a certain threshold
# - Figure out a way to re-propagate the orbit with different initial conditions (agent knows full state history a-priori and can choose at what epoch to perform an impulsive-maneuver. If a CAM is performed, the new initial state have to be computed and the orbit must be re-propagated.).