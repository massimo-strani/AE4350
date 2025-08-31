from gymnasium.spaces import Box

import numpy as np
import scipy
import yaml
import torch

from utils import *
from PPO import PPO

RANDOM_SEED = 42

GM = 398600 # [km3/s2]
NOMINAL_ORBIT = {
    'sma': 6871,  # [km]
    'ecc': 0.0001, # [-]
    'aop': 0      # [rad]
}

class Environment:
    def __init__(self, nominal_orbit: dict = NOMINAL_ORBIT, seed: int = RANDOM_SEED):
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = Box(low=-0.01, high=0.01, shape=(2,), dtype=np.float32)

        self.nominal_orbit = nominal_orbit

    def _propagate_body_to_epoch(self, sma: float, ecc: float, th: float, delta_t: float):
        n = np.sqrt(GM / sma ** 3)
        
        E = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(th / 2))
        M = E - ecc * np.sin(E)
        M += n * delta_t

        E = scipy.optimize.fsolve(lambda x: x - ecc * np.sin(x) - M, M)[0]
        th = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E / 2))

        return th

    def _propagate_system_to_epoch(self, sat_th: float, obj_th: float, delta_t: float):
        ths = [sat_th, obj_th]

        for i, orbital_elements in enumerate([self.sat_elements, self.obj_elements]):
            sma = orbital_elements['sma']
            ecc = orbital_elements['ecc']

            ths[i] = self._propagate_body_to_epoch(sma, ecc, ths[i], delta_t)

        return ths


    def _keplerian2cartesian(self, sma: float, ecc: float, aop: float, th: float):
        p = sma * (1 - ecc ** 2)
        
        r = p / (1 + ecc * np.cos(th))
        r_vec = r * np.array([[np.cos(th)], [np.sin(th)]])
        r_vec = np.array([[np.cos(aop), -np.sin(aop)],
                          [np.sin(aop),  np.cos(aop)]]) @ r_vec

        v_vec = np.sqrt(GM / p) * np.array([[ecc * np.sin(th)], [1 + ecc * np.cos(th)]])
        v_vec = np.array([[np.cos(aop + th), -np.sin(aop + th)],
                          [np.sin(aop + th),  np.cos(aop + th)]]) @ v_vec

        return r_vec.reshape(-1), v_vec.reshape(-1)

    def _cartesian2keplerian(self, r_vec: np.ndarray, v_vec: np.ndarray):
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # specific angular momentum
        h = np.cross(r_vec, v_vec)

        # eccentricity
        ecc_vec = np.array([v_vec[1] * h, -h * v_vec[0]]) / GM - r_vec / r
        ecc = np.linalg.norm(ecc_vec)

        # semi-major axis
        sma = (2 / r - v ** 2 / GM) ** -1

        # argument of periapsis
        aop = np.arctan2(ecc_vec[1], ecc_vec[0])

        # true anomaly at TCA
        th = np.arccos(np.dot(ecc_vec, r_vec) / (ecc * r))
        if np.dot(r_vec, v_vec) < 0:
            th = 2 * np.pi - th

        return {'sma': sma, 'ecc': ecc, 'aop': aop, 'th': th} 


    def get_observation(self):
        '''
        Returns:
          state: np.ndarray of shape (9,) representing the cartesian position and velocity of the satellite, its relative position and velocity from the debris body, and the cumulative delta-V.
        '''
        
        r_sat, v_sat = self._keplerian2cartesian(**self.sat_elements)
        r_obj, v_obj = self._keplerian2cartesian(**self.obj_elements)

        r_rel = r_obj - r_sat
        v_rel = v_obj - v_sat

        obs = np.hstack((r_sat, v_sat, r_rel, v_rel))
        return np.append(obs, self.cumsum_deltaV)


    def reset(self):
        # generate random TCA
        T = 2 * np.pi * np.sqrt(self.nominal_orbit['sma'] ** 3 / GM)
        self.TCA = np.random.uniform(T / 2, 2 * T)

        # compute true anomaly, cartesian position and velocity of the satellite at TCA
        self.sat_th_TCA = np.random.uniform(0, 2 * np.pi)
        sat_r_TCA, sat_v_TCA = self._keplerian2cartesian(**self.nominal_orbit, th=self.sat_th_TCA)

        # generate debris position and velocity at TCA
        r_offset = np.random.uniform(-1,1,(2,)) # (2,) [km]
        v_offset = np.random.uniform(-1,1,(2,)) # (2,) [km/s]

        obj_r_TCA = sat_r_TCA + r_offset # (2,)
        obj_v_TCA = sat_v_TCA + v_offset # (2,)

        # initialize satellite and debris orbital elements
        self.sat_elements = self.nominal_orbit.copy()
        self.obj_elements = self._cartesian2keplerian(obj_r_TCA, obj_v_TCA)
        
        self.obj_th_TCA = self.obj_elements['th']

        # propagate satellite and debris backward to initial epoch
        ths = self._propagate_system_to_epoch(self.sat_th_TCA, self.obj_th_TCA, -self.TCA)
        self.sat_elements['th'] = ths[0]
        self.obj_elements['th'] = ths[1]

        # Reset training variables
        self.state_history = [self.sat_elements.copy()]
        self.cumsum_deltaV = 0
        self.epoch = 0
        self.done = False

        return self.get_observation()

        
    def step(self, action: np.ndarray, time_step: float = 60):
        '''
          action: np.ndarray of shape (2,) representing the impulsive maneuver in km/s.
        '''

        # action is rescaled to match dimensions [km/s]
        action = self.action_space.low + .5 * (self.action_space.high - self.action_space.low) * (action + 1)

        COLLISION_THRESHOLD = 1 # [km]
        DELTAV_THRESHOLD = 0.02 # [km/s]

        reward = 0

        # UPDATE SYSTEM
        # Compute and store deltaV 
        deltaV = np.linalg.norm(action)
        self.cumsum_deltaV += deltaV

        # apply deltaV and recompute satellite's orbit
        sat_r, sat_v = self._keplerian2cartesian(**self.sat_elements)
        sat_v += action
        self.sat_elements = self._cartesian2keplerian(sat_r, sat_v)

        # propagate satellite and debris states one epoch forward
        ths = self._propagate_system_to_epoch(self.sat_elements['th'], self.obj_elements['th'], time_step)
        self.sat_elements['th'] = ths[0]
        self.obj_elements['th'] = ths[1]

        # store state in the history
        self.state_history.append(self.sat_elements.copy())

        # get observation and update info
        obs = self.get_observation()
        self.epoch += time_step

        info_dict = {
            'termination reason': None,
            'relative distance': None,
            'orbital elements': self.sat_elements,
            'action': action,
            'cumulative deltaV': self.cumsum_deltaV,
            'epoch': self.epoch
        }

        # CHECK COLLISION CONDITION
        relative_distance = np.linalg.norm(obs[4:6])
        info_dict['relative distance'] = relative_distance

        if relative_distance < COLLISION_THRESHOLD:
            # large penalty for collision
            reward -= 100
            self.done = True

            info_dict['termination reason'] = 'Collision detected'
            return obs, reward, self.done, info_dict

        # ORBIT DIVERGENCE REWARD
        delta_sma = np.abs(self.sat_elements['sma'] - self.nominal_orbit['sma']) / self.nominal_orbit['sma']
        delta_ecc = np.abs(self.sat_elements['ecc'] - self.nominal_orbit['ecc']) / (self.nominal_orbit['ecc'] + 1e-8)
        delta_aop = np.abs(self.sat_elements['aop'] - self.nominal_orbit['aop']) / np.pi

        reward -= (delta_sma + delta_ecc + delta_aop)

        # DeltaV REWARD
        reward -= 5 * deltaV / DELTAV_THRESHOLD

        if self.epoch > self.TCA:
            reward += 1 # small positive reward for missing the debris
            self.done = True

            info_dict['termination reason'] = 'CAM successful'

            return obs, reward, self.done, info_dict

        # MISS DISTANCE REWARD
        # propagate the two orbits to TCA to compute miss distance
        ths = [self.sat_elements['th'], self.obj_elements['th']]

        propagated_distance = np.zeros(int((self.TCA - self.epoch) / time_step) + 1)
        for i in range(propagated_distance.shape[0]):
            ths = self._propagate_system_to_epoch(*ths, time_step)

            sat_r, _ = self._keplerian2cartesian(self.sat_elements['sma'], self.sat_elements['ecc'], self.sat_elements['aop'], ths[0])
            obj_r, _ = self._keplerian2cartesian(self.obj_elements['sma'], self.obj_elements['ecc'], self.obj_elements['aop'], ths[1])

            propagated_distance[i] = np.linalg.norm(sat_r - obj_r)

        miss_distance = np.min(propagated_distance)

        # compute reward
        if miss_distance < COLLISION_THRESHOLD:
            reward -= (COLLISION_THRESHOLD - miss_distance) / COLLISION_THRESHOLD

        return obs, reward, self.done, info_dict


if __name__ == '__main__':
    env = Environment()

    for i in range(5):
        observation = env.reset()
        plot_nominal_orbits(env, show=True, save=f'plots/nominal_orbits_{i}.pdf')
