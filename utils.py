import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

def plot_nominal_orbits(env, show: bool = False, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6,6))

    th_TCA = [env.sat_th_TCA, env.obj_th_TCA]

    th = np.linspace(0, 2 * np.pi, 100)
    for i, orbital_elements in enumerate([env.sat_elements, env.obj_elements]):
        sma = orbital_elements['sma']
        ecc = orbital_elements['ecc']
        aop = orbital_elements['aop']

        r = sma * (1 - ecc ** 2) / (1 + ecc * np.cos(th))

        r_vec = r * np.array([np.cos(th), np.sin(th)])
        r_vec = np.array([[np.cos(aop), -np.sin(aop)],
                      [np.sin(aop),  np.cos(aop)]]) @ r_vec

        # plot orbit
        ax.plot(r_vec[0], r_vec[1], '--', label=['Satellite', 'Debris'][i], color=['tab:blue', 'tab:red'][i])

        # compute initial position and position at TCA
        th_0 = env._propagate_body_to_epoch(sma, ecc, th_TCA[i], -env.TCA)
        r_0, _ = env._keplerian2cartesian(sma, ecc, aop, th_0)
        r_TCA, _ = env._keplerian2cartesian(sma, ecc, aop, th_TCA[i])

        # plot initial position and position at TCA
        ax.plot(*r_0, 'o', label=['Satellite Start', 'Debris Start'][i], color=['tab:blue', 'tab:red'][i])
        ax.plot(*r_TCA, '^', label=['Satellite TCA', 'Debris TCA'][i], color=['tab:blue', 'tab:red'][i])

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper right')
    ax.grid()

    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)
        plt.close()
   