from pdb import run
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', size=14)


def plot_nominal_orbits(env, show: bool = False, save: str = None):
    fig, ax = plt.subplots(figsize=(6,6))

    th_TCA = [env.sat_th_TCA, env.obj_th_TCA]

    th = np.linspace(0, 2 * np.pi, 100)
    for i, orbital_elements in enumerate([env.nominal_orbit, env.obj_elements]):
        sma = orbital_elements['sma']
        ecc = orbital_elements['ecc']
        aop = orbital_elements['aop']

        r = sma * (1 - ecc ** 2) / (1 + ecc * np.cos(th))

        r_vec = r * np.array([np.cos(th), np.sin(th)])
        r_vec = np.array([[np.cos(aop), -np.sin(aop)],
                      [np.sin(aop),  np.cos(aop)]]) @ r_vec

        # plot orbit
        ax.plot(r_vec[0], r_vec[1], '--', label=['Satellite Orbit', 'Debris Orbit'][i], color=['tab:blue', 'tab:red'][i])

        # compute initial position and position at TCA
        th_0 = env._propagate_body_to_epoch(sma, ecc, th_TCA[i], -env.TCA)
        r_0, _ = env._keplerian2cartesian(sma, ecc, aop, th_0)
        r_TCA, _ = env._keplerian2cartesian(sma, ecc, aop, th_TCA[i])

        # plot initial position and position at TCA
        ax.plot(*r_0, 'o', label=['Satellite @ t$_0$', 'Debris @ t$_0$'][i], color=['tab:blue', 'tab:red'][i])

    ax.plot(*r_TCA, '^', label='Debris @ TCA', color='tab:red')
    ax.set_title('TCA : {:.1f} min'.format(env.TCA / 60))
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper right')
    ax.grid()

    fig.tight_layout()

    if show:
        plt.show()

    if save:
        fig.savefig(save)
        plt.close()

    return fig, ax

def plot_log(log_path: str, show: bool = False, save: str = None, figure: tuple[plt.Figure, plt.Axes] = (None, None), **kwargs):
    data = pd.read_csv(log_path)
    data = pd.DataFrame(data)

    label = kwargs.get('label', None)
    color = kwargs.get('color', None)

    # episodes = data[:,0]
    data[label] = data['reward'].rolling(window=10,win_type='triang',min_periods=1).mean()
    data['var'] = data['reward'].rolling(window=1,win_type='triang',min_periods=1).mean()

    if not any(figure):
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig, ax = figure

    data.plot(kind='line', x='episode', y=label, alpha=1, ax=ax, linewidth=1.5, color=color)
    data.plot(kind='line', x='episode', y='var', alpha=0.5, ax=ax, color=color)

    ax.set_ylabel('Average Episodic Reward')
    ax.set_xlabel('Episodes')
    
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    for i in range(len(handles)):
        if(i%2 == 0):
            new_handles.append(handles[i])
            new_labels.append(labels[i])
    ax.legend(new_handles, new_labels, loc='lower right')

    ax.grid(True)
    ax.set_xlim(right=2700)
    fig.tight_layout()

    if show:
        plt.show()

    if save:
        fig.savefig(save)
        plt.close()

    return fig, ax
   