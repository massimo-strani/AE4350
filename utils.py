import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

EARTH_RADIUS = 6371 # [km]

def rescale_plot(axes: plt.Axes):
    x_lim = axes.get_xlim()
    y_lim = axes.get_ylim()
    z_lim = axes.get_zlim()

    x_mid = np.mean(x_lim)
    y_mid = np.mean(y_lim)
    z_mid = np.mean(z_lim)

    radius = 0.5 * max([x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]])

    axes.set_xlim(x_mid - radius, x_mid + radius)
    axes.set_ylim(y_mid - radius, y_mid + radius)
    axes.set_zlim(z_mid - radius, z_mid + radius)

    axes.set_box_aspect([1, 1, 1])  # Equal aspect ratio

def plot_earth(axes: plt.Axes):
    # Create a sphere to represent the Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface of the Earth
    axes.plot_surface(x, y, z, color='b', alpha=0.3)


def visualize_episode(state_history: np.ndarray, save_path: str = 'plots/episode.mp4'):
    primary_state = state_history[:, 1:4]
    debris_state = state_history[:, 7:10]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    plot_earth(ax)

    trajectories = [
        ax.plot([], [], [], '-', label='Satellite')[0],
        ax.plot([], [], [], '-', label='Debris')[0]
    ]

    ax.figure.canvas.draw()
    # background = ax.figure.canvas.copy_from_bbox(ax.bbox)

    # Restore the background
    # ax.figure.canvas.restore_region(background)

    def update(frame):
        # Update the data for each trajectory
        for traj, state in zip(trajectories, [primary_state, debris_state]):
            traj.set_data(state[:frame, 0], state[:frame, 1])
            traj.set_3d_properties(state[:frame, 2])

        return trajectories

    ani = animation.FuncAnimation(fig, update, frames=len(state_history), interval=1, blit=True, repeat=False)
    ani.save(save_path, writer='ffmpeg', fps=30)
   