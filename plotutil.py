import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import LinearSegmentedColormap

def save_plot(args, step, x_real, x_fake, z_test):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.scatter(x_real[:, 0], x_real[:, 1], x_real[:, 2], s=0.1, c=x_real[:, 2], cmap=plt.cm.Spectral)
    ax.scatter(x_fake[:, 0], x_fake[:, 1], x_fake[:, 2], s=0.1, c=z_test[:, 1], cmap=plt.cm.jet)
    ax.set_aspect('equal')
    ax.view_init(15, -30)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    plt.savefig('out/{0:06d}.png'.format(step))
    plt.close()

def colormap(colors):
    values = range(len(colors))
    vmax = float(max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v/vmax, c))
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)
