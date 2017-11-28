import numpy as np
import re

PI = np.pi

class Dataset(object):

    def __init__(self, args):
        # Read point cloud
        pnts = []
        f = open(args.filename, 'r')
        for line in f:
            striped = line.strip()
            pnt = striped.split()
            pnts.append(map(float, pnt))
        f.close()
        pnts = np.array(pnts)

        self.training_data = pnts / abs(pnts).max()
        self.n_training_data = len(pnts)
        self.index_in_epoch = 0
        np.random.shuffle(self.training_data)

    def get_training_data(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.n_training_data:
            np.random.shuffle(self.training_data)
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.n_training_data
        end = self.index_in_epoch
        return self.training_data[start:end]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--filename', type=str, default='data/bunny.xyz')
        return parser.parse_args()

    args = parse_args()
    dataset = Dataset(args)
    data = dataset.training_data[:10000]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=0.1, c=data[:, 2], cmap=plt.cm.jet)
    ax.set_aspect('equal')
    ax.view_init(15, -30)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    plt.show()
