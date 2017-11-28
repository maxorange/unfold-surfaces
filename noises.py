import numpy as np
PI = np.pi

class Noise(object):

    def __init__(self, args, vrange):
        # Sample noise for test data
        n_ls = np.ceil(pow(args.n_test_data, 1./args.nz))
        ls = np.linspace(-vrange, vrange, n_ls)
        xs = [ls for i in range(args.nz)]
        mg = np.meshgrid(*xs)
        test_data = []
        for i in range(args.nz):
            flattened = mg[i].flatten()
            test_data.append(np.expand_dims(flattened, -1))
        self.test_data = np.concatenate(test_data, -1)
        self.nz = args.nz

class Uniform(Noise):

    def __init__(self, args):
        super(Uniform, self).__init__(args, 1)

    def sample(self, batch_size):
        return np.random.uniform(-1, 1, [batch_size, self.nz])

class Normal(Noise):

    def __init__(self, args):
        super(Normal, self).__init__(args, 3)

    def sample(self, batch_size):
        return np.random.normal(0, 1, [batch_size, self.nz])

class UniformCircle(object):

    def __init__(self, args):
        self.test_data = self.sample(args.n_test_data)

    def sample(self, batch_size):
        t = np.random.uniform(0, 2*PI, [batch_size, 1])
        r = np.sqrt(np.random.uniform(0, 1, [batch_size, 1]))
        x = r*np.cos(t)
        y = r*np.sin(t)
        return np.concatenate((x, y), -1)

class UniformSphericalSurface(object):

    def __init__(self, args):
        self.test_data = self.sample(args.n_test_data)

    def sample(self, batch_size):
        t = np.random.uniform(0, 2*PI, [batch_size, 1])
        z = np.random.uniform(-1, 1, [batch_size, 1])
        x = np.sqrt(1-z*z)*np.cos(t)
        y = np.sqrt(1-z*z)*np.sin(t)
        return np.concatenate((x, y, z), -1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--nz', type=int, default=2)
        parser.add_argument('--n_test_data', type=int, default=10000)
        return parser.parse_args()

    args = parse_args()
    noise = UniformSphericalSurface(args)
    data = noise.test_data

    # plt.scatter(data[:, 0], data[:, 1], 1)
    # plt.xlim([-4, 4])
    # plt.ylim([-4, 4])
    # plt.axes().set_aspect('equal', 'box')
    # plt.show()

    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=0.1, c=data[:, 2], cmap=plt.cm.jet)
    ax.set_aspect('equal')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()
