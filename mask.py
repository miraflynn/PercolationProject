import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

palette = sns.color_palette('muted')
colors = 'white', palette[1], palette[0]
cmap = LinearSegmentedColormap.from_list('cmap', colors)
cmap_no_empty = LinearSegmentedColormap.from_list('cmap', (palette[1], palette[0]))

class Mask:
    def __init__(self, l, w, k_size, p):
        self.array = np.zeros((l, w))
        self.kernel = np.zeros((k_size, k_size))
        self.make_grid(l, w, p)
        self.make_kernel(k_size)
        # self.particle_loc = (w/2, l)
        self.particle_loc = np.zeros((l,w))
        self.particle_loc[round(w/2),0] = 1


    @property
    def grid(self):
        return self.array

    def make_kernel(self, k_size):
        size = k_size
        shape = (k_size, k_size)
        # Variable radius kernel. I don't know how to do this with 2d list
        # comprehension stuff so I just for looped it. It also is only 
        # run once and kernel isn't very large.
        self.kernel = np.zeros(shape, dtype=np.int8)
        for i in range(0, k_size):
            for j in range(0, k_size):
                # print(i, j)
                self.kernel[i, j] = 1

    def make_grid(self, l, w, prob):
        """Make an array with two types of agents.
        
        n: width and height of the array
        probs: probability of generating a 0, 1, or 2
        
        return: NumPy array
        """

        grid = np.random.rand(l, w) > (1-prob)
        self.array = grid
        # choices = np.array([0, 1], dtype=np.int8)
        # self.array = np.random.choice(choices, (l, w), p=prob)

    def draw(self, plot_show=True, grid_lines=False):
        """
        Draws the grid.
        """

        # TODO: update from old shit

        # Make a copy because some implementations
        # of step perform updates in place.
        a = self.particle_loc.copy()
        n, m = a.shape
        plt.axis([0, m, 0, n])
        plt.xticks([])
        plt.yticks([])

        options = dict(interpolation='none', alpha=0.8)
        options['extent'] = [0, m, 0, n]
        plt.imshow(a, cmap if len(locs_where(self.particle_loc == 0)) > 0 else cmap_no_empty, **options)
        if grid_lines:
            ax = plt.gca()
            ax.set_xticks(np.arange(0, self.array.shape[0], 1))
            ax.set_yticks(np.arange(0, self.array.shape[1], 1))
            ax.grid(color='k')
        if plot_show:
            plt.show()

    def move(self, arr, source, dest):
        """Swap the agents at source and dest.

        source: location tuple
        dest: location tuple
        """
        arr[dest], arr[source] = arr[source].copy(), arr[dest].copy()  # Need these .copy for the Homo/Hetero model
        return arr

    def step(self):
        """Simulate one time step.

        threshold: percent of same-color neighbors needed to be happy

        returns the number of unhappy locations
        """

        options = dict(mode='same', boundary='wrap')
        source = locs_where(self.particle_loc == 1)[0]
        print(source)
        possible_locs = correlate2d(self.particle_loc, self.kernel, **options)
        # print(possible_locs)
        dest = random_loc(locs_where(possible_locs > 0))
        print(dest)
        self.particle_loc = self.move(self.particle_loc, source, dest)
        return self.particle_loc
        

        # unhappy_locs = self.find_unhappy(threshold)
        # if len(unhappy_locs) > 0:
        #     empty_locs = utils.locs_where(self.grid == 0)
        #     source = utils.random_loc(unhappy_locs)
        #     dest = utils.random_loc(empty_locs)
        #     self.move(source, dest)
        # return len(unhappy_locs)

    def loop(self, num_steps=1000):
        for _ in range(num_steps):
            self.step()
        return self

    # def loop_until_done(self, threshold=0.375, max_steps=10000):
    #     for i in (range(max_steps) if max_steps > 0 else itertools.count()):
    #         num_unhappy = self.step(threshold)
    #         if num_unhappy == 0:
    #             return self.avg_percent_same(), i
    #     return self.avg_percent_same(), max_steps


def locs_where(condition):
    """Find cells where a boolean array is True.

    condition: NumPy array

    return: list of coordinate pairs
    """
    ii, jj = np.nonzero(condition)
    return list(zip(ii, jj))


def random_loc(locs):
    """Choose a random element from a list of tuples.

    locs: list of tuples

    return: tuple
    """
    index = np.random.choice(len(locs))
    return locs[index]