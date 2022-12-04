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
        self.kernel = make_grid(k_size,k_size,1)
        self.fibers = make_grid(l, w, p)
        self.particle_loc = make_grid(l, w, 0)
        self.particle_loc[round(w/2),0] = 1

    def draw(self, plot_show=True, grid_lines=False):
        """
        Draws the grid.
        """

        # TODO: update from old shit

        # Make a copy because some implementations
        # of step perform updates in place.
        a = self.fibers.copy()
        p_loc = locs_where(self.particle_loc == 1)[0]
        a[p_loc] = 2 # Different color where particle is
        print(a)
        n, m = a.shape
        plt.axis([0, m, 0, n])
        plt.xticks([])
        plt.yticks([])

        options = dict(interpolation='none', alpha=0.8)
        options['extent'] = [0, m, 0, n]
        plt.imshow(a, cmap if len(locs_where(self.fibers == 0)) > 0 else cmap_no_empty, **options)
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
        arr[dest], arr[source] = arr[source].copy(), arr[dest].copy()
        return arr

    def step(self):
        """Simulate one time step.

        returns: particle location np.array

        """

        options = dict(mode='same', boundary='wrap') # Grid wraps around at boundaries

        source = locs_where(self.particle_loc == 1)[0] # Source is first location where self.particle_loc is 1.
        # locs_where returns an array, not a single loc

        dest = source
        # If there's a fiber at the source, then destination is same as source. If fibers == 0 at source the there is no fiber

        if self.fibers[source] == 0:
            possible_locs = correlate2d(self.particle_loc, self.kernel, **options) # Cross-correlate kernel and particle_loc to figure out where particle should go

            dest = random_loc(locs_where(possible_locs > 0)) # Choose a random particle location

        self.particle_loc = self.move(self.particle_loc, source, dest) # Move by switching source and dest

        return self.particle_loc

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

def make_grid(l, w, prob):
        """Make an array with two types of agents.
        
        n: width and height of the array
        probs: probability of generating a 1
        
        return: NumPy array
        """

        r = np.random.rand(l, w) > (1-prob)
        r = r.astype(int)
        print(r)
        return r

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