import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from time import sleep
import matplotlib.animation as animation
import random

# palette = sns.color_palette('muted')
palette = sns.color_palette('bright')
# colors = 'white', palette[1], palette[0]
colors = 'white', 'red', 'green', 'blue'
cmap = LinearSegmentedColormap.from_list('cmap', colors)
cmap_no_empty = LinearSegmentedColormap.from_list('cmap', (palette[1], palette[0]))

class Mask:
    def __init__(self, l, w, k_size, p):
        self.kernel = make_grid(k_size,k_size,1)
        self.fibers = make_grid(l, w, p)
        self.particles = [(1,0), (0,0)]

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
        # print(a)
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
    '''
    def move(self, arr, source, dest):
        """Swap the agents at source and dest.

        source: location tuple
        dest: location tuple
        """
        #arr[dest], arr[source] = arr[source].copy(), arr[dest].copy()
        
        return arr
    '''

    def step(self):
        """Simulate one time step.

        returns: particle location np.array

        """
        for particle in enumerate(len(self.particles)):
            particle_loc = make_grid(l, w, 0)
            particle_loc[self.particles[particle]] = 1
            options = dict(mode='same', boundary='wrap') # Grid wraps around at boundaries

            source = self.particles# Source is first location where self.particle_loc is 1.
            # locs_where returns an array, not a single loc

            dest = source
            # If there's a fiber at the source, then destination is same as source. If fibers == 0 at source the there is no fiber
            for particle in enumerate():
                if self.fibers[particle_locs] == 0:
                    possible_locs = correlate2d(self.particle_loc, self.kernel, **options) # Cross-correlate kernel and particle_loc to figure out where particle should go

                    dest = random_loc(locs_where(possible_locs > 0)) # Choose a random particle location

            particle = self.move(particle_loc, source, dest) # Move by switching source and dest

        return self.particle_lo

    def loop(self, num_steps=1000):
        for _ in range(num_steps):
            self.step()
        return self

    def animate(self, frames, interval=None, step=None):
        """Animate the automaton.
        
        frames: number of frames to draw
        interval: time between frames in seconds
        iters: number of steps between frames
        """
        if step is None:
            step = self.step
            
        plt.figure()
        try:
            for i in range(frames-1):
                self.draw()
                plt.show()
                if interval:
                    sleep(interval)
                step()
                #clear_output(wait=True)
                
            self.draw()
            plt.show()
        except KeyboardInterrupt:
            pass

    # def loop_until_done(self, threshold=0.375, max_steps=10000):
    #     for i in (range(max_steps) if max_steps > 0 else itertools.count()):
    #         num_unhappy = self.step(threshold)
    #         if num_unhappy == 0:
    #             return self.avg_percent_same(), i
    #     return self.avg_percent_same(), max_steps





class CA_percolation:
    def __init__(self, l, w, p):
        self.holes = make_grid(l, w, p)
        self.fluid = np.zeros((l, w), dtype=np.int8)
        self.fluid[0,0] = 1
        self.kernel = np.ones((3,3),dtype=np.int8)
        
    
    def step(self):
        self.fluid = (correlate2d(self.fluid, self.kernel, mode="same") * self.holes) > 0

    def draw(self, plot_show=True, grid_lines=False):
        a = self.holes == 0
        
        a = a + self.fluid * 2
        # print(a)
        n, m = a.shape
        plt.axis([0, m, 0, n])
        plt.xticks([])
        plt.yticks([])

        options = dict(interpolation='none', alpha=0.8)
        options['extent'] = [0, m, 0, n]
        plt.imshow(a, cmap, **options)
        if grid_lines:
            ax = plt.gca()
            ax.set_xticks(np.arange(0, self.array.shape[0], 1))
            ax.set_yticks(np.arange(0, self.array.shape[1], 1))
            ax.grid(color='k')
        if plot_show:
            plt.show()




MOVEMENT_KERNEL = np.asarray(
    [[1,1,1],
    [1,0,1],
    [0,0,0]],
    dtype= np.int8
)


class Particle:
    def __init__(self, grid, pos = (0,0), radius=1):
        self.pos = pos
        self.radius = 1
        self.grid = grid
        self.stuck = False
        self.stickyness = 1
        

    def move(self):
        if self.stuck:
            return True
        position_grid = np.zeros(np.shape(self.grid), dtype=np.int8)
        position_grid[self.pos] = 1
        movement_grid = correlate2d(position_grid, MOVEMENT_KERNEL, mode="same")
        # print(movement_grid)
        next_pos = random_loc(locs_where(movement_grid))

        self.stuck = self.grid[next_pos] == 1

        self.pos = next_pos
    def made_through(self, l):
        if self.pos[0] == l+1:
            return True
    
    

class AgentSimulation:
    def __init__(self, l, w, p, num_particles):
        self.grid = np.concatenate([np.zeros((2,w), dtype=np.int8),make_grid(l,w,p)])
        self.particles = self.construct_particles(num_particles, w)
        self.l = l
        # print(l)
    
    def construct_particles(self, num_particles, start_range):
        particles = []
        for i in range(num_particles):
            particles.append(Particle(self.grid,(0, random.randrange(start_range))))
        return particles
    
    def step(self):
        num_still_going = 0
        for particle in self.particles:
            particle.move()
            if(particle.stuck or particle.made_through(self.l)):
                pass
            else:
                num_still_going += 1

        return num_still_going

    def count_made_through(self):
        c = 0
        # a = []
        for particle in self.particles:
            if particle.made_through(self.l):
                c += 1
        return c
                
    
    def draw(self, plot_show=True, grid_lines=False, animate=False):
        a = self.grid.copy()

        for particle in self.particles:
            a[particle.pos] = 2
        # print(a)
        n, m = a.shape
        plt.axis([0, m, 0, n])
        plt.xticks([])
        plt.yticks([])

        options = dict(interpolation='none', alpha=0.8)
        options['extent'] = [0, m, 0, n]
        plt.imshow(a, cmap, **options)
        if grid_lines:
            ax = plt.gca()
            ax.set_xticks(np.arange(0, self.array.shape[0], 1))
            ax.set_yticks(np.arange(0, self.array.shape[1], 1))
            ax.grid(color='k')
        if plot_show:
            plt.show()

    def simulate(self, frames):
        states = []
        for i in range(frames):
            a = self.grid.copy()
            for particle in self.particles:
                a[particle.pos] = min(3, a[particle.pos] + 2)
            states.append(a)
            num_still_going = self.step()
            if num_still_going == 0:
                break
        # print(self.count_made_through())
        return states
    

def animate_frames(frames):

    fig, ax = plt.subplots()

    for i, frame in enumerate(frames):
        ax.clear()

        n, m = frame.shape
        ax.axis([0, m, 0, n])
        plt.xticks([])
        plt.yticks([])

        options = dict(interpolation='none', alpha=0.8)
        options['extent'] = [0, m, 0, n]
        ax.imshow(frame, cmap, **options)

        ax.set_title(f"frame {i}")
        # Note that using time.sleep does *not* work here!
        plt.pause(0.1)
    plt.pause(100)
        

        
        

        
        


def make_grid(l, w, prob, seed = 42069):
        """Make an array with two types of agents.
        
        n: width and height of the array
        probs: probability of generating a 1
        
        return: NumPy array
        """

        np.random.seed(seed)
        r = np.random.rand(l, w) > (1-prob)
        r = r.astype(np.int8)
        # print(r)
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

