import csv
import os
import multiprocessing as mp
import numpy as np
from time import sleep

from mask import Mask

from mask import CA_percolation

from mask import AgentSimulation
import mask
'''
def run_simulation():
    sim = setup_simulation()
    sim = sim.animate(100, interval=0.2)


def setup_simulation():
    sim = Mask(20,20,3,0.4)
    return sim

'''
'''
def agent_simulate(steps):
    sim = AgentSimulation(7, 7, 0.2, 1)
    for i in range(steps):
        sim.step()
        sim.draw(animate=True)
'''    

def simulate_one(n_runs, l, w, p, n_agents):
    for i in range(n_runs):
        sim = AgentSimulation(l, w, p, n_agents)
        sim.simulate(l*2)
        frac_through = 1 - (sim.count_made_through()/n_agents)

    avg_frac_through = np.mean(frac_through)
    print(f'Length: {l}, Width: {w}, P(fiber): {p}, Number of Particles: {n_agents}, Efficacy: {frac_through}')
    # return (l, w, p, n_agents, frac_through)
    return (l, p, frac_through)
    



if __name__ == "__main__":
    #run_simulation()
    '''sim = CA_percolation(30,30, 0.6)
    for i in range(100):
        sim.draw()
        sim.step()'''
    # sim = AgentSimulation(50, 200, 0.05, 1000)
    # frames = sim.simulate(200)
    # mask.animate_frames(frames)

    lens = [10, 20, 30, 40, 50]
    ps = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    n_agents = 1000
    width = 200
    n_runs = 20

    params = []
    for l in lens:
        for p in ps:
            params.append((n_runs, l, width, p, n_agents))
    
    print(len(params))

    with mp.Pool() as pool: # have the pool map each tuple onto calculate_s arguments
        # Starmap is just unpacking the tuple into multiple arguments
        # This will complete the operation with an optimal number of pool workers,
        # making use of all CPU cores.
        output_rows = pool.starmap(simulate_one, params)

    output_rows.sort()
    # headers = ["i", "length", "width", "p", "n_particles"]
    headers = ["length", "p", "frac_stopped"]
    output_rows.insert(0, headers)
    print(output_rows)
    
    # Write the output list to CSV
    wd = os.path.dirname(__file__)
    # path = os.path.join(wd, "data", "data.csv")
    with open('data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)



