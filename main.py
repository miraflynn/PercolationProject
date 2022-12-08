import csv
import os
import multiprocessing as mp
import numpy as np
from time import sleep

from mask import Mask

from mask import CA_percolation

from mask import AgentSimulation
'''
def run_simulation():
    sim = setup_simulation()
    sim = sim.animate(100, interval=0.2)


def setup_simulation():
    sim = Mask(20,20,3,0.4)
    return sim

'''

def agent_simulate(steps):
    sim = AgentSimulation(100, 100, 0.2, 20)
    for i in range(steps):
        sim.step()
        sim.draw()
    

if __name__ == "__main__":
    #run_simulation()
    '''sim = CA_percolation(30,30, 0.6)
    for i in range(100):
        sim.draw()
        sim.step()'''
    agent_simulate(100)


