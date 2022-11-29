import csv
import os
import multiprocessing as mp
import numpy as np

from mask import Mask

def run_simulation():
    sim = setup_simulation()
    sim = sim.loop()
    sim.draw()

def setup_simulation():
    sim = Mask(10,10,3,0.1)
    return sim


if __name__ == "__main__":
    run_simulation()