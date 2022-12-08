from itertools import combinations, product
from pprint import pprint
from random import random
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

def create_lattice(length, width, height):
    nodes = list(product(range(length), range(width), range(height)))
    edges = [
        set([n1, n2])
        for n1, n2
        in combinations(nodes, 2)
        if sum(
            abs(n1d - n2d)
            for n1d, n2d
            in zip(n1, n2)
        ) == 1
    ]

    starts = set([n for n in nodes if n[0] == 0])
    ends   = set([n for n in nodes if n[0] == length-1])

    return nodes, edges, starts, ends

def count_paths(visited, ends, edges):
    if not visited.isdisjoint(ends):
        return True

    return sum(
        count_paths(e | visited, ends, edges)
        for e
        in edges
        if len(e - visited) == 1
    )

def all_paths(starts, ends, edges):
    return sum(count_paths({s}, ends, edges) for s in starts)

def drop_edges(edges, p):
    for e in edges:
        if random() < p:
            yield e

if __name__ == "__main__":
    (length, width, height), p = map(int, sys.argv[1:4]), float(sys.argv[4])

    nodes, edges, starts, ends = create_lattice(length, width, height)

    N = 10000
    counts = [all_paths(starts, ends, drop_edges(edges, p)) for _ in range(N)]
    sns.ecdfplot(counts)
    plt.title(f"Length={length}, Width={width}, Height={height}, p={p}")
    plt.show()
