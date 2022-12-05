from itertools import combinations, product
from pprint import pprint
from random import random
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import sys

def simulate(length, width, height, p, n=10000):
    """
    Looking at a person wearing a mask:
    * Length is into the mask
    * Width is along their mouth
    * Height is the other one
    * p is the probability that any given edge is open
    * n is how many simulations to run
    """
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

    def path_exists(visited, edges):
        if not visited.isdisjoint(ends):
            return True

        for e in edges:
            if len(e - visited) == 1:
                return path_exists(e | visited, edges)

        return False

    assert all(path_exists({s}, edges) for s in starts)

    k = 0
    for _ in range(n):
        nedges = [e for e in edges if random() < p]
        k += any(path_exists({s}, nedges) for s in starts)

    return k / n

if __name__ == "__main__":
    length, width, height, p = sys.argv[1:]
    print(simulate(int(length), int(width), int(height), float(p)))

"""
Cl = 2
Cw = 4
Cp = 0.25

ls = range(1, 5)
ws = range(3, 13)
ps = np.linspace(0, 1, 11)

for p in np.linspace(0, 1, 11):
    print(f"{p}, {simulate(Cl, Cw, 0, p)}")
"""
"""
with open("Lfinal.csv", "w") as csv:
    csv.write("l,w,p,output\n")
    for l in ls:
        csv.write(f"{l},{Cw},{Cp},{simulate(l, Cw, 0, Cp)}\n")

    for w in ws:
        csv.write(f"{Cl},{w},{Cp},{simulate(Cl, w, 0, Cp)}\n")

    for p in ps:
        csv.write(f"{Cl},{Cw},{p},{simulate(Cl, Cw, 0, p)}\n")
"""
"""
def logistic(x, k, a):
    return 1/(1 + np.exp(-k*(x - a)))

ps = np.linspace(0, 1, 11)
ls = np.linspace(3, 12, 10).astype(np.uint8)
ks = []
ms = []

ps = np.linspace(0, 1, 11)
ls = range(3, 13)
ws = range(3, 13)

with open("Lfinal.csv", "w") as csv:
    csv.write("l,w,p,output\n")

    for l, w, p in product(ls, ws, ps):
        csv.write(f"{l},{w},{p},{simulate(l, w, 0, p)}\n")
"""
"""
for l in ls:
    ys = [simulate(l, 3, 0, p) for p in ps]
    popt, _ = curve_fit(logistic, ps, ys)
    ks.append(popt[0])


def linear(x, m, b):
    return m*x + b

popt, _ = curve_fit(linear, ls, ks)
print(popt)

plt.plot(ls, ks, 'o')
plt.plot(ls, linear(ls, *popt), 'r-')
#plt.plot(xs, logistic(xs, *popt), 'r-', label="k=%.2f, a=%.2f" % tuple(popt))
#plt.legend()
plt.show()"""