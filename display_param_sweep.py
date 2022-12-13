import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path = 'data.csv'

from main import *

lens = [10, 20, 30, 40, 50]
ps = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
n_agents = 1000
width = 200

df = pd.read_csv(path)

df = pd.pivot(df, index="length",columns = "p")
print(df)

# corr = df.corr()
sns.heatmap(df, cmap="Blues", annot=True, xticklabels=ps).set_xlabel("p")
# plt.show()

# plt.pcolor(df)
# plt.show()

# lens = [10, 20, 30, 40, 50]
# ps = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
# n_agents = 1000
# width = 200

# Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
# Cols = ['A', 'B', 'C', 'D']
# df = pd.DataFrame(simulate_one, index=ps, columns=lens)

# sns.heatmap(df, annot=False)






plt.show()
