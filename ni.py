import pandas as pd
import networkx as nx
import numpy as np
import os

work_dr = 'input_data'

f1 = pd.read_csv(os.path.join(work_dr, 'profiles.csv'), sep=',', index_col=0)
f2 = pd.read_csv('networks/sapiensIntPathproteinPairs', header=None, sep='\t', index_col=None)

f2[0] = f2[0].str.upper()
f2[1] = f2[1].str.upper()
f1 = f1[~f1.index.duplicated(keep='first')]
f1 = f1  # .iloc[:,-101:]
f1.index = f1.index.str.upper()
d1 = f1.fillna(0)  # .loc[np.sum(f1>0,axis=1)>20]#.iloc[:20]
Eca = []
print(f2)
print(f1)
for i in d1.columns:
    for j in d1.index:
        if pd.isna(j):
            print(j)
            continue
        if pd.isna(d1[i][j]):
            continue
        elif np.abs(d1[i][j]) > 0:
            Eca.append([i, j, np.exp(d1[i][j])])
print(pd.DataFrame(Eca).shape)
pd.DataFrame(Eca).to_csv(os.path.join(work_dr, 'sample_context.txt'), index=False, header=False)

Ega = []
context = d1.index
for j in f2.index:
    if f2.iloc[j][1] in context:
        Ega.append([f2.iloc[j][0], f2.iloc[j][1], 1])
    elif f2.iloc[j][0] in context:
        Ega.append([f2.iloc[j][1], f2.iloc[j][0], 1])
pd.DataFrame(Ega).to_csv(os.path.join(work_dr, 'protein_context.txt'), index=False, header=False)
print(len(set([i[0] for i in Ega])))
