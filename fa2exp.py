import numpy as np
import pandas as pd
import networkx as nx
import csv
from fa2_modified import ForceAtlas2
#import matplotlib.pyplot as plt
from sklearn import preprocessing

preparedDataFilename = 'MultiKE_EN_RU_15K_V1'

df = pd.read_csv('output/' + preparedDataFilename + '.csv', delimiter = ',')

G = nx.Graph()

positions = dict()

with open('output/' + preparedDataFilename + '.csv', encoding='utf8', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        G.add_node(int(row['Ent1_ID']), type=row['Type'], weight=90, pos=(float(row['X']), float(row['Y'])))
        positions[int(row['Ent1_ID'])] = (float(row['X']), float(row['Y']))
    for row in reader:
        id1 = int(row['Ent1_ID'])
        id2 = int(row['Ent2_ID'])
        G.add_edge(id1, id2)

with open('data/rel_triples_en_ru.csv', encoding='utf8', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id1 = df.loc[df['Ent1Full'] == str(row['Ent1_ID'])].head(1)['Ent1_ID'].values[0]
        id2 = df.loc[df['Ent1Full'] == str(row['Ent2_ID'])].head(1)['Ent1_ID'].values[0]
        G.add_edge(id1, id2)


forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=False,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=0.5,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=4000000000.0,
                        strongGravityMode=False,
                        gravity=0.0,

                        # Log
                        verbose=True)
pos = forceatlas2.forceatlas2_networkx_layout(G, pos=positions, iterations=200)

for i in range(0, len(df)):
    df.at[i, 'X'] = pos[i][0]
    df.at[i, 'Y'] = pos[i][1]

XY_Scaled = df[['X', 'Y']].to_numpy()

XY_Scaled = preprocessing.minmax_scale(XY_Scaled, feature_range=(-300, 300))

df['X'] = XY_Scaled[:, [0]]
df['Y'] = XY_Scaled[:, [1]]

df.to_csv('output/' + preparedDataFilename + '_graph.csv', index=False)

#nx.draw_networkx(G, pos, cmap=plt.get_cmap('jet'), node_size=50, with_labels=False)
#plt.show() 