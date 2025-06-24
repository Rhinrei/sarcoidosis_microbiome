import pandas as pd
from scipy.spatial.distance import braycurtis
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import matplotlib
matplotlib.use('TkAgg')

df_abundance = pd.read_csv("data/fungi_genus_absolute.csv", index_col=0).T
df_meta = pd.read_csv("data/fungi_metadata.csv", index_col=0)
df_abundance = df_abundance.join(df_meta)

pairs = [
    ('non-cultured', 'cultured'),
    ('BAL', 'biopsy'),
    ('control', 'washout')
]
#  pairs = list(combinations(materials, 2))

results = []

materials = df_abundance['material'].unique()
for material in materials:
    df_group = df_abundance[df_abundance['material'] == material]
    for (sample1, row1), (sample2, row2) in combinations(df_group.iterrows(), 2):
        vec1 = row1.drop('material')
        vec2 = row2.drop('material')
        dist = braycurtis(vec1, vec2)
        results.append({
            'Sample1': sample1,
            'Sample2': sample2,
            'Material': material,
            'BrayCurtisDissimilarity': dist
        })

result_df = pd.DataFrame(results)

plt.figure(figsize=(12, 8), constrained_layout=True)
ax = sns.violinplot(data=result_df, x='Material', y='BrayCurtisDissimilarity', inner='point')

annotator = Annotator(ax, pairs, data=result_df, x='Material', y='BrayCurtisDissimilarity')
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()


plt.title('Intra-group Bray-Curtis Dissimilarity by Material with Statistical Comparisons')
plt.xlabel('Material')
plt.ylabel('Bray-Curtis Dissimilarity')
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
