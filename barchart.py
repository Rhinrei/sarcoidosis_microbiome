import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('TkAgg')

df_abundance = pd.read_csv("data/fungi_genus_absolute.csv", index_col=0).T
df_meta = pd.read_csv("data/fungi_metadata.csv", index_col=0)
df_abundance = df_abundance.join(df_meta)
group_col = 'material'
taxa = df_abundance.columns.drop(group_col)

# standard error of the mean
grouped_mean = df_abundance.groupby(group_col)[taxa].mean()
grouped_sem = df_abundance.groupby(group_col)[taxa].sem()


def extract_genus(taxon):
    parts = taxon.split(';')
    for part in reversed(parts):
        if part.startswith('g__'):
            return part[3:]
    return 'Unknown'


grouped_mean.columns = grouped_mean.columns.to_series().map(extract_genus)
grouped_sem.columns = grouped_sem.columns.to_series().map(extract_genus)

top_n = 30
mean_abundance = grouped_mean.mean(axis=0)
top_taxa = mean_abundance.nlargest(top_n).index

grouped_top_mean = grouped_mean[top_taxa]
grouped_top_sem = grouped_sem[top_taxa]

materials = grouped_top_mean.index.tolist()
taxa = grouped_top_mean.columns.tolist()

fig, axes = plt.subplots(len(materials), 1, figsize=(max(15, len(taxa) * 0.5), 3 * len(materials)), sharex=True)

colors = plt.cm.tab10.colors
for i, material in enumerate(materials):
    ax = axes[i]
    data = grouped_top_mean.loc[material]
    error = grouped_top_sem.loc[material]
    color = colors[i % len(colors)]
    ax.bar(range(len(taxa)), data, yerr=error, capsize=3, color=color)
    ax.set_ylim(0, (grouped_top_mean.values + grouped_top_sem.values).max() * 1.1)
    ax.set_ylabel(material, rotation=0, labelpad=50, va='center', fontsize=12, color=color)
    if i == len(materials) - 1:
        ax.set_xticks(range(len(taxa)))
        ax.set_xticklabels(taxa, rotation=60, fontsize=8, ha='right')
    else:
        ax.set_xticks([])

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.show()
