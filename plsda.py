import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib

matplotlib.use('TkAgg')

fill_colors = ['lightblue', 'lightcoral']


def plot_ellipse(x, y, ax, n_std=2.0, facecolor='lightblue'):
    if len(x) < 2:
        return
    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvals)

    ellipse = Ellipse(
        (mean_x, mean_y), width, height, angle=angle,
        facecolor=facecolor, alpha=0.3,
        edgecolor='black', linewidth=1.5
    )
    ax.add_patch(ellipse)


df_abundance = pd.read_csv("data/fungi_genus_absolute.csv", index_col=0).T
df_meta = pd.read_csv("data/fungi_metadata.csv", index_col=0)
df_abundance = df_abundance.join(df_meta)
material_col = 'material'

pairs = [
    ('non-cultured', 'cultured'),
    ('BAL', 'biopsy'),
    ('control', 'washout')
]

plt.figure(figsize=(18, 6))

for i, (mat1, mat2) in enumerate(pairs, 1):
    df_pair = df_abundance[df_abundance[material_col].isin([mat1, mat2])]
    X = df_pair.drop(columns=[material_col])
    y = df_pair[material_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    pls = PLSRegression(n_components=2)
    pls.fit(X, y_encoded)
    X_scores = pls.x_scores_

    explained_variance = np.var(pls.x_scores_, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)

    ax = plt.subplot(1, 3, i)
    for class_num in range(len(le.classes_)):
        mask = y_encoded == class_num
        ax.scatter(
            X_scores[mask, 0],
            X_scores[mask, 1],
            label=le.classes_[class_num]
        )
        plot_ellipse(
            X_scores[mask, 0], X_scores[mask, 1], ax,
            facecolor=fill_colors[class_num]
        )

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    comp1_var = explained_variance_ratio[0] * 100
    comp2_var = explained_variance_ratio[1] * 100
    ax.set_xlabel(f'COMP1 ({comp1_var:.1f}%)')
    ax.set_ylabel(f'COMP2 ({comp2_var:.1f}%)')
    ax.set_title(f'PLS-DA: {mat1} vs {mat2}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
