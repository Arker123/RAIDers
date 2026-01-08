import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

INPUT_FILE = "omim_summary.xlsx"
OUTPUT_EXCEL = "omim_subtypes_clustered.xlsx"
OUTPUT_PLOT = "omim_subtypes_pca_colored.png"

TEXT_COLUMNS = [
    "Gene / Locus",
    "Associated Variant",
    "Functional Mechanism / Biological Effect",
    "Significance / Role"
]

N_CLUSTERS = 5
RANDOM_STATE = 42
DPI = 300

df = pd.read_excel(INPUT_FILE)

df["text_for_clustering"] = (
    df[TEXT_COLUMNS]
    .fillna("NA")
    .agg(" ".join, axis=1)
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    min_df=2,
    max_features=3000
)

X = vectorizer.fit_transform(df["text_for_clustering"])

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=20
)

df["Cluster"] = kmeans.fit_predict(X)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca.fit_transform(X.toarray())

plt.figure(figsize=(6, 5))

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=df["Cluster"],
    cmap="tab10"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Unsupervised Clustering of Rare Disease Subtypes")

legend = plt.legend(
    *scatter.legend_elements(),
    title="Cluster",
    loc="best"
)
plt.gca().add_artist(legend)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=DPI)
plt.show()

df.to_excel(OUTPUT_EXCEL, index=False)

print(f"Saved clustered table to: {OUTPUT_EXCEL}")
print(f"Saved PCA plot to: {OUTPUT_PLOT}")
