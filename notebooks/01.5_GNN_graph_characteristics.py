# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,notebooks//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: pytorch
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/HFooladi/GNNs-For-Chemists/blob/main/notebooks/01.5_GNN_graph_characteristics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="graph-char-title"
# # Molecular Graph Characteristics: What Do These Graphs Actually Look Like?
#
# ## Table of Contents
# 1. [Setup and Installation](#setup)
# 2. [Why Topology Matters for GNNs](#why)
# 3. [A Guided Tour: Three Example Molecules](#tour)
# 4. [Node and Edge Counts Across Datasets](#counts)
# 5. [Degree Distribution: Bounded by Valence](#degree)
# 6. [Sparsity: Molecules Are Almost Trees](#sparsity)
# 7. [Diameter and Shortest Paths](#diameter)
# 8. [Connected Components](#components)
# 9. [Receptive Field and the 3–5 Layer Rule](#depth)
# 10. [Summary](#summary)
# 11. [References](#references)
#
# > **Where this sits in the series:** This is a *supplementary* notebook that closes
# > the **01.x** representation series. By now you can build a molecular graph from
# > SMILES, draw it in 2D and 3D, and pick a framework (PyG, DGL, Jraph) to feed it
# > into a GNN. The natural next question is: **what do these graphs actually look
# > like — as graphs, not as drawings?** How big are they? How dense? How far apart
# > are atoms in the bond network? This notebook quantifies the answers across three
# > standard chemistry datasets — QM9, BACE, and ESOL — and uses those numbers to
# > derive the canonical "3–5 message-passing layers" rule from the data itself.

# + [markdown] id="graph-char-objectives"
# ## Learning Objectives
#
# After this notebook, you will be able to:
#
# - **Read** the basic vocabulary of graph theory through a chemist's lens: nodes,
#   edges, degree, density, diameter, shortest path, connected components.
# - **Recognise** the shape of "typical" molecular graphs — small (10–40 atoms),
#   extremely sparse (≈ tree-like), bounded degree (≤ 4), low diameter (5–15).
# - **Compare** molecular graphs against the graphs you may know from other
#   domains (social networks, the web) — they are *fundamentally* different.
# - **Derive**, from data, why chemistry GNNs almost always use 3–5
#   message-passing layers. The receptive field of `k` GCN layers is the
#   `k`-hop neighbourhood; a small `k` already covers most molecules.
#
# > **One-sentence intuition:** a molecular graph is a small, sparse, bounded-degree
# > graph whose diameter is usually under ten bonds — and that single sentence
# > shapes almost every architectural choice you'll see in the rest of the course.

# + [markdown] id="setup"
# ## 1. Setup and Installation <a name="setup"></a>
#
# We need RDKit (for SMILES → molecule conversion), NetworkX (for graph metrics
# like diameter and density), pandas (for tabular dataset summaries), and the
# matplotlib / seaborn / plotly trio for static and interactive plots.
# No PyTorch needed — every property here is a *topology* property, computable
# before any message passing happens.

# + colab={"base_uri": "https://localhost:8080/"} id="install-cell"
#@title Install required libraries
# !pip install -q rdkit networkx pandas matplotlib seaborn plotly

# + id="imports-cell"
#@title Import libraries and set plotting style
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import seaborn as sns

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Consistent plotting style across the tutorial series
sns.set_context("notebook", font_scale=1.2)
np.random.seed(42)

# Fixed dataset palette — same colours used in every figure below.
DATASET_COLORS = {"QM9": "#1f77b4", "BACE": "#ff7f0e", "ESOL": "#2ca02c"}

# + [markdown] id="setup-data"
# ### Where this notebook reads data from
#
# The notebook looks for the standard cached datasets that the rest of the
# course uses (`notebooks/data/`). If you're running this on Colab and the
# folder doesn't exist yet, the loader cells below fall back to public URLs.

# + id="data-path"
GITHUB_RAW_BASE = ("https://raw.githubusercontent.com/HFooladi/"
                   "GNNs-For-Chemists/main/notebooks/data_smiles")


def _find_dir(*names: str) -> Path | None:
    """Look for a directory by name, checking common locations relative
    to the notebook (CWD) and to the repo root."""
    for parent in (Path("."), Path(".."), Path("notebooks"), Path("../notebooks")):
        for name in names:
            p = parent / name
            if p.exists() and any(p.iterdir()):
                return p.resolve()
    return None


# Slim SMILES-only CSVs that ship with the repo — primary source for Colab.
SMILES_DIR = _find_dir("data_smiles")
# Full datasets cached locally (only present on the maintainer's machine).
DATA_DIR = _find_dir("data")
print(f"Slim SMILES dir: {SMILES_DIR if SMILES_DIR else 'not found locally'}")
print(f"Full data dir:   {DATA_DIR if DATA_DIR else 'not found locally'}")
print(f"GitHub fallback: {GITHUB_RAW_BASE}")


# + [markdown] id="why"
# ## 2. Why Topology Matters for GNNs <a name="why"></a>
#
# A Graph Neural Network is, at its core, a *message-passing* machine: every
# atom updates its hidden state by aggregating the states of its bonded
# neighbours, then we repeat. Two simple questions about the graph already
# constrain the design of the network:
#
# 1. **How big is the graph?** Small graphs mean small mini-batches even at
#    high batch sizes, fast per-molecule inference, and very cheap aggregation
#    (proportional to the number of edges, not the number of atom-pairs).
# 2. **How far apart are atoms in the graph?** A single message-passing layer
#    moves information one bond. To let an atom on one end of a molecule
#    "see" an atom on the other end, you need at least as many layers as the
#    graph diameter. But — as Notebook 02.1 shows — too many layers and the
#    representation oversmooths into uniform mush.
#
# The sweet spot in chemistry GNNs is a rule of thumb everyone quotes — 3 to
# 5 message-passing layers. The rest of this notebook is the
# **data-driven explanation** for that rule.

# + [markdown] id="tour"
# ## 3. A Guided Tour: Three Example Molecules <a name="tour"></a>
#
# Before scaling to thousands of molecules, let's anchor every concept in
# three concrete examples that we'll keep returning to:
#
# - **Methanol** (CH₃OH, 6 heavy + H atoms) — the minimum interesting molecule.
# - **Aspirin** (C₉H₈O₄, 13 heavy atoms) — drug-sized, one aromatic ring.
# - **A representative BACE molecule** (~30 heavy atoms) — drug-like, multi-ring,
#   one of the larger graphs we'll encounter.
#
# For each, we'll draw the chemical structure, draw the *graph* using RDKit's
# canonical 2D coordinates (so rings look like rings — same convention as
# Notebook 02.1), and print a stats table.

# + id="tour-utilities"
#@title Core utilities: SMILES → NetworkX, layout, and stats

# Atomic symbol table for nicer labels on plots and tables.
_ATOMIC_SYMBOL = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}


def smiles_to_nx(smiles: str, add_hs: bool = False) -> nx.Graph | None:
    """Convert a SMILES string to a NetworkX graph.

    Heavy-atom graphs by default (``add_hs=False``) — this is what almost
    every modern chemistry GNN actually sees. Pass ``add_hs=True`` to mirror
    Notebook 02.1's explicit-H view.

    Each node stores its atomic number under ``atomic_number``. If a 2D
    conformer can be computed, the canonical RDKit coordinates are also
    stored under ``pos`` so plots use chemical-drawing conventions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)
    try:
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        has_pos = True
    except Exception:
        has_pos = False

    G = nx.Graph()
    for atom in mol.GetAtoms():
        attrs = {"atomic_number": atom.GetAtomicNum()}
        if has_pos:
            p = conf.GetAtomPosition(atom.GetIdx())
            attrs["pos"] = (p.x, p.y)
        G.add_node(atom.GetIdx(), **attrs)
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G


def graph_stats(G: nx.Graph) -> dict:
    """Compute the topology statistics this notebook revolves around.

    Works for disconnected graphs (e.g. salts): diameter and average
    shortest path are reported on the *largest* connected component.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    n_components = nx.number_connected_components(G)

    if n_components == 0 or n == 0:
        diam, radius, avg_sp = 0, 0, 0.0
    else:
        largest = G.subgraph(max(nx.connected_components(G), key=len))
        if largest.number_of_nodes() < 2:
            diam, radius, avg_sp = 0, 0, 0.0
        else:
            diam = nx.diameter(largest)
            radius = nx.radius(largest)
            avg_sp = nx.average_shortest_path_length(largest)

    density = (2 * e) / (n * (n - 1)) if n > 1 else 0.0
    return {
        "n_nodes": n,
        "n_edges": e,
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(np.max(degrees)) if degrees else 0,
        "density": density,
        "diameter": diam,
        "radius": radius,
        "avg_shortest_path": avg_sp,
        "n_components": n_components,
    }


def draw_molecule_graph(G: nx.Graph, ax, title: str | None = None,
                        node_size: int = 380, highlight_path: list | None = None):
    """Draw a NetworkX molecule graph using RDKit 2D coordinates if available.

    If ``highlight_path`` is provided, the edges along that path are drawn
    in red on top of the base edge set — used in the diameter section to
    show the longest shortest path through a molecule.
    """
    if "pos" in next(iter(G.nodes(data=True)))[1]:
        pos = {i: G.nodes[i]["pos"] for i in G.nodes()}
    else:
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#444444",
                           width=1.4, alpha=0.85)
    if highlight_path and len(highlight_path) >= 2:
        path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=path_edges,
                               edge_color="#d62728", width=3.5, alpha=0.95)

    # Colour nodes by atomic number, with a chemistry-friendly palette.
    z = np.array([G.nodes[i]["atomic_number"] for i in G.nodes()])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=z, cmap="viridis",
                           vmin=1, vmax=20, node_size=node_size,
                           edgecolors="black", linewidths=0.7)
    labels = {i: _ATOMIC_SYMBOL.get(int(G.nodes[i]["atomic_number"]), "?")
              for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9,
                            font_color="white", font_weight="bold")
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=11)


# + id="tour-examples"
#@title Build the three running example molecules

EXAMPLES = {
    "Methanol": "CO",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    # A real BACE-1 inhibitor from the dataset — large, multi-ring, drug-like.
    "Drug-like (BACE)": "Fc1cc(F)cc(c1)CC(NC(=O)C(Cc2ccccc2)NC(=O)N3CCCC3)C(O)CN4CC(O)CC4",
}

graphs = {name: smiles_to_nx(smi, add_hs=False) for name, smi in EXAMPLES.items()}

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5),
                         gridspec_kw={"height_ratios": [1, 1.1]})
for col, (name, G) in enumerate(graphs.items()):
    # Row 0: chemical structure rendered by RDKit.
    mol = Chem.MolFromSmiles(EXAMPLES[name])
    img = Draw.MolToImage(mol, size=(360, 240))
    axes[0, col].imshow(img)
    axes[0, col].set_axis_off()
    axes[0, col].set_title(f"{name}\n({mol.GetNumHeavyAtoms()} heavy atoms)",
                           fontsize=11)
    # Row 1: graph view with the same RDKit layout.
    draw_molecule_graph(G, axes[1, col], title="graph view (heavy atoms only)")
fig.suptitle("Three running examples — chemical structure (top) vs. graph (bottom)",
             y=1.02, fontsize=13)
plt.tight_layout()
plt.show()

# + id="tour-stats-table"
stats_table = pd.DataFrame({name: graph_stats(G) for name, G in graphs.items()}).T
stats_table["density"] = stats_table["density"].map(lambda x: f"{x:.3f}")
stats_table["avg_degree"] = stats_table["avg_degree"].map(lambda x: f"{x:.2f}")
stats_table["avg_shortest_path"] = stats_table["avg_shortest_path"].map(lambda x: f"{x:.2f}")
display_cols = ["n_nodes", "n_edges", "avg_degree", "max_degree",
                "density", "diameter", "radius",
                "avg_shortest_path", "n_components"]
print("Topology statistics for the three examples:")
print(stats_table[display_cols].to_string())

# + [markdown] id="tour-interp"
# ### What to notice
#
# - **Sizes scale a lot.** Methanol is 2 heavy atoms; aspirin 13; the BACE-like
#   molecule is in the 30s. This 1–2 order of magnitude spread is typical
#   across chemistry datasets.
# - **`n_edges ≈ n_nodes`.** Even the largest molecule has only slightly more
#   bonds than atoms — a near-tree structure with a handful of rings adding
#   single extra edges each.
# - **Average degree is ~2.** Most heavy atoms have just two neighbours
#   (linear chains and ring carbons). Even ring-rich drug molecules don't push
#   the average past 2.2.
# - **Max degree is ≤ 4.** Carbon's tetravalence sets an upper bound across
#   all of organic chemistry. Compare with social networks where a single
#   "influencer" node can have a million neighbours.
# - **Density drops sharply with size.** Methanol is 1.0 (the complete-graph
#   maximum is reached for 2 nodes); aspirin is 0.17; the BACE molecule is
#   ~0.06. Bigger molecules are *much* sparser — we'll formalise this in §6.
# - **Diameter is small but not log-small.** Aspirin: 6 bonds. BACE: 14 bonds.
#   These numbers are central to the depth argument in §9.

# + [markdown] id="counts"
# ## 4. Node and Edge Counts Across Datasets <a name="counts"></a>
#
# Three molecules don't make a distribution. Let's load the standard chemistry
# datasets the rest of this course uses and compute topology statistics across
# thousands of real molecules.
#
# - **QM9** — ~134k small organic molecules with up to 9 heavy atoms; the
#   workhorse for quantum-property benchmarks. We subsample 10k for speed.
# - **BACE** — ~1.5k drug-like inhibitors of β-secretase. Molecules are big
#   (~30+ heavy atoms) and densely decorated with rings.
# - **ESOL** — ~1.1k small organic molecules with measured water solubilities;
#   sizes sit in between QM9 and BACE.

# + id="dataset-loaders"
#@title Dataset loaders (local cache → repo URL fallback)

def _load_smiles_csv(local_name: str, smiles_col: str = "smiles") -> list[str]:
    """Load a SMILES column from `notebooks/data_smiles/<local_name>`,
    falling back to the same file served from the repo's raw GitHub URL
    if it's not on disk (i.e. running on Colab without a local clone)."""
    if SMILES_DIR and (SMILES_DIR / local_name).exists():
        df = pd.read_csv(SMILES_DIR / local_name)
    else:
        url = f"{GITHUB_RAW_BASE}/{local_name}"
        df = pd.read_csv(url)
    return df[smiles_col].dropna().tolist()


def load_bace_smiles() -> list[str]:
    return _load_smiles_csv("bace_smiles.csv")


def load_esol_smiles() -> list[str]:
    return _load_smiles_csv("esol_smiles.csv")


def load_qm9_smiles(max_n: int = 10_000, seed: int = 42) -> list[str]:
    """Return up to ``max_n`` random QM9 SMILES.

    Three sources, in priority order:
    1. The slim 10 k SMILES sample in ``notebooks/data_smiles/`` (the
       file that ships with the repo — works on Colab out of the box).
    2. The full ``gdb9.sdf`` cached under ``notebooks/data/QM9/raw/`` —
       only present on the maintainer's machine; reservoir-sampled.
    3. The slim sample fetched from this repo's raw GitHub URL.
    """
    # 1. Local slim sample.
    if SMILES_DIR and (SMILES_DIR / "qm9_smiles_10k.csv").exists():
        df = pd.read_csv(SMILES_DIR / "qm9_smiles_10k.csv")
        smis = df["smiles"].dropna().tolist()
        return smis[:max_n] if max_n < len(smis) else smis

    # 2. Full SDF — only if available locally.
    if DATA_DIR and (DATA_DIR / "QM9/raw/gdb9.sdf").exists():
        suppl = Chem.SDMolSupplier(str(DATA_DIR / "QM9/raw/gdb9.sdf"),
                                   removeHs=True, sanitize=True)
        rng = np.random.default_rng(seed)
        reservoir: list[str] = []
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            if len(reservoir) < max_n:
                reservoir.append(smi)
            else:
                j = int(rng.integers(0, i + 1))
                if j < max_n:
                    reservoir[j] = smi
        return reservoir

    # 3. Colab fallback: pull the slim sample from the repo.
    url = f"{GITHUB_RAW_BASE}/qm9_smiles_10k.csv"
    df = pd.read_csv(url)
    smis = df["smiles"].dropna().tolist()
    return smis[:max_n] if max_n < len(smis) else smis


# + id="compute-stats"
#@title Compute topology stats across all three datasets
def stats_for_smiles_list(smiles_list: Iterable[str], dataset_name: str) -> pd.DataFrame:
    rows = []
    for smi in smiles_list:
        G = smiles_to_nx(smi, add_hs=False)
        if G is None or G.number_of_nodes() == 0:
            continue
        s = graph_stats(G)
        s["dataset"] = dataset_name
        rows.append(s)
    return pd.DataFrame(rows)


CACHE_PATH = Path("./graph_stats_cache.parquet")

if CACHE_PATH.exists():
    stats_df = pd.read_parquet(CACHE_PATH)
    print(f"Loaded cached stats from {CACHE_PATH}")
else:
    print("Computing stats — this takes ~1–2 minutes on the first run.")
    parts = []
    qm9_smiles = load_qm9_smiles(max_n=10_000)
    parts.append(stats_for_smiles_list(qm9_smiles, "QM9"))
    print(f"  QM9:  {len(parts[-1])} molecules")

    bace_smiles = load_bace_smiles()
    parts.append(stats_for_smiles_list(bace_smiles, "BACE"))
    print(f"  BACE: {len(parts[-1])} molecules")

    esol_smiles = load_esol_smiles()
    parts.append(stats_for_smiles_list(esol_smiles, "ESOL"))
    print(f"  ESOL: {len(parts[-1])} molecules")

    stats_df = pd.concat(parts, ignore_index=True)
    try:
        stats_df.to_parquet(CACHE_PATH)
        print(f"Cached to {CACHE_PATH}")
    except Exception:
        pass  # parquet engine missing on bare-bones Colab — not fatal.

print()
print("Stats DataFrame shape:", stats_df.shape)
print(stats_df.groupby("dataset")[
    ["n_nodes", "n_edges", "avg_degree", "density", "diameter"]
].mean().round(2))

# + [markdown] id="counts-plots"
# ### Distribution of molecule sizes
#
# Two stacked histograms: how many atoms (nodes) and how many bonds (edges)
# does each dataset's molecules contain?

# + id="size-histograms"
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for dataset, color in DATASET_COLORS.items():
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    axes[0].hist(subset["n_nodes"], bins=40, alpha=0.55, label=dataset,
                 color=color, density=True, edgecolor="white", linewidth=0.4)
    axes[1].hist(subset["n_edges"], bins=40, alpha=0.55, label=dataset,
                 color=color, density=True, edgecolor="white", linewidth=0.4)

axes[0].set_xlabel("Number of heavy atoms (nodes)")
axes[0].set_ylabel("density")
axes[0].set_title("Atom-count distribution per dataset")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Number of bonds (edges)")
axes[1].set_ylabel("density")
axes[1].set_title("Bond-count distribution per dataset")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# + [markdown] id="counts-scatter-md"
# ### Bonds vs. atoms: the near-tree relationship
#
# Plot every molecule as a single point on a `n_atoms` vs. `n_bonds` plane.
# If molecules were trees, all points would lie *exactly* on the line
# `n_bonds = n_atoms − 1`. The handful of extra bonds above that line is
# the number of rings (= number of "independent cycles", the
# **cyclomatic number**).

# + id="size-scatter"
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for dataset, color in DATASET_COLORS.items():
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    ax.scatter(subset["n_nodes"], subset["n_edges"],
               s=12, alpha=0.35, color=color, label=dataset, edgecolors="none")

# Reference line: a tree has n_bonds = n_nodes - 1.
xs = np.array([0, stats_df["n_nodes"].max() + 2])
ax.plot(xs, xs - 1, "k--", lw=1.2, alpha=0.7,
        label="tree: $E = N-1$ (no rings)")

ax.set_xlabel("Number of heavy atoms ($N$)")
ax.set_ylabel("Number of bonds ($E$)")
ax.set_title("Bonds scale almost linearly with atoms — molecules are nearly trees")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Fit and print the per-dataset slope.
print("\nLinear fit  E = a·N + b  per dataset:")
for dataset in DATASET_COLORS:
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) < 2:
        continue
    a, b = np.polyfit(subset["n_nodes"], subset["n_edges"], 1)
    rings = subset["n_edges"] - subset["n_nodes"] + 1
    print(f"  {dataset:5s}  slope a = {a:.3f}, intercept b = {b:+.2f}, "
          f"avg rings per molecule = {rings.mean():.2f}")

# + [markdown] id="counts-checkpoint"
# ### Checkpoint Exercise
#
# 1. **The slope of `n_bonds` vs. `n_atoms` is ≈ 1.05–1.10 across all three
#    datasets.** What does that slope tell you about the *average degree* of
#    a heavy atom? *(Hint: the handshake lemma — every edge contributes 2 to
#    the total degree sum.)*
#
# 2. **QM9's distribution stops sharply at 9 heavy atoms.** That's by
#    construction — QM9 enumerates *all* stable organic molecules up to 9
#    heavy atoms. What kind of bias does this introduce if you train a GNN on
#    QM9 and try to apply it to drug-like molecules from BACE?
#
# 3. **The intercept `b` of the fit is close to −1.** Why? Walk through what
#    happens for the simplest non-trivial graph: a 2-atom molecule with one
#    bond.

# + [markdown] id="degree"
# ## 5. Degree Distribution: Bounded by Valence <a name="degree"></a>
#
# In chemistry, the degree of a node — the number of bonds an atom has —
# is hard-capped by chemistry itself: carbon ≤ 4, nitrogen ≤ 3 (or 4 if
# protonated), oxygen ≤ 2, and so on. This makes molecular graphs
# *fundamentally different* from the kind of graphs studied in network
# science, where a power-law degree distribution dominates — a few hubs with
# thousands of connections, many leaves with just one.

# + id="degree-distribution"
def collect_degrees(smiles_list: Iterable[str]) -> np.ndarray:
    """Flat array of all atom degrees across all molecules in the list."""
    out = []
    for smi in smiles_list:
        G = smiles_to_nx(smi, add_hs=False)
        if G is None:
            continue
        out.extend(d for _, d in G.degree())
    return np.array(out)


# Reload SMILES lists (lightweight) so we don't depend on the cache.
try:
    qm9_smiles = qm9_smiles  # already in scope from §4
except NameError:
    try:
        qm9_smiles = load_qm9_smiles(max_n=10_000)
    except FileNotFoundError:
        qm9_smiles = []

try:
    bace_smiles
except NameError:
    bace_smiles = load_bace_smiles()
try:
    esol_smiles
except NameError:
    esol_smiles = load_esol_smiles()

degrees_by_dataset = {
    "QM9": collect_degrees(qm9_smiles) if qm9_smiles else np.array([]),
    "BACE": collect_degrees(bace_smiles),
    "ESOL": collect_degrees(esol_smiles),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: molecular degree distributions, stacked side-by-side.
max_deg = max(int(d.max()) for d in degrees_by_dataset.values() if len(d) > 0)
bins = np.arange(0.5, max_deg + 1.5)
for ax_offset, (dataset, color) in enumerate(DATASET_COLORS.items()):
    d = degrees_by_dataset.get(dataset, np.array([]))
    if len(d) == 0:
        continue
    axes[0].hist(d, bins=bins, alpha=0.55, density=True, label=dataset,
                 color=color, edgecolor="white", linewidth=0.6)
axes[0].set_xlabel("Atom degree (# bonded heavy-atom neighbours)")
axes[0].set_ylabel("Fraction of atoms")
axes[0].set_title("Molecules: bounded degree distribution (max ≤ 4)")
axes[0].set_xticks(range(1, max_deg + 1))
axes[0].grid(alpha=0.3)
axes[0].legend()

# Right: a Barabási–Albert (preferential-attachment) graph of comparable
# size shows the contrast — its degree distribution is heavy-tailed.
ba = nx.barabasi_albert_graph(n=5000, m=2, seed=42)
ba_degrees = np.array([d for _, d in ba.degree()])
axes[1].hist(ba_degrees, bins=40, alpha=0.7, color="#9467bd",
             edgecolor="white", linewidth=0.4, density=True,
             label="BA graph (n=5k, m=2)")
axes[1].set_yscale("log")
axes[1].set_xscale("log")
axes[1].set_xlabel("Node degree (log scale)")
axes[1].set_ylabel("density (log scale)")
axes[1].set_title("For contrast: a power-law social-network graph")
axes[1].legend()
axes[1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.show()

# + [markdown] id="degree-interp"
# ### What to notice
#
# - **Almost every atom has degree 1, 2, or 3.** Degree 4 (a fully
#   substituted carbon centre) is rare on its own. Anything beyond 4 is
#   essentially absent.
# - **Degree 2 is the most common value.** This corresponds to the inner
#   carbons of chains and the carbons inside aromatic rings — the most
#   abundant atomic environment in organic chemistry.
# - **The Barabási–Albert distribution is qualitatively different.** It is
#   *unbounded* and heavy-tailed: a handful of nodes have hundreds of
#   neighbours. No molecule looks like that — and that's why GNN designs
#   that work for social-network analytics often don't transfer straight to
#   chemistry without modification.
# - **Practical consequence:** because degree is bounded by a small constant,
#   the cost of one message-passing layer scales as `O(E) ≈ O(N)`, not
#   `O(N²)`. This is one of the reasons GNNs were practical for chemistry
#   before they were practical for protein structures or general graphs.

# + [markdown] id="degree-checkpoint"
# ### Checkpoint Exercise
#
# 1. **Predict** the most common heavy-atom degree in QM9 *before* looking at
#    the histogram. Then check. If you got it wrong, what was your mental
#    model?
# 2. **Why isn't degree-4 the most common, given how much carbon is in
#    organic chemistry?** *(Hint: most carbons in a chain or ring have one
#    or two bonds to hydrogen — hydrogens we removed when we built the
#    heavy-atom graph.)*

# + [markdown] id="sparsity"
# ## 6. Sparsity: Molecules Are Almost Trees <a name="sparsity"></a>
#
# **Density** is the standard graph-theoretic measure of how packed a graph
# is. For an undirected graph with `N` nodes and `E` edges:
#
# $$\rho \;=\; \frac{2E}{N(N-1)}.$$
#
# A complete graph has $\rho = 1$; a tree (the sparsest connected graph) has
# $E = N-1$, giving $\rho = 2/N$ for large $N$. Let's see where real molecules
# fall on this spectrum.

# + id="density-curve"
fig, ax = plt.subplots(figsize=(8, 5.5))
for dataset, color in DATASET_COLORS.items():
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    ax.scatter(subset["n_nodes"], subset["density"], s=14, alpha=0.4,
               color=color, label=dataset, edgecolors="none")

# Theoretical lower bound for a connected graph: a tree.
n_grid = np.arange(2, stats_df["n_nodes"].max() + 2)
ax.plot(n_grid, 2.0 / n_grid, "k--", lw=1.2, alpha=0.7,
        label=r"tree bound: $\rho = 2/N$")

ax.set_xlabel("Number of heavy atoms ($N$)")
ax.set_ylabel(r"Graph density $\rho = 2E / N(N-1)$")
ax.set_title("Real molecules hug the tree-density curve")
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# Quick comparison table — what does "sparse" look like in context?
comparison = pd.DataFrame([
    ("Erdős–Rényi random graph (p=0.5)",  0.50),
    ("Twitter follower graph (approx.)",  1e-4),
    ("Web graph (approx.)",                1e-7),
    ("Tree on 30 nodes",                   2 / 30),
    ("Tree on 100 nodes",                  2 / 100),
    ("Typical drug-like molecule (N≈30)", stats_df[
        (stats_df.dataset == "BACE") & (stats_df.n_nodes.between(28, 32))
    ]["density"].mean() if "BACE" in stats_df.dataset.values else 0.07),
    ("Typical QM9 molecule (N≈9)",        stats_df[
        (stats_df.dataset == "QM9") & (stats_df.n_nodes.between(8, 10))
    ]["density"].mean() if "QM9" in stats_df.dataset.values else 0.25),
], columns=["Graph type", "Density"])
print("\nDensity in context — different worlds of graphs:")
print(comparison.to_string(index=False))

# + [markdown] id="sparsity-interp"
# ### Why this matters for GNN design
#
# - **Molecular graphs hug the tree bound.** Adding rings bumps density up
#   slightly above $2/N$, but never close to $\sim 1/2$ (random) or even
#   $\sim 1/100$ (typical small-world). Molecules are an outlier even
#   among sparse-graph domains.
# - **Aggregation cost is proportional to E, not N².** A GCN layer on a
#   30-atom molecule costs ~30 edge messages, not ~900 attention scores.
#   This is why a 5-layer message-passing GNN on 100 molecules takes
#   milliseconds, even on a CPU.
# - **Full self-attention is mostly wasted compute on molecules.** A
#   vanilla Transformer over a 30-atom molecule computes 900 pairwise
#   attention scores, but only ~33 of those correspond to actual bonds.
#   The other 96 % are either capturing through-space chemistry (sometimes
#   useful, e.g. DimeNet, Notebook 08) or pure noise. This trade-off is the
#   reason Graph Transformers exist as a separate family.

# + [markdown] id="sparsity-checkpoint"
# ### Checkpoint Exercise
#
# A 30-atom drug has density ≈ 0.07. If you ran a vanilla Transformer
# (full $N \times N$ self-attention) on this molecule:
#
# 1. Roughly what fraction of attention weights correspond to actual bonded
#    pairs?
# 2. Why might it still be useful to attend to non-bonded pairs anyway?
#    *(Hint: think about hydrogen bonding, conformer geometry, and the way
#    DimeNet — Notebook 08 — uses distances rather than bonds.)*

# + [markdown] id="diameter"
# ## 7. Diameter and Shortest Paths <a name="diameter"></a>
#
# The **diameter** of a graph is the longest shortest path between any two
# nodes. For a chemist: how many bonds do you need to traverse to walk from
# the most isolated atom to the most isolated atom in the other direction?
#
# Diameter is the single most important topology number for choosing GNN
# depth — because after `k` message-passing layers, each atom has only
# seen information from at most `k` bonds away.

# + id="diameter-highlighted-path"
#@title Highlight the longest shortest path in our examples
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, (name, G) in zip(axes, graphs.items()):
    # Find a diameter path on the largest connected component.
    if G.number_of_nodes() == 0:
        continue
    cc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(cc_nodes)
    # Brute-force pair with longest shortest path.
    nodes = list(H.nodes())
    best_len, best_path = 0, [nodes[0]]
    for u in nodes:
        lengths = nx.single_source_shortest_path_length(H, u)
        v_far = max(lengths, key=lengths.get)
        if lengths[v_far] > best_len:
            best_len = lengths[v_far]
            best_path = nx.shortest_path(H, u, v_far)
    draw_molecule_graph(G, ax,
                        title=f"{name}\ndiameter = {best_len} bonds",
                        highlight_path=best_path)
fig.suptitle("The longest shortest path (in red) is the graph's diameter",
             y=1.05, fontsize=13)
plt.tight_layout()
plt.show()

# + [markdown] id="diameter-distribution-md"
# ### Diameter across thousands of molecules
#
# How much does the diameter vary across each dataset, and how does it scale
# with size?

# + id="diameter-distributions"
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# Left: per-dataset diameter histograms.
max_diam = int(stats_df["diameter"].max())
bins = np.arange(-0.5, max_diam + 1.5)
for dataset, color in DATASET_COLORS.items():
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    axes[0].hist(subset["diameter"], bins=bins, alpha=0.55, density=True,
                 label=dataset, color=color, edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("Graph diameter (bonds)")
axes[0].set_ylabel("density")
axes[0].set_title("Diameter distribution — narrow and small")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: diameter vs n_atoms scatter. Add a sqrt-N reference for contrast.
for dataset, color in DATASET_COLORS.items():
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    axes[1].scatter(subset["n_nodes"], subset["diameter"],
                    s=12, alpha=0.35, color=color, label=dataset,
                    edgecolors="none")

n_grid = np.arange(2, stats_df["n_nodes"].max() + 2)
axes[1].plot(n_grid, np.sqrt(n_grid), "k--", lw=1.2, alpha=0.7,
             label=r"$\sqrt{N}$ reference")
axes[1].plot(n_grid, np.log2(n_grid), "k:", lw=1.2, alpha=0.7,
             label=r"$\log_2 N$ (small-world)")
axes[1].set_xlabel("Number of heavy atoms ($N$)")
axes[1].set_ylabel("Graph diameter")
axes[1].set_title("Diameter scales much faster than $\\log N$")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# + [markdown] id="diameter-interp"
# ### What to notice
#
# - **QM9 diameters cluster at 3–6 bonds.** With ≤ 9 heavy atoms, you simply
#   can't have a longer chain than that.
# - **ESOL is 5–10 bonds; BACE pushes 8–15 bonds.** Drug-like molecules with
#   extended scaffolds and side-chains routinely have diameters in this
#   range.
# - **Diameter grows faster than `log N`** — closer to `√N` or even linear.
#   Social networks are *small-world*: their diameter scales as `log N`
#   thanks to long-range edges (e.g. "celebrity follows random user"). A
#   molecule has *no* long-range bonded edges; you can only get from one end
#   to the other by walking along the backbone, so distance accumulates.
# - **The average shortest path is roughly half the diameter.** Both numbers
#   tell the same story; we focus on diameter as the worst case.

# + [markdown] id="components"
# ## 8. Connected Components <a name="components"></a>
#
# A molecule that is a single connected graph is what we usually picture.
# But some entries in real datasets are **salts, co-crystals, or
# mixtures**: in SMILES, those are separated by `.`, and they parse as
# multiple connected components in the same molecular graph.
#
# This matters because message passing only flows along edges. Two
# disconnected components in the same molecule will *never* exchange
# information no matter how many layers you stack.

# + id="components-counts"
fig, ax = plt.subplots(figsize=(7.5, 4.0))
for dataset, color in DATASET_COLORS.items():
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    counts = subset["n_components"].value_counts().sort_index()
    ax.bar(counts.index + (list(DATASET_COLORS).index(dataset) - 1) * 0.25,
           counts.values / len(subset),
           width=0.22, color=color, label=dataset, alpha=0.9)
ax.set_xlabel("Number of connected components")
ax.set_ylabel("Fraction of molecules")
ax.set_title("Most molecules are one connected graph — but not all")
ax.set_yscale("log")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

n_multi = (stats_df.n_components > 1).sum()
print(f"\n{n_multi} of {len(stats_df)} molecules ({100*n_multi/len(stats_df):.2f} %) "
      f"have more than one connected component.")
print("These are typically salts (e.g., HCl salts of basic drugs) or "
      "co-crystals — handle them carefully in your training pipeline.")

# + [markdown] id="depth"
# ## 9. Receptive Field and the 3–5 Layer Rule <a name="depth"></a>
#
# This is the payoff section.
#
# After `k` rounds of message passing, each atom's hidden state contains
# information from atoms at most `k` bonds away — its **`k`-hop ego graph**.
# This is the **receptive field** of a `k`-layer GNN.
#
# Two natural coverage criteria, with different practical meanings:
#
# - **Radius criterion** ($k \geq \text{radius}(G)$): at least one atom — a
#   *central* atom — can see the entire molecule. For graph-level tasks
#   with a sum/mean readout, this is the relevant threshold: once even a
#   single node has seen everything, the readout has the information it
#   needs.
# - **Diameter criterion** ($k \geq \text{diameter}(G)$): *every* atom can
#   see every other atom. This is the harder requirement, needed for
#   per-atom (node-level) tasks where every atom must integrate global
#   context.
#
# Plot both curves, per dataset.

# + id="coverage-curve"
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

k_grid = np.arange(0, int(stats_df["diameter"].max()) + 2)

# Left: radius coverage (the looser, more practical criterion).
for dataset, color in DATASET_COLORS.items():
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    cov = [(subset["radius"] <= k).mean() for k in k_grid]
    axes[0].plot(k_grid, cov, marker="o", color=color, lw=2.0,
                 label=f"{dataset}  (median radius = "
                       f"{int(subset['radius'].median())})")
axes[0].axvspan(3, 5, color="grey", alpha=0.15)
axes[0].text(4, 0.05, "conventional\n3–5 layer range",
             ha="center", va="bottom", color="grey", fontsize=10)
axes[0].set_xlabel("Number of message-passing layers ($k$)")
axes[0].set_ylabel(r"Fraction of molecules with radius $\leq k$")
axes[0].set_title("Radius coverage — one central atom sees everything")
axes[0].set_ylim(-0.02, 1.02)
axes[0].grid(alpha=0.3)
axes[0].legend(loc="lower right")

# Right: diameter coverage (the stricter all-pairs criterion).
for dataset, color in DATASET_COLORS.items():
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    cov = [(subset["diameter"] <= k).mean() for k in k_grid]
    axes[1].plot(k_grid, cov, marker="s", color=color, lw=2.0,
                 label=f"{dataset}  (median diameter = "
                       f"{int(subset['diameter'].median())})")
axes[1].axvspan(3, 5, color="grey", alpha=0.15)
axes[1].text(4, 0.05, "conventional\n3–5 layer range",
             ha="center", va="bottom", color="grey", fontsize=10)
axes[1].set_xlabel("Number of message-passing layers ($k$)")
axes[1].set_ylabel(r"Fraction of molecules with diameter $\leq k$")
axes[1].set_title("Diameter coverage — every atom sees every other")
axes[1].set_ylim(-0.02, 1.02)
axes[1].grid(alpha=0.3)
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.show()

# Numerical readout: both criteria at k = 2..6.
print("\nFraction of molecules covered, by criterion and layer count:")
print(f"\n  RADIUS criterion (one central atom sees everything):")
print(f"  {'dataset':<6}  {'k=2':>6}  {'k=3':>6}  {'k=4':>6}  {'k=5':>6}  {'k=6':>6}")
for dataset in DATASET_COLORS:
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    vals = [(subset["radius"] <= k).mean() for k in [2, 3, 4, 5, 6]]
    print(f"  {dataset:<6}  " + "  ".join(f"{v:>6.1%}" for v in vals))

print(f"\n  DIAMETER criterion (every atom sees every other):")
print(f"  {'dataset':<6}  {'k=2':>6}  {'k=3':>6}  {'k=4':>6}  {'k=5':>6}  {'k=6':>6}")
for dataset in DATASET_COLORS:
    subset = stats_df[stats_df.dataset == dataset]
    if len(subset) == 0:
        continue
    vals = [(subset["diameter"] <= k).mean() for k in [2, 3, 4, 5, 6]]
    print(f"  {dataset:<6}  " + "  ".join(f"{v:>6.1%}" for v in vals))

# + [markdown] id="depth-interp"
# ### Reading the curves
#
# - **QM9 is essentially done by $k = 3$ for the radius criterion** (≈ 96 %)
#   and by $k = 5$ for the diameter criterion (≈ 82 %). For small
#   molecules a 3–5 layer GCN really does see the whole graph.
# - **ESOL sits in between.** Three layers cover radius for ~47 % of
#   molecules; five layers cover ~87 %. Six layers push past 94 %. The
#   diameter criterion lags as you'd expect.
# - **BACE is where the picture honestly breaks.** Drug-like molecules
#   are large (median radius ≈ 7, diameter ≈ 13–14), so a 3-layer GCN
#   reaches a tiny fraction of any molecule, and even a 6-layer model
#   only fully covers ~20 % under the radius criterion. **Pure
#   bond-graph message passing simply does not cover large molecules.**
# - **So what justifies 3–5 layers, given that?** The point isn't that
#   3–5 layers always covers the whole graph — it doesn't. The point is
#   that 3–5 layers is the **best reach you can buy before oversmoothing
#   (Notebook [02.1](02.1_GNN_oversmoothing.ipynb)) destroys the
#   representation**. Going to 8 or 10 layers wouldn't fix BACE; it
#   would collapse every atom to the same vector. For small molecules
#   (QM9, most of ESOL) the 3–5 sweet spot happens to coincide with full
#   coverage; for drug-sized molecules, it's the deepest model you can
#   train at all.
# - **This is also why the second half of the course exists.** When a
#   pure bond-graph 3–5 layer GCN under-reaches *and* a deeper one
#   over-smooths, you need a fundamentally different recipe — that's
#   what SchNet/DimeNet (Notebooks 07–08, distance-based), EGNN
#   (Notebook 09, geometric), and Graph Transformer (Notebook 10,
#   global attention) are doing.
#
# **Bottom line.** The 3–5 layer rule is *derived* from two facts: small
# typical molecules + oversmoothing of deeper layers. The radius
# distribution tells you *which* dataset that rule is comfortable with
# (QM9, ESOL) and which it strains against (BACE).
#
# > **What architectures do when this isn't enough.** For molecules where
# > the diameter is large or the relevant interactions are *through-space*
# > rather than *through-bond*, the message-passing-with-bond-edges design
# > runs out of steam. That's exactly what motivates the architectures in
# > the second half of the course:
# >
# > - **SchNet / DimeNet** (Notebooks 07–08): use 3D distances directly,
# >   shortcutting graph diameter altogether.
# > - **EGNN** (Notebook 09): equivariant on top of geometric features.
# > - **Graph Transformer / Graphormer** (Notebook 10): replace local
# >   message passing with global attention, paying $O(N^2)$ for instant
# >   coverage of any diameter.

# + [markdown] id="summary"
# ## 10. Summary <a name="summary"></a>
#
# Four numbers describe almost every molecular graph you'll meet in
# chemistry, and together they shape the design space of chemistry GNNs:
#
# | Property | Typical value | What it tells the network designer |
# |---|---|---|
# | Size ($N$) | 5–40 heavy atoms | Cheap per-molecule inference; large batch sizes are easy. |
# | Avg. degree | ≈ 2.0–2.2 | Aggregation is $O(N)$, not $O(N^2)$. |
# | Density ($\rho$) | $\approx 2/N$, near the tree bound | Local message passing is the natural operation. |
# | Diameter | 3–15 bonds | $k \in [3, 5]$ message-passing layers covers most molecules. |
#
# **Reflection prompt.** Suppose you had to design a GNN for **proteins**:
# typical size ~1000 residues, diameter ~30 in the residue contact graph.
# Which of the four conclusions above would you keep, and which would you
# revisit? *(Hint: the diameter argument doesn't survive contact with
# 1000-residue chains — and that's why protein GNNs look very different
# from chemistry GNNs.)*
#
# **Where to go next:**
#
# - **Notebook 02 — Message Passing.** With your new vocabulary, the
#   formal definition of GCN-style aggregation will feel concrete.
# - **Notebook 02.1 — Oversmoothing.** The reason "more layers" isn't a
#   free lunch — the other half of the depth argument made here.
# - **Notebooks 04–06 — GCN, GAT, GIN.** The first GNNs you'll
#   implement. Notice how they all stop at 3–5 layers.
# - **Notebooks 08–10 — DimeNet, EGNN, Graph Transformer.** The
#   architectures designed when the simple "bonds-only, 3–5 layer" picture
#   isn't enough.

# + [markdown] id="references"
# ## 11. References <a name="references"></a>
#
# 1. **Ramakrishnan et al. (2014)** — *Quantum chemistry structures and
#    properties of 134 kilo molecules.* Scientific Data 1, 140022. The QM9
#    dataset paper.
# 2. **Subramanian et al. (2016)** — *Computational modeling of β-secretase
#    1 (BACE-1) inhibitors.* J. Chem. Inf. Model. 56(10), 1936–1949. The
#    BACE dataset.
# 3. **Delaney (2004)** — *ESOL: estimating aqueous solubility directly
#    from molecular structure.* J. Chem. Inf. Comput. Sci. 44(3),
#    1000–1005. The ESOL solubility dataset.
# 4. **Wu et al. (2018)** — *MoleculeNet: a benchmark for molecular
#    machine learning.* Chemical Science 9, 513–530. Standardised
#    chemistry benchmarks including BACE and ESOL.
# 5. **Kipf & Welling (2017)** — *Semi-Supervised Classification with
#    Graph Convolutional Networks.* ICLR 2017. The receptive-field-as-
#    $k$-hop-neighbourhood view used here.
# 6. **Barabási & Albert (1999)** — *Emergence of scaling in random
#    networks.* Science 286(5439), 509–512. The power-law-degree contrast
#    we use in §5.
# 7. **Newman (2010)** — *Networks: An Introduction.* OUP. Standard
#    textbook treatment of density, diameter, and small-world graphs.
