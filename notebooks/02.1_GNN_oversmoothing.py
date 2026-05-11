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
# <a href="https://colab.research.google.com/github/HFooladi/GNNs-For-Chemists/blob/main/notebooks/02.1_GNN_oversmoothing.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="oversmoothing-title"
# # GNN Oversmoothing: When Deeper Is Worse
#
# ## Table of Contents
# 1. [Setup and Installation](#setup-and-installation)
# 2. [Recap: Message Passing in One Line](#recap)
# 3. [Why Oversmoothing Matters](#why-oversmoothing)
# 4. [Methanol Walk-through](#methanol)
# 5. [Scaling Up: Aspirin](#aspirin)
# 6. [Quantifying Oversmoothing](#quantify)
# 7. [The Pairwise-Distance View](#pairwise)
# 8. [Takeaway](#takeaway)
# 9. [References](#references)
#
# > **Where this sits in the series:** This is a *supplementary* notebook that extends
# > [Notebook 02 — Message Passing](02_GNN_message_passing.ipynb). In Notebook 02 we
# > saw one or two rounds of message passing. Here we ask: *what happens if we keep
# > going?*

# + [markdown] id="oversmoothing-objectives"
# ## Learning Objectives
#
# After this notebook, you will be able to:
#
# - **Explain** what oversmoothing is, in a single sentence a chemist would understand
# - **See** how repeated message passing makes every atom look the same — "all atoms turn grey"
# - **Quantify** that collapse with two simple diagnostics: feature standard deviation and pairwise distance
# - **Recognize** that very deep GNNs (10+ layers) often destroy the per-atom information needed for
#   tasks like partial-charge prediction, NMR-shift regression, or reactivity scoring
#
# > **One-sentence intuition:** Each round of message passing replaces an atom's feature with the
# > *average* of its neighbours. Average enough times, and every atom in the molecule converges to
# > the same number — at which point the network can no longer tell the atoms apart.

# + [markdown] id="setup-and-installation"
# ## 1. Setup and Installation <a name="setup-and-installation"></a>
#
# We only need RDKit (for the molecules), NetworkX (for the graph), and the usual NumPy /
# Matplotlib / Seaborn stack. No PyTorch needed — oversmoothing happens at the level of the
# *propagation operator*, before any learnable weights even appear.

# + colab={"base_uri": "https://localhost:8080/"} id="install-cell"
#@title Install required libraries
# !pip install -q rdkit

# + id="imports-cell"
#@title Import libraries and set plotting style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import seaborn as sns

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# Consistent plotting style across the tutorial series
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("Set2")
np.random.seed(42)

# + [markdown] id="recap"
# ## 2. Recap: Message Passing in One Line <a name="recap"></a>
#
# From [Notebook 02](02_GNN_message_passing.ipynb), recall that one round of (mean) message
# passing updates every atom's feature to the average of itself and its bonded neighbours:
#
# $$h_i^{(t+1)} \;=\; \frac{1}{|\mathcal{N}(i) \cup \{i\}|}\sum_{j \in \mathcal{N}(i) \cup \{i\}} h_j^{(t)}$$
#
# Including $i$ in the sum is the standard **self-loop** trick (Kipf & Welling, 2017): it keeps an
# atom's own feature in the mix at every step. We'll use mean aggregation with self-loops
# throughout this notebook — it's the simplest setup that *cleanly* exhibits oversmoothing.

# + [markdown] id="why-oversmoothing"
# ## 3. Why Oversmoothing Matters <a name="why-oversmoothing"></a>
#
# Mean aggregation is a *contraction*: averaging numbers can never make them more spread out, only
# less. Apply it once and neighbours grow more similar. Apply it many times and *every* atom
# becomes more similar to *every* other atom — the features converge to a single value, the
# graph's degree-weighted mean.
#
# **For a chemist:** imagine starting with each atom labelled by its atomic number — oxygen at
# Z = 8, carbon at Z = 6, hydrogen at Z = 1. After enough rounds of "ask your bonded neighbours
# their value and take the average," every atom in methanol holds the *same* number — somewhere
# between the highest and lowest starting value, depending on connectivity. The information
# *"I am the oxygen"* has been smoothed away. Worse, no amount of downstream fully-connected
# layers can recover it — it's gone from the representation.
#
# This is called **oversmoothing**, and it's the reason chemistry GNNs almost never use more than
# 3–5 message-passing layers. Let's see it happen.

# + id="utility-functions"
#@title Utility functions (graph construction, aggregation, plotting)

# Atomic symbol table for nicer plot labels
_ATOMIC_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}


def smiles2graph_nx(smiles: str) -> nx.Graph:
    """Convert a SMILES string to a NetworkX graph (with explicit Hs).

    Each node stores its atomic number under ``atomic_number`` and its
    canonical RDKit 2D coordinates under ``pos`` so the plot uses the
    standard chemical drawing layout (aromatic rings as hexagons, etc.)
    rather than a generic graph-theoretic layout.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    G = nx.Graph()
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        G.add_node(
            atom.GetIdx(),
            atomic_number=atom.GetAtomicNum(),
            pos=(p.x, p.y),
        )
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G


def initial_atomic_number_features(G: nx.Graph) -> np.ndarray:
    """Return a 1-D array of atomic numbers, indexed by node id."""
    return np.array(
        [float(G.nodes[i]["atomic_number"]) for i in sorted(G.nodes())],
        dtype=float,
    )


def apply_mean_aggregation(
    G: nx.Graph, features: np.ndarray, n_layers: int, include_self: bool = True
) -> list[np.ndarray]:
    """Apply repeated mean aggregation and return the full history.

    Uses the propagation matrix ``P = D^{-1} (A + I)`` (mean over self + neighbours)
    when ``include_self`` is True, otherwise ``P = D^{-1} A`` (pure neighbour mean).
    Returns a list of length ``n_layers + 1`` where index 0 is the input.
    """
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    if include_self:
        A = A + np.eye(A.shape[0])
    degrees = A.sum(axis=1, keepdims=True)
    P = A / np.clip(degrees, a_min=1e-12, a_max=None)
    history = [features.copy()]
    for _ in range(n_layers):
        history.append(P @ history[-1])
    return history


def plot_molecule_with_feature(
    G, feature, ax, pos, vmin, vmax, cmap="viridis", title=None, node_size=350
):
    """Draw a single molecular-graph panel coloured by a 1-D scalar feature.

    Returns the node PathCollection so a shared colorbar can be attached.
    """
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#444444", width=1.4, alpha=0.85)
    nodes_pc = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=feature, cmap=cmap,
        vmin=vmin, vmax=vmax, node_size=node_size, edgecolors="black", linewidths=0.7,
    )
    labels = {i: _ATOMIC_SYMBOL.get(int(G.nodes[i]["atomic_number"]), "?")
              for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8,
                            font_color="white", font_weight="bold")
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=11)
    return nodes_pc


def plot_oversmoothing_panels(G, feature_history, layers_to_show, suptitle):
    """Plot a row of molecule panels coloured by feature at each chosen layer.

    Shared colour scale comes from layer 0 (so colour drift toward the mean is visible).
    """
    pos = {i: G.nodes[i]["pos"] for i in G.nodes()}
    f0 = feature_history[0]
    vmin, vmax = float(f0.min()), float(f0.max())
    n = len(layers_to_show)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
    if n == 1:
        axes = [axes]
    sc = None
    for ax, layer in zip(axes, layers_to_show):
        sc = plot_molecule_with_feature(
            G, feature_history[layer], ax, pos, vmin=vmin, vmax=vmax,
            title=f"Layer {layer}",
        )
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.colorbar(sc, ax=axes, orientation="horizontal", shrink=0.55,
                 pad=0.08, label="feature value")
    plt.show()


def feature_std_curve(history):
    """Standard deviation of features across nodes, for each layer."""
    return np.array([float(np.std(h)) for h in history])


def pairwise_distance_matrix(feature: np.ndarray) -> np.ndarray:
    """|f_i - f_j| matrix for a 1-D feature vector."""
    return np.abs(feature[:, None] - feature[None, :])


def plot_distance_heatmaps(history, layers_to_show, atom_labels, suptitle):
    """Row of pairwise-distance heatmaps with shared colour scale."""
    vmax = float(pairwise_distance_matrix(history[0]).max())
    n = len(layers_to_show)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    im = None
    for ax, layer in zip(axes, layers_to_show):
        D = pairwise_distance_matrix(history[layer])
        im = ax.imshow(D, vmin=0.0, vmax=vmax, cmap="viridis")
        ax.set_title(f"Layer {layer}", fontsize=11)
        ax.set_xticks(range(len(atom_labels)))
        ax.set_yticks(range(len(atom_labels)))
        ax.set_xticklabels(atom_labels, fontsize=7, rotation=90)
        ax.set_yticklabels(atom_labels, fontsize=7)
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.8, pad=0.02,
                 label=r"$|f_i - f_j|$")
    plt.show()


def atom_label_list(G):
    """Atom labels like ['C1', 'O1', 'H1', 'H2', ...] for heatmap ticks."""
    counts = {}
    labels = []
    for i in sorted(G.nodes()):
        sym = _ATOMIC_SYMBOL.get(int(G.nodes[i]["atomic_number"]), "?")
        counts[sym] = counts.get(sym, 0) + 1
        labels.append(f"{sym}{counts[sym]}")
    return labels


# + [markdown] id="methanol"
# ## 4. Methanol Walk-through <a name="methanol"></a>
#
# Methanol (CH₃OH, 6 atoms with explicit hydrogens) is the same molecule we used in Notebook 02.
# Let's put atomic numbers on every atom and watch what mean aggregation does to them.

# + id="methanol-run"
G_meoh = smiles2graph_nx("CO")
features_meoh = initial_atomic_number_features(G_meoh)
history_meoh = apply_mean_aggregation(G_meoh, features_meoh, n_layers=20)

print("Initial features (atomic numbers):")
for i in sorted(G_meoh.nodes()):
    print(f"  Atom {i} ({_ATOMIC_SYMBOL[int(G_meoh.nodes[i]['atomic_number'])]}): "
          f"{features_meoh[i]:.2f}")
print(f"\nValue at every atom after 20 layers: {history_meoh[-1][0]:.4f}")
print(f"Max difference between atoms at layer 20: "
      f"{(history_meoh[-1].max() - history_meoh[-1].min()):.2e}")

# + id="methanol-panels"
plot_oversmoothing_panels(
    G_meoh, history_meoh, layers_to_show=[0, 1, 2, 5, 10, 20],
    suptitle="Methanol: atom features under repeated mean aggregation",
)

# + [markdown] id="methanol-interp"
# ### What you should see
#
# - **Layer 0**: oxygen stands out as the brightest node (Z = 8), the carbon is mid-tone (Z = 6),
#   the four hydrogens are dark (Z = 1). All four atom types are easily distinguished.
# - **Layers 1–2**: oxygen's neighbours start "borrowing" its value; the C–O bond softens the
#   contrast. Hydrogens attached to oxygen drift up; those attached to carbon stay low.
# - **Layer 5**: the colour gradient is already very narrow. You'd struggle to tell oxygen from
#   carbon by eye.
# - **Layers 10–20**: every node has converged to (essentially) the same value, ≈ 3.9 — the
#   connectivity-weighted mean of the starting atomic numbers. All atoms are visually
#   indistinguishable. **This is oversmoothing.**
#
# Methanol is so small that 5–10 rounds is enough to wipe out the chemical information. With a
# bigger graph this takes longer — but it still happens.

# + [markdown] id="aspirin"
# ## 5. Scaling Up: Aspirin <a name="aspirin"></a>
#
# Methanol is small enough that oversmoothing finishes in a handful of layers. Let's see the same
# phenomenon on a *real drug molecule*: **aspirin** (acetylsalicylic acid,
# SMILES `CC(=O)Oc1ccccc1C(=O)O`). With explicit hydrogens, aspirin has 21 atoms and three
# distinct chemical environments at layer 0 — carbons (Z = 6), oxygens (Z = 8), hydrogens
# (Z = 1) — including the ester linkage, the aromatic ring, and the carboxylic acid. It's a much
# richer starting state than methanol's, and a real molecule a chemist works with daily.

# + id="aspirin-run"
G_asp = smiles2graph_nx("CC(=O)Oc1ccccc1C(=O)O")
features_asp = initial_atomic_number_features(G_asp)
history_asp = apply_mean_aggregation(G_asp, features_asp, n_layers=20)

print(f"Aspirin: {G_asp.number_of_nodes()} atoms, "
      f"graph diameter = {nx.diameter(G_asp)} bond hops")
print(f"Feature std at layer 0:  {np.std(history_asp[0]):.3f}")
print(f"Feature std at layer 20: {np.std(history_asp[-1]):.4f}")
print(f"  -> features have collapsed by a factor of "
      f"{np.std(history_asp[0]) / np.std(history_asp[-1]):.0f}x in 20 layers")

# + id="aspirin-panels"
plot_oversmoothing_panels(
    G_asp, history_asp, layers_to_show=[0, 1, 2, 5, 10, 20],
    suptitle="Aspirin: atom features under repeated mean aggregation",
)

# + [markdown] id="aspirin-interp"
# ### What you should see
#
# - **Layer 0**: a *three-level* palette is obvious — the four oxygens of the ester and the
#   carboxylic acid glow brightest (Z = 8), the nine carbons of the methyl, the carbonyls, and
#   the aromatic ring sit in the middle (Z = 6), and the eight hydrogens are darkest (Z = 1).
#   The functional-group identity of every atom is readable straight off the colormap.
# - **Layers 1–2**: information starts bleeding across bonds. The aromatic ring (a 6-membered
#   cycle) mixes internally very fast because every ring carbon has multiple ring neighbours
#   pulling on it; the ester and acid oxygens drag their neighbouring carbons up; the methyl
#   hydrogens stay anomalously low because their only neighbour is a single carbon.
# - **Layer 5**: the three-level palette is gone. You'd struggle to point at the oxygen atoms.
# - **Layers 10–20**: uniform teal — the *carboxylic acid oxygen looks identical to the methyl
#   hydrogen*. **All chemical identity has been smoothed away.**
#
# Aspirin's collapse is slower than methanol's (more atoms to mix, longer diameter) but the
# endpoint is the same: a representation in which every atom carries the same number.

# + [markdown] id="quantify"
# ## 6. Quantifying Oversmoothing <a name="quantify"></a>
#
# Colour collapse is a visual diagnostic, but for a paper or a model report you want a single
# number. The simplest is the **standard deviation of node features**: high std-dev means the
# atoms are still distinguishable; std-dev → 0 means oversmoothed.

# + id="variance-curve"
n_layers = 20
std_meoh = feature_std_curve(history_meoh)
std_asp = feature_std_curve(history_asp)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(range(n_layers + 1), std_meoh, marker="o", label="methanol (6 atoms)")
ax.plot(range(n_layers + 1), std_asp, marker="s", label="aspirin (21 atoms)")
ax.set_yscale("log")
ax.set_xlabel("Message-passing layer")
ax.set_ylabel("std-dev of node features (log scale)")
ax.set_title("Feature standard deviation collapses with depth")
ax.axhline(1e-2, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(n_layers, 1.2e-2, "≈ uniform", color="grey", ha="right", va="bottom", fontsize=9)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# + [markdown] id="variance-interp"
# Both curves decay roughly exponentially with depth — straight lines on a log scale. Methanol
# collapses faster (smaller graph, shorter diameter); aspirin lags but follows the same fate. The
# *rate* of decay is governed by the graph's spectral gap — a topology property, not something
# the network architecture can change once mean aggregation has been chosen.

# + [markdown] id="pairwise"
# ## 7. The Pairwise-Distance View <a name="pairwise"></a>
#
# Standard deviation collapses many atoms to one number. To *see* exactly which atoms are
# becoming indistinguishable from which, we plot the matrix $|f_i - f_j|$ at each layer. Bright
# off-diagonal cells = pairs that can still be told apart. A uniformly dark matrix = total
# oversmoothing.

# + id="aspirin-heatmaps"
plot_distance_heatmaps(
    history_asp,
    layers_to_show=[0, 2, 5, 15],
    atom_labels=atom_label_list(G_asp),
    suptitle="Aspirin: pairwise feature distance $|f_i - f_j|$ across layers",
)

# + [markdown] id="pairwise-interp"
# ### What you should see
#
# - **Layer 0**: a sharp three-block structure — the O×H pairs are brightest (distance = 7),
#   the O×C and C×H pairs are mid-bright (distance = 2 and 5 respectively), and the C×C, O×O,
#   H×H diagonals are zero (each chemical type starts with one shared value).
# - **Layer 2**: the cross-block bands are softening, and within each block you can now see
#   *positional* structure — methyl carbons differ from carbonyl carbons differ from ring
#   carbons, because they have different neighbourhoods.
# - **Layer 5**: most cells are dim; the gross C/O/H structure is gone, only fine intra-block
#   positional differences remain.
# - **Layer 15**: the entire matrix is uniformly dark. *Every atom looks the same as every other
#   atom* — the network has lost its ability to identify which atom is which.
#
# This is what oversmoothing means in practice: any downstream predictor (a classifier, a
# regressor for partial charges, a node-level fingerprint) is reading from a representation in
# which all the atom-identifying information has been averaged away.

# + [markdown] id="checkpoint"
# ### Checkpoint Exercise
#
# Try to answer these before moving on:
#
# 1. **Predict benzene's behaviour.** Benzene (C₆H₆) is a small ring rather than a chain. Without
#    running the code, would you expect benzene to oversmooth *faster* or *slower* than n-hexane?
#    Why? *(Hint: how does the graph diameter of a 12-atom ring compare to a 20-atom chain?)*
#
# 2. **Self-loops.** In `apply_mean_aggregation`, we use `P = D⁻¹(A + I)`. What changes if you
#    set `include_self=False`? *(Hint: think about the C₂H₆ bipartite-style sub-pattern — what
#    happens to a hydrogen whose only neighbour is a carbon, and whose only neighbour's
#    neighbours are also hydrogens?)*
#
# 3. **Why this matters for partial charges.** Suppose we want a GNN to predict the partial
#    charge on each atom of methanol. If we naively stack 10 GCN layers, what is the likely
#    failure mode? Why might 3 layers actually work *better* than 10?

# + [markdown] id="takeaway"
# ## 8. Takeaway <a name="takeaway"></a>
#
# - **Mean message passing is a contraction.** Repeated application drives every node's feature
#   toward the graph mean, regardless of starting values.
# - **Oversmoothing is a topology effect, not a training issue.** It happens before any learnable
#   weights enter the picture. Adding parameters and training longer does not fix it.
# - **Graph size and diameter set the timescale.** Small, dense graphs (methanol) collapse in a
#   handful of layers. Larger, more topologically diverse graphs (aspirin) take more layers — but
#   they still collapse.
# - **Practical rule of thumb:** chemistry GNNs typically use 3–5 message-passing layers. Going
#   deeper rarely improves accuracy and usually hurts.
#
# **Common mitigations** (not implemented here — left as next-step reading):
#
# - **Skip / residual connections** — each layer adds its update to the input instead of
#   replacing it, preserving the original information.
# - **Jumping Knowledge networks (JKNet)** — read out from *all* intermediate layers, not just
#   the last one.
# - **PairNorm / GraphNorm** — explicit normalisation that re-spreads node features after each
#   layer.
# - **Use fewer layers** — often the simplest and most effective answer.
#
# These are explored in modern architectures introduced later in the series.

# + [markdown] id="references"
# ## 9. References <a name="references"></a>
#
# 1. **Kipf & Welling (2017)** — *Semi-Supervised Classification with Graph Convolutional
#    Networks.* ICLR 2017. Introduces the propagation rule with self-loops used here.
# 2. **Li, Han & Wu (2018)** — *Deeper Insights into Graph Convolutional Networks for
#    Semi-Supervised Learning.* AAAI 2018. First systematic analysis of oversmoothing.
# 3. **Oono & Suzuki (2020)** — *Graph Neural Networks Exponentially Lose Expressive Power for
#    Node Classification.* ICLR 2020. Rigorous theory: features converge to a low-dimensional
#    subspace at an exponential rate.
# 4. **Rusch, Bronstein & Mishra (2023)** — *A Survey on Oversmoothing in Graph Neural
#    Networks.* Comprehensive overview of causes and mitigations.
# 5. **Xu et al. (2018)** — *Representation Learning on Graphs with Jumping Knowledge
#    Networks.* ICML 2018. One of the standard mitigations.
