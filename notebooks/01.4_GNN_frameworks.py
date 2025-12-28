# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: pytorch
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/HFooladi/GNNs-For-Chemists/blob/main/notebooks/01.4_GNN_frameworks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="bXCYDK0NkdpC"
# # GNN Frameworks Comparison: PyTorch Geometric, DGL, and Jraph
#
# ## Table of Contents
# 1. [Setup and Installation](#setup-and-installation)
# 2. [Introduction and Learning Objectives](#introduction)
# 3. [The smiles2graph Function](#smiles2graph-function)
# 4. [Brief Introduction to JAX](#jax-introduction)
# 5. [PyTorch Geometric: The Data Object](#pytorch-geometric)
# 6. [Deep Graph Library: DGLGraph](#deep-graph-library)
# 7. [Jraph: GraphsTuple](#jraph)
# 8. [Side-by-Side Comparison](#comparison)
# 9. [API Differences and Design Philosophies](#api-differences)
# 10. [Framework Interoperability](#interoperability)
# 11. [Practical Example: Multiple Molecules](#practical-example)
# 12. [When to Choose Which Framework](#framework-selection)
# 13. [Checkpoint Exercises](#exercises)
# 14. [Conclusion and References](#conclusion)

# + [markdown] id="XHX_uMbYkkX7"
# ## 1. Setup and Installation <a name="setup-and-installation"></a>
#
# In this tutorial, we'll explore three major graph neural network frameworks:
# - **PyTorch Geometric (PyG)**: PyTorch-native, research-focused library
# - **Deep Graph Library (DGL)**: Multi-backend library supporting PyTorch, TensorFlow, and MXNet
# - **Jraph**: JAX-based library from Google DeepMind
#
# Each framework has its own design philosophy and data structures for representing graphs.

# + id="A7-zyVlhhvwk"
#@title Install required libraries
# Core dependencies
# !pip install -q rdkit

# PyTorch Geometric
# !pip install -q torch-geometric

# Deep Graph Library (PyTorch backend)
# !pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html

# JAX and Jraph
# !pip install -q jraph

# + id="N_1gfxN7iJsO"
#@title Import required libraries and check availability
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# RDKit for molecular handling
from rdkit import Chem
from rdkit.Chem import Draw

# NetworkX for graph visualization
import networkx as nx

# IPython display utilities
from IPython.display import display

# Set plotting style
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("Set2")

# Set random seeds for reproducibility
np.random.seed(42)

# + id="framework-availability"
#@title Check framework availability
# PyTorch Geometric
try:
    import torch
    from torch_geometric.data import Data as PyGData
    from torch_geometric.data import Batch as PyGBatch
    TORCH_GEOMETRIC_AVAILABLE = True
    torch.manual_seed(42)
    print("PyTorch Geometric: Available")
    print(f"  PyTorch version: {torch.__version__}")
except ImportError as e:
    TORCH_GEOMETRIC_AVAILABLE = False
    print(f"PyTorch Geometric: Not available ({e})")

# Deep Graph Library
try:
    import dgl
    DGL_AVAILABLE = True
    print(f"DGL: Available (version {dgl.__version__})")
except ImportError as e:
    DGL_AVAILABLE = False
    print(f"DGL: Not available ({e})")

# JAX and Jraph
# Note: JAX on Colab with GPU can have initialization issues.
# We force CPU backend for this tutorial since we're just demonstrating data structures.
try:
    import os
    # Force JAX to use CPU to avoid GPU/CUDA version mismatch issues on Colab
    os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax
    import jax.numpy as jnp
    import jraph
    JRAPH_AVAILABLE = True
    print(f"Jraph: Available")
    print(f"  JAX version: {jax.__version__}")
    print(f"  JAX backend: {jax.default_backend()}")
except ImportError as e:
    JRAPH_AVAILABLE = False
    print(f"Jraph: Not available ({e})")
except Exception as e:
    JRAPH_AVAILABLE = False
    print(f"Jraph: Error during initialization ({e})")

print("\n" + "="*50)
if all([TORCH_GEOMETRIC_AVAILABLE, DGL_AVAILABLE, JRAPH_AVAILABLE]):
    print("All frameworks available! Ready for full comparison.")
else:
    print("Some frameworks missing. Examples will use available ones.")

# + [markdown] id="introduction"
# ## 2. Introduction and Learning Objectives <a name="introduction"></a>
#
# In the previous tutorials (01, 01.1, 01.2, 01.3), we learned how to represent molecules as graphs with:
# - **Nodes** representing atoms with feature vectors
# - **Edges** representing bonds with feature vectors
#
# We also explored different graph representations: 2D connectivity, 3D spatial graphs, dual graphs, and fragment graphs.
#
# Now, we face a practical question: **which framework should we use to implement GNNs?**
#
# Three major frameworks dominate the GNN landscape:
#
# | Framework | Backend | Key Strength | Primary Use Case |
# |-----------|---------|--------------|------------------|
# | **PyTorch Geometric** | PyTorch | Research, Datasets | Academic papers, Benchmarking |
# | **DGL** | Multi (PyTorch/TF/MXNet) | Flexibility, Scale | Production, Large graphs |
# | **Jraph** | JAX | Performance, Functional | HPC, TPU acceleration |
#
# ### Learning Objectives
#
# By the end of this tutorial, you will be able to:
# - **Understand** the core data structures of each framework (Data, DGLGraph, GraphsTuple)
# - **Convert** molecular graphs to each framework's format
# - **Recognize** API differences and design philosophies
# - **Choose** the right framework for your specific use case
# - **Convert** graphs between frameworks when needed

# + [markdown] id="smiles2graph-section"
# ## 3. The smiles2graph Function <a name="smiles2graph-function"></a>
#
# Before we dive into the frameworks, we need a function to convert molecules to graph data.
# This is an enhanced version of the function from Tutorial 01, producing:
# - **21 node features** per atom
# - **6 edge features** per bond
#
# The function returns numpy arrays that we'll then convert to each framework's format.

# + id="smiles2graph-implementation"
def smiles2graph(smiles: str):
    """
    Convert a SMILES string to graph representation with comprehensive features.

    This function produces framework-agnostic numpy arrays that can be converted
    to PyTorch Geometric, DGL, or Jraph format.

    Args:
        smiles (str): SMILES string of the molecule

    Returns:
        tuple: (node_features, edge_indices, edge_features)
            - node_features: np.array of shape [num_nodes, 21]
            - edge_indices: list of (src, dst) tuples
            - edge_features: np.array of shape [num_edges, 6]

    Node Features (21 dimensions):
        - Atom type one-hot (11 dims): C, O, N, H, F, P, S, Cl, Br, I, Other
        - Formal charge (1 dim)
        - Aromaticity (1 dim)
        - Ring membership (1 dim)
        - Degree (1 dim)
        - Total hydrogens (1 dim)
        - Radical electrons (1 dim)
        - Hybridization one-hot (4 dims): SP, SP2, SP3, Other

    Edge Features (6 dimensions):
        - Bond type one-hot (4 dims): Single, Double, Triple, Aromatic
        - Conjugation (1 dim)
        - Ring membership (1 dim)
    """
    # Create RDKit molecule from SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Add hydrogens to explicit representation for complete molecular structure
    mol = Chem.AddHs(mol)

    # Define mapping of bond types to indices for one-hot encoding
    bond_type_to_idx = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }

    # Get total number of atoms in the molecule
    n_atoms = mol.GetNumAtoms()

    # Initialize list to store node (atom) features
    node_features = []
    atom_symbols = []  # Store symbols for visualization

    for atom in mol.GetAtoms():
        # Extract basic atomic properties
        atom_type = atom.GetSymbol()
        atom_symbols.append(atom_type)
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())

        # Create one-hot encoding for common atom types (11 features)
        atom_types = ['C', 'O', 'N', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        atom_type_onehot = [1 if atom_type == t else 0 for t in atom_types]
        if atom_type not in atom_types:
            atom_type_onehot.append(1)  # "Other" category
        else:
            atom_type_onehot.append(0)

        # Create one-hot encoding for hybridization states (4 features)
        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3
        ]
        hybridization_onehot = [1 if hybridization == h else 0 for h in hybridization_types]
        if hybridization not in hybridization_types:
            hybridization_onehot.append(1)  # "Other" hybridization
        else:
            hybridization_onehot.append(0)

        # Combine all atomic features into a single feature vector (21 dims total)
        features = atom_type_onehot + [
            formal_charge,                    # Formal charge of the atom
            is_aromatic,                      # Whether atom is part of an aromatic system
            is_in_ring,                       # Whether atom is part of a ring
            atom.GetDegree(),                 # Number of directly bonded neighbors
            atom.GetTotalNumHs(),             # Total number of hydrogens
            atom.GetNumRadicalElectrons()     # Number of unpaired electrons
        ] + hybridization_onehot

        node_features.append(features)

    # Convert node features to numpy array
    node_features = np.array(node_features, dtype=np.float32)

    # Initialize lists for edge information
    edge_features = []
    edge_indices = []

    # Process each bond in the molecule
    for bond in mol.GetBonds():
        # Get indices of atoms involved in the bond
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # Create one-hot encoding for bond type (4 dims)
        bond_type = bond.GetBondType()
        bond_type_onehot = np.zeros(len(bond_type_to_idx), dtype=np.float32)
        if bond_type in bond_type_to_idx:
            bond_type_onehot[bond_type_to_idx[bond_type]] = 1

        # Extract additional bond properties
        is_conjugated = float(bond.GetIsConjugated())
        is_in_ring = float(bond.IsInRing())

        # Combine all bond features (6 dims total)
        features = np.concatenate([bond_type_onehot, [is_conjugated, is_in_ring]])

        # Add edge in both directions (undirected graph representation)
        edge_features.append(features)
        edge_indices.append((begin_idx, end_idx))

        edge_features.append(features)  # Same feature for reverse direction
        edge_indices.append((end_idx, begin_idx))

    # Convert edge features to numpy array
    if edge_features:
        edge_features = np.array(edge_features, dtype=np.float32)
    else:
        edge_features = np.empty((0, 6), dtype=np.float32)

    return node_features, edge_indices, edge_features, atom_symbols


# + id="test-smiles2graph"
#@title Test smiles2graph with Ethanol
# Let's use Ethanol (CCO) as our running example throughout this tutorial

smiles = "CCO"  # Ethanol
molecule_name = "Ethanol"

# Convert to graph
node_features, edge_indices, edge_features, atom_symbols = smiles2graph(smiles)

# Display the molecule structure
mol = Chem.MolFromSmiles(smiles)
mol_with_h = Chem.AddHs(mol)
display(Draw.MolToImage(mol_with_h, size=(300, 200)))

# Print graph statistics
print(f"\n{'='*50}")
print(f"Molecule: {molecule_name} (SMILES: {smiles})")
print(f"{'='*50}")
print(f"\nGraph Statistics:")
print(f"  Number of nodes (atoms): {len(node_features)}")
print(f"  Number of edges (bonds, bidirectional): {len(edge_indices)}")
print(f"  Node feature dimensions: {node_features.shape[1]}")
print(f"  Edge feature dimensions: {edge_features.shape[1]}")
print(f"\nAtoms: {atom_symbols}")
print(f"\nNode features shape: {node_features.shape}")
print(f"Edge features shape: {edge_features.shape}")

# + id="visualize-features"
#@title Visualize node features
# Create a heatmap of node features for ethanol

feature_names = [
    'C', 'O', 'N', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Other',  # Atom types (11)
    'Charge', 'Aromatic', 'InRing', 'Degree', 'TotalH', 'Radical',  # Properties (6)
    'SP', 'SP2', 'SP3', 'HybOther'  # Hybridization (4)
]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    node_features,
    cmap='YlOrRd',
    annot=True,
    fmt='.0f',
    xticklabels=feature_names,
    yticklabels=[f"{i}: {s}" for i, s in enumerate(atom_symbols)],
    ax=ax,
    cbar_kws={'label': 'Feature Value'}
)
ax.set_xlabel('Features')
ax.set_ylabel('Atoms')
ax.set_title(f'Node Feature Matrix for {molecule_name}')
plt.tight_layout()
plt.show()

# + [markdown] id="jax-introduction"
# ## 4. Brief Introduction to JAX <a name="jax-introduction"></a>
#
# Before we explore Jraph, it's helpful to understand JAX basics since it has a different programming paradigm from PyTorch.
#
# ### What is JAX?
#
# JAX is Google's library for high-performance numerical computing. It differs from PyTorch in several key ways:
#
# | Aspect | PyTorch | JAX |
# |--------|---------|-----|
# | **Programming Style** | Object-oriented, stateful | Functional, stateless |
# | **Arrays** | Mutable tensors | Immutable arrays |
# | **Gradients** | `loss.backward()` | `jax.grad(fn)` returns a function |
# | **Compilation** | Eager by default | JIT compilation with `@jax.jit` |
# | **Hardware** | GPU, CPU | GPU, CPU, **TPU native** |
#
# ### Key JAX Concepts for Jraph
#
# 1. **`jax.numpy` (jnp)**: NumPy-compatible API with GPU/TPU acceleration
# 2. **Immutability**: Arrays can't be modified in-place; operations return new arrays
# 3. **Pure Functions**: Functions should have no side effects for JIT compilation
# 4. **JIT Compilation**: `@jax.jit` decorator compiles functions for speed

# + id="jax-demo"
#@title Quick JAX demonstration
if JRAPH_AVAILABLE:
    try:
        print("JAX Basics Demonstration")
        print("="*50)

        # jax.numpy works like numpy
        x = jnp.array([1.0, 2.0, 3.0])
        print(f"\n1. Creating arrays with jnp:")
        print(f"   x = jnp.array([1.0, 2.0, 3.0])")
        print(f"   x = {x}")
        print(f"   Type: {type(x)}")

        # Arrays are immutable
        print(f"\n2. Arrays are immutable:")
        print(f"   In NumPy: x[0] = 10  # This works")
        print(f"   In JAX: x = x.at[0].set(10)  # Returns NEW array")
        x_new = x.at[0].set(10)
        print(f"   Original x: {x}")
        print(f"   New array: {x_new}")

        # JIT compilation
        print(f"\n3. JIT Compilation:")

        def slow_fn(x):
            return jnp.sum(x ** 2)

        fast_fn = jax.jit(slow_fn)

        print(f"   slow_fn(x) = {slow_fn(x)}")
        print(f"   fast_fn(x) = {fast_fn(x)}  # Same result, faster!")

        print("\n" + "="*50)
        print("These concepts are important for understanding Jraph!")

    except Exception as e:
        print(f"JAX demonstration failed: {e}")
        print("\nThis can happen due to GPU/CUDA issues on Colab.")
        print("The Jraph data structure examples below will still work!")
        print("\nKey JAX concepts to know:")
        print("  1. jax.numpy (jnp) is like numpy but with GPU/TPU support")
        print("  2. JAX arrays are immutable (use x.at[i].set(v) to 'modify')")
        print("  3. @jax.jit decorator compiles functions for speed")
else:
    print("JAX not available. Install with: pip install jax jaxlib")

# + [markdown] id="pytorch-geometric-section"
# ## 5. PyTorch Geometric: The Data Object <a name="pytorch-geometric"></a>
#
# PyTorch Geometric (PyG) is the most widely used GNN library in research. Its core data structure is the `Data` object.
#
# ### The Data Object Structure
#
# ```python
# Data(
#     x=[num_nodes, num_node_features],      # Node feature matrix
#     edge_index=[2, num_edges],              # Edge connectivity in COO format
#     edge_attr=[num_edges, num_edge_features], # Edge feature matrix
#     y=...,                                   # Target labels (optional)
#     pos=...,                                 # Node positions (optional)
#     **kwargs                                 # Any additional attributes
# )
# ```
#
# ### Understanding COO Format
#
# PyG uses **COO (Coordinate) format** for edge indices, which is more memory-efficient than adjacency matrices for sparse graphs like molecules:
#
# ```
# edge_index = [[0, 1, 1, 2],   # Source nodes
#               [1, 0, 2, 1]]   # Destination nodes
# ```
#
# This represents edges: 0→1, 1→0, 1→2, 2→1

# + id="pyg-implementation"
#@title Create PyG Data object from molecular graph

def create_pyg_data(smiles: str, y=None):
    """
    Create a PyTorch Geometric Data object from a SMILES string.

    Args:
        smiles: SMILES string of the molecule
        y: Optional target label

    Returns:
        torch_geometric.data.Data: PyG Data object
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not available")

    # Get graph data from smiles2graph
    node_features, edge_indices, edge_features, atom_symbols = smiles2graph(smiles)

    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)

    # Convert edge indices to COO format [2, num_edges]
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Create Data object
    data = PyGData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles
    )

    # Add target if provided
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    return data


# + id="pyg-demo"
#@title Demonstrate PyG Data object
if TORCH_GEOMETRIC_AVAILABLE:
    # Create PyG Data for Ethanol
    pyg_data = create_pyg_data("CCO")

    print("PyTorch Geometric Data Object for Ethanol")
    print("="*50)

    print(f"\n1. Accessing the Data object:")
    print(f"   pyg_data = {pyg_data}")

    print(f"\n2. Node features (x):")
    print(f"   Shape: {pyg_data.x.shape}")
    print(f"   Type: {type(pyg_data.x)}")
    print(f"   Access: pyg_data.x")

    print(f"\n3. Edge index (COO format):")
    print(f"   Shape: {pyg_data.edge_index.shape}")
    print(f"   First 4 edges:")
    print(f"   Source nodes:      {pyg_data.edge_index[0, :4].tolist()}")
    print(f"   Destination nodes: {pyg_data.edge_index[1, :4].tolist()}")

    print(f"\n4. Edge features:")
    print(f"   Shape: {pyg_data.edge_attr.shape}")
    print(f"   Access: pyg_data.edge_attr")

    print(f"\n5. Useful properties:")
    print(f"   pyg_data.num_nodes = {pyg_data.num_nodes}")
    print(f"   pyg_data.num_edges = {pyg_data.num_edges}")
    print(f"   pyg_data.num_node_features = {pyg_data.num_node_features}")
    print(f"   pyg_data.num_edge_features = {pyg_data.num_edge_features}")

    print(f"\n6. Custom attributes:")
    print(f"   pyg_data.smiles = '{pyg_data.smiles}'")
else:
    print("PyTorch Geometric not available")

# + id="pyg-visualization"
#@title Visualize PyG edge_index format
if TORCH_GEOMETRIC_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Molecule structure as graph
    ax1 = axes[0]
    node_features, edge_indices, _, atom_symbols = smiles2graph("CCO")

    G = nx.Graph()
    for i, symbol in enumerate(atom_symbols):
        G.add_node(i, label=symbol)

    # Add unique edges only (not bidirectional)
    seen_edges = set()
    for src, dst in edge_indices:
        edge = tuple(sorted([src, dst]))
        if edge not in seen_edges:
            G.add_edge(src, dst)
            seen_edges.add(edge)

    pos = nx.spring_layout(G, seed=42)
    colors = ['#FF6B6B' if s == 'C' else '#4ECDC4' if s == 'O' else '#95E1D3' for s in atom_symbols]

    nx.draw(G, pos, ax=ax1, node_color=colors, node_size=500,
            labels={i: s for i, s in enumerate(atom_symbols)},
            font_size=10, font_weight='bold')
    ax1.set_title("Ethanol Molecular Graph")

    # Right: Edge index visualization
    ax2 = axes[1]
    edge_index = pyg_data.edge_index.numpy()

    # Create a visual representation of edge_index
    edge_display = np.zeros((2, min(8, edge_index.shape[1])))
    edge_display[:] = edge_index[:, :8]

    sns.heatmap(edge_display, annot=True, fmt='.0f', cmap='Blues',
                yticklabels=['Source (row 0)', 'Dest (row 1)'],
                xticklabels=[f'Edge {i}' for i in range(edge_display.shape[1])],
                ax=ax2, cbar=False)
    ax2.set_title("PyG edge_index (COO format)\nShape: [2, num_edges]")

    plt.tight_layout()
    plt.show()

    print("\nKey Insight: PyG uses [2, num_edges] shape for edge_index")
    print("Row 0: source nodes, Row 1: destination nodes")
    print("Each column represents one directed edge")

# + [markdown] id="dgl-section"
# ## 6. Deep Graph Library: DGLGraph <a name="deep-graph-library"></a>
#
# DGL (Deep Graph Library) uses a different approach with `DGLGraph` objects that store features in dictionaries.
#
# ### The DGLGraph Structure
#
# ```python
# g = dgl.graph((source_nodes, dest_nodes))
# g.ndata['h'] = node_features    # Node features as dictionary
# g.edata['e'] = edge_features    # Edge features as dictionary
# ```
#
# ### Key DGL Concepts
#
# - **`ndata`**: Dictionary for node features (can have multiple feature sets)
# - **`edata`**: Dictionary for edge features
# - **Heterogeneous graphs**: Native support for multiple node/edge types
# - **Multi-backend**: Works with PyTorch, TensorFlow, and MXNet

# + id="dgl-implementation"
#@title Create DGL graph from molecular graph

def create_dgl_graph(smiles: str):
    """
    Create a DGL DGLGraph from a SMILES string.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        dgl.DGLGraph: DGL graph object
    """
    if not DGL_AVAILABLE:
        raise ImportError("DGL not available")

    # Get graph data from smiles2graph
    node_features, edge_indices, edge_features, atom_symbols = smiles2graph(smiles)

    # Extract source and destination nodes
    if len(edge_indices) > 0:
        src_nodes = [e[0] for e in edge_indices]
        dst_nodes = [e[1] for e in edge_indices]
    else:
        src_nodes = []
        dst_nodes = []

    # Create DGL graph from edge list
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(node_features))

    # Add node features using ndata dictionary
    g.ndata['h'] = torch.tensor(node_features, dtype=torch.float)

    # Add edge features using edata dictionary
    g.edata['e'] = torch.tensor(edge_features, dtype=torch.float)

    # Store SMILES as graph-level data
    g.smiles = smiles

    return g


# + id="dgl-demo"
#@title Demonstrate DGL DGLGraph object
if DGL_AVAILABLE:
    # Create DGL graph for Ethanol
    dgl_graph = create_dgl_graph("CCO")

    print("DGL DGLGraph Object for Ethanol")
    print("="*50)

    print(f"\n1. The DGLGraph object:")
    print(f"   dgl_graph = {dgl_graph}")

    print(f"\n2. Node features (ndata dictionary):")
    print(f"   Keys: {list(dgl_graph.ndata.keys())}")
    print(f"   Access: dgl_graph.ndata['h']")
    print(f"   Shape: {dgl_graph.ndata['h'].shape}")

    print(f"\n3. Edge features (edata dictionary):")
    print(f"   Keys: {list(dgl_graph.edata.keys())}")
    print(f"   Access: dgl_graph.edata['e']")
    print(f"   Shape: {dgl_graph.edata['e'].shape}")

    print(f"\n4. Edge connectivity:")
    src, dst = dgl_graph.edges()
    print(f"   Access: src, dst = dgl_graph.edges()")
    print(f"   First 4 edges:")
    print(f"   Source nodes:      {src[:4].tolist()}")
    print(f"   Destination nodes: {dst[:4].tolist()}")

    print(f"\n5. Useful methods:")
    print(f"   dgl_graph.num_nodes() = {dgl_graph.num_nodes()}")
    print(f"   dgl_graph.num_edges() = {dgl_graph.num_edges()}")

    print(f"\n6. Difference from PyG:")
    print(f"   - Features stored in dictionaries (ndata, edata)")
    print(f"   - Can have multiple feature sets: ndata['h'], ndata['pos'], etc.")
    print(f"   - Edges accessed via .edges() method, not edge_index attribute")
else:
    print("DGL not available")

# + id="dgl-visualization"
#@title Visualize DGL structure
if DGL_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ndata visualization
    ax1 = axes[0]
    ndata_example = {
        'h': 'Node features [9, 21]',
        'pos': '3D coordinates [9, 3]',
        'charge': 'Atomic charges [9, 1]'
    }

    y_pos = np.arange(len(ndata_example))
    ax1.barh(y_pos, [1]*len(ndata_example), color='#3498db', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"ndata['{k}']\n{v}" for k, v in ndata_example.items()])
    ax1.set_xlim(0, 2)
    ax1.set_xlabel('')
    ax1.set_title("DGL ndata: Node Feature Dictionary\n(Multiple feature sets possible)")
    ax1.set_xticks([])

    # Right: edata visualization
    ax2 = axes[1]
    edata_example = {
        'e': 'Edge features [16, 6]',
        'dist': 'Bond distances [16, 1]',
        'type': 'Bond types [16, 4]'
    }

    y_pos = np.arange(len(edata_example))
    ax2.barh(y_pos, [1]*len(edata_example), color='#e74c3c', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"edata['{k}']\n{v}" for k, v in edata_example.items()])
    ax2.set_xlim(0, 2)
    ax2.set_xlabel('')
    ax2.set_title("DGL edata: Edge Feature Dictionary\n(Multiple feature sets possible)")
    ax2.set_xticks([])

    plt.tight_layout()
    plt.show()

    print("\nKey Insight: DGL uses dictionaries for flexible feature storage")
    print("You can store multiple feature types under different keys!")

# + [markdown] id="jraph-section"
# ## 7. Jraph: GraphsTuple <a name="jraph"></a>
#
# Jraph (pronounced "giraffe") is Google DeepMind's GNN library built on JAX. It uses `GraphsTuple`, a named tuple that's designed for functional programming.
#
# ### The GraphsTuple Structure
#
# ```python
# GraphsTuple(
#     nodes,      # Node features [total_nodes, node_features]
#     edges,      # Edge features [total_edges, edge_features]
#     senders,    # Source node indices [total_edges]
#     receivers,  # Destination node indices [total_edges]
#     n_node,     # Number of nodes per graph [num_graphs]
#     n_edge,     # Number of edges per graph [num_graphs]
#     globals     # Graph-level features [num_graphs, global_features]
# )
# ```
#
# ### Why Senders/Receivers?
#
# Jraph uses "senders" and "receivers" terminology because it thinks of graphs from a **message passing** perspective:
# - **Senders**: Nodes that send messages (source nodes)
# - **Receivers**: Nodes that receive messages (destination nodes)
#
# This naming makes the message passing paradigm explicit in the data structure.

# + id="jraph-implementation"
#@title Create Jraph GraphsTuple from molecular graph

def create_jraph_graph(smiles: str):
    """
    Create a Jraph GraphsTuple from a SMILES string.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        jraph.GraphsTuple: Jraph graph object
    """
    if not JRAPH_AVAILABLE:
        raise ImportError("Jraph not available")

    # Get graph data from smiles2graph
    node_features, edge_indices, edge_features, atom_symbols = smiles2graph(smiles)

    # Extract senders (source) and receivers (destination)
    if len(edge_indices) > 0:
        senders = jnp.array([e[0] for e in edge_indices])
        receivers = jnp.array([e[1] for e in edge_indices])
    else:
        senders = jnp.array([], dtype=jnp.int32)
        receivers = jnp.array([], dtype=jnp.int32)

    # Create GraphsTuple
    graph = jraph.GraphsTuple(
        nodes=jnp.array(node_features),
        edges=jnp.array(edge_features) if len(edge_features) > 0 else None,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([len(node_features)]),  # Array for batching
        n_edge=jnp.array([len(edge_indices)]),   # Array for batching
        globals=None  # No graph-level features
    )

    return graph


# + id="jraph-demo"
#@title Demonstrate Jraph GraphsTuple
if JRAPH_AVAILABLE:
    try:
        # Create Jraph graph for Ethanol
        jraph_graph = create_jraph_graph("CCO")

        print("Jraph GraphsTuple for Ethanol")
        print("="*50)

        print(f"\n1. The GraphsTuple namedtuple:")
        print(f"   Type: {type(jraph_graph)}")
        print(f"   Fields: {jraph_graph._fields}")

        print(f"\n2. Node features (nodes):")
        print(f"   Shape: {jraph_graph.nodes.shape}")
        print(f"   Type: {type(jraph_graph.nodes)}")
        print(f"   Access: jraph_graph.nodes")

        print(f"\n3. Edge connectivity (senders/receivers):")
        print(f"   Senders shape: {jraph_graph.senders.shape}")
        print(f"   Receivers shape: {jraph_graph.receivers.shape}")
        print(f"   First 4 edges:")
        print(f"   Senders (source):    {jraph_graph.senders[:4].tolist()}")
        print(f"   Receivers (dest):    {jraph_graph.receivers[:4].tolist()}")

        print(f"\n4. Edge features (edges):")
        if jraph_graph.edges is not None:
            print(f"   Shape: {jraph_graph.edges.shape}")
        else:
            print(f"   None (no edge features)")

        print(f"\n5. Batch information:")
        print(f"   n_node: {jraph_graph.n_node}  (nodes per graph)")
        print(f"   n_edge: {jraph_graph.n_edge}  (edges per graph)")

        print(f"\n6. Graph-level features:")
        print(f"   globals: {jraph_graph.globals}")

        print(f"\n7. Key differences from PyG/DGL:")
        print(f"   - Uses JAX arrays (jnp.array), not PyTorch tensors")
        print(f"   - Named 'senders/receivers' instead of 'src/dst' or 'edge_index'")
        print(f"   - n_node/n_edge arrays enable efficient batching")
        print(f"   - Immutable structure (functional paradigm)")
    except Exception as e:
        print(f"Jraph demonstration failed: {e}")
        print("This can happen due to JAX/GPU issues. See Section 4 notes.")
else:
    print("Jraph not available")

# + id="jraph-visualization"
#@title Visualize Jraph structure and message passing perspective
if JRAPH_AVAILABLE:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a visual representation of GraphsTuple
    components = [
        ('nodes', 'Node features\n[9, 21]', '#3498db'),
        ('edges', 'Edge features\n[16, 6]', '#e74c3c'),
        ('senders', 'Source indices\n[16]', '#2ecc71'),
        ('receivers', 'Dest indices\n[16]', '#9b59b6'),
        ('n_node', 'Nodes per graph\n[1]', '#f39c12'),
        ('n_edge', 'Edges per graph\n[1]', '#f39c12'),
        ('globals', 'Graph features\nNone', '#95a5a6'),
    ]

    y_pos = np.arange(len(components))
    colors = [c[2] for c in components]

    bars = ax.barh(y_pos, [1]*len(components), color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{c[0]}\n{c[1]}" for c in components])
    ax.set_xlim(0, 2)
    ax.set_title("Jraph GraphsTuple Structure\n(Named tuple with 7 fields)")
    ax.set_xticks([])

    # Add message passing annotation
    ax.annotate('Message Passing\nPerspective',
                xy=(1.5, 2.5), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.annotate('', xy=(1.3, 2), xytext=(1.3, 3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.show()

    print("\nKey Insight: Jraph names edges as senders→receivers")
    print("This reflects the message passing paradigm explicitly!")

# + [markdown] id="comparison-section"
# ## 8. Side-by-Side Comparison <a name="comparison"></a>
#
# Now let's compare all three frameworks with the same molecule to see the differences clearly.

# + id="comparison-function"
#@title Compare all frameworks

def compare_frameworks(smiles: str, molecule_name: str = "Molecule"):
    """
    Create and compare graph representations across all available frameworks.

    Args:
        smiles: SMILES string
        molecule_name: Name for display

    Returns:
        dict: Dictionary with framework representations
    """
    results = {'smiles': smiles, 'name': molecule_name}

    # Get base graph data
    node_features, edge_indices, edge_features, atom_symbols = smiles2graph(smiles)
    results['num_nodes'] = len(node_features)
    results['num_edges'] = len(edge_indices)

    # PyTorch Geometric
    if TORCH_GEOMETRIC_AVAILABLE:
        pyg_data = create_pyg_data(smiles)
        results['pyg'] = pyg_data

    # DGL
    if DGL_AVAILABLE:
        dgl_graph = create_dgl_graph(smiles)
        results['dgl'] = dgl_graph

    # Jraph
    if JRAPH_AVAILABLE:
        try:
            jraph_graph = create_jraph_graph(smiles)
            results['jraph'] = jraph_graph
        except Exception as e:
            print(f"Note: Jraph creation failed ({e}). Skipping Jraph comparison.")

    return results


# + id="comparison-demo"
#@title Side-by-side comparison for Ethanol
comparison = compare_frameworks("CCO", "Ethanol")

print("Side-by-Side Framework Comparison: Ethanol")
print("="*60)

# Create comparison table
data = {
    'Aspect': [
        'Node Features Access',
        'Edge Index Access',
        'Edge Features Access',
        'Num Nodes',
        'Num Edges',
        'Data Type',
        'Paradigm'
    ]
}

if TORCH_GEOMETRIC_AVAILABLE:
    data['PyTorch Geometric'] = [
        'data.x',
        'data.edge_index',
        'data.edge_attr',
        str(comparison['pyg'].num_nodes),
        str(comparison['pyg'].num_edges),
        'torch.Tensor',
        'Object-oriented'
    ]

if DGL_AVAILABLE:
    data['DGL'] = [
        "g.ndata['h']",
        'g.edges()',
        "g.edata['e']",
        str(comparison['dgl'].num_nodes()),
        str(comparison['dgl'].num_edges()),
        'torch.Tensor',
        'Dictionary-based'
    ]

if JRAPH_AVAILABLE and 'jraph' in comparison:
    data['Jraph'] = [
        'graph.nodes',
        'senders, receivers',
        'graph.edges',
        str(int(comparison['jraph'].n_node[0])),
        str(int(comparison['jraph'].n_edge[0])),
        'jax.Array',
        'Functional'
    ]

df = pd.DataFrame(data)
df.set_index('Aspect', inplace=True)
print(df.to_string())

# + id="comparison-visualization"
#@title Visual comparison of edge representation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

node_features, edge_indices, _, atom_symbols = smiles2graph("CCO")
first_edges = edge_indices[:6]  # First 6 edges

# PyG style
ax1 = axes[0]
if TORCH_GEOMETRIC_AVAILABLE:
    pyg_edge_index = comparison['pyg'].edge_index[:, :6].numpy()
    ax1.matshow(pyg_edge_index, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(6):
            ax1.text(j, i, str(int(pyg_edge_index[i, j])), ha='center', va='center', fontsize=12)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Source', 'Dest'])
    ax1.set_xticks(range(6))
    ax1.set_xticklabels([f'E{i}' for i in range(6)])
ax1.set_title('PyG: edge_index\n[2, num_edges]')

# DGL style
ax2 = axes[1]
if DGL_AVAILABLE:
    src, dst = comparison['dgl'].edges()
    dgl_display = np.vstack([src[:6].numpy(), dst[:6].numpy()])
    ax2.matshow(dgl_display, cmap='Greens', aspect='auto')
    for i in range(2):
        for j in range(6):
            ax2.text(j, i, str(int(dgl_display[i, j])), ha='center', va='center', fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Source', 'Dest'])
    ax2.set_xticks(range(6))
    ax2.set_xticklabels([f'E{i}' for i in range(6)])
ax2.set_title('DGL: g.edges()\nReturns (src, dst) tuple')

# Jraph style
ax3 = axes[2]
if JRAPH_AVAILABLE and 'jraph' in comparison:
    jraph_display = np.vstack([
        comparison['jraph'].senders[:6],
        comparison['jraph'].receivers[:6]
    ])
    ax3.matshow(jraph_display, cmap='Purples', aspect='auto')
    for i in range(2):
        for j in range(6):
            ax3.text(j, i, str(int(jraph_display[i, j])), ha='center', va='center', fontsize=12)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Senders', 'Receivers'])
    ax3.set_xticks(range(6))
    ax3.set_xticklabels([f'E{i}' for i in range(6)])
    ax3.set_title('Jraph: senders/receivers\nSeparate arrays')
else:
    ax3.text(0.5, 0.5, 'Jraph not available\n(JAX/GPU issue)', ha='center', va='center',
             transform=ax3.transAxes, fontsize=12)
    ax3.set_title('Jraph: senders/receivers\n(unavailable)')

plt.tight_layout()
plt.show()

print("\nSame edges, three different representations!")
print("All represent the same molecular graph structure.")

# + [markdown] id="api-differences-section"
# ## 9. API Differences and Design Philosophies <a name="api-differences"></a>
#
# Each framework has distinct design choices that affect how you write code.

# + id="api-comparison-table"
#@title Detailed API comparison

api_comparison = {
    'Operation': [
        'Create empty graph',
        'Add node features',
        'Add edge features',
        'Get number of nodes',
        'Get number of edges',
        'Add self-loops',
        'Batch multiple graphs',
        'Convert to NetworkX',
        'GPU transfer'
    ],
    'PyTorch Geometric': [
        'Data()',
        'data.x = features',
        'data.edge_attr = features',
        'data.num_nodes',
        'data.num_edges',
        'add_self_loops(edge_index)',
        'Batch.from_data_list([...])',
        'to_networkx(data)',
        'data.to(device)'
    ],
    'DGL': [
        'dgl.graph(([], []))',
        "g.ndata['h'] = features",
        "g.edata['e'] = features",
        'g.num_nodes()',
        'g.num_edges()',
        'dgl.add_self_loop(g)',
        'dgl.batch([...])',
        'g.to_networkx()',
        'g.to(device)'
    ],
    'Jraph': [
        'GraphsTuple(...)',
        'graph._replace(nodes=...)',
        'graph._replace(edges=...)',
        'graph.n_node.sum()',
        'graph.n_edge.sum()',
        'Manual implementation',
        'jraph.batch([...])',
        'Manual conversion',
        'jax.device_put(graph)'
    ]
}

api_df = pd.DataFrame(api_comparison)
api_df.set_index('Operation', inplace=True)

print("API Comparison: Common Operations")
print("="*80)
print(api_df.to_string())

# + id="design-philosophy"
#@title Design philosophy comparison
print("\nDesign Philosophy Comparison")
print("="*60)

print("""
PyTorch Geometric
-----------------
Philosophy: "Research-first, PyTorch-native"
Strengths:
  - Largest collection of implemented GNN architectures
  - MoleculeNet and other datasets built-in
  - Most academic papers use PyG
  - Simple, Pythonic API
Weaknesses:
  - PyTorch only
  - Some scalability challenges with very large graphs

DGL
---
Philosophy: "Flexible, production-ready, scalable"
Strengths:
  - Multi-backend (PyTorch, TensorFlow, MXNet)
  - Excellent for heterogeneous graphs
  - Good for very large graphs
  - Strong industry adoption
Weaknesses:
  - Dictionary-based API can be verbose
  - Smaller research community than PyG

Jraph
-----
Philosophy: "Functional, high-performance, JAX-native"
Strengths:
  - Native TPU support
  - JIT compilation for speed
  - Elegant functional programming model
  - Great for research requiring custom gradients
Weaknesses:
  - Smallest ecosystem
  - JAX learning curve
  - Fewer pre-built architectures
  - Debugging can be challenging
""")

# + [markdown] id="interoperability-section"
# ## 10. Framework Interoperability <a name="interoperability"></a>
#
# Sometimes you need to convert graphs between frameworks. Here are utility functions for common conversions.

# + id="conversion-functions"
#@title Framework conversion functions

def pyg_to_dgl(pyg_data):
    """Convert PyTorch Geometric Data to DGL Graph."""
    if not (TORCH_GEOMETRIC_AVAILABLE and DGL_AVAILABLE):
        raise ImportError("Both PyG and DGL required")

    edge_index = pyg_data.edge_index
    src, dst = edge_index[0], edge_index[1]

    g = dgl.graph((src, dst), num_nodes=pyg_data.num_nodes)

    if pyg_data.x is not None:
        g.ndata['h'] = pyg_data.x
    if pyg_data.edge_attr is not None:
        g.edata['e'] = pyg_data.edge_attr

    return g


def dgl_to_pyg(dgl_graph):
    """Convert DGL Graph to PyTorch Geometric Data."""
    if not (TORCH_GEOMETRIC_AVAILABLE and DGL_AVAILABLE):
        raise ImportError("Both PyG and DGL required")

    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst])

    x = dgl_graph.ndata.get('h')
    edge_attr = dgl_graph.edata.get('e')

    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)


def pyg_to_jraph(pyg_data):
    """Convert PyTorch Geometric Data to Jraph GraphsTuple."""
    if not (TORCH_GEOMETRIC_AVAILABLE and JRAPH_AVAILABLE):
        raise ImportError("Both PyG and Jraph required")

    nodes = jnp.array(pyg_data.x.numpy()) if pyg_data.x is not None else None
    edges = jnp.array(pyg_data.edge_attr.numpy()) if pyg_data.edge_attr is not None else None
    senders = jnp.array(pyg_data.edge_index[0].numpy())
    receivers = jnp.array(pyg_data.edge_index[1].numpy())

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([pyg_data.num_nodes]),
        n_edge=jnp.array([pyg_data.num_edges]),
        globals=None
    )


def jraph_to_pyg(jraph_graph):
    """Convert Jraph GraphsTuple to PyTorch Geometric Data."""
    if not (TORCH_GEOMETRIC_AVAILABLE and JRAPH_AVAILABLE):
        raise ImportError("Both PyG and Jraph required")

    x = torch.tensor(np.array(jraph_graph.nodes)) if jraph_graph.nodes is not None else None
    edge_attr = torch.tensor(np.array(jraph_graph.edges)) if jraph_graph.edges is not None else None

    senders = np.array(jraph_graph.senders)
    receivers = np.array(jraph_graph.receivers)
    edge_index = torch.tensor(np.stack([senders, receivers]), dtype=torch.long)

    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)


# + id="conversion-demo"
#@title Test conversions
if TORCH_GEOMETRIC_AVAILABLE and DGL_AVAILABLE:
    print("Testing PyG <-> DGL conversion")
    print("="*40)

    # Create PyG data
    pyg_original = create_pyg_data("CCO")
    print(f"Original PyG: {pyg_original.num_nodes} nodes, {pyg_original.num_edges} edges")

    # Convert to DGL
    dgl_converted = pyg_to_dgl(pyg_original)
    print(f"Converted DGL: {dgl_converted.num_nodes()} nodes, {dgl_converted.num_edges()} edges")

    # Convert back to PyG
    pyg_roundtrip = dgl_to_pyg(dgl_converted)
    print(f"Roundtrip PyG: {pyg_roundtrip.num_nodes} nodes, {pyg_roundtrip.num_edges} edges")

    # Verify
    if torch.allclose(pyg_original.x, pyg_roundtrip.x):
        print("\nNode features preserved!")
    if torch.allclose(pyg_original.edge_attr, pyg_roundtrip.edge_attr):
        print("Edge features preserved!")

if TORCH_GEOMETRIC_AVAILABLE and JRAPH_AVAILABLE:
    try:
        print("\n" + "="*40)
        print("Testing PyG <-> Jraph conversion")
        print("="*40)

        # Create PyG data
        pyg_original = create_pyg_data("CCO")
        print(f"Original PyG: {pyg_original.num_nodes} nodes, {pyg_original.num_edges} edges")

        # Convert to Jraph
        jraph_converted = pyg_to_jraph(pyg_original)
        print(f"Converted Jraph: {int(jraph_converted.n_node[0])} nodes, {int(jraph_converted.n_edge[0])} edges")

        # Convert back to PyG
        pyg_roundtrip = jraph_to_pyg(jraph_converted)
        print(f"Roundtrip PyG: {pyg_roundtrip.num_nodes} nodes, {pyg_roundtrip.num_edges} edges")

        print("\nConversions successful!")
    except Exception as e:
        print(f"\nPyG <-> Jraph conversion failed: {e}")
        print("This is likely due to JAX initialization issues on GPU.")

# + [markdown] id="practical-example-section"
# ## 11. Practical Example: Multiple Molecules <a name="practical-example"></a>
#
# Let's apply our framework comparison to molecules of increasing complexity.

# + id="multiple-molecules"
#@title Compare frameworks across multiple molecules
molecules = [
    ("CCO", "Ethanol"),
    ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
]

print("Framework Comparison: Multiple Molecules")
print("="*70)

results = []
for smiles, name in molecules:
    comparison = compare_frameworks(smiles, name)
    results.append({
        'Molecule': name,
        'SMILES': smiles,
        'Nodes': comparison['num_nodes'],
        'Edges': comparison['num_edges'],
        'Node Features': 21,
        'Edge Features': 6
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# + id="molecule-complexity-viz"
#@title Visualize molecule complexity
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (smiles, name) in enumerate(molecules):
    ax = axes[idx]
    mol = Chem.MolFromSmiles(smiles)
    mol_with_h = Chem.AddHs(mol)

    # Get graph stats
    node_features, edge_indices, _, _ = smiles2graph(smiles)

    # Draw molecule
    img = Draw.MolToImage(mol, size=(250, 200))
    ax.imshow(img)
    ax.set_title(f"{name}\n{len(node_features)} atoms, {len(edge_indices)} edges")
    ax.axis('off')

plt.suptitle("Molecules of Increasing Complexity", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# + id="complexity-comparison"
#@title Graph size comparison
molecules_extended = [
    ("C", "Methane"),
    ("CCO", "Ethanol"),
    ("c1ccccc1", "Benzene"),
    ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
]

stats = []
for smiles, name in molecules_extended:
    node_features, edge_indices, _, _ = smiles2graph(smiles)
    stats.append({
        'Molecule': name,
        'Nodes': len(node_features),
        'Edges': len(edge_indices),
        'Node Features Total': len(node_features) * 21,
        'Edge Features Total': len(edge_indices) * 6
    })

stats_df = pd.DataFrame(stats)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(stats_df))
width = 0.35

bars1 = ax.bar(x - width/2, stats_df['Nodes'], width, label='Nodes', color='#3498db')
bars2 = ax.bar(x + width/2, stats_df['Edges'], width, label='Edges', color='#e74c3c')

ax.set_xlabel('Molecule')
ax.set_ylabel('Count')
ax.set_title('Graph Size Comparison')
ax.set_xticks(x)
ax.set_xticklabels(stats_df['Molecule'], rotation=45, ha='right')
ax.legend()

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# + [markdown] id="framework-selection-section"
# ## 12. When to Choose Which Framework <a name="framework-selection"></a>
#
# Use this decision guide to choose the right framework for your project.

# + id="decision-matrix"
#@title Framework selection guide
print("Framework Selection Guide")
print("="*70)

decision_matrix = {
    'Use Case': [
        'Academic research / paper reproduction',
        'Molecular property prediction (standard benchmarks)',
        'Production deployment',
        'Very large graphs (millions of nodes)',
        'Heterogeneous graphs (multiple node/edge types)',
        'High-performance computing / TPU',
        'Quick prototyping',
        'Custom gradient computation',
        'Multi-backend requirement (TF + PyTorch)',
    ],
    'Recommended': [
        'PyTorch Geometric',
        'PyTorch Geometric',
        'DGL',
        'DGL',
        'DGL',
        'Jraph',
        'PyTorch Geometric',
        'Jraph',
        'DGL',
    ],
    'Reason': [
        'Most papers use PyG, easy to find implementations',
        'MoleculeNet datasets built-in',
        'Better scalability, industry adoption',
        'Designed for billion-edge graphs',
        'Native heterograph support',
        'JAX TPU support, JIT compilation',
        'Simpler API, more examples',
        'JAX automatic differentiation flexibility',
        'Only multi-backend option',
    ]
}

decision_df = pd.DataFrame(decision_matrix)
print(decision_df.to_string(index=False))

# + id="summary-visualization"
#@title Visual summary
fig, ax = plt.subplots(figsize=(10, 6))

frameworks = ['PyTorch Geometric', 'DGL', 'Jraph']
categories = ['Ease of Use', 'Research Support', 'Production Ready', 'Performance', 'Ecosystem']

# Scores (subjective, based on community consensus)
scores = {
    'PyTorch Geometric': [5, 5, 3, 4, 5],
    'DGL': [4, 4, 5, 4, 4],
    'Jraph': [3, 3, 3, 5, 2],
}

x = np.arange(len(categories))
width = 0.25

for i, (framework, score) in enumerate(scores.items()):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, score, width, label=framework, alpha=0.8)

ax.set_ylabel('Score (1-5)')
ax.set_title('Framework Comparison (Subjective Assessment)')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 6)
ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Average')

plt.tight_layout()
plt.show()

print("\nNote: These scores are subjective and based on general community consensus.")
print("Your specific use case may have different priorities!")

# + [markdown] id="exercises-section"
# ## 13. Checkpoint Exercises <a name="exercises"></a>
#
# Test your understanding with these exercises.

# + [markdown] id="exercise-1"
# ### Exercise 1: Create Graph Representations
#
# Create PyG, DGL, and Jraph representations for **Ibuprofen** (SMILES: `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O`).
#
# Verify that all three frameworks report the same number of nodes and edges.

# + id="exercise-1-solution"
# Your solution here
ibuprofen_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

# Uncomment and complete:
# pyg_ibuprofen = create_pyg_data(ibuprofen_smiles)
# dgl_ibuprofen = create_dgl_graph(ibuprofen_smiles)
# jraph_ibuprofen = create_jraph_graph(ibuprofen_smiles)

# Print results:
# print(f"PyG nodes: {pyg_ibuprofen.num_nodes}")
# ...

# + [markdown] id="exercise-2"
# ### Exercise 2: Framework Conversion
#
# 1. Create a PyG Data object for Benzene (`c1ccccc1`)
# 2. Convert it to DGL format
# 3. Add a new node feature called 'position' with random 3D coordinates
# 4. Convert back to PyG format
#
# Hint: In DGL, you can add new features with `g.ndata['position'] = ...`

# + id="exercise-2-solution"
# Your solution here
benzene_smiles = "c1ccccc1"

# Step 1: Create PyG data
# pyg_benzene = create_pyg_data(benzene_smiles)

# Step 2: Convert to DGL
# dgl_benzene = pyg_to_dgl(pyg_benzene)

# Step 3: Add position feature
# positions = torch.randn(dgl_benzene.num_nodes(), 3)
# dgl_benzene.ndata['position'] = positions

# Step 4: Convert back (note: custom conversion needed to preserve position)

# + [markdown] id="exercise-3"
# ### Exercise 3: Batch Processing
#
# Create a batch of 3 molecules (Methane, Ethanol, Propane) in PyG format.
# Use `Batch.from_data_list()` to combine them.
#
# Print:
# - Total number of nodes in the batch
# - The `batch` attribute (which tells you which graph each node belongs to)

# + id="exercise-3-solution"
# Your solution here
smiles_list = ["C", "CCO", "CCC"]  # Methane, Ethanol, Propane

# Create individual graphs
# graphs = [create_pyg_data(s) for s in smiles_list]

# Batch them
# from torch_geometric.data import Batch
# batch = Batch.from_data_list(graphs)

# Print results
# print(f"Total nodes: {batch.num_nodes}")
# print(f"Batch attribute: {batch.batch}")

# + [markdown] id="conclusion-section"
# ## 14. Conclusion and References <a name="conclusion"></a>
#
# ### Summary
#
# In this tutorial, we explored three major GNN frameworks:
#
# | Framework | Data Structure | Key Characteristic |
# |-----------|---------------|-------------------|
# | **PyTorch Geometric** | `Data` object | Research-focused, PyTorch-native |
# | **DGL** | `DGLGraph` | Multi-backend, dictionary-based features |
# | **Jraph** | `GraphsTuple` | JAX-based, functional programming |
#
# ### Key Takeaways
#
# 1. **All frameworks can represent the same graphs** - they just use different data structures
# 2. **PyG is best for research** - most paper implementations, built-in datasets
# 3. **DGL is best for production** - scalability, multi-backend support
# 4. **Jraph is best for high-performance** - TPU support, JIT compilation
# 5. **Conversion between frameworks is straightforward** - through numpy arrays
#
# ### What's Next?
#
# Now that you understand how to represent molecular graphs in different frameworks, you're ready to:
# - **Tutorial 02**: Implement message passing and GNN layers
# - **Tutorials 03-10**: Explore specific GNN architectures (GCN, GAT, GIN, SchNet, etc.)
#
# ### References
#
# 1. Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.
#    - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
#
# 2. Wang, M., et al. (2019). Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks. *arXiv preprint arXiv:1909.01315*.
#    - DGL: https://www.dgl.ai/
#
# 3. Godwin, J., et al. (2020). Jraph: A library for graph neural networks in jax. *GitHub repository*.
#    - Jraph: https://github.com/google-deepmind/jraph
#
# 4. Bradbury, J., et al. (2018). JAX: composable transformations of Python+NumPy programs.
#    - JAX: https://github.com/google/jax
#
# 5. Sanchez-Lengeling, B., et al. (2021). A Gentle Introduction to Graph Neural Networks. *Distill*.
#    - Interactive tutorial: https://distill.pub/2021/gnn-intro/
