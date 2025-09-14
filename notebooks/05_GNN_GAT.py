# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: pytorch
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/HFooladi/GNNs-For-Chemists/blob/main/notebooks/05_GNN_GAT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="Pi8JlOwjeQjw"
# # Graph Attention Networks (GAT) Tutorial for Chemists and Pharmacists
#
# ## Table of Contents
# 1. [Introduction](#introduction)
# 2. [Setup and Installation](#setup-and-installation)
# 3. [Understanding Molecular Graphs](#understand-graph)
# 4. [Graph Neural Networks: Basic Concepts](#gnn-basics)
# 5. [The Attention Mechanism: Why It Matters](#attention-mechanism)
# 6. [Implementing a Graph Attention Network (GAT)](#implement-gat)
# 7. [Understanding Multi-Head Attention](#understand-multi-head)
# 8. [Implementing a Complete GAT Model for Molecular Property Prediction](#implement-gat-property-prediction)
# 9. [Visualizing Attention Weights in Molecules](#visualize-attention)
# 10. [Comparing Single-Head vs Multi-Head Attention Performance](#compare-single-multi-head)
# 11. [Visualizing Feature Transformation Through the Network](#visualize-feature-transformation)
# 12. [Interactive Visualization of Attention Mechanism](#interactive-visualiztion)
# 13. [Conclusion and Further Research Directions](#conclusion)

# + [markdown] id="B7yeqnMLQyVX"
# ## 1. Introduction to Graph Neural Networks and Attention Mechanisms  <a name="introduction"></a>
#
# In this notebook, we'll explore Graph Attention Networks (GAT), a powerful graph neural network architecture particularly useful for molecular data. This tutorial is specifically designed for chemists and pharmacists who want to understand how these models work for molecular property prediction and drug discovery.

# + [markdown] id="dFe5wKCJQ5Co"
# ## 2. Setup and Installation <a name="setup-and-installation"></a>
#
# First, let's install the necessary packages:

# + cellView="form" colab={"base_uri": "https://localhost:8080/"} id="8UJWZUJWeP6f" outputId="dab67029-23c0-4f8d-defc-353d67305370"
#@title Intstall necessary libraries
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
# !pip install -q rdkit
# !pip install -q networkx

# + cellView="form" colab={"base_uri": "https://localhost:8080/"} id="1Ej6ZhFBdbPz" outputId="62350c69-7318-4457-81ea-7041ceb26561"
#@title Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, MessagePassing
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import to_networkx, softmax

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import io
from PIL import Image
import random
from IPython.display import HTML

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# + [markdown] id="eSaeOEFIe6GJ"
# ## 3. Understanding Molecular Graphs <a name="understand-graph"></a>
#
# Molecules are naturally represented as graphs where atoms are nodes and bonds are edges. Let's visualize a simple molecule as a graph:

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="887XhJ1Be6jl" outputId="51376bd7-9303-4677-8439-496c434f9cdb"
def atom_features(atom):
    """
    Extract a feature vector for an RDKit atom.

    Features included:
        - Atomic number
        - Chirality tag (encoded as integer)
        - Degree (number of directly-bonded atoms)
        - Formal charge
        - Total number of hydrogens
        - Number of radical electrons
        - Hybridization (encoded as integer)
        - Aromaticity (0 or 1)
        - Ring membership (0 or 1)

    Args:
        atom (rdkit.Chem.rdchem.Atom): An RDKit Atom object.

    Returns:
        torch.Tensor: Feature tensor of shape (9,) with dtype long.
    """
    return torch.tensor([
        atom.GetAtomicNum(),                    # Atomic number
        int(atom.GetChiralTag()),               # Chirality
        atom.GetDegree(),                       # Degree
        atom.GetFormalCharge(),                 # Formal charge
        atom.GetTotalNumHs(),                   # Number of hydrogens
        atom.GetNumRadicalElectrons(),          # Radical electrons
        int(atom.GetHybridization()),           # Hybridization
        int(atom.GetIsAromatic()),              # Aromaticity
        int(atom.IsInRing())                    # Ring membership
    ], dtype=torch.long)


def bond_features(bond):
    """
    Extract a feature vector for an RDKit bond.

    Features included:
        - Bond type as double (e.g., 1.0 for single, 2.0 for double)
        - Conjugation (0 or 1)
        - Ring membership (0 or 1)

    Args:
        bond (rdkit.Chem.rdchem.Bond): An RDKit Bond object.

    Returns:
        torch.Tensor: Feature tensor of shape (3,) with dtype long.
    """
    return torch.tensor([
        int(bond.GetBondTypeAsDouble()),        # Bond type
        int(bond.GetIsConjugated()),            # Conjugation
        int(bond.IsInRing())                    # Ring membership
    ], dtype=torch.long)

def mol_to_graph(smiles):
    """
    Converts a SMILES into a PyTorch Geometric graph data object.

    Nodes represent atoms with features, and edges represent bonds with features.
    The graph is undirected: each bond adds two directed edges (i->j and j->i).

    Args:
        smiles (str): SMILES representing the molecule.

    Returns:
        torch_geometric.data.Data: Graph data object containing:
            - x: Node feature matrix [num_nodes, 9]
            - edge_index: Edge list [2, num_edges]
            - edge_attr: Edge feature matrix [num_edges, 3]
            - smiles: Original SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Node features
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()], dim=0)

    # Edge index and edge features
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Add both directions for undirected graph
        edge_index.append((i, j))
        edge_index.append((j, i))

        edge_attr.append(bond_features(bond))
        edge_attr.append(bond_features(bond))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr, dim=0) if edge_attr else None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    return data


def visualize_molecule(smiles, title="Molecule"):
    """Visualize a molecule using RDKit"""
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    # Draw molecule
    fig, ax = plt.subplots(figsize=(5, 5))
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = drawer.GetDrawingText()

    # Convert the image data to a PIL Image
    pil_image = Image.open(io.BytesIO(img))

    # Display the image
    plt.imshow(pil_image)
    plt.axis('off')
    plt.title(title)
    plt.show()

def visualize_molecular_graph(smiles, title="Molecular Graph"):
    """
    Visualizes the 2D structure of a molecule using RDKit and networkx and displays it.

    Args:
        smiles (str): SMILES representing the molecule.
        title (str): Plot title.
    """
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    data = mol_to_graph(smiles)
    G = to_networkx(data, to_undirected=True)

    # Get the 2D coordinates from RDKit
    pos = {}
    for i, atom in enumerate(mol.GetAtoms()):
        pos[i] = mol.GetConformer().GetAtomPosition(i)
        pos[i] = (pos[i].x, -pos[i].y)  # Flip y for better visualization

    plt.figure(figsize=(6, 6))

    # Get atom labels
    atom_labels = {i: atom.GetSymbol() for i, atom in enumerate(mol.GetAtoms())}

    # Get atom features for node coloring
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # Draw the graph
    nx.draw(G, pos,
            labels=atom_labels,
            with_labels=True,
            node_color=atom_features,
            cmap=plt.cm.viridis,
            node_size=500,
            font_size=10,
            font_color='white',
            edge_color='gray')

    plt.title(title)
    plt.axis('off')
    plt.show()

# Example: Aspirin
aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
visualize_molecule(aspirin_smiles, "Aspirin")
visualize_molecular_graph(aspirin_smiles, "Aspirin as a Graph")

# Example: Paracetamol (Acetaminophen)
paracetamol_smiles = "CC(=O)NC1=CC=C(C=C1)O"
visualize_molecule(paracetamol_smiles, "Paracetamol")
visualize_molecular_graph(paracetamol_smiles, "Paracetamol as a Graph")


# + [markdown] id="tEDQA8RlfE8u"
# ## 4. Graph Neural Networks: Basic Concepts <a name="gnn-basics"></a>
#
# Before diving into GATs, let's understand the basic concept of message passing in graph neural networks:

# + colab={"base_uri": "https://localhost:8080/", "height": 720} id="YOvVEVhifGDb" outputId="4213af31-734b-4857-cc54-85878698af33"
class BasicMessagePassing(MessagePassing):
    """
    A simple custom message passing layer using PyG's MessagePassing framework.

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.

    Notes: 
        - This is a simplified example. Receiving messages from all neighbors.
        - The aggregation function is just the sum of the messages.
    """
    def __init__(self, in_channels, out_channels):
        super(BasicMessagePassing, self).__init__(aggr='add')  # "add" aggregation
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass of the message passing layer.

        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels].
            edge_index (Tensor): Edge index [2, num_edges].

        Returns:
            Tensor: Updated node features [num_nodes, out_channels].
        """
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Transform node features
        x = self.lin(x.float())

        # Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        """
        Constructs messages from source nodes.

        Args:
            x_j (Tensor): Features of source nodes [num_edges, out_channels].

        Returns:
            Tensor: Messages to be aggregated.
        """
        # x_j has shape [E, out_channels]
        # Simple message function that just returns node features
        return x_j

    def update(self, aggr_out):
        """
        Updates node embeddings after aggregation.

        Args:
            aggr_out (Tensor): Aggregated messages for each node [num_nodes, out_channels].

        Returns:
            Tensor: Updated node features.
        """
        # aggr_out has shape [N, out_channels]
        # No update function, just return the aggregated messages
        return aggr_out

def simulate_basic_message_passing(data, input_dim=9, output_dim=9):
    """
    Apply the basic message passing layer to a molecular graph.

    Args:
        data (Data): PyG Data object for a molecule.
        input_dim (int): Input feature dimension.
        output_dim (int): Output feature dimension.

    Returns:
        Tuple[Tensor, Tensor]: Original and updated node feature matrices.
    """
    # Initialize a simple message passing layer
    mp_layer = BasicMessagePassing(input_dim, output_dim)

    # Original node features
    original_features = data.x

    # Apply message passing
    updated_features = mp_layer(data.x, data.edge_index)

    return original_features, updated_features

# Create a visualization of message passing on aspirin
aspirin_data = mol_to_graph(aspirin_smiles)
orig_feat, updated_feat = simulate_basic_message_passing(aspirin_data)

print("Original node features (first 3 nodes):")
print(orig_feat[:3])
print("\nUpdated node features after message passing (first 3 nodes):")
print(updated_feat[:3])

# Visualize the difference using a heatmap
def plot_feature_comparison(original, updated, title="Feature Comparison"):
    """Plot a comparison of original and updated features"""
    # Compute the difference
    diff = (updated - original).abs().mean(dim=1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot original features
    im1 = ax1.imshow(original.detach().numpy(), cmap='viridis')
    ax1.set_title("Original Features")
    ax1.set_xlabel("Feature Dimension")
    ax1.set_ylabel("Node ID")
    plt.colorbar(im1, ax=ax1)

    # Plot updated features
    im2 = ax2.imshow(updated.detach().numpy(), cmap='viridis')
    ax2.set_title("Updated Features After Message Passing")
    ax2.set_xlabel("Feature Dimension")
    ax2.set_ylabel("Node ID")
    plt.colorbar(im2, ax=ax2)

    # Plot the difference in a separate visualization
    ax3.bar(range(len(diff)), diff.detach().numpy())
    ax3.set_title("Average Absolute Difference Per Node")
    ax3.set_xlabel("Node ID")
    ax3.set_ylabel("Average Absolute Difference")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_feature_comparison(orig_feat, updated_feat, "Basic Message Passing on Aspirin")


# + [markdown] id="j0EsbspNfOXF"
# ## 5. The Attention Mechanism: Why It Matters <a name="attention-mechanism"></a>
#
# Let's compare traditional averaging-based message passing with attention-based message passing:

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1gVD_TCsfLLr" outputId="e741e26d-01bf-41c0-9ce5-5d7451e1f95e"
class SimpleAveragingLayer(MessagePassing):
    """
    A layer that simply averages neighbor features

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.

    Notes: 
        - This is a simplified example. Receiving messages from all neighbors.
        - The aggregation function is just the average of the messages.
    """
    def __init__(self, in_channels, out_channels):
        super(SimpleAveragingLayer, self).__init__(aggr='mean')  # "mean" aggregation
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Transform node features
        x = self.lin(x.float())

        # Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Simple message function
        return x_j

    def update(self, aggr_out):
        # No update function
        return aggr_out

class SimpleAttentionLayer(MessagePassing):
    """
    A simplified attention layer with single head for demonstration

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.

    Notes: 
        - This is a simplified example. Receiving messages from all neighbors.
        - The aggregation function is just the sum of the messages with a learned attention mechanism.
        - The attention mechanism is a simple linear layer that takes the concatenation of the target and source node features and outputs a single attention coefficient.
        - The attention coefficient is then normalized to sum to 1.
        - We consider one head for simplicity.
    """
    def __init__(self, in_channels, out_channels):
        super(SimpleAttentionLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.att = nn.Linear(2 * out_channels, 1)
        self.alpha = None  # Store attention weights

    def forward(self, x, edge_index, return_attention_weights=False):
        # Transform node features
        x = self.lin(x.float())
        
        # Start propagating messages with attention
        out = self.propagate(edge_index, x=x)
        
        if return_attention_weights:
            return out, (edge_index, self.alpha)
        return out

    def message(self, x_i, x_j, index):
        # Concatenate features of target and source nodes
        x = torch.cat([x_i, x_j], dim=-1)
        
        # Compute attention coefficient
        alpha = self.att(x)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # Normalize attention coefficients
        alpha = softmax(alpha, index)
        
        # Store attention weights
        self.alpha = alpha
        
        # Apply attention weights to source features
        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out

def compare_averaging_vs_attention(data, input_dim=9, output_dim=9):
    """Compare averaging vs attention on a molecular graph"""
    # Initialize layers
    avg_layer = SimpleAveragingLayer(input_dim, output_dim)
    att_layer = SimpleAttentionLayer(input_dim, output_dim)

    # Apply both methods
    avg_features = avg_layer(data.x, data.edge_index)
    att_features, (edge_index, att_weights) = att_layer(data.x, data.edge_index, return_attention_weights=True)

    return data.x, avg_features, att_features, att_weights

# Compare on aspirin
original, avg_feat, att_feat, att_weights = compare_averaging_vs_attention(aspirin_data)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot original features
axes[0].imshow(original.detach().numpy(), cmap='viridis')
axes[0].set_title("Original Features")
axes[0].set_xlabel("Feature Dimension")
axes[0].set_ylabel("Node ID")

# Plot averaging features
axes[1].imshow(avg_feat.detach().numpy(), cmap='viridis')
axes[1].set_title("After Averaging Aggregation")
axes[1].set_xlabel("Feature Dimension")

# Plot attention features
im = axes[2].imshow(att_feat.detach().numpy(), cmap='viridis')
axes[2].set_title("After Attention Aggregation")
axes[2].set_xlabel("Feature Dimension")

plt.colorbar(im, ax=axes[2])
plt.suptitle("Comparison: Original vs. Averaging vs. Attention")
plt.tight_layout()
plt.show()

# Visualize the attention weights on the molecular graph
def visualize_attention_on_graph(smiles, attention_weights, title="Attention Weights"):
    """Visualize attention weights on a molecular graph"""
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    data = mol_to_graph(smiles)
    G = to_networkx(data, to_undirected=False)  # Directed graph for attention

    # Get the 2D coordinates from RDKit
    pos = {}
    for i, atom in enumerate(mol.GetAtoms()):
        pos[i] = mol.GetConformer().GetAtomPosition(i)
        pos[i] = (pos[i].x, -pos[i].y)  # Flip y for better visualization

    plt.figure(figsize=(8, 8))

    # Get atom labels
    atom_labels = {i: atom.GetSymbol() for i, atom in enumerate(mol.GetAtoms())}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color='lightblue',
                          node_size=500)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=atom_labels, font_size=12)

    # Normalize attention weights for visualization
    if attention_weights is not None:
        att_weights = attention_weights.detach().numpy().flatten()
        # Create a mapping from edge indices to attention weights
        edge_att = {}
        for i, (src, dst) in enumerate(data.edge_index.t().tolist()):
            edge_att[(src, dst)] = att_weights[i]

        # Create edge list with weights
        edges, weights = zip(*edge_att.items())

        # Normalize weights for visualization
        min_width = 1
        max_width = 5
        norm_weights = [min_width + (w - min(weights)) * (max_width - min_width) / (max(weights) - min(weights) + 1e-6) for w in weights]

        # Draw edges with varying width based on attention
        nx.draw_networkx_edges(G, pos,
                              edgelist=edges,
                              width=norm_weights,
                              edge_color='gray',
                              alpha=0.7,
                              arrowsize=15,
                              node_size=500,
                              connectionstyle='arc3,rad=0.1')
    else:
        # Draw edges without attention weights
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)

    plt.title(title)
    plt.axis('off')
    plt.show()

# Reshape attention weights to match edges
edge_att_weights = torch.zeros(aspirin_data.edge_index.size(1))
for i in range(len(att_weights)):
    edge_att_weights[i] = att_weights[i]

visualize_attention_on_graph(aspirin_smiles, edge_att_weights, "Attention Weights on Aspirin")


# + [markdown] id="1jzM0Un9g28R"
# ### 5.1 Key Differences Between Attention and Simple Averaging
#

# + colab={"base_uri": "https://localhost:8080/", "height": 608} id="BDFUDS7Wftd2" outputId="cc249c78-db22-4da3-ed7f-79d9ddb42e53"
def create_difference_visualization():
    """
    Create a visual explanation of the difference between
    averaging aggregation and attention-based aggregation in GNNs.

    This function generates a simple 4-node graph centered on node 1,
    showing how it aggregates messages from its neighbors:
    - Left subplot: simple averaging (equal weights).
    - Right subplot: attention (varying weights based on learned importance).
    """
    # Create a simple synthetic graph for demonstration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Example graph layout
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([(1, 2), (1, 3), (1, 4)])

    pos = {1: (0.5, 0.5), 2: (0.2, 0.8), 3: (0.8, 0.8), 4: (0.5, 0.2)}

    # Draw averaging aggregation
    ax1.set_title("Simple Averaging Aggregation", fontsize=14)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=12)

    # Draw edges with equal importance
    nx.draw_networkx_edges(G, pos, ax=ax1, width=2.0, alpha=0.7,
                         arrowsize=15, edge_color='gray')

    # Add equal weight labels
    edge_labels = {(1, 2): '1/3', (1, 3): '1/3', (1, 4): '1/3'}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                font_size=12, ax=ax1)

    # Draw attention-based aggregation
    ax2.set_title("Attention-Based Aggregation", fontsize=14)
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=12)

    # Draw edges with different widths to represent attention
    nx.draw_networkx_edges(G, pos, ax=ax2,
                         width=[4.0, 2.0, 1.0],
                         edge_color=['red', 'blue', 'gray'],
                         alpha=0.7, arrowsize=15)

    # Add attention weight labels
    edge_labels = {(1, 2): '0.6', (1, 3): '0.3', (1, 4): '0.1'}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                font_size=12, ax=ax2)

    # Add explanation text
    ax1.text(0.5, -0.1, "All neighbors contribute equally\nto the central node's update",
             ha='center', va='center', transform=ax1.transAxes, fontsize=12)

    ax2.text(0.5, -0.1, "Neighbors contribute based on\nlearned importance (attention weights)",
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()
    plt.show()

create_difference_visualization()


# + [markdown] id="7SRy69Zek1Gg"
# ## 6. Implementing a Graph Attention Network (GAT) <a name="implement-gat"></a>
#
# Now let's implement a full GAT model using PyTorch Geometric:

# + colab={"base_uri": "https://localhost:8080/"} id="nwgVOrl8g6oQ" outputId="88af14e9-9de7-48ef-cf99-3d05878f9986"
class GATLayer(nn.Module):
    """
    Custom GAT layer for demonstration and visualization

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        concat (bool): Whether to concatenate the attention heads.
        add_self_loops (bool): Whether to add self-loops to the edge index.
    """
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, concat=True, add_self_loops=True):
        super(GATLayer, self).__init__()
        self.gat = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            add_self_loops=add_self_loops
        )
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, return_attention=False):
        # For visualization purposes, we'll use the GATConv's return_attention_weights
        if return_attention:
            out, attention_weights = self.gat(x.float(), edge_index, return_attention_weights=True)
            return out, attention_weights
        else:
            return self.gat(x.float(), edge_index)

class SimpleGAT(nn.Module):
    """
    A simple GAT model with the ability to return attention weights

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden features per node.
        out_channels (int): Number of output features per node.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        concat (bool): Whether to concatenate the attention heads.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, concat=True):
            super(SimpleGAT, self).__init__()
            self.dropout = dropout
            
            # First GAT layer
            self.conv1 = GATLayer(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=concat)
            
            # Second GAT layer - adjust input channels based on concatenation
            conv2_in_channels = hidden_channels * heads if concat else hidden_channels
            self.conv2 = GATLayer(conv2_in_channels, out_channels, heads=1, dropout=dropout, concat=concat)

    def forward(self, x, edge_index, return_attention=False):
        # For the first layer
        if return_attention:
            x, attention_weights_1 = self.conv1(x, edge_index, return_attention=True)
        else:
            x = self.conv1(x, edge_index)

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # For the second layer
        if return_attention:
            x, attention_weights_2 = self.conv2(x, edge_index, return_attention=True)
            return x, (attention_weights_1, attention_weights_2)
        else:
            x = self.conv2(x, edge_index)
            return x

# For molecular data, we'll use a more appropriate model
class MolecularGAT(nn.Module):
    """GAT model designed for molecular property prediction
    
    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden features per node.
        out_channels (int): Number of output features per node.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        concat (bool): Whether to concatenate the attention heads.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.1, concat=True):
        super(MolecularGAT, self).__init__()
        self.dropout = dropout

        # First GAT layer with multi-head attention
        self.gat1 = GATLayer(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=concat)
        # Second GAT layer
        conv2_in_channels = hidden_channels * heads if concat else hidden_channels
        self.gat2 = GATLayer(conv2_in_channels, hidden_channels, heads=1, dropout=dropout, concat=concat)
        # Output layer
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GAT layer
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GAT layer
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = torch.mean(x, dim=0)

        # Final linear layer
        x = self.lin(x)

        return x

# Initialize a simple model for our aspirin molecule
in_channels = aspirin_data.x.size(1)  # Number of input features
hidden_channels = 8
out_channels = 8

# Try with single head attention
single_head_model = SimpleGAT(in_channels, hidden_channels, out_channels, heads=1)
# Try with multi-head attention
multi_head_model = SimpleGAT(in_channels, hidden_channels, out_channels, heads=4)

# Run both models
single_head_out = single_head_model(aspirin_data.x, aspirin_data.edge_index)
multi_head_out = multi_head_model(aspirin_data.x, aspirin_data.edge_index)

print("Output with single head attention:", single_head_out.shape)
print("Output with multi-head attention:", multi_head_out.shape)


# Get outputs and attention weights from the multi_head model
x, attention_weights = multi_head_model(aspirin_data.x, aspirin_data.edge_index, return_attention=True)

# Unpack attention weights for each layer
attention_weights_layer1, attention_weights_layer2 = attention_weights

# Each attention weight is a tuple with (edge_index, attention_values)
edge_index_layer1, attn_values_layer1 = attention_weights_layer1
edge_index_layer2, attn_values_layer2 = attention_weights_layer2

print("Layer 1 attention shape:", attn_values_layer1.shape)
print("Layer 2 attention shape:", attn_values_layer2.shape)


# + [markdown] id="9TLqDt3dk73U"
# ## 7. Understanding Multi-Head Attention <a name="understand-multi-head"></a>
#
# Let's visualize and understand the importance of multi-head attention:

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Qp7t60-4k5uA" outputId="b848d553-254a-44ac-b628-23e83ff7607a"
def visualize_multi_head_attention(data, heads=4, hidden_dim=8, add_self_loops=True):
    """
    Visualize multi-head attention on a molecular graph

    Args:
        data (torch_geometric.data.Data): The input data object containing the molecular graph.
        heads (int): The number of attention heads to visualize.
        hidden_dim (int): The dimension of the hidden features.
        add_self_loops (bool): Whether to add self-loops to the edge index. Default is True.
    """
    # Create a custom GAT layer that will return attention weights
    gat_layer = GATLayer(data.x.size(1), hidden_dim, heads=heads, add_self_loops=add_self_loops)

    # Forward pass with attention weights
    _, attention_weights = gat_layer(data.x, data.edge_index, return_attention=True)

    # Extract source, target, and attention weights
    edge_index, att_weights = attention_weights

    # Reshape attention weights for visualization
    # att_weights is of shape [num_edges, num_heads]
    num_edges = edge_index.size(1)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, heads, figsize=(heads * 5, 5))
    if heads == 1:
        axes = [axes]

    # Get molecule for visualization
    mol = Chem.MolFromSmiles(data.smiles)
    AllChem.Compute2DCoords(mol)

    # Get the 2D coordinates from RDKit
    pos = {}
    for i, atom in enumerate(mol.GetAtoms()):
        pos[i] = mol.GetConformer().GetAtomPosition(i)
        pos[i] = (pos[i].x, -pos[i].y)

    # Get atom labels
    atom_labels = {i: atom.GetSymbol() for i, atom in enumerate(mol.GetAtoms())}

    # Create NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(data.x.size(0)))

    # Visualize each attention head
    for h in range(heads):
        ax = axes[h]

        # Get attention weights for this head
        head_weights = att_weights[:, h].detach().numpy()

        # Create edge list with attention weights for this head
        edges = []
        weights = []
        for i in range(num_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            weight = head_weights[i]
            edges.append((src, dst))
            weights.append(weight)

        # Normalize weights for visualization
        min_width = 0.5
        max_width = 4.0
        if len(weights) > 0:  # Ensure there are edges
            norm_weights = [min_width + (w - min(weights)) * (max_width - min_width) /
                            (max(weights) - min(weights) + 1e-6) for w in weights]
        else:
            norm_weights = []

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                              node_color='lightblue',
                              node_size=500,
                              ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, labels=atom_labels, font_size=10, ax=ax)

        # Draw edges with attention weights
        if edges:  # Ensure there are edges
            nx.draw_networkx_edges(G, pos,
                                 edgelist=edges,
                                 width=norm_weights,
                                 edge_color='gray',
                                 alpha=0.7,
                                 arrowsize=10,
                                 ax=ax,
                                 node_size=500,
                                 connectionstyle='arc3,rad=0.1')

        ax.set_title(f"Attention Head {h+1}")
        ax.axis('off')

    plt.suptitle(f"Multi-Head Attention ({heads} heads) on Molecular Graph", fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize multi-head attention on aspirin
visualize_multi_head_attention(aspirin_data, heads=4, add_self_loops=True)

# Create a visual explanation of why multi-head attention is important
def explain_multi_head_advantages():
    """Create a visual explanation of why multi-head attention is beneficial"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Create a simple molecule for demonstration
    G = nx.Graph()
    G.add_nodes_from(range(1, 7))  # 6 nodes representing atoms

    # Add edges
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
    G.add_edges_from(edges)

    # Define positions for drawing (hexagon layout)
    pos = {
        1: (0, 1),
        2: (0.866, 0.5),
        3: (0.866, -0.5),
        4: (0, -1),
        5: (-0.866, -0.5),
        6: (-0.866, 0.5)
    }

    # Single head attention
    ax1.set_title("Single Head Attention", fontsize=14)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=12)

    # Draw edges with varying widths to represent attention
    edge_widths = [4, 3, 2, 1, 2, 3]  # Attention weights for visualization
    nx.draw_networkx_edges(G, pos, ax=ax1,
                         width=edge_widths,
                         alpha=0.7,
                         edge_color='gray')

    # Add annotation to explain the limitation
    ax1.text(0, -1.5, "Single attention mechanism must divide\nits focus across all relationships",
             ha='center', fontsize=12)

    # Multi-head attention
    ax2.set_title("Multi-Head Attention", fontsize=14)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=12)

    # Create distinct edge styles for different attention heads
    # Head 1: Focus on bonds 1-2, 4-5
    head1_edges = [(1, 2), (4, 5)]
    nx.draw_networkx_edges(G, pos, ax=ax2,
                         edgelist=head1_edges,
                         width=3.0,
                         alpha=0.8,
                         edge_color='red',
                         style='solid',
                         label='Head 1')

    # Head 2: Focus on bonds 2-3, 5-6
    head2_edges = [(2, 3), (5, 6)]
    nx.draw_networkx_edges(G, pos, ax=ax2,
                         edgelist=head2_edges,
                         width=3.0,
                         alpha=0.8,
                         edge_color='blue',
                         style='dashed',
                         label='Head 2')

    # Head 3: Focus on bonds 3-4, 6-1
    head3_edges = [(3, 4), (6, 1)]
    nx.draw_networkx_edges(G, pos, ax=ax2,
                         edgelist=head3_edges,
                         width=3.0,
                         alpha=0.8,
                         edge_color='green',
                         style='dotted',
                         label='Head 3')

    ax2.legend(fontsize=10)

    # Add annotation to explain the advantage
    ax2.text(0, -1.5, "Multiple attention heads can specialize\nin different chemical relationships simultaneously",
             ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

explain_multi_head_advantages()

# Create a more technical explanation with chemical context
def explain_multi_head_chemistry_context():
    """Explain the importance of multi-head attention in molecular contexts"""
    plt.figure(figsize=(10, 8))

    # Create a text explanation with diagram
    explanation = """
    # Why Multi-Head Attention is Critical for Molecular Graphs

    ## Different Attention Heads Can Capture:

    1. **Functional Group Interactions**
       - Head 1: Focus on hydrogen bonding patterns
       - Head 2: Focus on π-π stacking interactions
       - Head 3: Focus on hydrophobic interactions

    2. **Multi-Scale Chemical Properties**
       - Head 1: Local atomic environment (1-2 bonds)
       - Head 2: Medium-range effects (3-4 bonds)
       - Head 3: Global molecular shape and electron distribution

    3. **Different Chemical Contexts**
       - Head 1: Identify aromatic ring systems
       - Head 2: Detect hydrogen bond donors/acceptors
       - Head 3: Recognize electronegativity patterns

    ## Benefits in Drug Discovery Applications:

    - More comprehensive feature detection
    - Better handling of complex structure-activity relationships
    - Improved generalization to new molecules
    - Enhanced interpretability of important substructures
    """

    plt.text(0.1, 0.1, explanation, fontsize=14,
             verticalalignment='bottom', horizontalalignment='left',
             family='monospace')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

explain_multi_head_chemistry_context()

# + [markdown] id="5FX2RxJQlOd2"
# ## 8. Implementing a Complete GAT Model for Molecular Property Prediction <a name="implement-gat-property-prediction"></a>
#
# Now let's implement a complete GAT model for molecular property prediction and train it on a real dataset:

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="og6QrfkZlCwl" outputId="37505937-0dad-419e-801d-133a12ac6a77"
# ===================================================
# 1. Load and Preprocess the Dataset
# ===================================================
# Load a dataset from MoleculeNet for training
print("Loading ESOL dataset (water solubility data)...")
dataset = MoleculeNet(root='data', name='ESOL')
print(f"Dataset loaded: {len(dataset)} molecules")

# Split the dataset
torch.manual_seed(42)
indices = torch.randperm(len(dataset))
train_idx = indices[:int(0.8 * len(dataset))]
val_idx = indices[int(0.8 * len(dataset)):int(0.9 * len(dataset))]
test_idx = indices[int(0.9 * len(dataset)):]

train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]
test_dataset = dataset[test_idx]

print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

# Create data loaders
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ===================================================
# 2. Define GAT Model for Regression
# ===================================================

# Define our GAT model for regression
class MolecularGATForRegression(nn.Module):
    """GAT model designed for molecular property prediction
    
    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden features per node.
        out_channels (int): Number of output features per node.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, heads=4, dropout=0.2):
        super(MolecularGATForRegression, self).__init__()
        self.dropout = dropout
        
        # First GAT layer
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        
        # Second GAT layer
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels * 2, heads=2, dropout=dropout)
        
        # Calculate the input dimension for lin1 based on pooling operations
        pool_out_dim = hidden_channels * 4 * 3  # 3 pooling operations
        self.lin1 = nn.Linear(pool_out_dim, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # First GAT layer
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GAT layer
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = torch.cat([
            global_add_pool(x, batch),
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)

        # Output MLP
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x

# Need to import pooling functions
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# ===================================================
# 3. Initialize Model and Optimizer
# ===================================================

# Check the input feature dimensions from the dataset
sample_data = dataset[0]
print(f"Node features: {sample_data.x.shape}")

# Initialize model
in_channels = sample_data.x.shape[1]
model = MolecularGATForRegression(in_channels, hidden_channels=32, heads=4)
model = model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.MSELoss()

# Training loop
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# Evaluation function
def evaluate(loader):
    """
    Evaluate the model on a dataset loader.

    Returns:
        - average MSE loss
        - predicted values
        - ground truth values
    """
    model.eval()
    total_loss = 0
    pred_list, true_list = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            pred_list.append(out.cpu())
            true_list.append(data.y.cpu())

    preds = torch.cat(pred_list, dim=0)
    targets = torch.cat(true_list, dim=0)

    return total_loss / len(loader.dataset), preds, targets

# Train the model
print("Training the GAT model...")
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience, patience_counter = 10, 0

for epoch in range(1, 101):  # 100 epochs
    train_loss = train()
    val_loss, _, _ = evaluate(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_gat_model.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    print(f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Load the best model and evaluate on test set
model.load_state_dict(torch.load('best_gat_model.pt'))
test_loss, test_preds, test_targets = evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Visualize the training process
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot predictions vs actual values
plt.figure(figsize=(8, 8))
plt.scatter(test_targets.numpy(), test_preds.numpy(), alpha=0.5)
plt.plot([-10, 2], [-10, 2], 'r--')  # Perfect prediction line
plt.xlabel('Actual log(Solubility)')
plt.ylabel('Predicted log(Solubility)')
plt.title('GAT Model Predictions on Test Set')
plt.grid(True)
plt.show()

# + [markdown] id="Q6M4fXWIsvQm"
# ## 9. Visualizing Attention Weights in Molecules <a name="visualize-attention"></a>
#
# Now let's visualize the attention weights on some test molecules to understand what the model has learned:

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="ud-jbtGmlUAK" outputId="ccc3f752-97fe-4592-b832-1992695378d2"
import inspect

def visualize_model_attention(model, data, smiles, num_heads=4, use_gpu=False):
    """
    Visualize what the model is attending to

    Args:
        model (torch.nn.Module): The model to visualize.
        data (torch_geometric.data.Data): The input data object containing the molecular graph.
        smiles (str): The SMILES string of the molecule to visualize.
        num_heads (int): The number of attention heads to visualize.
        use_gpu (bool): Whether to use GPU for visualization.
    """
    model.eval()

    # Set the device
    viz_device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    model = model.to(viz_device)

    # Move data to the correct device
    x = data.x.to(viz_device)
    edge_index = data.edge_index.to(viz_device)

    # Create a fake batch index (all zeros since we have only one graph)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=viz_device)

    # Forward pass with attention weights
    with torch.no_grad():
        # Check if the model has the expected return_attention parameter
        try:
            forward_params = list(inspect.getfullargspec(model.forward).args)
            has_return_attention = 'return_attention' in forward_params
        except:
            has_return_attention = False

        if has_return_attention:
            # This works for our SimpleGAT model
            _, attention_weights = model(x, edge_index, return_attention=True)
            layer1_attn, layer2_attn = attention_weights
        else:
            # Fall back to hook-based method for other models
            attention_weights = []

            def hook_fn(module, input, output):
                # The attention weights are the second element of the tuple
                # if return_attention_weights=True
                if isinstance(output, tuple) and len(output) == 2:
                    attention_weights.append(output[1])

            # Register the hook on GAT layers
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, GATConv) or isinstance(module, GATLayer):
                    hooks.append(module.register_forward_hook(hook_fn))

            # Forward pass
            _ = model(x, edge_index, batch)

            # Remove the hooks
            for hook in hooks:
                hook.remove()

            if not attention_weights:
                print("No attention weights were captured. Check the model architecture.")
                return

            # Use the first layer's attention
            layer1_attn = attention_weights[0]
            layer2_attn = attention_weights[1] if len(attention_weights) > 1 else None

    # Get the attention weights from the first layer
    edge_index, att_weights = layer1_attn

    # Move tensors to CPU for visualization
    edge_index = edge_index.cpu()
    att_weights = att_weights.cpu()

    # Determine number of heads to visualize (could be 1 for single-head attention)
    if len(att_weights.shape) > 1:
        num_heads = min(att_weights.size(1), num_heads)
    else:
        num_heads = 1
        # Reshape for consistent processing
        att_weights = att_weights.unsqueeze(1)

    # Create molecule for visualization
    mol = Chem.MolFromSmiles(smiles)

    # Calculate 2D coordinates if they don't exist
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)

    # Get positions for drawing
    pos = {}
    for i, atom in enumerate(mol.GetAtoms()):
        position = mol.GetConformer().GetAtomPosition(i)
        pos[i] = (position.x, -position.y)  # Flip y for better visualization

    # Create NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(data.x.size(0)))

    # Get atom labels
    atom_labels = {i: atom.GetSymbol() for i, atom in enumerate(mol.GetAtoms())}

    # Visualize each attention head
    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 5, 5))
    if num_heads == 1:
        axes = [axes]

    # Visualize each attention head
    for h in range(num_heads):
        ax = axes[h]

        # Get attention weights for this head
        if num_heads > 1:
            head_weights = att_weights[:, h].detach().numpy()
        else:
            head_weights = att_weights.squeeze().detach().numpy()

        # Create edge list with attention weights
        edges = []
        weights = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            weight = head_weights[i] if i < len(head_weights) else 0.0
            edges.append((src, dst))
            weights.append(weight)

        # Normalize weights for visualization
        if weights:
            min_width = 0.5
            max_width = 4.0
            min_weight = min(weights)
            max_weight = max(weights)
            range_weight = max_weight - min_weight
            if range_weight > 1e-6:
                norm_weights = [(w - min_weight) * (max_width - min_width) / range_weight + min_width for w in weights]
            else:
                norm_weights = [min_width for _ in weights]
        else:
            norm_weights = []

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                              node_color='lightblue',
                              node_size=500,
                              ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, labels=atom_labels, font_size=10, ax=ax)

        # Draw edges with attention weights
        if edges:
            cmap = plt.cm.viridis
            edges_collection = nx.draw_networkx_edges(G, pos,
                                                    edgelist=edges,
                                                    width=norm_weights,
                                                    edge_color=weights,
                                                    edge_cmap=cmap,
                                                    alpha=0.7,
                                                    arrowsize=10,
                                                    ax=ax,
                                                    node_size=500,
                                                    connectionstyle='arc3,rad=0.1')  # Curved edges

            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
            cbar.set_label('Attention Weight')

        ax.set_title(f"Attention Head {h+1}")
        ax.axis('off')

    # Add molecule image in an inset
    if num_heads > 0:
        img_ax = fig.add_axes([0.01, 0.65, 0.2, 0.3])
        img = Draw.MolToImage(mol, size=(300, 300))
        img_ax.imshow(img)
        img_ax.axis('off')

    model_name = model.__class__.__name__
    plt.suptitle(f"{model_name}'s Attention Weights on {Chem.MolToSmiles(mol, isomericSmiles=False)}", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Also visualize second layer if available
    if layer2_attn is not None:
        edge_index2, att_weights2 = layer2_attn
        edge_index2 = edge_index2.cpu()
        att_weights2 = att_weights2.cpu()

        # Determine number of heads for second layer
        if len(att_weights2.shape) > 1:
            num_heads2 = min(att_weights2.size(1), num_heads)
        else:
            num_heads2 = 1
            att_weights2 = att_weights2.unsqueeze(1)

        # Create second layer visualization
        fig2, axes2 = plt.subplots(1, num_heads2, figsize=(num_heads2 * 5, 5))
        if num_heads2 == 1:
            axes2 = [axes2]

        # Similar visualization code for second layer...
        for h in range(num_heads2):
            ax = axes2[h]

            # Get attention weights for this head in layer 2
            if num_heads2 > 1:
                head_weights = att_weights2[:, h].detach().numpy()
            else:
                head_weights = att_weights2.squeeze().detach().numpy()

            # Create edge list with attention weights
            edges = []
            weights = []
            for i in range(edge_index2.size(1)):
                src, dst = edge_index2[0, i].item(), edge_index2[1, i].item()
                weight = head_weights[i] if i < len(head_weights) else 0.0
                edges.append((src, dst))
                weights.append(weight)

            # Normalize weights for visualization
            if weights:
                min_width = 0.5
                max_width = 4.0
                min_weight = min(weights)
                max_weight = max(weights)
                range_weight = max_weight - min_weight
                if range_weight > 1e-6:
                    norm_weights = [(w - min_weight) * (max_width - min_width) / range_weight + min_width for w in weights]
                else:
                    norm_weights = [min_width for _ in weights]
            else:
                norm_weights = []

            # Draw nodes
            nx.draw_networkx_nodes(G, pos,
                                  node_color='lightpink',  # Different color for layer 2
                                  node_size=500,
                                  ax=ax)

            # Draw labels
            nx.draw_networkx_labels(G, pos, labels=atom_labels, font_size=10, ax=ax)

            # Draw edges with attention weights
            if edges:
                cmap = plt.cm.plasma  # Different colormap for layer 2
                edges_collection = nx.draw_networkx_edges(G, pos,
                                                        edgelist=edges,
                                                        width=norm_weights,
                                                        edge_color=weights,
                                                        edge_cmap=cmap,
                                                        alpha=0.7,
                                                        arrowsize=10,
                                                        ax=ax,
                                                        node_size=500,
                                                        connectionstyle='arc3,rad=0.1')

                # Add a colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
                cbar.set_label('Attention Weight')

            ax.set_title(f"Layer 2 - Attention Head {h+1}")
            ax.axis('off')

        # Add molecule image in an inset for layer 2 visualization
        if num_heads2 > 0:
            img_ax = fig2.add_axes([0.01, 0.65, 0.2, 0.3])
            img = Draw.MolToImage(mol, size=(300, 300))
            img_ax.imshow(img)
            img_ax.axis('off')

        plt.suptitle(f"{model_name}'s Layer 2 Attention Weights", fontsize=16)
        plt.tight_layout()
        plt.show()

# Usage for a specific molecule
def visualize_gat_on_molecule(smiles="CC(=O)Oc1ccccc1C(=O)O", use_multi_head=True):
    """
    Visualize attention weights for a specific molecule with single or multi-head attention
    
    Args:
        smiles (str): The SMILES string of the molecule to visualize.
        use_multi_head (bool): Whether to use multi-head attention.
    """
    # Create molecule graph
    import inspect
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    import networkx as nx
    import matplotlib.pyplot as plt


    # Get molecule graph data
    data = mol_to_graph(smiles)

    # Get number of features
    in_channels = data.x.size(1)
    hidden_channels = 8
    out_channels = 8

    # Create GAT model - single or multi-head 
    if use_multi_head:
        model = SimpleGAT(in_channels, hidden_channels, out_channels, heads=4)
        print("Using multi-head attention (4 heads)")
    else:
        model = SimpleGAT(in_channels, hidden_channels, out_channels, heads=1)
        print("Using single-head attention")

    # Visualize attention weights
    visualize_model_attention(model, data, smiles,
                             num_heads=4 if use_multi_head else 1,
                             use_gpu=False)

    return model, data

# Example usage - visualize for aspirin with both single and multi-head attention
print("Visualizing attention for Aspirin:")
Aspirin = "CC(=O)Oc1ccccc1C(=O)O"
aspirin_single_head_model, _= visualize_gat_on_molecule(Aspirin, use_multi_head=False)
aspirin_multi_head_model, _ = visualize_gat_on_molecule(Aspirin, use_multi_head=True)

# Try with other molecules
paracetamol = "CC(=O)Nc1ccc(O)cc1"
ibuprofen = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"

print("\nVisualizing attention for Paracetamol:")
visualize_gat_on_molecule(paracetamol, use_multi_head=True)

print("\nVisualizing attention for Ibuprofen:")
visualize_gat_on_molecule(ibuprofen, use_multi_head=True)


# + [markdown] id="teyREwBWs3Aj"
# ## 10. Comparing Single-Head vs Multi-Head Attention Performance <a name="compare-single-multi-head"></a>
#
# Let's train models with different numbers of attention heads and compare their performance:

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="45qiYeVXs3QG" outputId="928a6d20-fdaf-42b8-cdc8-82923805e1aa"
def train_and_evaluate_model(model, optimizer, criterion, train_loader, val_loader, test_loader, epochs, heads):
    """
    Train a GAT model for molecular regression and evaluate its performance.

    Args:
        model (nn.Module): The GAT model instance.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function (e.g., MSELoss).
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        epochs (int): Number of training epochs.
        heads (int): Number of attention heads in GAT layers.

    Returns:
        dict: Dictionary containing training/validation losses, final performance, and model.
    """
    model = model.to(device)


    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        train_loss = total_loss / len(train_loader.dataset)

        # Evaluate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x.float(), data.edge_index, data.batch)
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs

        val_loss = val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Evaluate on test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = criterion(out, data.y)
            test_loss += loss.item() * data.num_graphs

    test_loss = test_loss / len(test_loader.dataset)

    return {
        'heads': heads,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'test_loss': test_loss,
        'model': model,
    }

# Train models with different numbers of attention heads
head_configs = [1, 2, 4, 8]
model_results = []

for heads in head_configs:
    epochs = 30
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Initialize model
    model = MolecularGATForRegression(in_channels, hidden_channels=32, heads=heads)
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.MSELoss()
    print(f"Training model with {heads} attention heads...")
    results = train_and_evaluate_model(model, optimizer, criterion, train_loader, val_loader, test_loader, epochs, heads)
    model_results.append(results)
    print(f"Heads: {heads}, Test Loss: {results['test_loss']:.4f}")

# Visualize the comparison
plt.figure(figsize=(12, 10))

# Plot training curves
plt.subplot(2, 1, 1)
for result in model_results:
    plt.plot(result['train_losses'], label=f"{result['heads']} Heads (Train)")
    plt.plot(result['val_losses'], linestyle='--', label=f"{result['heads']} Heads (Val)")

plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss for Different Numbers of Attention Heads')
plt.legend()
plt.grid(True)

# Plot final test performance
plt.subplot(2, 1, 2)
heads = [result['heads'] for result in model_results]
test_losses = [result['test_loss'] for result in model_results]

bars = plt.bar(heads, test_losses)
plt.xlabel('Number of Attention Heads')
plt.ylabel('Test Loss (MSE)')
plt.title('Test Performance vs. Number of Attention Heads')
plt.xticks(heads)
plt.grid(axis='y')

# Add value labels on bars
for bar, val in zip(bars, test_losses):
    plt.text(bar.get_x() + bar.get_width()/2.0,
             bar.get_height() + 0.005,
             f'{val:.4f}',
             ha='center')

plt.tight_layout()
plt.show()


# + [markdown] id="-E7RduAZs8tV"
# ## 11. Visualizing Feature Transformation Through the Network <a name="visualize-feature-transformation"></a>
#
# Let's see how node features evolve through the network:

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="c5FIa26os6Su" outputId="7d855cc1-eb1f-426a-879c-7eba34732edf"
def visualize_feature_transformation(model, data, smiles, use_gpu=False):
    """
    Visualize how node features are transformed through the GAT layers
    
    Args:
        model (torch.nn.Module): The model to visualize.
        data (torch_geometric.data.Data): The input data object containing the molecular graph.
        smiles (str): The SMILES string of the molecule to visualize.
        use_gpu (bool): Whether to use GPU for computation.
    """
    try:
        model.eval()

        # Set the device with error handling
        try:
            viz_device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
            model = model.to(viz_device)
        except RuntimeError as e:
            print(f"Error setting device: {e}")
            viz_device = torch.device('cpu')
            model = model.to(viz_device)

        # Create hooks to get intermediate activations
        activations = {}

        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook

        # Find the GAT layers in the model
        gat_layers = []
        for name, module in model.named_modules():
            if isinstance(module, GATConv):
                gat_layers.append((name, module))

        if len(gat_layers) < 2:
            print("Model needs at least 2 GAT layers for feature transformation visualization")
            return None

        # Register hooks
        hooks = []
        for name, module in gat_layers[:2]:
            hooks.append(module.register_forward_hook(get_activation(name)))

        # Forward pass with error handling
        try:
            with torch.no_grad():
                x = data.x.to(viz_device)
                edge_index = data.edge_index.to(viz_device)
                batch = torch.zeros(x.size(0), dtype=torch.long, device=viz_device)
                _ = model(x, edge_index, batch)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return None
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

        # Get the input features and activations
        input_features = data.x.cpu().numpy()
        print("Input features: ", input_features.shape)

        if len(activations) < 2:
            print("Could not capture activations from both GAT layers")
            return None

        # Get the activations
        gat_names = list(activations.keys())
        gat1_features = activations[gat_names[0]].cpu().numpy()
        gat2_features = activations[gat_names[1]].cpu().numpy()

        print("GAT1 features: ", gat1_features.shape)
        print("GAT2 features: ", gat2_features.shape)

        # Get the molecule and atom symbols
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES string: {smiles}")
            return None

        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # Create color map for different atom types
        unique_symbols = list(set(atom_symbols))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_symbols)))
        symbol_to_color = {sym: colors[i] for i, sym in enumerate(unique_symbols)}
        node_colors = [symbol_to_color[sym] for sym in atom_symbols]

        # Reduce dimensionality with error handling
        try:
            if input_features.shape[1] > 2:
                pca = PCA(n_components=2)
                input_pca = pca.fit_transform(input_features)
            else:
                input_pca = input_features

            # Use t-SNE with error handling
            perplexity = min(10, input_features.shape[0]-1)
            if perplexity < 1:
                print("Not enough samples for t-SNE")
                return None

            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            gat1_tsne = tsne.fit_transform(gat1_features)
            gat2_tsne = tsne.fit_transform(gat2_features)
        except Exception as e:
            print(f"Error during dimensionality reduction: {e}")
            return None

        # Plot the transformations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot input features
        for i, (x, y) in enumerate(input_pca):
            axes[0].scatter(x, y, s=100, alpha=0.8, color=node_colors[i])
            axes[0].text(x, y, atom_symbols[i], ha='center', va='center')
        axes[0].set_title('Input Features (PCA)')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')

        # Plot features after first GAT layer
        for i, (x, y) in enumerate(gat1_tsne):
            axes[1].scatter(x, y, s=100, alpha=0.8, color=node_colors[i])
            axes[1].text(x, y, atom_symbols[i], ha='center', va='center')
        axes[1].set_title('Features After First GAT Layer (t-SNE)')
        axes[1].set_xlabel('Dimension 1')
        axes[1].set_ylabel('Dimension 2')

        # Plot features after second GAT layer
        for i, (x, y) in enumerate(gat2_tsne):
            axes[2].scatter(x, y, s=100, alpha=0.8, color=node_colors[i])
            axes[2].text(x, y, atom_symbols[i], ha='center', va='center')
        axes[2].set_title('Features After Second GAT Layer (t-SNE)')
        axes[2].set_xlabel('Dimension 1')
        axes[2].set_ylabel('Dimension 2')

        # Add legend for atom types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=symbol_to_color[sym], 
                                    label=sym, markersize=10) 
                         for sym in unique_symbols]
        fig.legend(handles=legend_elements, loc='upper right')

        plt.suptitle(f'Feature Transformation Through GAT Layers for {smiles}', fontsize=16)
        plt.tight_layout()
        plt.show()

        return True

    except Exception as e:
        print(f"Error in visualization: {e}")
        return None

# Create and train two small models for visualization
def train_small_model_for_vis(heads=4):
    """
    Create a small model and visualize feature transformations on a test molecule
    
    Args:
        heads (int): The number of attention heads to use in the model.
    """
    # Create a graph from the SMILES
    sample = dataset[0]

    # Get number of features
    num_features = sample.x.size(1)

    # Create a small GATv2 model for visualization
    class SmallGAT(nn.Module):
        """GAT model designed for molecular property prediction"""
        def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
            super(SmallGAT, self).__init__()
            # First GAT layer with multi-head attention
            self.gat1 = GATLayer(in_channels, hidden_channels, heads=heads, dropout=dropout)
            # Second GAT layer, typically we use concat=False at the last layer
            self.gat2 = GATLayer(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
            # Output layer
            self.lin = nn.Linear(hidden_channels, out_channels)

        def forward(self, x, edge_index, batch=None):
            # Apply first GAT layer with multi-head attention
            x = F.elu(self.gat1(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)

            # Apply second GAT layer
            x = F.elu(self.gat2(x, edge_index))

            # Global pooling (mean of all node features)
            if batch is not None:
                x = global_mean_pool(x, batch)
            else:
                x = torch.mean(x, dim=0)

            # Final linear layer
            x = self.lin(x)

            return x

    # Create the model
    viz_model = SmallGAT(num_features, hidden_channels=8, out_channels=1, heads=heads, dropout=0.1)

    epochs = 50
    lr = 0.001
    weight_decay = 5e-4
    # Train briefly (just to get some reasonable parameters)
    optimizer = torch.optim.Adam(viz_model.parameters(), lr=lr, weight_decay=weight_decay)
    result = train_and_evaluate_model(viz_model, optimizer, criterion, train_loader, val_loader, test_loader, epochs, heads=heads)
    return result['model']

# Visualize feature transformations for one of our test molecules
# Get models
print("Training model with 4 heads..")
four_head_model = train_small_model_for_vis(heads=4)
print("Training model with 8 heads..")
eight_head_model = train_small_model_for_vis(heads=8)

# Example molecules
example_molecules = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
}

# Visualize feature evolution
aspirin_smiles = example_molecules['Aspirin']
aspirin_data = mol_to_graph(aspirin_smiles)
print(f"Aspirin data: {aspirin_data}")

print("Four head features: ")
four_head_features = visualize_feature_transformation(four_head_model, aspirin_data, aspirin_smiles)

print("Eight head features: ")
eight_head_features = visualize_feature_transformation(eight_head_model, aspirin_data, aspirin_smiles)


# + [markdown] id="IVu33MvatDK2"
# ## 12. Interactive Visualization of Attention Mechanism <a name="interactive-visualiztion"></a>
#
# Let's create an interactive demonstration to understand how attention works:

# + colab={"base_uri": "https://localhost:8080/", "height": 817} id="4TCro08MtDtT" outputId="bf7c3a05-5625-4f0e-cb19-a4c9f7ae3ca7"
from IPython.display import HTML
import base64
from matplotlib.animation import FuncAnimation

def create_attention_animation(smiles):
    """
    Create an animation showing how attention weights propagate information
    
    Args:
        smiles (str): The SMILES string of the molecule to visualize.

    Returns:
        ani (matplotlib.animation.FuncAnimation): The animation object.
    
    Raises:
        Exception: If the molecule is not found or cannot be visualized.

    Note:
        This is just a simple example for demonstration purposes.
        in reality, each step of the attention mechanism is computed in parallel,
        and the animation is just a visualization of the attention weights.
    """
    # Create molecule graph
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    data = mol_to_graph(smiles)

    # Create a simplified attention layer
    att_layer = SimpleAttentionLayer(data.x.size(1), data.x.size(1))

    # Get attention weights
    _, (edge_index, weights) = att_layer(data.x, data.edge_index, return_attention_weights=True)

    # Create networkx graph
    G = nx.DiGraph()
    G.add_nodes_from(range(data.x.size(0)))

    # Get positions
    pos = {}
    for i, atom in enumerate(mol.GetAtoms()):
        pos[i] = mol.GetConformer().GetAtomPosition(i)
        pos[i] = (pos[i].x, -pos[i].y)

    # Get edges with attention weights
    edges = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))]
    edge_weights = weights.detach().numpy().flatten()

    # Get atom labels
    atom_labels = {i: atom.GetSymbol() for i, atom in enumerate(mol.GetAtoms())}

    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize the plot
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=atom_labels, font_size=12, ax=ax)

    # Animation function
    def update(frame):
        ax.clear()

        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, labels=atom_labels, font_size=12, ax=ax)

        # Draw edges with attention up to current frame
        if frame > 0:
            # Scale weights for better visualization
            min_width = 1
            max_width = 5
            norm_weights = [min_width + (w - min(edge_weights)) * (max_width - min_width) /
                           (max(edge_weights) - min(edge_weights) + 1e-6) for w in edge_weights[:frame]]

            # Draw edges with attention weights
            nx.draw_networkx_edges(G, pos,
                                 edgelist=edges[:frame],
                                 width=norm_weights,
                                 edge_color=edge_weights[:frame],
                                 edge_cmap=plt.cm.viridis,
                                 alpha=0.8,
                                 arrowsize=15,
                                 ax=ax,
                                 node_size=500,
                                 connectionstyle='arc3,rad=0.1')

        ax.set_title(f"Attention Propagation (Step {frame}/{len(edges)})")
        ax.axis('off')

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(edges)+1, interval=500)

    # To display the animation, save it as a GIF and display with HTML
    plt.close()
    return ani

# Create animation for aspirin
aspirin_ani = create_attention_animation(example_molecules['Aspirin'])

# Save as GIF
aspirin_ani.save('aspirin_attention.gif', writer='pillow', fps=2)

# Display the animation
from IPython.display import Image
Image('aspirin_attention.gif')

# + [markdown] id="eaH_DVKxtHJ0"
# ## 13. Conclusion and Further Research Directions <a name="conclusion"></a>
#
# Graph Attention Networks (GATs) provide powerful tools for molecular modeling and property prediction by allowing the model to focus on the most relevant parts of a molecule. In this tutorial, we've covered:
#
# 1. The basics of representing molecules as graphs
# 2. How Graph Neural Networks process molecular data
# 3. The importance of attention mechanisms vs. simple averaging
# 4. The benefits of multi-head attention for capturing different aspects of molecular structure
# 5. Implementing and training a GAT model for molecular property prediction
#
# For chemists and pharmacists, GATs offer an interpretable deep learning approach that aligns well with chemical intuition about molecular structure. The attention weights can provide insights into which atomic interactions are most relevant for specific properties, potentially guiding the design of new molecules.
#
# ### Further Research Directions:
#
# - Combining GATs with other molecular descriptors
# - Applying GATs to protein-ligand interactions
# - Using GATs for de novo molecular design
# - Exploring edge attention mechanisms for bond-specific features
# - Hierarchical GATs for modeling molecular substructures
#
# ### References:
#
# 1. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.
# 2. Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., ... & Barzilay, R. (2019). Analyzing learned molecular representations for property prediction. Journal of chemical information and modeling, 59(8), 3370-3388.
# 3. Xiong, Z., Wang, D., Liu, X., Zhong, F., Wan, X., Li, X., ... & Fu, T. (2019). Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism. Journal of medicinal chemistry, 63(16), 8749-8760.

# + id="3mq4ecqstKBg"

