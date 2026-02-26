# ---
# jupyter:
#   jupytext:
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
# <a href="https://colab.research.google.com/github/HFooladi/GNNs-For-Chemists/blob/main/notebooks/01.3_GNN_fragment_representation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="bXCYDK0NkdpC"
# # Graph Neural Networks for Molecular Representation Part 2: Fragment-Based Approaches
#
# ## Table of Contents
# 1. [Setup and Review](#setup-and-review)
# 2. [From Atoms to Fragments: Why Fragment-Based Representations?](#from-atoms-to-fragments-why-fragment-based-representations)
# 3. [Theoretical Background: Fragments as Graph Nodes](#theoretical-background-fragments-as-graph-nodes)
# 4. [Functional Group-Based Fragmentation](#functional-group-based-fragmentation)
# 5. [Ring System-Based Fragmentation](#ring-system-based-fragmentation)
# 6. [BRICS Fragmentation](#brics-fragmentation)
# 7. [Comparing Fragment-Based vs Atom-Based Representations](#comparing-fragment-based-vs-atom-based-representations)
# 8. [PyTorch Geometric Implementation for Fragment Graphs](#pytorch-geometric-implementation-for-fragment-graphs)
# 9. [Fragment Property Prediction](#fragment-property-prediction)
# 10. [Advanced Visualization Techniques](#advanced-visualization-techniques)
# 11. [Conclusion](#conclusion)

# + [markdown] id="XHX_uMbYkkX7"
# ## 1. Setup and Review <a name="setup-and-review"></a>
#
# In **Part 1** of this tutorial series, we learned how to represent molecules as graphs where:
# - **Atoms** were the nodes
# - **Chemical bonds** were the edges
# - Each atom had features (element type, charge, hybridization, etc.)
# - Each bond had features (bond type, aromaticity, etc.)
#
# In **Part 2**, we'll explore a more flexible approach where:
# - **Molecular fragments** (groups of atoms) become the nodes
# - **Connections between fragments** become the edges
# - Each fragment has collective properties
# - Fragment interactions capture higher-level chemical relationships
#
# This approach is particularly powerful for:
# - **Drug discovery**: Identifying pharmacophores and functional groups
# - **Chemical synthesis**: Understanding reaction patterns and building blocks
# - **Material science**: Analyzing repeating structural units
# - **Toxicity prediction**: Recognizing toxic substructures

# + colab={"base_uri": "https://localhost:8080/"} id="A7-zyVlhhvwk" outputId="11f359c0-1a20-4e60-c6ed-e5425d1df826"
#@title Install required libraries
# !pip install -q rdkit
# !pip install -q torch_geometric

# + colab={"base_uri": "https://localhost:8080/"} id="N_1gfxN7iJsO" outputId="ce82ca66-1544-47b8-c391-32f4c6afecd1"
#@title Import required libraries
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import itertools

# RDKit for molecular handling
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import BRICS, Fragments, rdqueries
from rdkit.Chem.Draw import rdMolDraw2D

# PyTorch and PyTorch Geometric
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import to_networkx

# NetworkX for graph visualization
import networkx as nx

# Set plotting style
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("Set2")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("All libraries imported successfully!")

# + [markdown] id="r1xzRB8EkUtQ"
# ## 2. From Atoms to Fragments: Why Fragment-Based Representations? <a name="from-atoms-to-fragments-why-fragment-based-representations"></a>
#
# ### The Limitation of Atom-Based Representations
#
# While atom-based graphs capture precise molecular structure, they have some limitations:
#
# 1. **High granularity**: Large molecules result in very large graphs
# 2. **Missing chemical intuition**: Important functional groups are scattered across multiple nodes
# 3. **Computational complexity**: Processing scales with the number of atoms
# 4. **Chemical interpretation**: Hard to relate predictions back to known chemical concepts
#
# ### Advantages of Fragment-Based Representations
#
# Fragment-based representations offer several benefits:
#
# 1. **Chemical relevance**: Nodes represent meaningful chemical units (functional groups, rings)
# 2. **Reduced complexity**: Fewer nodes for large molecules
# 3. **Interpretability**: Easier to understand which chemical features drive predictions
# 4. **Prior knowledge**: Incorporates known structure-activity relationships
# 5. **Transferability**: Fragments can be learned and applied across different molecules
#
# Let's start with a simple example to illustrate the concept:

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="conceptual_comparison" outputId=""
def create_conceptual_comparison():
    """
    Create a conceptual comparison between atom-based and fragment-based representations
    using aspirin as an example molecule.
    
    The function creates a 2x2 grid of plots showing:
    1. Description of atom-based representation (top left)
    2. Description of fragment-based representation (top right) 
    3. Chemical interpretation comparison (bottom left)
    4. Best use cases for each approach (bottom right)
    """
    # Create figure with 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Create aspirin molecule for demonstration
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    mol = Chem.MolFromSmiles(aspirin_smiles)
    mol = Chem.AddHs(mol)  # Add hydrogen atoms
    
    # Top left: Atom-based representation description
    axes[0, 0].text(0.5, 0.5, "Atom-Based Representation\n\n• 21 nodes (atoms)\n• 21 edges (bonds)\n• Each node = 1 atom\n• Fine-grained structure", 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Atom-Based Graph", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Top right: Fragment-based representation description
    axes[0, 1].text(0.5, 0.5, "Fragment-Based Representation\n\n• 3-5 nodes (fragments)\n• 2-4 edges (connections)\n• Each node = functional group\n• Chemically meaningful", 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Fragment-Based Graph", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom left: Chemical interpretation comparison
    axes[1, 0].text(0.5, 0.7, "Chemical Interpretation", ha='center', fontsize=14, fontweight='bold')
    axes[1, 0].text(0.5, 0.5, "Atom-based:\n'Carbon at position 5 connects to...'\n\nFragment-based:\n'Benzene ring connects to\nacetyl ester group'", 
                   ha='center', va='center', fontsize=11)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Bottom right: Best use cases for each approach
    axes[1, 1].text(0.5, 0.7, "Best Use Cases", ha='center', fontsize=14, fontweight='bold')
    axes[1, 1].text(0.5, 0.45, "Atom-based:\n• Precise property prediction\n• Quantum calculations\n• Small molecules\n\nFragment-based:\n• Drug discovery\n• Large molecules\n• Interpretable models\n• Prior knowledge integration", 
                   ha='center', va='center', fontsize=10)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

# Create and display the comparison visualization
create_conceptual_comparison()

# + [markdown] id="VzsxoQxEm9dM"
# ## 3. Theoretical Background: Fragments as Graph Nodes <a name="theoretical-background-fragments-as-graph-nodes"></a>
#
# ### What is a Molecular Fragment?
#
# A **molecular fragment** is a connected subgraph of atoms that represents a meaningful chemical unit. Fragments can be defined in several ways:
#
# 1. **Functional Groups**: Chemically reactive groups (e.g., -OH, -COOH, -NH2)
# 2. **Ring Systems**: Cyclic structures (e.g., benzene, cyclohexane)
# 3. **Pharmacophores**: 3D arrangements responsible for biological activity
# 4. **Synthetic Building Blocks**: Units used in chemical synthesis
# 5. **Structural Motifs**: Recurring patterns in molecular databases
#
# ### Mathematical Representation
#
# For a fragment-based graph F:
# - **Nodes V_F**: Set of molecular fragments
# - **Edges E_F**: Connections between fragments
# - **Node Features X_F**: Properties of each fragment
# - **Edge Features E_attr_F**: Properties of fragment connections
#
# The transformation from atom-based to fragment-based representation involves:
# 1. **Fragmentation**: Partitioning atoms into meaningful groups
# 2. **Fragment Feature Extraction**: Computing properties for each fragment
# 3. **Connection Identification**: Determining how fragments are linked
# 4. **Edge Feature Computation**: Characterizing fragment interactions

# + [markdown] id="fragmentation_methods"
# ### Types of Fragmentation Methods
#
# Let's explore three main approaches to molecular fragmentation:

# + id="fragmentation_overview" colab={"base_uri": "https://localhost:8080/", "height": 600"
def visualize_fragmentation_methods():
    """
    Visualize different fragmentation approaches on the same molecule.
    
    This function creates a 2x2 grid visualization showing:
    - The original ibuprofen molecule
    - Three text boxes describing different fragmentation methods:
        1. Functional Group-Based
        2. Ring System-Based  
        3. BRICS Fragmentation
    
    Each method description includes key characteristics and what the nodes represent.
    """
    # Use ibuprofen as an example since it has clear functional groups (carboxylic acid, aromatic ring)
    ibuprofen_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    # Convert SMILES string to RDKit molecule object
    mol = Chem.MolFromSmiles(ibuprofen_smiles)
    
    # Create 2x2 subplot grid with specified figure size
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left plot: Display the original ibuprofen molecule
    img = Draw.MolToImage(mol, size=(300, 300))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Molecule: Ibuprofen", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Define descriptions for each fragmentation method
    # Each description includes:
    # - Method name
    # - Key components/characteristics 
    # - What the nodes represent in the graph
    methods = [
        "Functional Group-Based\n\n• Carboxylic acid group\n• Aromatic ring\n• Alkyl chains\n\nNodes: Chemical functions",
        "Ring System-Based\n\n• Benzene ring system\n• Aliphatic fragments\n• Connection points\n\nNodes: Ring vs non-ring",
        "BRICS Fragmentation\n\n• Systematic bond breaking\n• Realistic fragments\n• Retrosynthetic analysis\n\nNodes: Synthetic building blocks"
    ]
    
    # Define grid positions and colors for each method description
    positions = [(0, 1), (1, 0), (1, 1)]  # Positions in the 2x2 grid
    colors = ['lightcoral', 'lightsalmon', 'lightblue']  # Pastel colors for each box
    
    # Create text boxes for each fragmentation method
    for i, (method, pos, color) in enumerate(zip(methods, positions, colors)):
        # Add text with colored background box
        axes[pos].text(0.5, 0.5, method, 
                      ha='center',  # Horizontal alignment
                      va='center',  # Vertical alignment
                      fontsize=11,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor=color))
        # Set axis limits and turn off axis display
        axes[pos].set_xlim(0, 1)
        axes[pos].set_ylim(0, 1)
        axes[pos].axis('off')
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

# Create and display the visualization
visualize_fragmentation_methods()

# + [markdown] id="esDZEn1lsUUr"
# ## 4. Functional Group-Based Fragmentation <a name="functional-group-based-fragmentation"></a>
#
# Functional groups are the most chemically intuitive way to fragment molecules. They represent reactive sites and determine many molecular properties.

# + id="functional_group_implementation"
def identify_functional_groups(mol: Chem.rdchem.Mol) -> tuple[dict, dict]:
    """
    Identify functional groups in a molecule using SMARTS patterns.
    
    Args:
        mol: RDKit molecule object to analyze
        
    Returns:
        tuple containing:
            fragments (dict): Dictionary mapping fragment IDs to functional group information
                Each fragment has keys:
                - name: Name of the functional group
                - description: Description of the functional group 
                - atoms: List of atom indices in the fragment
                - type: Either 'functional_group' or 'backbone'
            atom_to_fragment (dict): Dictionary mapping atom indices to their fragment ID
    """
    
    # Define functional group SMARTS patterns
    # Each pattern is (name, SMARTS, description)
    # SMARTS patterns define substructure queries to match functional groups
    functional_groups = [
        ("Carboxylic_Acid", "[CX3](=O)[OX2H1]", "Carboxylic acid"),
        ("Ester", "[CX3](=O)[OX2][#6]", "Ester group"), 
        ("Alcohol", "[OX2H][#6]", "Alcohol/hydroxyl"),
        ("Phenol", "[OX2H][c]", "Phenolic hydroxyl"),
        ("Aldehyde", "[CX3H1](=O)[#6]", "Aldehyde"),
        ("Ketone", "[#6][CX3](=O)[#6]", "Ketone"),
        ("Amine_Primary", "[NX3;H2][#6]", "Primary amine"),
        ("Amine_Secondary", "[NX3;H1]([#6])[#6]", "Secondary amine"), 
        ("Amine_Tertiary", "[NX3;H0]([#6])([#6])[#6]", "Tertiary amine"),
        ("Aromatic_Ring", "c1ccccc1", "Benzene ring"),
        ("Ether", "[OX2]([#6])[#6]", "Ether linkage"),
        ("Amide", "[NX3][CX3](=[OX1])[#6]", "Amide group"),
        ("Nitrile", "[NX1]#[CX2]", "Nitrile"),
        ("Nitro", "[NX3+]([OX1-])[OX1-]", "Nitro group")
    ]
    
    fragments = {}  # Store fragment information
    fragment_id = 0  # Counter for fragment IDs
    atom_to_fragment = {}  # Map atoms to their fragment
    
    # Find matches for each functional group pattern
    for fg_name, smarts, description in functional_groups:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            # Find all matches of the pattern in the molecule
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                # Only create fragment if atoms aren't already assigned
                if not any(atom_idx in atom_to_fragment for atom_idx in match):
                    fragments[fragment_id] = {
                        'name': fg_name,
                        'description': description,
                        'atoms': list(match),
                        'type': 'functional_group'
                    }
                    # Mark atoms as belonging to this fragment
                    for atom_idx in match:
                        atom_to_fragment[atom_idx] = fragment_id
                    fragment_id += 1
    
    # Handle remaining atoms not in functional groups
    unassigned_atoms = []
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx not in atom_to_fragment:
            unassigned_atoms.append(atom_idx)
    
    # Group unassigned atoms into connected components
    if unassigned_atoms:
        # Find bonds between unassigned atoms
        remaining_bonds = []
        for bond in mol.GetBonds():
            if (bond.GetBeginAtomIdx() in unassigned_atoms and 
                bond.GetEndAtomIdx() in unassigned_atoms):
                remaining_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        
        # Find connected components using depth-first search
        components = []
        visited = set()
        
        for atom_idx in unassigned_atoms:
            if atom_idx not in visited:
                component = []
                stack = [atom_idx]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        # Add connected unassigned atoms
                        for begin, end in remaining_bonds:
                            if begin == current and end not in visited:
                                stack.append(end)
                            elif end == current and begin not in visited:
                                stack.append(begin)
                
                if component:
                    components.append(component)
        
        # Create backbone fragments from components
        for component in components:
            if len(component) > 0:
                fragments[fragment_id] = {
                    'name': 'Backbone',
                    'description': 'Molecular backbone',
                    'atoms': component,
                    'type': 'backbone'
                }
                for atom_idx in component:
                    atom_to_fragment[atom_idx] = fragment_id
                fragment_id += 1
    
    return fragments, atom_to_fragment

def create_functional_group_graph(mol: Chem.rdchem.Mol, 
                                fragments: dict,
                                atom_to_fragment: dict) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Create a graph representation where functional groups are nodes.
    
    Args:
        mol: RDKit molecule object
        fragments: Fragment dictionary from identify_functional_groups
        atom_to_fragment: Mapping from atoms to fragments
        
    Returns:
        tuple containing:
            node_features (np.ndarray): Feature matrix for nodes/fragments
            adjacency (np.ndarray): Adjacency matrix showing fragment connections
            edge_features (list): List of edge features (empty in current implementation)
            node_labels (list): List of fragment names for each node
    """
    n_fragments = len(fragments)
    
    # Create node features for fragments
    node_features = []
    node_labels = []
    
    # Define features for each fragment
    for frag_id in range(n_fragments):
        if frag_id in fragments:
            frag = fragments[frag_id]
            
            # Basic numeric features
            features = [
                len(frag['atoms']),  # Size of fragment
                1 if frag['type'] == 'functional_group' else 0,  # Binary functional group indicator
                1 if frag['type'] == 'backbone' else 0,  # Binary backbone indicator
            ]
            
            # One-hot encoding of functional group types
            fg_types = ['Carboxylic_Acid', 'Ester', 'Alcohol', 'Phenol', 'Aldehyde', 
                       'Ketone', 'Amine_Primary', 'Amine_Secondary', 'Amine_Tertiary', 
                       'Aromatic_Ring', 'Ether', 'Amide', 'Other']
            
            # Create one-hot encoding
            fg_encoding = [0] * len(fg_types)
            if frag['name'] in fg_types:
                fg_encoding[fg_types.index(frag['name'])] = 1
            else:
                fg_encoding[-1] = 1  # Mark as "Other"
            
            features.extend(fg_encoding)
            node_features.append(features)
            node_labels.append(frag['name'])
    
    node_features = np.array(node_features)
    
    # Create adjacency matrix showing fragment connections
    adjacency = np.zeros((n_fragments, n_fragments))
    edge_features = []
    
    # Find connections between fragments through bonds
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        
        if atom1 in atom_to_fragment and atom2 in atom_to_fragment:
            frag1 = atom_to_fragment[atom1]
            frag2 = atom_to_fragment[atom2]
            
            # Add edges between different fragments
            if frag1 != frag2:
                adjacency[frag1, frag2] = 1
                adjacency[frag2, frag1] = 1  # Undirected graph
    
    return node_features, adjacency, edge_features, node_labels

# Test with aspirin
aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
aspirin_mol = Chem.MolFromSmiles(aspirin_smiles)

fragments, atom_to_fragment = identify_functional_groups(aspirin_mol)
node_features, adjacency, edge_features, node_labels = create_functional_group_graph(
    aspirin_mol, fragments, atom_to_fragment
)

print("Functional Groups in Aspirin:")
for frag_id, frag in fragments.items():
    print(f"Fragment {frag_id}: {frag['name']} - {frag['description']} (atoms: {frag['atoms']})")

print(f"\nFragment graph has {len(fragments)} nodes")
print("Node labels:", node_labels)

# + [markdown] id="visualize_functional_groups"
# Let's visualize how functional group fragmentation works:

# + id="visualize_functional_groups_implementation" colab={"base_uri": "https://localhost:8080/", "height": 800"
def visualize_functional_group_fragmentation(mol, fragments, atom_to_fragment, molecule_name="Molecule"):
    """
    Visualize functional group-based fragmentation of a molecule.
    
    Args:
        mol: RDKit molecule object to visualize
        fragments: Dictionary mapping fragment IDs to fragment information
        atom_to_fragment: Dictionary mapping atom indices to fragment IDs
        molecule_name: Name of the molecule for display purposes
        
    Creates a 2x2 figure showing:
        1. Original molecule structure
        2. Molecule with highlighted functional groups 
        3. Graph representation of fragments and their connections
        4. Text details about each fragment
    """
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Original molecule visualization
    img = Draw.MolToImage(mol, size=(350, 350))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"Original {molecule_name}", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Highlighted functional groups visualization
    ax = axes[0, 1]
    
    # Create a copy to avoid modifying original molecule
    mol_copy = Chem.Mol(mol)
    
    # Generate distinct colors for different fragments using matplotlib colormap
    colors = plt.cm.Set3(np.linspace(0, 1, len(fragments)))
    highlight_atoms = []
    highlight_colors = {}
    
    # Assign colors to atoms in each fragment
    for i, (frag_id, frag) in enumerate(fragments.items()):
        color = colors[i]
        for atom_idx in frag['atoms']:
            highlight_atoms.append(atom_idx)
            highlight_colors[atom_idx] = color
    
    # Draw molecule with RDKit's 2D drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(350, 350)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_highlighted = drawer.GetDrawingText()
    
    # Display molecule (highlighting functionality simplified)
    axes[0, 1].imshow(img)
    axes[0, 1].set_title("Functional Groups Identified", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Fragment graph visualization
    ax = axes[1, 0]
    
    # Only create graph if multiple fragments exist
    if len(fragments) > 1:
        # Create NetworkX graph representation
        G = nx.Graph()
        
        # Add nodes representing fragments
        for frag_id, frag in fragments.items():
            G.add_node(frag_id, label=frag['name'])
        
        # Add edges between connected fragments
        for i in range(len(fragments)):
            for j in range(i+1, len(fragments)):
                if adjacency[i, j] == 1:
                    G.add_edge(i, j)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Color nodes based on fragment type
        node_colors = []
        for node in G.nodes():
            if fragments[node]['type'] == 'functional_group':
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax)
        
        # Add fragment name labels (truncated to 8 chars)
        labels = {frag_id: frag['name'][:8] for frag_id, frag in fragments.items()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title("Fragment Graph", fontsize=12, fontweight='bold')
    else:
        # Display message if only one fragment found
        ax.text(0.5, 0.5, "Only one fragment\nidentified", ha='center', va='center')
        ax.set_title("Fragment Graph", fontsize=12, fontweight='bold')
    
    ax.axis('off')
    
    # 4. Fragment details text display
    ax = axes[1, 1]
    ax.axis('off')
    
    # Build text description of fragments
    fragment_text = "Identified Fragments:\n\n"
    for frag_id, frag in fragments.items():
        fragment_text += f"{frag_id}: {frag['name']}\n"
        fragment_text += f"   Type: {frag['type']}\n"
        fragment_text += f"   Size: {len(frag['atoms'])} atoms\n\n"
    
    # Display fragment details in a text box
    ax.text(0.05, 0.95, fragment_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    ax.set_title("Fragment Details", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Test visualization with aspirin molecule
visualize_functional_group_fragmentation(aspirin_mol, fragments, atom_to_fragment, "Aspirin")

# Test visualization with a more complex molecule (Ibuprofen)
print("\n" + "="*50)
print("Testing with Ibuprofen:")

# Create and analyze Ibuprofen molecule
ibuprofen_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
ibuprofen_mol = Chem.MolFromSmiles(ibuprofen_smiles)

# Identify fragments and create graph representation
ibuprofen_fragments, ibuprofen_atom_to_fragment = identify_functional_groups(ibuprofen_mol)
ibuprofen_node_features, ibuprofen_adjacency, _, ibuprofen_node_labels = create_functional_group_graph(
    ibuprofen_mol, ibuprofen_fragments, ibuprofen_atom_to_fragment
)

# Visualize Ibuprofen fragmentation
visualize_functional_group_fragmentation(ibuprofen_mol, ibuprofen_fragments, 
                                       ibuprofen_atom_to_fragment, "Ibuprofen")

# + [markdown] id="ring_system_fragmentation"
# ## 5. Ring System-Based Fragmentation <a name="ring-system-based-fragmentation"></a>
#
# Ring systems are crucial structural elements in organic chemistry. They often determine:
# - **Molecular rigidity** and shape
# - **Aromaticity** and electronic properties  
# - **Pharmacological activity** (many drugs contain rings)
# - **Synthetic accessibility**

# + id="ring_system_implementation"
def identify_ring_systems(mol: Chem.rdchem.Mol) -> tuple[dict, dict]:
    """
    Identify ring systems and non-ring fragments in a molecule.
    
    This function analyzes a molecule to find:
    1. Ring systems (both aromatic and aliphatic)
    2. Fused ring systems
    3. Non-ring fragments (aliphatic chains)
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        tuple containing:
            fragments (dict): Dictionary mapping fragment IDs to ring system information
                Keys are fragment IDs (int)
                Values are dicts containing:
                    - name: Type of fragment
                    - description: Detailed description
                    - atoms: List of atom indices
                    - type: 'ring_system' or 'non_ring'
                    - is_aromatic: Boolean
                    - ring_count: Number of rings
            atom_to_fragment (dict): Maps atom indices to their fragment IDs
    """
    # Get ring information from RDKit
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    
    fragments = {}
    fragment_id = 0
    atom_to_fragment = {}
    
    # Process ring systems if they exist
    if atom_rings:
        # Merge overlapping rings into ring systems using sets
        ring_systems = []
        
        for ring in atom_rings:
            ring_set = set(ring)
            merged = False
            
            # Check if current ring overlaps with any existing system
            for i, existing_system in enumerate(ring_systems):
                if ring_set & existing_system:  # Set intersection checks overlap
                    ring_systems[i] = existing_system | ring_set  # Union merges systems
                    merged = True
                    break
            
            if not merged:
                ring_systems.append(ring_set)
        
        # Create fragments for each identified ring system
        for ring_system in ring_systems:
            ring_atoms = list(ring_system)
            
            # Determine ring system properties
            is_aromatic = any(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() 
                            for atom_idx in ring_atoms)
            ring_count = len([ring for ring in atom_rings if set(ring) & ring_system])
            
            # Classify ring system type
            if is_aromatic:
                if ring_count == 1:
                    ring_type = "Aromatic_Ring"
                    description = "Single aromatic ring"
                else:
                    ring_type = "Aromatic_System"
                    description = f"Fused aromatic system ({ring_count} rings)"
            else:
                if ring_count == 1:
                    ring_type = "Aliphatic_Ring"
                    description = "Single aliphatic ring"
                else:
                    ring_type = "Aliphatic_System"  
                    description = f"Fused aliphatic system ({ring_count} rings)"
            
            # Store fragment information
            fragments[fragment_id] = {
                'name': ring_type,
                'description': description,
                'atoms': ring_atoms,
                'type': 'ring_system',
                'is_aromatic': is_aromatic,
                'ring_count': ring_count
            }
            
            # Map atoms to their fragment
            for atom_idx in ring_atoms:
                atom_to_fragment[atom_idx] = fragment_id
            
            fragment_id += 1
    
    # Handle non-ring atoms
    non_ring_atoms = [atom_idx for atom_idx in range(mol.GetNumAtoms()) 
                     if atom_idx not in atom_to_fragment]
    
    # Group non-ring atoms into connected components
    if non_ring_atoms:
        # Find bonds between non-ring atoms
        non_ring_bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                         for bond in mol.GetBonds()
                         if (bond.GetBeginAtomIdx() in non_ring_atoms and 
                             bond.GetEndAtomIdx() in non_ring_atoms)]
        
        # Find connected components using DFS
        components = []
        visited = set()
        
        for atom_idx in non_ring_atoms:
            if atom_idx not in visited:
                component = []
                stack = [atom_idx]
                
                # DFS to find connected non-ring atoms
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        # Add connected non-ring atoms to stack
                        for begin, end in non_ring_bonds:
                            if begin == current and end not in visited:
                                stack.append(end)
                            elif end == current and begin not in visited:
                                stack.append(begin)
                
                if component:
                    components.append(component)
        
        # Create fragments for non-ring components
        for component in components:
            if len(component) > 0:
                fragments[fragment_id] = {
                    'name': 'Aliphatic_Chain',
                    'description': f'Non-ring fragment ({len(component)} atoms)',
                    'atoms': component,
                    'type': 'non_ring',
                    'is_aromatic': False,
                    'ring_count': 0
                }
                
                for atom_idx in component:
                    atom_to_fragment[atom_idx] = fragment_id
                
                fragment_id += 1
    
    return fragments, atom_to_fragment

def create_ring_system_graph(mol: Chem.rdchem.Mol, 
                           fragments: dict, 
                           atom_to_fragment: dict) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Create a graph representation where ring systems are nodes.
    
    Args:
        mol: RDKit molecule object
        fragments: Dictionary of fragment information from identify_ring_systems()
        atom_to_fragment: Mapping of atoms to fragments from identify_ring_systems()
        
    Returns:
        tuple containing:
            node_features (np.ndarray): Matrix of node features
            adjacency (np.ndarray): Adjacency matrix
            edge_features (list): Empty list (placeholder for future use)
            node_labels (list): List of fragment names
    """
    n_fragments = len(fragments)
    
    # Create node features for ring fragments
    node_features = []
    node_labels = []
    
    # Define possible ring types for one-hot encoding
    ring_types = ['Aromatic_Ring', 'Aromatic_System', 'Aliphatic_Ring', 
                  'Aliphatic_System', 'Aliphatic_Chain', 'Other']
    
    for frag_id in range(n_fragments):
        if frag_id in fragments:
            frag = fragments[frag_id]
            
            # Construct feature vector
            features = [
                len(frag['atoms']),  # Size of fragment
                1 if frag['type'] == 'ring_system' else 0,  # Ring system indicator
                1 if frag.get('is_aromatic', False) else 0,  # Aromaticity
                frag.get('ring_count', 0),  # Number of rings
                1 if frag['type'] == 'non_ring' else 0,  # Non-ring indicator
            ]
            
            # Add one-hot encoding of ring type
            ring_encoding = [1 if frag['name'] == ring_type else 0 
                           for ring_type in ring_types[:-1]]
            ring_encoding.append(1 if frag['name'] not in ring_types[:-1] else 0)
            
            features.extend(ring_encoding)
            node_features.append(features)
            node_labels.append(frag['name'])
    
    node_features = np.array(node_features)
    
    # Create adjacency matrix for fragment connectivity
    adjacency = np.zeros((n_fragments, n_fragments))
    
    # Find connections between fragments through bonds
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        if atom1 in atom_to_fragment and atom2 in atom_to_fragment:
            frag1 = atom_to_fragment[atom1]
            frag2 = atom_to_fragment[atom2]
            
            # Add edge if atoms belong to different fragments
            if frag1 != frag2:
                adjacency[frag1, frag2] = 1
                adjacency[frag2, frag1] = 1  # Symmetric adjacency
    
    return node_features, adjacency, [], node_labels

# Test with naphthalene (fused aromatic rings)
naphthalene_smiles = "c1ccc2ccccc2c1"
naphthalene_mol = Chem.MolFromSmiles(naphthalene_smiles)

naphthalene_fragments, naphthalene_atom_to_fragment = identify_ring_systems(naphthalene_mol)

print("Ring Systems in Naphthalene:")
for frag_id, frag in naphthalene_fragments.items():
    print(f"Fragment {frag_id}: {frag['name']} - {frag['description']}")
    print(f"  Aromatic: {frag['is_aromatic']}, Ring count: {frag['ring_count']}")

# + id="visualize_ring_systems" colab={"base_uri": "https://localhost:8080/", "height": 1200"
def visualize_ring_system_fragmentation(mol: Chem.rdchem.Mol, 
                                      fragments: dict, 
                                      atom_to_fragment: dict, 
                                      molecule_name: str = "Molecule") -> None:
    """
    Visualize ring system-based fragmentation of a molecule with multiple views.

    Args:
        mol: RDKit molecule object to visualize
        fragments: Dictionary mapping fragment IDs to fragment information
        atom_to_fragment: Dictionary mapping atom indices to fragment IDs
        molecule_name: Name of the molecule for display purposes

    Returns:
        None - Displays matplotlib figure with 4 subplots:
        1. Original molecule structure
        2. Ring systems highlighted 
        3. Fragment connectivity graph
        4. Fragment details text
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Original molecule view
    img = Draw.MolToImage(mol, size=(350, 350))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"Original {molecule_name}", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Ring systems view (placeholder - same as original for now)
    axes[0, 1].imshow(img)  # TODO: Add actual ring system highlighting
    axes[0, 1].set_title("Ring Systems Identified", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Fragment connectivity graph
    ax = axes[1, 0]
    
    if len(fragments) > 1:
        # Create NetworkX graph representation
        G = nx.Graph()
        
        # Add fragment nodes
        for frag_id, frag in fragments.items():
            G.add_node(frag_id, label=frag['name'])
        
        # Get adjacency matrix and add edges between connected fragments
        node_features, adjacency, _, node_labels = create_ring_system_graph(
            mol, fragments, atom_to_fragment
        )
        
        for i in range(len(fragments)):
            for j in range(i+1, len(fragments)):
                if adjacency[i, j] == 1:
                    G.add_edge(i, j)
        
        # Set up graph layout
        pos = nx.spring_layout(G, seed=42)
        
        # Color nodes based on fragment type:
        # - Red shades for ring systems (light coral for aromatic, salmon for aliphatic)
        # - Light blue for non-ring fragments
        node_colors = []
        for node in G.nodes():
            if fragments[node]['type'] == 'ring_system':
                if fragments[node]['is_aromatic']:
                    node_colors.append('lightcoral')
                else:
                    node_colors.append('lightsalmon')
            else:
                node_colors.append('lightblue')
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax)
        
        # Add fragment labels (truncated to 8 chars for readability)
        labels = {frag_id: frag['name'][:8] for frag_id, frag in fragments.items()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title("Ring System Graph", fontsize=12, fontweight='bold')
    else:
        # Handle single fragment case
        ax.text(0.5, 0.5, "Only one fragment\nidentified", ha='center', va='center')
        ax.set_title("Ring System Graph", fontsize=12, fontweight='bold')
    
    ax.axis('off')
    
    # 4. Fragment details text display
    ax = axes[1, 1]
    ax.axis('off')
    
    # Build detailed fragment information text
    fragment_text = "Ring System Analysis:\n\n"
    for frag_id, frag in fragments.items():
        fragment_text += f"{frag_id}: {frag['name']}\n"
        fragment_text += f"   Size: {len(frag['atoms'])} atoms\n"
        if frag['type'] == 'ring_system':
            fragment_text += f"   Aromatic: {frag['is_aromatic']}\n"
            fragment_text += f"   Rings: {frag['ring_count']}\n"
        fragment_text += "\n"
    
    # Display fragment details in a formatted text box
    ax.text(0.05, 0.95, fragment_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan"))
    ax.set_title("Ring System Details", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Test visualization with naphthalene
visualize_ring_system_fragmentation(naphthalene_mol, naphthalene_fragments, 
                                   naphthalene_atom_to_fragment, "Naphthalene")

# Test with more complex molecule containing mixed ring systems
complex_smiles = "C1CCC(CC1)C2=CC=C(C=C2)C3CCCCC3"  # Cyclohexyl-benzene-cyclohexyl
complex_mol = Chem.MolFromSmiles(complex_smiles)

complex_fragments, complex_atom_to_fragment = identify_ring_systems(complex_mol)
visualize_ring_system_fragmentation(complex_mol, complex_fragments, 
                                   complex_atom_to_fragment, "Complex Ring System")

# + [markdown] id="brics_fragmentation"
# ## 6. BRICS Fragmentation <a name="brics-fragmentation"></a>
#
# **BRICS** (Breaking of Retrosynthetically Interesting Chemical Substructures) is a systematic approach to molecular fragmentation that:
#
# - Breaks bonds at positions that are **chemically meaningful** for synthesis
# - Creates fragments that could realistically be **synthetic building blocks**
# - Is widely used in **drug discovery** and **chemical space exploration**
# - Provides **standardized fragmentation** across different molecules

# + id="brics_implementation"
def brics_fragmentation(mol: Chem.rdchem.Mol) -> tuple[dict, dict]:
    """
    Perform BRICS fragmentation on a molecule.
    
    Args:
        mol: RDKit molecule object to fragment
        
    Returns:
        tuple containing:
            - fragments: Dictionary mapping fragment IDs to BRICS fragment information
            - atom_to_fragment: Dictionary mapping atom indices to fragment IDs
    """
    # Perform BRICS decomposition
    try:
        # Get BRICS fragments as SMILES strings (not Mol objects)
        brics_fragments = BRICS.BRICSDecompose(mol, returnMols=False)
        fragment_smiles = list(brics_fragments)
    except:
        # Fallback if BRICS fails - use whole molecule as single fragment
        fragment_smiles = [Chem.MolToSmiles(mol)]
    
    fragments = {}
    atom_to_fragment = {}
    
    # If only one fragment (no decomposition), use the whole molecule
    if len(fragment_smiles) <= 1:
        # Create single fragment containing all atoms
        fragments[0] = {
            'name': 'Whole_Molecule',
            'description': 'Cannot be decomposed by BRICS',
            'atoms': list(range(mol.GetNumAtoms())),
            'type': 'complete',
            'smiles': Chem.MolToSmiles(mol)
        }
        # Map all atoms to fragment 0
        for atom_idx in range(mol.GetNumAtoms()):
            atom_to_fragment[atom_idx] = 0
        return fragments, atom_to_fragment
    
    # For multiple fragments, process each BRICS fragment
    fragment_id = 0
    
    for frag_smiles in fragment_smiles:
        # Clean up BRICS notation by removing dummy atoms marked with *
        clean_smiles = frag_smiles.replace('[*]', '')
        if clean_smiles:  # Only process non-empty fragments
            fragments[fragment_id] = {
                'name': f'BRICS_Fragment_{fragment_id}',
                'description': f'BRICS fragment: {clean_smiles}',
                'atoms': [],  # Will be filled later
                'type': 'brics_fragment',
                'smiles': clean_smiles
            }
            fragment_id += 1
    
    # Create simplified atom-to-fragment mapping
    # Distribute atoms evenly across fragments
    if len(fragments) > 0:
        atoms_per_fragment = max(1, mol.GetNumAtoms() // len(fragments))
        
        for atom_idx in range(mol.GetNumAtoms()):
            # Assign atom to fragment, ensuring we don't exceed fragment count
            frag_id = min(atom_idx // atoms_per_fragment, len(fragments) - 1)
            atom_to_fragment[atom_idx] = frag_id
            fragments[frag_id]['atoms'].append(atom_idx)
    
    return fragments, atom_to_fragment

def improved_brics_fragmentation(mol: Chem.rdchem.Mol) -> tuple[dict, dict]:
    """
    Improved BRICS fragmentation with better error handling.
    
    Args:
        mol: RDKit molecule object to fragment
        
    Returns:
        tuple containing:
            - fragments: Dictionary mapping fragment IDs to fragment information
            - atom_to_fragment: Dictionary mapping atom indices to fragment IDs
    """
    try:
        # Attempt BRICS decomposition
        brics_fragments = BRICS.BRICSDecompose(mol, returnMols=False)
        fragment_smiles = list(brics_fragments)
        
        # Clean fragments by removing dummy atoms and empty fragments
        clean_fragments = []
        for smiles in fragment_smiles:
            clean_smiles = smiles.replace('[*]', '').strip()
            if clean_smiles and clean_smiles != '':
                clean_fragments.append(clean_smiles)
        
        # If no meaningful decomposition, use fallback
        if len(clean_fragments) <= 1:
            return brics_fragmentation_fallback(mol)
        
        # Create fragment dictionary entries
        fragments = {}
        atom_to_fragment = {}
        
        for i, clean_smiles in enumerate(clean_fragments):
            fragments[i] = {
                'name': f'BRICS_{i}',
                'description': f'BRICS: {clean_smiles}',
                'atoms': [],
                'type': 'brics_fragment',
                'smiles': clean_smiles
            }
        
        # Distribute atoms evenly across fragments
        atoms_per_fragment = max(1, mol.GetNumAtoms() // len(fragments))
        for atom_idx in range(mol.GetNumAtoms()):
            frag_id = min(atom_idx // atoms_per_fragment, len(fragments) - 1)
            atom_to_fragment[atom_idx] = frag_id
            fragments[frag_id]['atoms'].append(atom_idx)
        
        return fragments, atom_to_fragment
        
    except Exception as e:
        print(f"BRICS failed: {e}. Using fallback method.")
        return brics_fragmentation_fallback(mol)

def brics_fragmentation_fallback(mol: Chem.rdchem.Mol) -> tuple[dict, dict]:
    """
    Fallback method when BRICS fails - treats molecule as single fragment.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        tuple containing:
            - fragments: Dictionary with single fragment containing whole molecule
            - atom_to_fragment: Dictionary mapping all atoms to fragment 0
    """
    # Create single fragment containing all atoms
    fragments = {0: {
        'name': 'Whole_Molecule',
        'description': 'BRICS decomposition not possible',
        'atoms': list(range(mol.GetNumAtoms())),
        'type': 'complete',
        'smiles': Chem.MolToSmiles(mol)
    }}
    
    # Map all atoms to fragment 0
    atom_to_fragment = {i: 0 for i in range(mol.GetNumAtoms())}
    return fragments, atom_to_fragment

def create_brics_graph(mol: Chem.rdchem.Mol, 
                      fragments: dict, 
                      atom_to_fragment: dict) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Create a graph representation from BRICS fragments.
    
    Args:
        mol: RDKit molecule object
        fragments: Dictionary of fragment information
        atom_to_fragment: Mapping of atoms to fragments
        
    Returns:
        tuple containing:
            - node_features: numpy array of fragment features
            - adjacency: numpy array of fragment connectivity
            - edge_features: list of edge features (empty in current implementation)
            - node_labels: list of fragment names
    """
    n_fragments = len(fragments)
    
    # Create node features and labels
    node_features = []
    node_labels = []
    
    for frag_id in range(n_fragments):
        if frag_id in fragments:
            frag = fragments[frag_id]
            
            # Extract features for each fragment
            features = [
                len(frag['atoms']),  # Size of fragment
                1 if frag['type'] == 'brics_fragment' else 0,  # Fragment type flag
                len(frag['smiles']) if 'smiles' in frag else 0,  # Complexity measure
            ]
            
            node_features.append(features)
            node_labels.append(frag['name'])
    
    # Convert to numpy array with proper shape handling
    node_features = np.array(node_features) if node_features else np.array([]).reshape(0, 3)
    
    # Create adjacency matrix
    adjacency = np.zeros((n_fragments, n_fragments)) if n_fragments > 0 else np.array([])
    
    # Build connectivity between fragments
    if n_fragments > 1:
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()
            
            # Check if atoms belong to different fragments
            if atom1 in atom_to_fragment and atom2 in atom_to_fragment:
                frag1 = atom_to_fragment[atom1]
                frag2 = atom_to_fragment[atom2]
                
                # Create undirected edge between different fragments
                if frag1 != frag2:
                    adjacency[frag1, frag2] = 1
                    adjacency[frag2, frag1] = 1
    
    return node_features, adjacency, [], node_labels

# Test BRICS fragmentation with improved function
test_molecules = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
}

print("BRICS Fragmentation Results:")
print("=" * 50)

for name, smiles in test_molecules.items():
    mol = Chem.MolFromSmiles(smiles)
    fragments, atom_to_fragment = improved_brics_fragmentation(mol)
    
    print(f"\n{name}:")
    print(f"Original SMILES: {smiles}")
    print(f"Number of BRICS fragments: {len(fragments)}")
    
    for frag_id, frag in fragments.items():
        print(f"  Fragment {frag_id}: {frag['description']}")

# + id="visualize_brics" colab={"base_uri": "https://localhost:8080/", "height": 800"
def visualize_brics_fragmentation(mol: Chem.rdchem.Mol, 
                                fragments: dict, 
                                atom_to_fragment: dict, 
                                molecule_name: str = "Molecule") -> None:
    """
    Visualize BRICS-based molecular fragmentation with multiple views.
    
    Args:
        mol: RDKit molecule object to visualize
        fragments: Dictionary mapping fragment IDs to fragment information
        atom_to_fragment: Dictionary mapping atom indices to fragment IDs
        molecule_name: Name of the molecule for display purposes
        
    Returns:
        None - displays matplotlib visualization
        
    The visualization includes 4 subplots:
    1. Original molecule structure
    2. BRICS fragment SMILES strings
    3. Fragment connectivity graph 
    4. Analysis of fragmentation results
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Original molecule visualization
    img = Draw.MolToImage(mol, size=(350, 350))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"Original {molecule_name}", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. BRICS fragments as text display
    ax = axes[0, 1]
    ax.axis('off')
    
    # Build fragment text display
    fragment_text = "BRICS Fragments:\n\n"
    for frag_id, frag in fragments.items():
        if 'smiles' in frag and frag['smiles'] and frag['smiles'].strip():
            fragment_text += f"{frag_id}: {frag['smiles']}\n"
        else:
            fragment_text += f"{frag_id}: {frag['name']}\n"
    
    # Display fragment text in a box
    ax.text(0.1, 0.9, fragment_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    ax.set_title("BRICS Fragment SMILES", fontsize=12, fontweight='bold')
    
    # 3. Fragment connectivity graph
    ax = axes[1, 0]
    
    if len(fragments) > 1:
        # Get graph representation data
        node_features, adjacency, _, node_labels = create_brics_graph(
            mol, fragments, atom_to_fragment
        )
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes to graph
        for frag_id in fragments.keys():
            G.add_node(frag_id)
        
        # Add edges based on adjacency matrix
        for i in range(len(fragments)):
            for j in range(i+1, len(fragments)):
                if len(adjacency) > 0 and adjacency[i, j] == 1:
                    G.add_edge(i, j)
        
        # Draw graph if it has nodes
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, seed=42)  # Deterministic layout
            nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=800, ax=ax)
            nx.draw_networkx_edges(G, pos, ax=ax)
            
            # Add fragment labels
            labels = {frag_id: f"F{frag_id}" for frag_id in fragments.keys()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
        
        ax.set_title("BRICS Fragment Graph", fontsize=12, fontweight='bold')
    else:
        # Handle single fragment case
        ax.text(0.5, 0.5, "Single fragment\n(no decomposition)", ha='center', va='center')
        ax.set_title("BRICS Fragment Graph", fontsize=12, fontweight='bold')
    
    ax.axis('off')
    
    # 4. Analysis summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Build analysis text
    analysis_text = "BRICS Analysis:\n\n"
    analysis_text += f"Total fragments: {len(fragments)}\n"
    analysis_text += f"Original atoms: {mol.GetNumAtoms()}\n"
    
    if len(fragments) > 1:
        # Calculate and display fragment statistics
        avg_size = sum(len(frag['atoms']) for frag in fragments.values()) / len(fragments)
        analysis_text += f"Avg fragment size: {avg_size:.1f}\n"
        analysis_text += "\nFragment types:\n"
        for frag_id, frag in fragments.items():
            analysis_text += f"  {frag_id}: {frag['type']}\n"
    else:
        analysis_text += "\nNo decomposition possible\n(molecule too simple or\nno BRICS-breakable bonds)"
    
    # Display analysis text in a box
    ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan"))
    ax.set_title("BRICS Analysis", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Test visualization on example molecules
for name, smiles in test_molecules.items():
    mol = Chem.MolFromSmiles(smiles)
    fragments, atom_to_fragment = improved_brics_fragmentation(mol)
    visualize_brics_fragmentation(mol, fragments, atom_to_fragment, name)

# + [markdown] id="comparison_section"
# ## 7. Comparing Fragment-Based vs Atom-Based Representations <a name="comparing-fragment-based-vs-atom-based-representations"></a>
#
# Now let's directly compare the different representation approaches on the same molecules to understand their trade-offs:

# + id="comparison_implementation" colab={"base_uri": "https://localhost:8080/", "height": 1000"
def compare_representations(smiles: str, molecule_name: str) -> dict:
    """
    Compare atom-based vs different fragment-based molecular representations.

    Args:
        smiles: SMILES string representation of the molecule
        molecule_name: Name of the molecule for display purposes

    Returns:
        dict: Dictionary containing counts of different representations with keys:
            - atom_count: Number of atoms
            - fg_count: Number of functional group fragments 
            - ring_count: Number of ring system fragments
            - brics_count: Number of BRICS fragments
    """
    # Convert SMILES to RDKit molecule and add hydrogens
    mol = Chem.MolFromSmiles(smiles)
    mol_with_h = Chem.AddHs(mol)
    
    # Get all different molecular representations
    # 1. Atom-based representation (simplified from Part 1)
    n_atoms = mol_with_h.GetNumAtoms()
    
    # 2. Functional group-based fragmentation
    fg_fragments, fg_atom_to_fragment = identify_functional_groups(mol)
    
    # 3. Ring system-based fragmentation
    ring_fragments, ring_atom_to_fragment = identify_ring_systems(mol)
    
    # 4. BRICS-based fragmentation
    brics_fragments, brics_atom_to_fragment = improved_brics_fragmentation(mol)
    
    # Create visualization comparing all representations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot original molecule structure
    img = Draw.MolToImage(mol, size=(300, 300))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"{molecule_name}\n(Original Structure)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Create comparison table
    ax = axes[0, 1]
    ax.axis('off')
    
    # Define comparison metrics for each representation
    comparison_data = {
        'Representation': ['Atom-based', 'Functional Groups', 'Ring Systems', 'BRICS'],
        'Nodes': [n_atoms, len(fg_fragments), len(ring_fragments), len(brics_fragments)],
        'Granularity': ['Fine', 'Medium', 'Medium', 'Coarse'],
        'Chemical Meaning': ['Low', 'High', 'High', 'Medium']
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Format table text
    table_text = "Representation Comparison:\n\n"
    for _, row in df.iterrows():
        table_text += f"{row['Representation']:15s}: {row['Nodes']:2d} nodes\n"
        table_text += f"{'':15s}  {row['Granularity']}, {row['Chemical Meaning']} meaning\n\n"
    
    # Display comparison table
    ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    ax.set_title("Quantitative Comparison", fontsize=12, fontweight='bold')
    
    # Create bar plot comparing graph complexity
    ax = axes[0, 2]
    methods = ['Atom-based', 'Functional\nGroups', 'Ring\nSystems', 'BRICS']
    node_counts = [n_atoms, len(fg_fragments), len(ring_fragments), len(brics_fragments)]
    colors = ['lightblue', 'lightcoral', 'lightsalmon', 'lightgreen']
    
    bars = ax.bar(methods, node_counts, color=colors)
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Graph Complexity', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for bar, count in zip(bars, node_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    # Visualize individual fragment representations
    representations = [
        ("Functional Groups", fg_fragments, 'lightcoral'),
        ("Ring Systems", ring_fragments, 'lightsalmon'), 
        ("BRICS", brics_fragments, 'lightgreen')
    ]
    
    # Create network graphs for each fragmentation method
    for i, (rep_name, fragments, color) in enumerate(representations):
        ax = axes[1, i]
        
        if len(fragments) > 1:
            # Create networkx graph
            G = nx.Graph()
            for frag_id in fragments.keys():
                G.add_node(frag_id)
            
            # Add edges based on fragment adjacency
            if rep_name == "Functional Groups":
                _, adjacency, _, _ = create_functional_group_graph(mol, fragments, fg_atom_to_fragment)
            elif rep_name == "Ring Systems":
                _, adjacency, _, _ = create_ring_system_graph(mol, fragments, ring_atom_to_fragment)
            else:  # BRICS
                _, adjacency, _, _ = create_brics_graph(mol, fragments, brics_atom_to_fragment)
            
            # Add edges between adjacent fragments
            for i in range(len(fragments)):
                for j in range(i+1, len(fragments)):
                    if len(adjacency) > 0 and i < len(adjacency) and j < len(adjacency[0]) and adjacency[i, j] == 1:
                        G.add_edge(i, j)
            
            # Draw network graph if nodes exist
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, seed=42)
                nx.draw_networkx_nodes(G, pos, node_color=color, node_size=600, ax=ax)
                nx.draw_networkx_edges(G, pos, ax=ax)
                
                # Create and add node labels
                labels = {}
                for frag_id, frag in fragments.items():
                    if rep_name == "BRICS" and 'smiles' in frag:
                        labels[frag_id] = frag['smiles'][:6] + "..." if len(frag['smiles']) > 6 else frag['smiles']
                    else:
                        labels[frag_id] = frag['name'][:8]
                
                nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        else:
            # Handle single fragment case
            ax.text(0.5, 0.5, f"Single {rep_name.lower()}\nfragment", ha='center', va='center')
        
        ax.set_title(f"{rep_name}\n({len(fragments)} fragments)", fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Return dictionary with fragment counts
    return {
        'atom_count': n_atoms,
        'fg_count': len(fg_fragments),
        'ring_count': len(ring_fragments), 
        'brics_count': len(brics_fragments)
    }

# Test molecules for comparison
test_molecules_comparison = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Morphine": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
}

# Compare representations for each test molecule
comparison_results = {}
for name, smiles in test_molecules_comparison.items():
    print(f"\nAnalyzing {name}...")
    comparison_results[name] = compare_representations(smiles, name)

# Print summary comparison table
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)

summary_df = pd.DataFrame(comparison_results).T
summary_df.columns = ['Atoms', 'Functional Groups', 'Ring Systems', 'BRICS']
print(summary_df)


# + [markdown] id="pyg_implementation"
# ## 8. PyTorch Geometric Implementation for Fragment Graphs <a name="pytorch-geometric-implementation-for-fragment-graphs"></a>
#
# Now let's implement fragment-based graphs in PyTorch Geometric format, extending our work from Part 1:

# + id="pyg_fragment_implementation"
def fragment_to_pyg(mol: Chem.rdchem.Mol, fragmentation_method: str = 'functional_groups') -> Data:
    """
    Convert a molecule to PyTorch Geometric Data object using fragment-based representation.
    
    Args:
        mol: RDKit molecule object to be fragmented
        fragmentation_method: Method to use for fragmenting the molecule. 
                            Options are 'functional_groups', 'ring_systems', or 'brics'
        
    Returns:
        torch_geometric.data.Data: PyG Data object containing:
            - x: Node features tensor
            - edge_index: Edge connectivity tensor 
            - edge_attr: Edge features tensor
            - num_nodes: Number of fragment nodes
            - fragmentation_method: Method used for fragmentation
            - fragment_labels: Labels for each fragment
            - smiles: SMILES string of input molecule
            
    Raises:
        ValueError: If an invalid fragmentation method is provided
    """
    
    # Choose fragmentation method and get fragments + atom mappings
    if fragmentation_method == 'functional_groups':
        fragments, atom_to_fragment = identify_functional_groups(mol)
        # Create graph representation using functional group fragments
        node_features, adjacency, edge_features, node_labels = create_functional_group_graph(
            mol, fragments, atom_to_fragment
        )
    elif fragmentation_method == 'ring_systems':
        fragments, atom_to_fragment = identify_ring_systems(mol)
        # Create graph representation using ring system fragments
        node_features, adjacency, edge_features, node_labels = create_ring_system_graph(
            mol, fragments, atom_to_fragment
        )
    elif fragmentation_method == 'brics':
        fragments, atom_to_fragment = brics_fragmentation(mol)
        # Create graph representation using BRICS fragments
        node_features, adjacency, edge_features, node_labels = create_brics_graph(
            mol, fragments, atom_to_fragment
        )
    else:
        raise ValueError(f"Unknown fragmentation method: {fragmentation_method}")
    
    # Convert node features to PyTorch tensor
    if len(node_features) > 0:
        x = torch.tensor(node_features, dtype=torch.float)
    else:
        # Create empty tensor with expected feature dimension if no nodes
        x = torch.empty((0, 3), dtype=torch.float)
    
    # Create edge index tensor from adjacency matrix
    edge_indices = []
    if len(adjacency) > 0:
        for i in range(len(adjacency)):
            for j in range(len(adjacency)):
                if adjacency[i, j] == 1:
                    edge_indices.append([i, j])
    
    # Convert edge indices to PyTorch tensor
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        # Create empty edge index tensor if no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Create edge feature tensor (currently using simple ones)
    if len(edge_indices) > 0:
        edge_attr = torch.ones((len(edge_indices), 1), dtype=torch.float)
    else:
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    
    # Create and return PyG Data object with all graph information
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(fragments),
        fragmentation_method=fragmentation_method,
        fragment_labels=node_labels,
        smiles=Chem.MolToSmiles(mol)
    )
    
    return data

def compare_pyg_representations(smiles: str, molecule_name: str) -> None:
    """
    Compare PyG representations using different fragmentation methods.
    
    Args:
        smiles: SMILES string of molecule to analyze
        molecule_name: Name of the molecule for display purposes
        
    Prints:
        Detailed comparison of different fragmentation methods including:
        - Number of nodes
        - Number of edges
        - Node feature dimensions
        - Edge feature dimensions
        - Fragment labels
    """
    mol = Chem.MolFromSmiles(smiles)
    
    print(f"\nPyTorch Geometric Representation Comparison: {molecule_name}")
    print("=" * 60)
    
    methods = ['functional_groups', 'ring_systems', 'brics']
    
    for method in methods:
        try:
            # Generate PyG representation for each method
            data = fragment_to_pyg(mol, method)
            print(f"\n{method.replace('_', ' ').title()}:")
            print(f"  Nodes: {data.num_nodes}")
            print(f"  Edges: {data.edge_index.shape[1]}")
            print(f"  Node features shape: {data.x.shape}")
            print(f"  Edge features shape: {data.edge_attr.shape}")
            if hasattr(data, 'fragment_labels'):
                print(f"  Fragment labels: {data.fragment_labels}")
            
        except Exception as e:
            print(f"\n{method.replace('_', ' ').title()}: Error - {str(e)}")

# Test PyG implementations with example molecules
test_molecules_pyg = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
}

for name, smiles in test_molecules_pyg.items():
    compare_pyg_representations(smiles, name)

# Create and analyze a specific example
print("\n" + "="*60)
print("DETAILED PyG ANALYSIS: Aspirin with Functional Groups")
print("="*60)

aspirin_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
aspirin_fg_data = fragment_to_pyg(aspirin_mol, 'functional_groups')

print("\nAspirin Fragment Graph (PyG Format):")
print(aspirin_fg_data)

print("\nNode Features:")
print(aspirin_fg_data.x)
print("\nEdge Index:")
print(aspirin_fg_data.edge_index)

# + [markdown] id="fragment_properties"
# ## 9. Fragment Property Prediction <a name="fragment-property-prediction"></a>
#
# One key advantage of fragment-based representations is the ability to predict and interpret molecular properties at the fragment level:

# + id="fragment_properties_implementation" colab={"base_uri": "https://localhost:8080/", "height": 800"
def fragment_to_pyg(mol: Chem.rdchem.Mol, fragmentation_method: str = 'functional_groups') -> Data:
    """
    Convert a molecule to PyTorch Geometric Data object using fragment-based representation.
    
    Args:
        mol: RDKit molecule object to be fragmented and converted
        fragmentation_method: Method to use for fragmenting the molecule. Options are:
            - 'functional_groups': Fragment by functional groups like -OH, -COOH etc.
            - 'ring_systems': Fragment by ring systems in the molecule
            - 'brics': Use BRICS fragmentation rules
        
    Returns:
        torch_geometric.data.Data: PyG Data object containing:
            - x: Node features tensor
            - edge_index: Edge connectivity tensor 
            - edge_attr: Edge features tensor
            - num_nodes: Number of fragment nodes
            - fragmentation_method: Method used for fragmentation
            - fragment_labels: Labels for each fragment
            - smiles: SMILES string of input molecule
            
    Raises:
        ValueError: If an invalid fragmentation method is specified
    """
    
    # Choose fragmentation method and get fragments
    if fragmentation_method == 'functional_groups':
        fragments, atom_to_fragment = identify_functional_groups(mol)
        node_features, adjacency, edge_features, node_labels = create_functional_group_graph(
            mol, fragments, atom_to_fragment
        )
    elif fragmentation_method == 'ring_systems':
        fragments, atom_to_fragment = identify_ring_systems(mol)
        node_features, adjacency, edge_features, node_labels = create_ring_system_graph(
            mol, fragments, atom_to_fragment
        )
    elif fragmentation_method == 'brics':
        fragments, atom_to_fragment = improved_brics_fragmentation(mol)
        node_features, adjacency, edge_features, node_labels = create_brics_graph(
            mol, fragments, atom_to_fragment
        )
    else:
        raise ValueError(f"Unknown fragmentation method: {fragmentation_method}")
    
    # Convert node features to PyTorch tensor
    if len(node_features) > 0:
        x = torch.tensor(node_features, dtype=torch.float)
    else:
        # Create empty tensor with 3 feature dimensions if no nodes
        x = torch.empty((0, 3), dtype=torch.float)
    
    # Create edge index tensor from adjacency matrix
    edge_indices = []
    if len(adjacency) > 0:
        for i in range(len(adjacency)):
            for j in range(len(adjacency)):
                if adjacency[i, j] == 1:
                    edge_indices.append([i, j])
    
    # Convert edge indices to PyTorch tensor
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Create edge feature tensor (currently just ones)
    if len(edge_indices) > 0:
        edge_attr = torch.ones((len(edge_indices), 1), dtype=torch.float)
    else:
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    
    # Create and return PyG Data object with all attributes
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(fragments),
        fragmentation_method=fragmentation_method,
        fragment_labels=node_labels,
        smiles=Chem.MolToSmiles(mol)
    )
    
    return data

def compare_pyg_representations(smiles: str, molecule_name: str) -> None:
    """
    Compare PyG representations using different fragmentation methods.
    
    Args:
        smiles: SMILES string of the molecule to analyze
        molecule_name: Name of the molecule for display purposes
        
    Prints:
        Detailed comparison of the PyG representations including:
        - Number of nodes
        - Number of edges
        - Node feature dimensions
        - Edge feature dimensions
        - Fragment labels
    """
    mol = Chem.MolFromSmiles(smiles)
    
    print(f"\nPyTorch Geometric Representation Comparison: {molecule_name}")
    print("=" * 60)
    
    # Try each fragmentation method
    methods = ['functional_groups', 'ring_systems', 'brics']
    for method in methods:
        try:
            # Generate PyG representation for each method
            data = fragment_to_pyg(mol, method)
            print(f"\n{method.replace('_', ' ').title()}:")
            print(f"  Nodes: {data.num_nodes}")
            print(f"  Edges: {data.edge_index.shape[1]}")
            print(f"  Node features shape: {data.x.shape}")
            print(f"  Edge features shape: {data.edge_attr.shape}")
            if hasattr(data, 'fragment_labels'):
                print(f"  Fragment labels: {data.fragment_labels}")
            
        except Exception as e:
            print(f"\n{method.replace('_', ' ').title()}: Error - {str(e)}")

# Test PyG implementations with example molecules
test_molecules_pyg = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
}

for name, smiles in test_molecules_pyg.items():
    compare_pyg_representations(smiles, name)

# Create and analyze a specific example
print("\n" + "="*60)
print("DETAILED PyG ANALYSIS: Aspirin with Functional Groups")
print("="*60)

aspirin_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
aspirin_fg_data = fragment_to_pyg(aspirin_mol, 'functional_groups')

print("\nAspirin Fragment Graph (PyG Format):")
print(aspirin_fg_data)

print("\nNode Features:")
print(aspirin_fg_data.x)
print("\nEdge Index:")
print(aspirin_fg_data.edge_index)

# + [markdown] id="advanced_visualization"
# ## 10. Advanced Visualization Techniques <a name="advanced-visualization-techniques"></a>
#
# Let's create some advanced visualizations to better understand fragment-based representations:

# + id="advanced_viz_implementation" colab={"base_uri": "https://localhost:8080/", "height": 1200"
from typing import Dict

def create_comprehensive_fragment_analysis(smiles: str, molecule_name: str) -> None:
    """
    Create a comprehensive analysis comparing all fragmentation methods.
    
    Args:
        smiles (str): SMILES string representation of the molecule
        molecule_name (str): Name of the molecule for display purposes
        
    Returns:
        None: Displays visualization plots
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # Get fragmentations using different methods
    fg_fragments, fg_atom_to_fragment = identify_functional_groups(mol)
    ring_fragments, ring_atom_to_fragment = identify_ring_systems(mol)
    brics_fragments, brics_atom_to_fragment = improved_brics_fragmentation(mol)
    
    # Analyze chemical properties for each fragment type
    fg_properties = analyze_fragment_properties(mol, fg_fragments, 'functional_groups')
    ring_properties = analyze_fragment_properties(mol, ring_fragments, 'ring_systems')
    brics_properties = analyze_fragment_properties(mol, brics_fragments, 'brics')
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Original molecule visualization (large)
    ax1 = plt.subplot(3, 4, (1, 2))
    img = Draw.MolToImage(mol, size=(400, 400))
    ax1.imshow(img)
    ax1.set_title(f"{molecule_name}\n({mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds)", 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: Radar chart comparing methods
    ax2 = plt.subplot(3, 4, 3, projection='polar')
    methods = ['Functional\nGroups', 'Ring\nSystems', 'BRICS']
    node_counts = [len(fg_fragments), len(ring_fragments), len(brics_fragments)]
    
    # Normalize counts for radar chart
    max_nodes = max(node_counts) if max(node_counts) > 0 else 1
    normalized_counts = [count / max_nodes for count in node_counts]
    
    # Create circular plot points
    angles = np.linspace(0, 2 * np.pi, len(methods), endpoint=False).tolist()
    normalized_counts += normalized_counts[:1]  # Complete the circle
    angles += angles[:1]
    
    ax2.plot(angles, normalized_counts, 'o-', linewidth=2, color='blue')
    ax2.fill(angles, normalized_counts, alpha=0.25, color='blue')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(methods)
    ax2.set_title("Fragmentation\nComparison", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    
    # Plot 3: Bar chart of fragment complexity
    ax3 = plt.subplot(3, 4, 4)
    methods_short = ['FG', 'Ring', 'BRICS']
    complexities = [len(fg_fragments), len(ring_fragments), len(brics_fragments)]
    
    bars = ax3.bar(methods_short, complexities, color=['lightcoral', 'lightsalmon', 'lightgreen'])
    ax3.set_ylabel('Number of Fragments')
    ax3.set_title('Graph Complexity', fontweight='bold')
    
    # Add value labels to bars
    for bar, complexity in zip(bars, complexities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{complexity}', ha='center', va='bottom')
    
    # Plots 4-6: Individual fragment network graphs
    fragment_data = [
        (fg_fragments, "Functional Groups", 'lightcoral', fg_atom_to_fragment, create_functional_group_graph),
        (ring_fragments, "Ring Systems", 'lightsalmon', ring_atom_to_fragment, create_ring_system_graph),
        (brics_fragments, "BRICS", 'lightgreen', brics_atom_to_fragment, create_brics_graph)
    ]
    
    positions = [(2, 1), (2, 2), (2, 3)]
    
    for (fragments, method_name, color, atom_to_fragment, graph_func), (row, col) in zip(fragment_data, positions):
        ax = plt.subplot(3, 4, row * 4 + col)
        
        if len(fragments) > 1:
            # Create networkx graph
            G = nx.Graph()
            for frag_id in fragments.keys():
                G.add_node(frag_id)
            
            # Get adjacency matrix and add edges
            _, adjacency, _, _ = graph_func(mol, fragments, atom_to_fragment)
            
            for i in range(len(fragments)):
                for j in range(i+1, len(fragments)):
                    if len(adjacency) > 0 and i < len(adjacency) and j < len(adjacency[0]) and adjacency[i, j] == 1:
                        G.add_edge(i, j)
            
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, seed=42)
                nx.draw_networkx_nodes(G, pos, node_color=color, node_size=400, ax=ax)
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6)
                
                # Add fragment size labels
                labels = {frag_id: str(len(frag['atoms'])) if method_name in ["Functional Groups", "Ring Systems"]
                         else str(frag_id) for frag_id, frag in fragments.items()}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        else:
            ax.text(0.5, 0.5, f"Single\nfragment", ha='center', va='center')
        
        ax.set_title(f"{method_name}\n({len(fragments)} fragments)", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Plot 7: Fragment size distribution violin plot
    ax7 = plt.subplot(3, 4, (3, 1))
    all_properties = [fg_properties, ring_properties, brics_properties]
    method_labels = ['FG', 'Ring', 'BRICS']
    
    fragment_sizes = []
    method_ids = []
    
    for i, (properties, label) in enumerate(zip(all_properties, method_labels)):
        if properties:
            sizes = [prop['size'] for prop in properties.values()]
            fragment_sizes.extend(sizes)
            method_ids.extend([i] * len(sizes))
    
    if fragment_sizes:
        parts = ax7.violinplot([
            [prop['size'] for prop in props.values()] if props else [0]
            for props in all_properties
        ], positions=[1, 2, 3], showmeans=True)
        
        colors = ['lightcoral', 'lightsalmon', 'lightgreen']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax7.set_xticks([1, 2, 3])
        ax7.set_xticklabels(method_labels)
        ax7.set_ylabel('Fragment Size')
        ax7.set_title('Size Distribution', fontweight='bold')
    
    # Plot 8: Chemical property heatmap
    ax8 = plt.subplot(3, 4, (3, 2))
    heatmap_data = []
    heatmap_labels = []
    
    for properties, method in [(fg_properties, 'FG'), (ring_properties, 'Ring'), (brics_properties, 'BRICS')]:
        if properties:
            for frag_id, props in properties.items():
                heatmap_data.append([
                    props['size'],
                    props['polarity_score'],
                    props['aromaticity_ratio'] * 10,  # Scale for visibility
                    props['carbon_count']
                ])
                heatmap_labels.append(f"{method}_{frag_id}")
    
    if heatmap_data:
        heatmap_array = np.array(heatmap_data)
        
        # Normalize each property column
        for col in range(heatmap_array.shape[1]):
            col_max = heatmap_array[:, col].max()
            if col_max > 0:
                heatmap_array[:, col] = heatmap_array[:, col] / col_max
        
        im = ax8.imshow(heatmap_array.T, cmap='viridis', aspect='auto')
        ax8.set_xticks(range(len(heatmap_labels)))
        ax8.set_xticklabels(heatmap_labels, rotation=45, ha='right')
        ax8.set_yticks(range(4))
        ax8.set_yticklabels(['Size', 'Polarity', 'Aromatic', 'Carbons'])
        ax8.set_title('Fragment Properties\n(normalized)', fontweight='bold')
        
        plt.colorbar(im, ax=ax8, shrink=0.6)
    
    plt.tight_layout()
    plt.show()

def analyze_fragment_properties(mol: Chem.Mol, fragments: Dict, method: str) -> Dict:
    """
    Analyze chemical properties of fragments.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        fragments (Dict): Dictionary of fragments
        method (str): Fragmentation method name
        
    Returns:
        Dict: Properties for each fragment
    """
    properties = {}
    for frag_id, fragment in fragments.items():
        # Create submolecule from fragment atoms
        atom_indices = list(fragment['atoms'])
        submol = Chem.PathToSubmol(mol, atom_indices)
        
        # Calculate properties
        properties[frag_id] = {
            'size': len(atom_indices),
            'polarity_score': calculate_polarity_score(submol),
            'aromaticity_ratio': calculate_aromaticity_ratio(submol),
            'carbon_count': count_carbons(submol)
        }
    return properties

def calculate_polarity_score(mol: Chem.Mol) -> float:
    """
    Calculate polarity score based on electronegative atoms.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        
    Returns:
        float: Polarity score
    """
    polar_atoms = ['N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P']
    total_atoms = mol.GetNumAtoms()
    polar_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in polar_atoms)
    return polar_count / total_atoms if total_atoms > 0 else 0

def calculate_aromaticity_ratio(mol: Chem.Mol) -> float:
    """
    Calculate ratio of aromatic atoms to total atoms.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        
    Returns:
        float: Aromaticity ratio
    """
    total_atoms = mol.GetNumAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return aromatic_atoms / total_atoms if total_atoms > 0 else 0

def count_carbons(mol: Chem.Mol) -> int:
    """
    Count number of carbon atoms in molecule.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        
    Returns:
        int: Number of carbon atoms
    """
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

# Test the visualization with example molecules
interesting_molecules = {
    "Morphine": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "Penicillin G": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
}

for name, smiles in interesting_molecules.items():
    print(f"\nComprehensive Analysis: {name}")
    create_comprehensive_fragment_analysis(smiles, name)

# + [markdown] id="conclusion"
# ## 11. Conclusion <a name="conclusion"></a>
#
# ### Summary of Fragment-Based Molecular Representations
#
# In this tutorial, we explored how to represent molecules as graphs where **molecular fragments** serve as nodes instead of individual atoms. This approach offers several key advantages:
#
# #### **Key Concepts Learned:**
#
# 1. **Multiple Fragmentation Strategies**: We implemented three different approaches:
#    - **Functional Group-Based**: Identifies chemically reactive groups (alcohols, carboxylic acids, etc.)
#    - **Ring System-Based**: Separates aromatic/aliphatic rings from linear chains
#    - **BRICS Fragmentation**: Systematic decomposition based on synthetic chemistry
#
# 2. **Flexibility of Graph Representations**: The same molecule can be represented at different levels of granularity:
#    - Fine-grained: Atom-based graphs (many nodes, precise structure)
#    - Medium-grained: Fragment-based graphs (fewer nodes, chemical meaning)
#    - Coarse-grained: Whole-molecule features (traditional descriptors)
#
# 3. **Chemical Interpretability**: Fragment-based representations make it easier to:
#    - Understand which chemical features drive predictions
#    - Relate model outputs to known structure-activity relationships
#    - Design new molecules based on successful fragments
#
# 4. **PyTorch Geometric Integration**: Fragment graphs can be seamlessly used with modern GNN frameworks
#
# #### **When to Use Fragment-Based vs Atom-Based Representations:**
#
# **Fragment-Based is Better For:**
# - Large molecules (proteins, polymers)
# - Drug discovery (identifying pharmacophores)
# - Interpretable models
# - Prior knowledge integration
# - Computational efficiency
#
# **Atom-Based is Better For:**
# - Small molecules
# - Precise property prediction
# - Quantum chemical calculations
# - When fine structural details matter
#
# #### **Practical Applications:**
#
# 1. **Drug Discovery**: Fragment-based graphs can identify important pharmacophores and guide medicinal chemistry
# 2. **Material Science**: Ring systems and polymer units can be treated as functional fragments
# 3. **Chemical Synthesis**: BRICS fragmentation provides realistic synthetic building blocks
# 4. **Toxicity Prediction**: Known toxic substructures can be encoded as fragment features
#
# ### **Next Steps:**
#
# The flexibility demonstrated here shows that graph neural networks can be adapted to match the **level of chemical understanding** appropriate for your specific problem. Consider:
#
# - **Hierarchical representations** that combine multiple granularities
# - **Domain-specific fragmentation** for your particular application
# - **Learned fragmentation** where the model discovers optimal fragments
# - **Multi-scale models** that process both atom and fragment information
#
# Fragment-based molecular graphs represent a powerful bridge between traditional chemical knowledge and modern machine learning, enabling more interpretable and chemically-informed AI models for molecular science.

# + [markdown] id="exercise_final"
# ### ✅ Final Exercise: Design Your Own Fragmentation Method
#
# **Challenge**: Create a custom fragmentation method for a specific chemical application:
#
# 1. **Choose a domain**: 
#    - Drug discovery (focus on pharmacophores)
#    - Material science (focus on polymer units)
#    - Catalysis (focus on active sites)
#    - Your own area of interest
#
# 2. **Design fragmentation rules**: 
#    - What makes a meaningful fragment in your domain?
#    - How should fragments be connected?
#    - What features are important for each fragment?
#
# 3. **Implement and test**:
#    - Code your fragmentation method using RDKit
#    - Test on relevant molecules from your domain
#    - Compare with the methods we've implemented
#
# 4. **Evaluate**:
#    - Does your method create chemically meaningful fragments?
#    - How does the graph complexity compare to other methods?
#    - What insights does your representation provide?
#
# This exercise will help you understand how to adapt graph representations to specific chemical problems and domains.
# -

print("Tutorial Part 2 Complete!")
print("You now understand fragment-based molecular graph representations!")
print("Ready to tackle domain-specific applications and advanced GNN architectures.")


