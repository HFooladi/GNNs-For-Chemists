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
# <a href="https://colab.research.google.com/github/HFooladi/GNNs-For-Chemists/blob/main/notebooks/01.2_GNN_alternative_representations.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="bXCYDK0NkdpC"
# # Alternative Molecular Graph Representations: Beyond Atoms and Bonds
#
# ## Table of Contents
# 1. [Setup and Installation](#setup-and-installation)
# 2. [Introduction to Alternative Graph Representations](#introduction)
#    - 2.1 Quick Recap: What We've Learned So Far
#    - 2.2 What Traditional Graphs Miss
#    - 2.3 Learning Objectives
#    - 2.4 A Concrete Example: Acetic Acid
# 3. [The Dual Graph Solution](#theoretical-background)
#    - 3.1 The Key Insight
#    - 3.2 Understanding the Dual Graph Concept
#    - 3.3 Step-by-Step Transformation Tutorial
#    - 3.4 Side-by-Side Comparison
#    - 3.5 Interactive 3D Dual Graph Demo
#    - 3.6 Preview: Dihedral Angles with Butane
#    - 3.7 Mathematical Formulation
# 4. [Bond-as-Node Graph Construction](#bond-as-node-graph-construction)
# 5. [Angle-Based Edge Creation](#angle-based-edge-creation)
# 6. [Dihedral Angles and Torsional Information](#dihedral-angles-and-torsional-information)
# 7. [Implementing GeoGNN-Inspired Dual Graphs](#implementing-gemnet-inspired-dual-graphs)
# 8. [Visualization of Alternative Representations](#visualization-of-alternative-representations)
# 9. [Comparing Different Graph Paradigms](#comparing-different-graph-paradigms)
# 10. [Advanced Features and Applications](#advanced-features-and-applications)
# 11. [Conclusion](#conclusion)

# + [markdown] id="XHX_uMbYkkX7"
# ## 1. Setup and Installation <a name="setup-and-installation"></a>
#
# In this tutorial, we'll explore alternative molecular graph representations inspired by the GeoGNN paper (Fang et al., Nature Machine Intelligence, 2022). These approaches go beyond the traditional "atoms as nodes, bonds as edges" paradigm to capture richer geometric information.

# + colab={"base_uri": "https://localhost:8080/"} id="A7-zyVlhhvwk" outputId="11f359c0-1a20-4e60-c6ed-e5425d1df826"
#@title install required libraries
# !pip install -q rdkit
# !pip install -q torch_geometric
# !pip install -q plotly
# !pip install -q networkx
# !pip install -q py3dmol

# + [markdown] id="cKHLDrQ1mwDo"
# Import the required libraries:

# + colab={"base_uri": "https://localhost:8080/"} id="N_1gfxN7iJsO" outputId="ce82ca66-1544-47b8-c391-32f4c6afecd1"
#@title Import required libraries
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# RDKit for molecular handling
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# PyTorch and PyTorch Geometric
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

# 3D molecular visualization
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
    print("✅ py3Dmol available - enhanced 3D molecular visualization enabled!")
except ImportError:
    PY3DMOL_AVAILABLE = False
    print("⚠️ py3Dmol not available - install with: pip install py3dmol")
    print("Some advanced 3D visualizations will use fallback methods")

# Math and utilities
import itertools
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("Set2")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("Libraries imported successfully!")
print(f"RDKit version: {rdkit.__version__}")
print(f"PyTorch version: {torch.__version__}")

# + [markdown] id="r1xzRB8EkUtQ"
# ## 2. Introduction to Alternative Graph Representations <a name="introduction"></a>
#
# ### 2.1 Quick Recap: What We've Learned So Far
#
# Before diving into alternative representations, let's recall what we covered in the previous notebooks:
#
# **From Notebook 01 (Basic Molecular Graphs)**:
# - Molecules can be represented as graphs where **atoms are nodes** and **bonds are edges**
# - We learned to create adjacency matrices and node feature matrices
# - This "traditional graph" captures the **topology** (connectivity) of a molecule
#
# **From Notebook 01.1 (3D Representations)**:
# - We added **3D coordinates** to our molecular graphs
# - We saw that molecules with identical 2D graphs can have different 3D shapes (conformers, stereoisomers)
# - 3D information is crucial for drug design, stereochemistry, and property prediction
#
# **The Missing Piece**:
# While 3D coordinates give us spatial positions, we're still treating **atoms as the fundamental units**.
# But chemists often think in terms of **bonds and angles** - and that's what we'll explore today!
#
# ### 2.2 What Traditional Graphs Miss
#
# Traditional molecular graphs represent molecules as:
# - **Nodes**: Atoms
# - **Edges**: Chemical bonds
#
# However, this representation has important limitations:
# 1. **Missing geometric information**: Bond angles (e.g., the ~121° carbonyl angle) are not explicitly captured
# 2. **Bond types look the same structurally**: A C-C single bond and C=O double bond are both just "edges" in the graph
# 3. **No spatial relationships**: How bonds relate to each other in 3D space is lost
#
# Let's see this concretely with an example - **acetic acid (CH3COOH)** - which has both single AND double bonds.
#
# ### 2.3 Learning Objectives
#
# By the end of this tutorial, you will be able to:
# - **Understand** why traditional graphs miss important geometric information
# - **Construct** bond-as-node molecular graphs (where bonds become the fundamental units)
# - **Calculate** bond angles and dihedral angles from molecular structures
# - **Implement** dual graph representations inspired by GeoGNN (Fang et al., Nature Machine Intelligence, 2022)
# - **Visualize** the transformation from traditional to dual graph representations
# - **Compare** what each representation captures

# + [markdown] id="acetic-acid-demo-intro"
# ### 2.4 A Concrete Example: What's Missing in Acetic Acid?
#
# Let's visualize exactly what information traditional graphs fail to capture, using acetic acid (CH3COOH).
# This molecule is perfect for our demonstration because it contains:
# - **Single bonds**: C-C, C-O, O-H
# - **A double bond**: C=O (the carbonyl group)
# - **Important angles**: The ~121° carbonyl angle that defines the carboxylic acid geometry

# + colab={"base_uri": "https://localhost:8080/", "height": 500} id="acetic-acid-demo"
def demonstrate_limitation_with_acetic_acid():
    """
    Demonstrate why traditional graphs miss important chemical information,
    using acetic acid (CC(=O)O) which has BOTH single and double bonds.

    This is the "aha moment" - showing what geometry information is lost
    when we only encode atoms and bonds.

    Returns:
        py3Dmol viewer or None if py3Dmol is not available
    """
    print("Why Traditional Graphs Fall Short: The Acetic Acid Example")
    print("=" * 65)

    # Create acetic acid with 3D coordinates
    smiles = "CC(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id == -1:
        print("Could not generate 3D coordinates")
        return None
    AllChem.MMFFOptimizeMolecule(mol)

    # Get conformer for position calculations
    conf = mol.GetConformer()

    # Analyze what traditional graph captures
    print("\n1. What Traditional Graph (G) Captures:")
    print(f"   - {mol.GetNumAtoms()} atoms (nodes) including hydrogens")
    print(f"   - {mol.GetNumBonds()} bonds (edges)")

    # Count bond types
    single_bonds = 0
    double_bonds = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            single_bonds += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            double_bonds += 1

    print(f"   - Bond types: {single_bonds} single bonds, {double_bonds} double bond")
    print("   - Atom types: C, C, O, O, H, H, H, H")

    # What's missing
    print("\n2. What Traditional Graph (G) MISSES:")
    print("   - The C-C=O bond angle (~121° for sp2 carbonyl carbon)")
    print("   - The O-C-O angle in the carboxyl group (~120°)")
    print("   - The C-C-O angle for the methyl attachment")
    print("   - The C=O double bond's planarity vs C-C single bond's rotation")

    print("\n3. WHY THIS MATTERS for Machine Learning:")
    print("   - In a traditional graph, single bonds and double bonds are just 'edges'")
    print("   - The ~121° carbonyl angle is crucial for reactivity prediction")
    print("   - A GNN using only topology treats ALL bonds similarly in the graph structure!")
    print("   - Geometric information must be EXPLICITLY encoded - that's what dual graphs do!")

    # Create py3Dmol visualization
    if not PY3DMOL_AVAILABLE:
        print("\npy3Dmol not available - skipping 3D visualization")
        return None

    print("\nCreating 3D visualization...")

    # Create side-by-side viewer
    viewer = py3Dmol.view(width=1200, height=450, viewergrid=(1, 2))
    mol_block = Chem.MolToMolBlock(mol)

    # LEFT PANEL: Traditional view (what we capture)
    viewer.addModel(mol_block, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {
        'stick': {'colorscheme': 'Jmol', 'radius': 0.2},
        'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}
    }, viewer=(0, 0))

    viewer.addLabel('Traditional Graph View\n\nAtoms = Nodes\nBonds = Edges\n\nThis is what we encode...',
                   {'position': {'x': 0, 'y': 3.5, 'z': 0},
                    'backgroundColor': '#2196F3',
                    'fontColor': 'white', 'fontSize': 12}, viewer=(0, 0))

    # RIGHT PANEL: Highlight what we're missing (angles)
    viewer.addModel(mol_block, 'mol', viewer=(0, 1))
    viewer.setStyle({'model': -1}, {
        'stick': {'colorscheme': 'Jmol', 'radius': 0.15},
        'sphere': {'colorscheme': 'Jmol', 'scale': 0.25}
    }, viewer=(0, 1))

    # Find the carbonyl carbon (connected to =O)
    carbonyl_c_idx = None
    carbonyl_o_idx = None
    hydroxyl_o_idx = None
    methyl_c_idx = None

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            for bond in atom.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    other = bond.GetOtherAtom(atom)
                    if other.GetSymbol() == 'O':
                        carbonyl_c_idx = atom.GetIdx()
                        carbonyl_o_idx = other.GetIdx()
                        break

    # Find methyl carbon and hydroxyl oxygen
    if carbonyl_c_idx is not None:
        carbonyl_c = mol.GetAtomWithIdx(carbonyl_c_idx)
        for neighbor in carbonyl_c.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                methyl_c_idx = neighbor.GetIdx()
            elif neighbor.GetSymbol() == 'O' and neighbor.GetIdx() != carbonyl_o_idx:
                hydroxyl_o_idx = neighbor.GetIdx()

    # Highlight the key angle (C-C=O) if we found the atoms
    if carbonyl_c_idx is not None and carbonyl_o_idx is not None and methyl_c_idx is not None:
        # Get positions
        carbonyl_c_pos = conf.GetAtomPosition(carbonyl_c_idx)
        carbonyl_o_pos = conf.GetAtomPosition(carbonyl_o_idx)
        methyl_c_pos = conf.GetAtomPosition(methyl_c_idx)

        # Draw thick lines to show the angle
        viewer.addCylinder({'start': {'x': float(carbonyl_c_pos.x), 'y': float(carbonyl_c_pos.y), 'z': float(carbonyl_c_pos.z)},
                          'end': {'x': float(carbonyl_o_pos.x), 'y': float(carbonyl_o_pos.y), 'z': float(carbonyl_o_pos.z)},
                          'radius': 0.2, 'color': '#F44336', 'alpha': 0.9}, viewer=(0, 1))

        viewer.addCylinder({'start': {'x': float(carbonyl_c_pos.x), 'y': float(carbonyl_c_pos.y), 'z': float(carbonyl_c_pos.z)},
                          'end': {'x': float(methyl_c_pos.x), 'y': float(methyl_c_pos.y), 'z': float(methyl_c_pos.z)},
                          'radius': 0.2, 'color': '#F44336', 'alpha': 0.9}, viewer=(0, 1))

        # Calculate and display the angle
        vec1 = np.array([carbonyl_o_pos.x - carbonyl_c_pos.x,
                        carbonyl_o_pos.y - carbonyl_c_pos.y,
                        carbonyl_o_pos.z - carbonyl_c_pos.z])
        vec2 = np.array([methyl_c_pos.x - carbonyl_c_pos.x,
                        methyl_c_pos.y - carbonyl_c_pos.y,
                        methyl_c_pos.z - carbonyl_c_pos.z])

        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        # Add angle label near the vertex
        viewer.addLabel(f'C-C=O angle\n~{angle_deg:.0f}deg',
                       {'position': {'x': float(carbonyl_c_pos.x),
                                   'y': float(carbonyl_c_pos.y + 1.5),
                                   'z': float(carbonyl_c_pos.z)},
                        'backgroundColor': '#F44336',
                        'fontColor': 'white', 'fontSize': 11}, viewer=(0, 1))

    # Highlight O-C-O angle if we found hydroxyl oxygen
    if carbonyl_c_idx is not None and carbonyl_o_idx is not None and hydroxyl_o_idx is not None:
        hydroxyl_o_pos = conf.GetAtomPosition(hydroxyl_o_idx)

        viewer.addCylinder({'start': {'x': float(carbonyl_c_pos.x), 'y': float(carbonyl_c_pos.y), 'z': float(carbonyl_c_pos.z)},
                          'end': {'x': float(hydroxyl_o_pos.x), 'y': float(hydroxyl_o_pos.y), 'z': float(hydroxyl_o_pos.z)},
                          'radius': 0.15, 'color': '#4CAF50', 'alpha': 0.9}, viewer=(0, 1))

        # Calculate O-C-O angle
        vec1 = np.array([carbonyl_o_pos.x - carbonyl_c_pos.x,
                        carbonyl_o_pos.y - carbonyl_c_pos.y,
                        carbonyl_o_pos.z - carbonyl_c_pos.z])
        vec3 = np.array([hydroxyl_o_pos.x - carbonyl_c_pos.x,
                        hydroxyl_o_pos.y - carbonyl_c_pos.y,
                        hydroxyl_o_pos.z - carbonyl_c_pos.z])

        cos_angle2 = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
        angle_deg2 = np.degrees(np.arccos(np.clip(cos_angle2, -1.0, 1.0)))

        viewer.addLabel(f'O-C-O angle\n~{angle_deg2:.0f}deg',
                       {'position': {'x': float(carbonyl_c_pos.x - 1.5),
                                   'y': float(carbonyl_c_pos.y),
                                   'z': float(carbonyl_c_pos.z)},
                        'backgroundColor': '#4CAF50',
                        'fontColor': 'white', 'fontSize': 11}, viewer=(0, 1))

    viewer.addLabel('What We\'re MISSING!\n\nBond angles define geometry\nThis info is NOT in the graph!',
                   {'position': {'x': 0, 'y': 3.5, 'z': 0},
                    'backgroundColor': '#FF5722',
                    'fontColor': 'white', 'fontSize': 12}, viewer=(0, 1))

    viewer.zoomTo()

    print("\nLeft: What traditional graph captures (topology)")
    print("Right: What it MISSES (bond angles shown in red/green)")
    print("\nThe solution? Make BONDS the nodes, and ANGLES the edges!")

    return viewer

# Run the demonstration
acetic_acid_demo_viewer = demonstrate_limitation_with_acetic_acid()
if acetic_acid_demo_viewer:
    acetic_acid_demo_viewer.show()

# + [markdown] id="VzsxoQxEm9dM"
# ## 3. The Dual Graph Solution <a name="theoretical-background"></a>
#
# Now that we understand what's missing from traditional graphs, let's see how **dual graphs** solve this problem.

# + [markdown] id="key-insight-table"
# ### 3.1 The Key Insight
#
# Here's the fundamental transformation:
#
# | | **Traditional Graph (G)** | **Dual/Bond-Angle Graph (H)** |
# |---|---------------------------|-------------------------------|
# | **Nodes** | Atoms | **Bonds** |
# | **Edges** | Bonds | **Angles** (between adjacent bonds) |
# | **Captures** | Topology (connectivity) | **Geometry** (shape) |
# | **Example** | "Carbon connected to Oxygen" | "C-C bond at 121° to C=O bond" |
#
# **Why this works**: By making bonds the fundamental units, we can explicitly encode the angles between them - which is exactly the geometric information we were missing!
#
# Think about it from a chemistry perspective:
# - Traditional graphs tell you **what atoms are connected**
# - Dual graphs tell you **how those connections are arranged in space**

# + [markdown] id="zF4rEulu6gUI"
# ### 3.2 Understanding the Dual Graph Concept
#
# Let's formalize what we mean by "dual graphs" in molecular representation:
#
# **Traditional Graph (G)**:
# - Nodes (V_G): Atoms {A₁, A₂, A₃, ...}
# - Edges (E_G): Bonds {B₁₂, B₂₃, B₃₄, ...}
#
# **Bond-Angle Graph (H)**:
# - Nodes (V_H): Bonds {B₁₂, B₂₃, B₃₄, ...}
# - Edges (E_H): Bond angles {∠(B₁₂, B₂₃), ∠(B₂₃, B₃₄), ...}
#
# This creates a **hierarchical representation** where bonds in the first graph become nodes in the second graph.
# The key insight is that **two bonds share an edge if and only if they share a common atom**.

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="uM4e1dRlndV_" outputId="conceptual-diagram"
def illustrate_dual_graph_concept():
    """
    Create a conceptual diagram showing traditional vs dual graph representations.
    Uses acetic acid (CH3-C(=O)-OH) as the example molecule.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Example molecule: Acetic acid core (simplified: C-C(=O)-O)
    # Traditional representation
    ax1 = axes[0, 0]
    ax1.set_title('Traditional Graph (G)\nAtoms as Nodes, Bonds as Edges\n(Acetic Acid Core)', fontsize=12, weight='bold')

    # Draw atoms as nodes - simplified acetic acid structure
    # CH3 - C(=O) - OH  ->  C1 - C2 = O1, C2 - O2
    atom_positions = [(0, 0), (1.2, 0), (1.8, 0.6), (1.8, -0.6)]
    atom_labels = ['C₁\n(CH₃)', 'C₂', 'O₁\n(=O)', 'O₂\n(-OH)']
    atom_colors = ['skyblue', 'skyblue', 'salmon', 'salmon']

    for i, (pos, label, color) in enumerate(zip(atom_positions, atom_labels, atom_colors)):
        circle = plt.Circle(pos, 0.15, color=color, zorder=3)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], label, ha='center', va='center', fontsize=9, weight='bold')

    # Draw bonds as edges
    ax1.plot([0, 1.2], [0, 0], 'k-', linewidth=3)  # C-C single
    ax1.plot([1.2, 1.8], [0, 0.6], 'k-', linewidth=5)  # C=O double (thicker)
    ax1.plot([1.2, 1.8], [0, -0.6], 'k-', linewidth=3)  # C-O single

    ax1.text(0.6, -0.2, 'C-C', ha='center', fontsize=9, style='italic')
    ax1.text(1.65, 0.35, 'C=O', ha='center', fontsize=9, style='italic', color='red')
    ax1.text(1.65, -0.35, 'C-O', ha='center', fontsize=9, style='italic')

    ax1.set_xlim(-0.4, 2.3)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Graph statistics for traditional
    ax2 = axes[0, 1]
    ax2.set_title('Traditional Graph Properties', fontsize=12, weight='bold')
    ax2.text(0.1, 0.85, 'Nodes (Atoms): 4 (heavy atoms)', fontsize=11)
    ax2.text(0.1, 0.75, 'Edges (Bonds): 3', fontsize=11)
    ax2.text(0.1, 0.65, 'Information Captured:', fontsize=11, weight='bold')
    ax2.text(0.15, 0.55, '• Atom types (C, O)', fontsize=10)
    ax2.text(0.15, 0.45, '• Bond connectivity', fontsize=10)
    ax2.text(0.15, 0.35, '• Bond types (single/double)', fontsize=10)
    ax2.text(0.1, 0.22, 'Missing Information:', fontsize=11, weight='bold', color='red')
    ax2.text(0.15, 0.12, '• C-C=O angle (~121°)', fontsize=10, color='red')
    ax2.text(0.15, 0.02, '• O-C-O angle (~120°)', fontsize=10, color='red')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Bond-angle representation
    ax3 = axes[1, 0]
    ax3.set_title('Bond-Angle Graph (H)\nBonds as Nodes, Angles as Edges', fontsize=12, weight='bold')

    # Draw bonds as nodes - positioned to show structure
    bond_positions = [(0, 0), (1, 0.5), (1, -0.5)]  # C-C, C=O, C-O
    bond_labels = ['C-C\n(single)', 'C=O\n(double)', 'C-O\n(single)']
    bond_colors = ['lightblue', 'lightcoral', 'lightgreen']

    for i, (pos, label, color) in enumerate(zip(bond_positions, bond_labels, bond_colors)):
        square = plt.Rectangle((pos[0]-0.18, pos[1]-0.18), 0.36, 0.36,
                              color=color, zorder=3)
        ax3.add_patch(square)
        ax3.text(pos[0], pos[1], label, ha='center', va='center', fontsize=8, weight='bold')

    # Draw angle edges
    ax3.plot([0, 1], [0, 0.5], 'r-', linewidth=3)  # C-C to C=O angle
    ax3.plot([0, 1], [0, -0.5], 'b-', linewidth=3)  # C-C to C-O angle
    ax3.plot([1, 1], [0.5, -0.5], 'g-', linewidth=3)  # C=O to C-O angle (O-C-O)

    ax3.text(0.3, 0.35, '~121°', ha='center', fontsize=9, style='italic', color='red')
    ax3.text(0.3, -0.35, '~110°', ha='center', fontsize=9, style='italic', color='blue')
    ax3.text(1.25, 0, '~120°', ha='center', fontsize=9, style='italic', color='green')

    ax3.set_xlim(-0.5, 1.7)
    ax3.set_ylim(-0.9, 0.9)
    ax3.set_aspect('equal')
    ax3.axis('off')

    # Graph statistics for bond-angle
    ax4 = axes[1, 1]
    ax4.set_title('Bond-Angle Graph Properties', fontsize=12, weight='bold')
    ax4.text(0.1, 0.85, 'Nodes (Bonds): 3', fontsize=11)
    ax4.text(0.1, 0.75, 'Edges (Angles): 3', fontsize=11)
    ax4.text(0.1, 0.65, 'Information Captured:', fontsize=11, weight='bold')
    ax4.text(0.15, 0.55, '• Bond lengths', fontsize=10)
    ax4.text(0.15, 0.45, '• Bond types (as node features)', fontsize=10)
    ax4.text(0.15, 0.35, '• Bond angles (as edge features!)', fontsize=10)
    ax4.text(0.1, 0.22, 'Key Advantage:', fontsize=11, weight='bold', color='green')
    ax4.text(0.15, 0.12, '• Geometry explicitly encoded', fontsize=10, color='green')
    ax4.text(0.15, 0.02, '• Double bond distinct from single', fontsize=10, color='green')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

illustrate_dual_graph_concept()

# + [markdown] id="transformation-tutorial-intro"
# ### 3.3 Step-by-Step Transformation Tutorial
#
# Now let's see the transformation from traditional graph to dual graph **step by step** using acetic acid.
# This interactive 3D visualization shows exactly how we convert atoms/bonds to bonds/angles.

# + colab={"base_uri": "https://localhost:8080/", "height": 700} id="transformation-tutorial-code"
def create_transformation_tutorial_acetic_acid():
    """
    Step-by-step tutorial showing how to transform acetic acid from
    traditional graph to dual graph representation.

    Uses a 2x3 py3Dmol grid for visual progression:
    - Top row: Steps 1-3 of the transformation
    - Bottom row: The resulting graphs

    Returns:
        py3Dmol viewer or None if py3Dmol is not available
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available - cannot create transformation tutorial")
        return None

    print("Step-by-Step Transformation: Traditional to Dual Graph")
    print("Using Acetic Acid (CC(=O)O) as our example")
    print("=" * 60)

    # Create acetic acid
    smiles = "CC(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id == -1:
        print("Could not generate 3D coordinates")
        return None
    AllChem.MMFFOptimizeMolecule(mol)

    mol_block = Chem.MolToMolBlock(mol)
    conf = mol.GetConformer()

    # Create 2x3 viewer grid
    viewer = py3Dmol.view(width=1200, height=700, viewergrid=(2, 3))

    # Define colors - special colors for different bond types
    single_bond_color = '#2196F3'  # Blue for single bonds
    double_bond_color = '#F44336'  # Red for double bond

    # === STEP 1: Start with the molecule ===
    # Panel (0,0): Original molecule
    viewer.addModel(mol_block, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {
        'stick': {'colorscheme': 'Jmol', 'radius': 0.2},
        'sphere': {'colorscheme': 'Jmol', 'scale': 0.35}
    }, viewer=(0, 0))
    viewer.addLabel('STEP 1: Our Molecule\nAcetic Acid (CH3COOH)\n8 atoms, 7 bonds',
                   {'position': {'x': 0, 'y': 4, 'z': 0},
                    'backgroundColor': '#2196F3',
                    'fontColor': 'white', 'fontSize': 11}, viewer=(0, 0))

    # === STEP 2: Identify the bonds (future nodes) ===
    # Panel (0,1): Highlight each bond with a colored sphere at its center
    viewer.addModel(mol_block, 'mol', viewer=(0, 1))
    viewer.setStyle({'model': -1}, {'line': {'color': 'lightgray', 'width': 2}}, viewer=(0, 1))

    bond_positions = []
    bond_info_list = []

    for i, bond in enumerate(mol.GetBonds()):
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        begin_pos = conf.GetAtomPosition(begin_atom.GetIdx())
        end_pos = conf.GetAtomPosition(end_atom.GetIdx())

        center_x = (begin_pos.x + end_pos.x) / 2
        center_y = (begin_pos.y + end_pos.y) / 2
        center_z = (begin_pos.z + end_pos.z) / 2

        bond_positions.append([center_x, center_y, center_z])

        # Use different colors for single vs double bonds
        is_double = bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
        color = double_bond_color if is_double else single_bond_color

        bond_info_list.append({
            'begin_idx': begin_atom.GetIdx(),
            'end_idx': end_atom.GetIdx(),
            'begin_sym': begin_atom.GetSymbol(),
            'end_sym': end_atom.GetSymbol(),
            'is_double': is_double,
            'color': color
        })

        # Add sphere at bond center
        viewer.addSphere({'center': {'x': float(center_x), 'y': float(center_y), 'z': float(center_z)},
                         'radius': 0.35, 'color': color, 'alpha': 0.9}, viewer=(0, 1))

    viewer.addLabel('STEP 2: Identify Bonds\nBlue = single bonds\nRed = double bond (C=O)\nThese become NODES!',
                   {'position': {'x': 0, 'y': 4, 'z': 0},
                    'backgroundColor': '#4CAF50',
                    'fontColor': 'white', 'fontSize': 11}, viewer=(0, 1))

    # === STEP 3: Find shared atoms (create edges) ===
    # Panel (0,2): Show atoms that connect multiple bonds
    viewer.addModel(mol_block, 'mol', viewer=(0, 2))
    viewer.setStyle({'model': -1}, {'line': {'color': 'lightgray', 'width': 1}}, viewer=(0, 2))

    # Find atoms connected to multiple bonds
    for atom in mol.GetAtoms():
        bonds_on_atom = list(atom.GetBonds())
        if len(bonds_on_atom) > 1:
            pos = conf.GetAtomPosition(atom.GetIdx())
            viewer.addSphere({'center': {'x': float(pos.x), 'y': float(pos.y), 'z': float(pos.z)},
                             'radius': 0.5, 'color': 'gold', 'alpha': 0.8}, viewer=(0, 2))

    viewer.addLabel('STEP 3: Find Shared Atoms\nGold = atoms connecting 2+ bonds\nThese create EDGES\n(angle connections)',
                   {'position': {'x': 0, 'y': 4, 'z': 0},
                    'backgroundColor': '#FF9800',
                    'fontColor': 'white', 'fontSize': 11}, viewer=(0, 2))

    # === Bottom Row: Building the Dual Graph ===

    # Panel (1,0): Traditional graph summary
    viewer.addModel(mol_block, 'mol', viewer=(1, 0))
    viewer.setStyle({'model': -1}, {
        'stick': {'colorscheme': 'Jmol', 'radius': 0.15},
        'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}
    }, viewer=(1, 0))
    viewer.addLabel('Traditional Graph (G)\n8 nodes (atoms)\n7 edges (bonds)\nNo angle information',
                   {'position': {'x': 0, 'y': 4, 'z': 0},
                    'backgroundColor': '#607D8B',
                    'fontColor': 'white', 'fontSize': 11}, viewer=(1, 0))

    # Panel (1,1): Dual graph nodes only
    viewer.addModel(mol_block, 'mol', viewer=(1, 1))
    viewer.setStyle({'model': -1}, {'line': {'color': '#E0E0E0', 'width': 1}}, viewer=(1, 1))

    # Add bond nodes
    for i, (center, info) in enumerate(zip(bond_positions, bond_info_list)):
        viewer.addSphere({'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                         'radius': 0.35, 'color': info['color'], 'alpha': 0.9}, viewer=(1, 1))

    viewer.addLabel('Dual Graph (H) - Nodes\n7 nodes (bonds)\nBlue=single, Red=double\nReady for edges...',
                   {'position': {'x': 0, 'y': 4, 'z': 0},
                    'backgroundColor': '#9C27B0',
                    'fontColor': 'white', 'fontSize': 11}, viewer=(1, 1))

    # Panel (1,2): Complete dual graph with angle edges
    viewer.addModel(mol_block, 'mol', viewer=(1, 2))
    viewer.setStyle({'model': -1}, {'line': {'color': '#E0E0E0', 'width': 1}}, viewer=(1, 2))

    # Add bond nodes
    for i, (center, info) in enumerate(zip(bond_positions, bond_info_list)):
        viewer.addSphere({'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                         'radius': 0.3, 'color': info['color'], 'alpha': 0.9}, viewer=(1, 2))

    # Add angle edges (connect bonds that share an atom)
    edge_count = 0
    atom_to_bonds = defaultdict(list)
    for i, info in enumerate(bond_info_list):
        atom_to_bonds[info['begin_idx']].append(i)
        atom_to_bonds[info['end_idx']].append(i)

    for atom_idx, bond_indices in atom_to_bonds.items():
        for j in range(len(bond_indices)):
            for k in range(j + 1, len(bond_indices)):
                b1_idx = bond_indices[j]
                b2_idx = bond_indices[k]
                pos1 = bond_positions[b1_idx]
                pos2 = bond_positions[b2_idx]

                viewer.addCylinder({'start': {'x': float(pos1[0]), 'y': float(pos1[1]), 'z': float(pos1[2])},
                                  'end': {'x': float(pos2[0]), 'y': float(pos2[1]), 'z': float(pos2[2])},
                                  'radius': 0.06, 'color': 'black', 'alpha': 0.7}, viewer=(1, 2))
                edge_count += 1

    viewer.addLabel(f'COMPLETE Dual Graph (H)\n7 nodes (bonds)\n{edge_count} edges (angles!)\nGeometry encoded!',
                   {'position': {'x': 0, 'y': 4, 'z': 0},
                    'backgroundColor': '#4CAF50',
                    'fontColor': 'white', 'fontSize': 11}, viewer=(1, 2))

    viewer.zoomTo()

    print("\nTRANSFORMATION COMPLETE!")
    print("-" * 40)
    print("Traditional Graph (G): 8 atoms as nodes, 7 bonds as edges")
    print(f"Dual Graph (H): 7 bonds as nodes, {edge_count} angles as edges")
    print("\nThe dual graph explicitly encodes geometric relationships!")
    print("Notice: The C=O double bond (red) is now a distinct node.")

    return viewer

# Create the transformation tutorial
transformation_viewer = create_transformation_tutorial_acetic_acid()
if transformation_viewer:
    transformation_viewer.show()

# + [markdown] id="side-by-side-intro"
# ### 3.4 Side-by-Side Comparison: Before and After
#
# Let's directly compare what each representation captures with clear info panels:

# + colab={"base_uri": "https://localhost:8080/", "height": 500} id="side-by-side-code"
def create_side_by_side_comparison(smiles="CC(=O)O", molecule_name="Acetic Acid"):
    """
    Create a clear before/after comparison showing:
    - Left: What traditional graph captures
    - Right: What dual graph captures

    Returns:
        py3Dmol viewer or None if py3Dmol is not available
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available")
        return None

    print(f"Side-by-Side Comparison: Traditional vs Dual Graph")
    print(f"Molecule: {molecule_name} ({smiles})")
    print("=" * 60)

    # Create molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id == -1:
        print("Could not generate 3D coordinates")
        return None
    AllChem.MMFFOptimizeMolecule(mol)

    mol_block = Chem.MolToMolBlock(mol)
    conf = mol.GetConformer()

    # Create side-by-side viewer
    viewer = py3Dmol.view(width=1200, height=500, viewergrid=(1, 2))

    # === LEFT PANEL: Traditional Graph ===
    viewer.addModel(mol_block, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {
        'stick': {'colorscheme': 'Jmol', 'radius': 0.2},
        'sphere': {'colorscheme': 'Jmol', 'scale': 0.35}
    }, viewer=(0, 0))

    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # Count bond types
    n_single = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.SINGLE)
    n_double = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.DOUBLE)

    info_text_left = f"""TRADITIONAL GRAPH (G)

Nodes: {n_atoms} (atoms)
Edges: {n_bonds} (bonds)

CAPTURES:
  Atom types (C, O, H)
  Bond connectivity
  Bond types ({n_single} single, {n_double} double)

MISSES:
  Bond angles
  3D geometry
  Spatial relationships"""

    viewer.addLabel(info_text_left,
                   {'position': {'x': 5, 'y': 2, 'z': 0},
                    'backgroundColor': '#E3F2FD',
                    'fontColor': 'black', 'fontSize': 10,
                    'alignment': 'left'}, viewer=(0, 0))

    # === RIGHT PANEL: Dual Graph ===
    viewer.addModel(mol_block, 'mol', viewer=(0, 1))
    viewer.setStyle({'model': -1}, {'line': {'color': '#BDBDBD', 'width': 1}}, viewer=(0, 1))

    # Add bond nodes with different colors for single/double
    bond_positions = []
    for bond in mol.GetBonds():
        begin_pos = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        end_pos = conf.GetAtomPosition(bond.GetEndAtomIdx())

        center = [(begin_pos.x + end_pos.x) / 2,
                  (begin_pos.y + end_pos.y) / 2,
                  (begin_pos.z + end_pos.z) / 2]
        bond_positions.append(center)

        is_double = bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
        color = '#F44336' if is_double else '#2196F3'

        viewer.addSphere({'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                         'radius': 0.35, 'color': color, 'alpha': 0.9}, viewer=(0, 1))

    # Build atom_to_bonds mapping for angle edges
    atom_to_bonds = defaultdict(list)
    for i, bond in enumerate(mol.GetBonds()):
        atom_to_bonds[bond.GetBeginAtomIdx()].append(i)
        atom_to_bonds[bond.GetEndAtomIdx()].append(i)

    # Add angle edges and calculate angles
    n_angle_edges = 0
    angles = []

    for atom_idx, bond_indices in atom_to_bonds.items():
        for j in range(len(bond_indices)):
            for k in range(j + 1, len(bond_indices)):
                b1_idx = bond_indices[j]
                b2_idx = bond_indices[k]
                pos1 = bond_positions[b1_idx]
                pos2 = bond_positions[b2_idx]

                viewer.addCylinder({'start': {'x': float(pos1[0]), 'y': float(pos1[1]), 'z': float(pos1[2])},
                                  'end': {'x': float(pos2[0]), 'y': float(pos2[1]), 'z': float(pos2[2])},
                                  'radius': 0.06, 'color': 'black', 'alpha': 0.7}, viewer=(0, 1))
                n_angle_edges += 1

                # Calculate angle (simplified)
                bond1 = mol.GetBondWithIdx(b1_idx)
                bond2 = mol.GetBondWithIdx(b2_idx)

                # Find shared atom
                atoms1 = {bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()}
                atoms2 = {bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx()}
                shared = atoms1 & atoms2

                if shared:
                    shared_idx = list(shared)[0]
                    other1 = list(atoms1 - shared)[0]
                    other2 = list(atoms2 - shared)[0]

                    pos_shared = conf.GetAtomPosition(shared_idx)
                    pos_other1 = conf.GetAtomPosition(other1)
                    pos_other2 = conf.GetAtomPosition(other2)

                    vec1 = np.array([pos_other1.x - pos_shared.x, pos_other1.y - pos_shared.y, pos_other1.z - pos_shared.z])
                    vec2 = np.array([pos_other2.x - pos_shared.x, pos_other2.y - pos_shared.y, pos_other2.z - pos_shared.z])

                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                    angles.append(angle)

    mean_angle = np.mean(angles) if angles else 0

    info_text_right = f"""DUAL GRAPH (H)

Nodes: {n_bonds} (bonds)
Edges: {n_angle_edges} (angles)

CAPTURES:
  Bond lengths
  Bond types (as features)
  Bond angles (~{mean_angle:.0f}deg avg)

ENABLES:
  Geometry-aware learning
  Shape-based predictions
  Better property models!"""

    viewer.addLabel(info_text_right,
                   {'position': {'x': 5, 'y': 2, 'z': 0},
                    'backgroundColor': '#E8F5E9',
                    'fontColor': 'black', 'fontSize': 10,
                    'alignment': 'left'}, viewer=(0, 1))

    viewer.zoomTo()

    print("\nKEY COMPARISON:")
    print(f"Traditional: {n_atoms} nodes, {n_bonds} edges (topology only)")
    print(f"Dual:        {n_bonds} nodes, {n_angle_edges} edges (includes geometry!)")
    print(f"\nThe dual graph has {n_angle_edges/n_bonds:.1f}x more edges per node, encoding angle information.")
    print(f"Mean bond angle: {mean_angle:.1f} degrees")

    return viewer

# Create the side-by-side comparison
comparison_viewer = create_side_by_side_comparison("CC(=O)O", "Acetic Acid")
if comparison_viewer:
    comparison_viewer.show()

# + [markdown] id="dual-graph-3d-demo"
# ### 3.5 Interactive 3D Dual Graph Demo
#
# Now let's create an interactive 3D demonstration that you can explore with any molecule:

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="dual-graph-py3dmol"
def create_dual_graph_3d_demo(smiles="CC(=O)O", molecule_name="Acetic Acid"):
    """
    Create an interactive 3D demonstration of dual graph concept using py3Dmol.
    Shows traditional graph and highlights how bonds become nodes.

    Args:
        smiles: SMILES string of the molecule (default: acetic acid)
        molecule_name: Display name for the molecule
    """
    if not PY3DMOL_AVAILABLE:
        print("⚠️ py3Dmol not available - cannot create 3D dual graph demo")
        return None
    
    print(f"🔬 Creating 3D Dual Graph Demonstration for {molecule_name}")
    print("=" * 60)
    
    # Create molecule with 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id == -1:
        print(f"❌ Could not generate 3D coordinates for {smiles}")
        return None
    
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Create side-by-side viewer: traditional view vs bond-centered view
    viewer = py3Dmol.view(width=1200, height=500, viewergrid=(1, 2))
    
    mol_block = Chem.MolToMolBlock(mol)
    
    # LEFT PANEL: Traditional molecular view
    viewer.addModel(mol_block, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}, 
                                   'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}}, viewer=(0, 0))
    
    viewer.addLabel('Traditional Graph (G)\\nAtoms as Nodes\\nBonds as Edges', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 
                    'backgroundColor': 'lightblue',
                    'fontColor': 'black', 'fontSize': 14}, viewer=(0, 0))
    
    # RIGHT PANEL: Bond-centered view showing bonds as nodes
    viewer.addModel(mol_block, 'mol', viewer=(0, 1))
    # Make atoms smaller and bonds more prominent
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.3}, 
                                   'sphere': {'colorscheme': 'Jmol', 'scale': 0.2}}, viewer=(0, 1))
    
    # Add spheres at bond centers to represent bonds as nodes
    conf = mol.GetConformer()
    bond_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
    
    for i, bond in enumerate(mol.GetBonds()):
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        begin_pos = conf.GetAtomPosition(begin_atom.GetIdx())
        end_pos = conf.GetAtomPosition(end_atom.GetIdx())
        
        # Calculate bond center
        center_x = (begin_pos.x + end_pos.x) / 2
        center_y = (begin_pos.y + end_pos.y) / 2
        center_z = (begin_pos.z + end_pos.z) / 2
        
        # Add sphere at bond center
        color = bond_colors[i % len(bond_colors)]
        viewer.addSphere({'center': {'x': float(center_x), 'y': float(center_y), 'z': float(center_z)},
                         'radius': 0.4, 'color': color, 'alpha': 0.8}, viewer=(0, 1))
        
        # Add label for bond
        bond_label = f"{begin_atom.GetSymbol()}-{end_atom.GetSymbol()}"
        viewer.addLabel(bond_label, 
                       {'position': {'x': float(center_x), 'y': float(center_y + 0.8), 'z': float(center_z)}, 
                        'backgroundColor': color, 'fontColor': 'white', 'fontSize': 10}, viewer=(0, 1))
    
    viewer.addLabel('Bond-Angle Graph (H)\\nBonds as Nodes\\nColorful spheres = Bond nodes', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 
                    'backgroundColor': 'lightcoral',
                    'fontColor': 'black', 'fontSize': 14}, viewer=(0, 1))
    
    viewer.zoomTo()
    
    print("✅ 3D Dual Graph demonstration ready!")
    print("💡 Left panel: Traditional atom-bond representation")
    print("💡 Right panel: Bond-as-node representation (colored spheres)")
    print("🔄 Try rotating both views to understand the concept!")
    
    return viewer

# Create the 3D dual graph demo
demo_viewer = create_dual_graph_3d_demo("CC(=O)O", "Acetic Acid")
if demo_viewer:
    demo_viewer.show()

# + [markdown] id="dihedral-preview-intro"
# ### 3.6 Preview: Dihedral Angles and Conformational Flexibility
#
# So far we've focused on **bond angles** (3 atoms). But for molecules with 4+ atoms in a row,
# we can also capture **dihedral angles** - the angle between two planes defined by 4 consecutive atoms.
#
# This is crucial for understanding:
# - **Molecular conformations**: gauche vs anti arrangements
# - **Rotational barriers**: energy required to rotate around bonds
# - **Drug-receptor binding**: flexible molecules must adopt specific conformations
#
# Let's visualize this with **butane (CCCC)**, which has 4 carbons in a row:

# + colab={"base_uri": "https://localhost:8080/", "height": 500} id="dihedral-preview-code"
def visualize_dihedral_with_butane():
    """
    Demonstrate dihedral angles using butane (CCCC), which has
    4 carbons in a row - perfect for showing 4-atom dihedral calculation.

    Shows the C-C-C-C backbone and the planes that define the dihedral angle.

    Returns:
        py3Dmol viewer or None if py3Dmol is not available
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available - cannot create dihedral visualization")
        return None

    print("Dihedral Angles Preview: Butane (CCCC)")
    print("=" * 50)

    # Create butane with 3D coordinates
    smiles = "CCCC"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id == -1:
        print("Could not generate 3D coordinates")
        return None
    AllChem.MMFFOptimizeMolecule(mol)

    mol_block = Chem.MolToMolBlock(mol)
    conf = mol.GetConformer()

    # Find the 4 carbons (they'll be indices 0-3)
    carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'][:4]

    # Get positions of the 4 carbons
    c_positions = []
    for idx in carbon_indices:
        pos = conf.GetAtomPosition(idx)
        c_positions.append([pos.x, pos.y, pos.z])

    # Calculate dihedral angle
    p1, p2, p3, p4 = [np.array(p) for p in c_positions]

    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    n1_norm = n1 / np.linalg.norm(n1) if np.linalg.norm(n1) > 0 else n1
    n2_norm = n2 / np.linalg.norm(n2) if np.linalg.norm(n2) > 0 else n2

    cos_dihedral = np.dot(n1_norm, n2_norm)
    dihedral_angle = np.degrees(np.arccos(np.clip(cos_dihedral, -1.0, 1.0)))

    # Determine sign
    if np.dot(np.cross(n1_norm, n2_norm), v2) < 0:
        dihedral_angle = -dihedral_angle

    print(f"\nButane has 4 carbons: C1-C2-C3-C4")
    print(f"The dihedral angle (C1-C2-C3-C4) is: {dihedral_angle:.1f} degrees")
    print(f"\nCommon conformations:")
    print(f"  - Anti (180°): Most stable, hydrogens far apart")
    print(f"  - Gauche (~60°): Less stable, some steric strain")
    print(f"  - Eclipsed (0°): Least stable, high energy barrier")

    # Create viewer
    viewer = py3Dmol.view(width=1200, height=500, viewergrid=(1, 2))

    # LEFT PANEL: Show the molecule with backbone highlighted
    viewer.addModel(mol_block, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {
        'stick': {'colorscheme': 'Jmol', 'radius': 0.15},
        'sphere': {'colorscheme': 'Jmol', 'scale': 0.25}
    }, viewer=(0, 0))

    # Highlight the C-C-C-C backbone
    for i in range(len(c_positions)):
        viewer.addSphere({'center': {'x': float(c_positions[i][0]),
                                    'y': float(c_positions[i][1]),
                                    'z': float(c_positions[i][2])},
                         'radius': 0.5, 'color': '#F44336', 'alpha': 0.6}, viewer=(0, 0))

    # Draw backbone connections
    for i in range(len(c_positions) - 1):
        viewer.addCylinder({'start': {'x': float(c_positions[i][0]),
                                     'y': float(c_positions[i][1]),
                                     'z': float(c_positions[i][2])},
                          'end': {'x': float(c_positions[i+1][0]),
                                 'y': float(c_positions[i+1][1]),
                                 'z': float(c_positions[i+1][2])},
                          'radius': 0.2, 'color': '#F44336', 'alpha': 0.8}, viewer=(0, 0))

    viewer.addLabel(f'Butane: The 4-Carbon Backbone\n\nRed spheres = C1-C2-C3-C4\nThis defines our dihedral',
                   {'position': {'x': 0, 'y': 4, 'z': 0},
                    'backgroundColor': '#2196F3',
                    'fontColor': 'white', 'fontSize': 11}, viewer=(0, 0))

    # RIGHT PANEL: Show the planes
    viewer.addModel(mol_block, 'mol', viewer=(0, 1))
    viewer.setStyle({'model': -1}, {'line': {'color': 'lightgray', 'width': 1}}, viewer=(0, 1))

    # Highlight the two planes with different colors
    # Plane 1: C1-C2-C3 (blue)
    plane1_center = np.mean([c_positions[0], c_positions[1], c_positions[2]], axis=0)
    viewer.addSphere({'center': {'x': float(plane1_center[0]),
                                'y': float(plane1_center[1]),
                                'z': float(plane1_center[2])},
                     'radius': 0.3, 'color': '#2196F3', 'alpha': 0.7}, viewer=(0, 1))

    # Draw plane 1 bonds
    for i in range(2):
        viewer.addCylinder({'start': {'x': float(c_positions[i][0]),
                                     'y': float(c_positions[i][1]),
                                     'z': float(c_positions[i][2])},
                          'end': {'x': float(c_positions[i+1][0]),
                                 'y': float(c_positions[i+1][1]),
                                 'z': float(c_positions[i+1][2])},
                          'radius': 0.18, 'color': '#2196F3', 'alpha': 0.9}, viewer=(0, 1))

    # Plane 2: C2-C3-C4 (green)
    plane2_center = np.mean([c_positions[1], c_positions[2], c_positions[3]], axis=0)
    viewer.addSphere({'center': {'x': float(plane2_center[0]),
                                'y': float(plane2_center[1]),
                                'z': float(plane2_center[2])},
                     'radius': 0.3, 'color': '#4CAF50', 'alpha': 0.7}, viewer=(0, 1))

    # Draw plane 2 bonds
    for i in range(1, 3):
        viewer.addCylinder({'start': {'x': float(c_positions[i][0]),
                                     'y': float(c_positions[i][1]),
                                     'z': float(c_positions[i][2])},
                          'end': {'x': float(c_positions[i+1][0]),
                                 'y': float(c_positions[i+1][1]),
                                 'z': float(c_positions[i+1][2])},
                          'radius': 0.18, 'color': '#4CAF50', 'alpha': 0.9}, viewer=(0, 1))

    # Add dihedral angle label
    center_pos = np.mean(c_positions, axis=0)
    viewer.addLabel(f'Dihedral Angle: {abs(dihedral_angle):.0f}deg\n\nBlue = Plane 1 (C1-C2-C3)\nGreen = Plane 2 (C2-C3-C4)\n\nAngle between these planes!',
                   {'position': {'x': float(center_pos[0]),
                               'y': float(center_pos[1] + 3),
                               'z': float(center_pos[2])},
                    'backgroundColor': '#FF9800',
                    'fontColor': 'white', 'fontSize': 11}, viewer=(0, 1))

    viewer.zoomTo()

    print(f"\nVisualization shows:")
    print(f"  Left: Butane with C-C-C-C backbone highlighted")
    print(f"  Right: The two planes that define the dihedral angle")
    print(f"\nWe'll explore dihedrals in detail in Section 6!")

    return viewer

# Create dihedral preview
dihedral_preview_viewer = visualize_dihedral_with_butane()
if dihedral_preview_viewer:
    dihedral_preview_viewer.show()

# + [markdown] id="8hNh4BnQnQYL"
# ### 3.7 Mathematical Formulation
#
# For a molecule with atoms A = {a₁, a₂, ..., aₙ} and bonds B = {b₁, b₂, ..., bₘ}:
#
# **Traditional Graph G = (V_G, E_G)**:
# - V_G = A (atoms as nodes)
# - E_G = B (bonds as edges)
#
# **Bond-Angle Graph H = (V_H, E_H)**:
# - V_H = B (bonds as nodes)
# - E_H = {(bᵢ, bⱼ) | bᵢ and bⱼ share a common atom}
#
# **Edge Features in H**:
# - Bond angle: θᵢⱼ = angle between bonds bᵢ and bⱼ
# - Geometric relationship: spatial arrangement information

# + [markdown] id="esDZEn1lsUUr"
# ## 4. Bond-as-Node Graph Construction <a name="bond-as-node-graph-construction"></a>
#
# Let's implement the construction of bond-as-node graphs step by step.

# + colab={"base_uri": "https://localhost:8080/"} id="XL33mziBv859" outputId="bond-construction"
def extract_bonds_from_molecule(mol):
    """
    Extract all bonds from a molecule with their properties.
    
    Args:
        mol: RDKit molecule object with 3D coordinates
    
    Returns:
        list: List of bond dictionaries with detailed information
    """
    bonds = []
    
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        bond_info = {
            'bond_idx': bond.GetIdx(),
            'begin_atom_idx': begin_atom.GetIdx(),
            'end_atom_idx': end_atom.GetIdx(),
            'begin_atom_symbol': begin_atom.GetSymbol(),
            'end_atom_symbol': end_atom.GetSymbol(),
            'bond_type': bond.GetBondType(),
            'is_aromatic': bond.GetIsAromatic(),
            'is_in_ring': bond.IsInRing(),
            'bond_order': bond.GetBondTypeAsDouble()
        }
        
        # Calculate bond length if 3D coordinates are available
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            begin_pos = conf.GetAtomPosition(begin_atom.GetIdx())
            end_pos = conf.GetAtomPosition(end_atom.GetIdx())
            
            bond_length = np.sqrt(
                (begin_pos.x - end_pos.x)**2 + 
                (begin_pos.y - end_pos.y)**2 + 
                (begin_pos.z - end_pos.z)**2
            )
            bond_info['bond_length'] = bond_length
            
            # Store 3D positions for angle calculations
            bond_info['begin_pos'] = np.array([begin_pos.x, begin_pos.y, begin_pos.z])
            bond_info['end_pos'] = np.array([end_pos.x, end_pos.y, end_pos.z])
            bond_info['bond_vector'] = bond_info['end_pos'] - bond_info['begin_pos']
        
        bonds.append(bond_info)
    
    return bonds

def create_bond_features(bonds):
    """
    Create feature vectors for bonds to be used as node features.
    
    Args:
        bonds: List of bond dictionaries
    
    Returns:
        numpy.ndarray: Bond feature matrix
    """
    bond_features = []
    
    for bond in bonds:
        # One-hot encoding for bond types
        bond_type_features = [0, 0, 0, 0]  # [single, double, triple, aromatic]
        
        if bond['bond_type'] == Chem.rdchem.BondType.SINGLE:
            bond_type_features[0] = 1
        elif bond['bond_type'] == Chem.rdchem.BondType.DOUBLE:
            bond_type_features[1] = 1
        elif bond['bond_type'] == Chem.rdchem.BondType.TRIPLE:
            bond_type_features[2] = 1
        elif bond['bond_type'] == Chem.rdchem.BondType.AROMATIC:
            bond_type_features[3] = 1
        
        # Additional bond features
        features = bond_type_features + [
            float(bond['is_aromatic']),
            float(bond['is_in_ring']),
            bond['bond_order'],
            bond.get('bond_length', 0.0)  # 0.0 if no 3D coords
        ]
        
        bond_features.append(features)
    
    return np.array(bond_features)

# Test bond extraction with example molecules
test_molecules = {
    "Methanol": "CO",
    "Ethanol": "CCO",
    "Propane": "CCC",
    "Benzene": "c1ccccc1",
    "Cyclopropane": "C1CC1"
}

print("Bond Extraction Analysis:")
print("=" * 50)

for name, smiles in test_molecules.items():
    # Create molecule and add 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id != -1:
        AllChem.MMFFOptimizeMolecule(mol)
    
    # Extract bonds
    bonds = extract_bonds_from_molecule(mol)
    bond_features = create_bond_features(bonds)
    
    print(f"\n{name} ({smiles}):")
    print(f"  Atoms: {mol.GetNumAtoms()}")
    print(f"  Bonds: {len(bonds)}")
    print(f"  Bond feature matrix shape: {bond_features.shape}")
    
    # Show first few bonds
    print("  Bond details:")
    for i, bond in enumerate(bonds[:3]):  # Show first 3 bonds
        bond_desc = f"{bond['begin_atom_symbol']}-{bond['end_atom_symbol']}"
        bond_type = str(bond['bond_type']).split('.')[-1]
        length = bond.get('bond_length', 0.0)
        print(f"    Bond {i}: {bond_desc} ({bond_type}, {length:.2f} Å)")

# + [markdown] id="bond-extraction-3d"
# ### 3D Visualization of Bond Features
#
# Let's create an interactive visualization showing how bond features are extracted and represented:

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="visualize-bond-features"
def visualize_bond_features_3d(smiles="c1ccccc1", molecule_name="Benzene"):
    """
    Create 3D visualization showing bond features and properties.
    """
    if not PY3DMOL_AVAILABLE:
        print("⚠️ py3Dmol not available - cannot create bond features visualization")
        return None
    
    print(f"🔗 Visualizing Bond Features for {molecule_name}")
    print("=" * 50)
    
    # Create molecule with 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id == -1:
        print(f"❌ Could not generate 3D coordinates for {smiles}")
        return None
    
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Extract bond information
    bonds = extract_bonds_from_molecule(mol)
    
    # Create viewer
    viewer = py3Dmol.view(width=1000, height=500)
    mol_block = Chem.MolToMolBlock(mol)
    viewer.addModel(mol_block, 'mol')
    
    # Base molecular structure
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.15}, 
                                   'sphere': {'colorscheme': 'Jmol', 'scale': 0.25}})
    
    # Color-code bonds by type
    conf = mol.GetConformer()
    bond_type_colors = {
        Chem.rdchem.BondType.SINGLE: 'blue',
        Chem.rdchem.BondType.DOUBLE: 'red', 
        Chem.rdchem.BondType.TRIPLE: 'green',
        Chem.rdchem.BondType.AROMATIC: 'purple'
    }
    
    # Add information about each bond
    for i, bond_info in enumerate(bonds[:8]):  # Limit to first 8 bonds for clarity
        begin_pos = conf.GetAtomPosition(bond_info['begin_atom_idx'])
        end_pos = conf.GetAtomPosition(bond_info['end_atom_idx'])
        
        # Calculate bond center
        center_x = (begin_pos.x + end_pos.x) / 2
        center_y = (begin_pos.y + end_pos.y) / 2
        center_z = (begin_pos.z + end_pos.z) / 2
        
        # Get bond color
        bond_color = bond_type_colors.get(bond_info['bond_type'], 'gray')
        
        # Add bond information sphere
        viewer.addSphere({'center': {'x': float(center_x), 'y': float(center_y), 'z': float(center_z)},
                         'radius': 0.3, 'color': bond_color, 'alpha': 0.7})
        
        # Create bond label with key information
        bond_desc = f"{bond_info['begin_atom_symbol']}-{bond_info['end_atom_symbol']}"
        bond_type_str = str(bond_info['bond_type']).split('.')[-1]
        length = bond_info.get('bond_length', 0.0)
        
        label_text = f"Bond {i+1}: {bond_desc}\n{bond_type_str}\n{length:.2f} Å"
        
        viewer.addLabel(label_text,
                       {'position': {'x': float(center_x), 'y': float(center_y + 1.0), 'z': float(center_z)}, 
                        'backgroundColor': bond_color, 'fontColor': 'white', 'fontSize': 10})
    
    # Add legend
    viewer.addLabel('Bond Feature Visualization\n\nBond Types:\n🔵 Single\n🔴 Double\n🟢 Triple\n🟣 Aromatic\n\nSpheres = Bond Centers\nLabels = Bond Properties', 
                   {'position': {'x': 5, 'y': 3, 'z': 0}, 
                    'backgroundColor': 'lightgray',
                    'fontColor': 'black', 'fontSize': 12})
    
    viewer.zoomTo()
    
    print(f"✅ Bond features visualization ready for {molecule_name}!")
    print("💡 Each colored sphere represents a bond with its properties")
    print("🔄 Rotate to see all bond features in 3D space")
    
    return viewer

# Create bond features visualization
bond_viz = visualize_bond_features_3d("c1ccccc1", "Benzene")
if bond_viz:
    bond_viz.show()

# + [markdown] id="JlzUrrQJxqJK"
# ### Creating Bond-Node Adjacency
#
# Now let's create the adjacency relationships between bond nodes based on shared atoms:

# + colab={"base_uri": "https://localhost:8080/"} id="bond-adjacency"
def create_bond_adjacency(bonds):
    """
    Create adjacency matrix for bond-as-node graph.
    Bonds are connected if they share a common atom.
    
    Args:
        bonds: List of bond dictionaries
    
    Returns:
        tuple: (adjacency_matrix, edge_list, shared_atoms)
    """
    n_bonds = len(bonds)
    adjacency = np.zeros((n_bonds, n_bonds))
    edge_list = []
    shared_atoms = []
    
    # Create atom-to-bonds mapping
    atom_to_bonds = defaultdict(list)
    for bond_idx, bond in enumerate(bonds):
        atom_to_bonds[bond['begin_atom_idx']].append(bond_idx)
        atom_to_bonds[bond['end_atom_idx']].append(bond_idx)
    
    # Connect bonds that share atoms
    for atom_idx, bond_indices in atom_to_bonds.items():
        # Connect all pairs of bonds that share this atom
        for i in range(len(bond_indices)):
            for j in range(i + 1, len(bond_indices)):
                bond_i, bond_j = bond_indices[i], bond_indices[j]
                
                adjacency[bond_i, bond_j] = 1
                adjacency[bond_j, bond_i] = 1  # Symmetric
                
                edge_list.append((bond_i, bond_j))
                edge_list.append((bond_j, bond_i))  # Add both directions
                
                shared_atoms.append(atom_idx)
                shared_atoms.append(atom_idx)  # For both directions
    
    return adjacency, edge_list, shared_atoms

def analyze_bond_graph(bonds, adjacency):
    """
    Analyze the properties of the bond-as-node graph.
    """
    n_bonds = len(bonds)
    n_edges = np.sum(adjacency) // 2  # Divide by 2 because adjacency is symmetric
    
    # Calculate average degree
    degrees = np.sum(adjacency, axis=1)
    avg_degree = np.mean(degrees)
    
    # Find highly connected bonds (hubs)
    max_degree = np.max(degrees)
    hub_bonds = np.where(degrees == max_degree)[0]
    
    return {
        'n_bonds': n_bonds,
        'n_edges': n_edges,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'hub_bonds': hub_bonds,
        'degrees': degrees
    }

# Test bond adjacency construction
print("\nBond-as-Node Graph Analysis:")
print("=" * 40)

for name, smiles in list(test_molecules.items())[:3]:  # Test first 3 molecules
    # Create molecule with 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id != -1:
        AllChem.MMFFOptimizeMolecule(mol)
    
    # Extract bonds and create bond graph
    bonds = extract_bonds_from_molecule(mol)
    adjacency, edge_list, shared_atoms = create_bond_adjacency(bonds)
    analysis = analyze_bond_graph(bonds, adjacency)
    
    print(f"\n{name}:")
    print(f"  Traditional graph: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
    print(f"  Bond graph: {analysis['n_bonds']} nodes, {analysis['n_edges']} edges")
    print(f"  Average bond degree: {analysis['avg_degree']:.2f}")
    print(f"  Max bond degree: {analysis['max_degree']}")
    
    # Identify most connected bonds
    if len(analysis['hub_bonds']) > 0:
        hub_bond = bonds[analysis['hub_bonds'][0]]
        print(f"  Most connected bond: {hub_bond['begin_atom_symbol']}-{hub_bond['end_atom_symbol']}")

# + [markdown] id="bond-adjacency-3d"
# ### 3D Visualization of Bond Adjacency Network
#
# Now let's visualize how bonds connect to each other through shared atoms:

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="bond-network-3d"
def visualize_bond_adjacency_3d(smiles="CCCO", molecule_name="Propanol"):
    """
    Create 3D visualization of bond adjacency network.
    Shows how bonds connect through shared atoms.
    """
    if not PY3DMOL_AVAILABLE:
        print("⚠️ py3Dmol not available - cannot create bond adjacency visualization")
        return None
    
    print(f"🕸️ Visualizing Bond Adjacency Network for {molecule_name}")
    print("=" * 55)
    
    # Create molecule with 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id == -1:
        print(f"❌ Could not generate 3D coordinates for {smiles}")
        return None
    
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Extract bonds and create adjacency
    bonds = extract_bonds_from_molecule(mol)
    adjacency, edge_list, shared_atoms = create_bond_adjacency(bonds)
    
    # Create viewer with molecular structure
    viewer = py3Dmol.view(width=1000, height=500)
    mol_block = Chem.MolToMolBlock(mol)
    viewer.addModel(mol_block, 'mol')
    
    # Show molecular structure as wireframe
    viewer.setStyle({'model': -1}, {'line': {'color': 'lightgray', 'width': 2}})
    
    # Add bond nodes (spheres at bond centers)
    conf = mol.GetConformer()
    bond_positions = []
    bond_colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan']
    
    for i, bond_info in enumerate(bonds):
        begin_pos = conf.GetAtomPosition(bond_info['begin_atom_idx'])
        end_pos = conf.GetAtomPosition(bond_info['end_atom_idx'])
        
        # Calculate bond center
        center = np.array([(begin_pos.x + end_pos.x) / 2, 
                          (begin_pos.y + end_pos.y) / 2, 
                          (begin_pos.z + end_pos.z) / 2])
        bond_positions.append(center)
        
        # Add bond node
        color = bond_colors[i % len(bond_colors)]
        viewer.addSphere({'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                         'radius': 0.4, 'color': color, 'alpha': 0.8})
        
        # Label bond
        bond_desc = f"{bond_info['begin_atom_symbol']}-{bond_info['end_atom_symbol']}"
        viewer.addLabel(f"B{i}: {bond_desc}",
                       {'position': {'x': float(center[0]), 'y': float(center[1] + 0.8), 'z': float(center[2])}, 
                        'backgroundColor': color, 'fontColor': 'white', 'fontSize': 10})
    
    # Add bond-bond connections (angle edges)
    for i in range(0, len(edge_list), 2):  # Skip every other (bidirectional)
        bond_i_idx, bond_j_idx = edge_list[i]
        
        if bond_i_idx < len(bond_positions) and bond_j_idx < len(bond_positions):
            pos_i = bond_positions[bond_i_idx]
            pos_j = bond_positions[bond_j_idx]
            
            # Add connection line between bond centers
            viewer.addCylinder({'start': {'x': float(pos_i[0]), 'y': float(pos_i[1]), 'z': float(pos_i[2])},
                              'end': {'x': float(pos_j[0]), 'y': float(pos_j[1]), 'z': float(pos_j[2])},
                              'radius': 0.1, 'color': 'black', 'alpha': 0.6})
    
    # Add explanation
    viewer.addLabel('Bond Adjacency Network\n\n🔴🔵🟢 Colored spheres = Bonds as nodes\n⚫ Black lines = Bond connections\n(Bonds connected if they share an atom)', 
                   {'position': {'x': 0, 'y': 5, 'z': 0}, 
                    'backgroundColor': 'lightyellow',
                    'fontColor': 'black', 'fontSize': 12})
    
    viewer.zoomTo()
    
    print(f"✅ Bond adjacency network visualization ready!")
    print(f"💡 {len(bonds)} bonds shown as colored spheres")
    print(f"💡 {len(edge_list)//2} bond-bond connections shown as black lines")
    print("🔄 Rotate to see the full bond network structure")
    
    return viewer

# Create bond adjacency visualization  
network_viz = visualize_bond_adjacency_3d("CCCO", "Propanol")
if network_viz:
    network_viz.show()

# + [markdown] id="mXkhsu7myPUC"
# ## 5. Angle-Based Edge Creation <a name="angle-based-edge-creation"></a>
#
# Now we'll implement the calculation of bond angles to create edge features in the bond-angle graph:

# + colab={"base_uri": "https://localhost:8080/"} id="angle-calculation"
def calculate_bond_angle(bond1, bond2, shared_atom_idx):
    """
    Calculate the angle between two bonds that share a common atom.
    
    Args:
        bond1, bond2: Bond dictionaries with 3D information
        shared_atom_idx: Index of the shared atom
    
    Returns:
        float: Bond angle in degrees
    """
    # Get bond vectors
    if 'bond_vector' not in bond1 or 'bond_vector' not in bond2:
        return 0.0  # Return 0 if no 3D coordinates
    
    # Determine direction of vectors from shared atom
    if bond1['begin_atom_idx'] == shared_atom_idx:
        vec1 = bond1['bond_vector']
    else:
        vec1 = -bond1['bond_vector']
    
    if bond2['begin_atom_idx'] == shared_atom_idx:
        vec2 = bond2['bond_vector']
    else:
        vec2 = -bond2['bond_vector']
    
    # Calculate angle using dot product
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def create_angle_based_edges(bonds, edge_list, shared_atoms):
    """
    Create edge features based on bond angles.
    
    Args:
        bonds: List of bond dictionaries
        edge_list: List of bond pairs (edges in bond graph)
        shared_atoms: List of shared atoms for each edge
    
    Returns:
        numpy.ndarray: Edge feature matrix with angle information
    """
    edge_features = []
    
    for i in range(0, len(edge_list), 2):  # Process pairs (since we have bidirectional edges)
        bond_i_idx, bond_j_idx = edge_list[i]
        shared_atom_idx = shared_atoms[i]
        
        bond_i = bonds[bond_i_idx]
        bond_j = bonds[bond_j_idx]
        
        # Calculate bond angle
        angle = calculate_bond_angle(bond_i, bond_j, shared_atom_idx)
        
        # Create edge features
        # [angle_cos, angle_sin, angle_degrees, is_acute, is_obtuse]
        angle_rad = np.radians(angle)
        features = [
            np.cos(angle_rad),
            np.sin(angle_rad),
            angle / 180.0,  # Normalized angle
            float(angle < 90),  # Is acute
            float(angle > 90)   # Is obtuse
        ]
        
        # Add features for both directions
        edge_features.append(features)
        edge_features.append(features)
    
    return np.array(edge_features)

def analyze_molecular_angles(mol, bonds, edge_list, shared_atoms):
    """
    Analyze the distribution of bond angles in a molecule.
    """
    angles = []
    
    for i in range(0, len(edge_list), 2):
        bond_i_idx, bond_j_idx = edge_list[i]
        shared_atom_idx = shared_atoms[i]
        
        bond_i = bonds[bond_i_idx]
        bond_j = bonds[bond_j_idx]
        
        angle = calculate_bond_angle(bond_i, bond_j, shared_atom_idx)
        if angle > 0:  # Valid angle
            angles.append(angle)
    
    if angles:
        return {
            'angles': angles,
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'min_angle': np.min(angles),
            'max_angle': np.max(angles),
            'n_angles': len(angles)
        }
    else:
        return {'n_angles': 0}

# Test angle calculations
print("Bond Angle Analysis:")
print("=" * 30)

for name, smiles in list(test_molecules.items())[:4]:
    # Create molecule with 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id != -1:
        AllChem.MMFFOptimizeMolecule(mol)
    
    # Extract bonds and create bond graph
    bonds = extract_bonds_from_molecule(mol)
    adjacency, edge_list, shared_atoms = create_bond_adjacency(bonds)
    
    # Calculate angles
    angle_analysis = analyze_molecular_angles(mol, bonds, edge_list, shared_atoms)
    
    print(f"\n{name}:")
    if angle_analysis['n_angles'] > 0:
        print(f"  Number of bond angles: {angle_analysis['n_angles']}")
        print(f"  Mean angle: {angle_analysis['mean_angle']:.1f}°")
        print(f"  Angle range: {angle_analysis['min_angle']:.1f}° - {angle_analysis['max_angle']:.1f}°")
        print(f"  Standard deviation: {angle_analysis['std_angle']:.1f}°")
    else:
        print("  No bond angles (linear molecule or no 3D coordinates)")

# + [markdown] id="bond-angles-3d"
# ### 3D Visualization of Bond Angles
#
# Let's create an interactive visualization that shows bond angles explicitly:

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="bond-angles-viz"
def visualize_bond_angles_3d(smiles="CCO", molecule_name="Ethanol"):
    """
    Create 3D visualization showing bond angles between connected bonds.
    """
    if not PY3DMOL_AVAILABLE:
        print("⚠️ py3Dmol not available - cannot create bond angles visualization")
        return None
    
    print(f"📐 Visualizing Bond Angles for {molecule_name}")
    print("=" * 45)
    
    # Create molecule with 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id == -1:
        print(f"❌ Could not generate 3D coordinates for {smiles}")
        return None
    
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Extract bonds and angles
    bonds = extract_bonds_from_molecule(mol)
    adjacency, edge_list, shared_atoms = create_bond_adjacency(bonds)
    angle_analysis = analyze_molecular_angles(mol, bonds, edge_list, shared_atoms)
    
    # Create viewer
    viewer = py3Dmol.view(width=1200, height=500, viewergrid=(1, 2))
    mol_block = Chem.MolToMolBlock(mol)
    
    # LEFT PANEL: Molecular structure with highlighted angles
    viewer.addModel(mol_block, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}, 
                                   'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}}, viewer=(0, 0))
    
    # Highlight atoms involved in angles
    conf = mol.GetConformer()
    angle_count = 0
    
    for i in range(0, min(len(edge_list), 10), 2):  # Show first 5 angles max
        bond_i_idx, bond_j_idx = edge_list[i]
        shared_atom_idx = shared_atoms[i]
        
        bond_i = bonds[bond_i_idx]
        bond_j = bonds[bond_j_idx]
        
        angle = calculate_bond_angle(bond_i, bond_j, shared_atom_idx)
        if angle > 0:
            angle_count += 1
            
            # Get positions
            shared_pos = conf.GetAtomPosition(shared_atom_idx)
            
            # Get other atoms in bonds
            other_atom_i = bond_i['end_atom_idx'] if bond_i['begin_atom_idx'] == shared_atom_idx else bond_i['begin_atom_idx']
            other_atom_j = bond_j['end_atom_idx'] if bond_j['begin_atom_idx'] == shared_atom_idx else bond_j['begin_atom_idx']
            
            other_pos_i = conf.GetAtomPosition(other_atom_i)
            other_pos_j = conf.GetAtomPosition(other_atom_j)
            
            # Highlight the angle with colored lines
            angle_color = ['red', 'blue', 'green', 'orange', 'purple'][angle_count % 5]
            
            # Add thick lines to show the angle
            viewer.addCylinder({'start': {'x': float(shared_pos.x), 'y': float(shared_pos.y), 'z': float(shared_pos.z)},
                              'end': {'x': float(other_pos_i.x), 'y': float(other_pos_i.y), 'z': float(other_pos_i.z)},
                              'radius': 0.15, 'color': angle_color, 'alpha': 0.8}, viewer=(0, 0))
            
            viewer.addCylinder({'start': {'x': float(shared_pos.x), 'y': float(shared_pos.y), 'z': float(shared_pos.z)},
                              'end': {'x': float(other_pos_j.x), 'y': float(other_pos_j.y), 'z': float(other_pos_j.z)},
                              'radius': 0.15, 'color': angle_color, 'alpha': 0.8}, viewer=(0, 0))
            
            # Add angle label
            label_pos_x = (shared_pos.x + other_pos_i.x + other_pos_j.x) / 3
            label_pos_y = (shared_pos.y + other_pos_i.y + other_pos_j.y) / 3 + 1.0
            label_pos_z = (shared_pos.z + other_pos_i.z + other_pos_j.z) / 3
            
            shared_atom_symbol = mol.GetAtomWithIdx(shared_atom_idx).GetSymbol()
            other_symbol_i = mol.GetAtomWithIdx(other_atom_i).GetSymbol()
            other_symbol_j = mol.GetAtomWithIdx(other_atom_j).GetSymbol()
            
            angle_label = f"∠{other_symbol_i}-{shared_atom_symbol}-{other_symbol_j}\n{angle:.1f}°"
            viewer.addLabel(angle_label,
                           {'position': {'x': float(label_pos_x), 'y': float(label_pos_y), 'z': float(label_pos_z)}, 
                            'backgroundColor': angle_color, 'fontColor': 'white', 'fontSize': 10}, viewer=(0, 0))
    
    viewer.addLabel('Bond Angles in 3D\nColored lines show angles\nLabels show angle values', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 
                    'backgroundColor': 'lightblue',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(0, 0))
    
    # RIGHT PANEL: Abstract bond-angle graph representation
    viewer.addModel(mol_block, 'mol', viewer=(0, 1))
    # Make atoms very small and bonds prominent
    viewer.setStyle({'model': -1}, {'line': {'color': 'lightgray', 'width': 1}}, viewer=(0, 1))
    
    # Show bonds as nodes and angles as edges
    bond_centers = []
    for i, bond_info in enumerate(bonds):
        begin_pos = conf.GetAtomPosition(bond_info['begin_atom_idx'])
        end_pos = conf.GetAtomPosition(bond_info['end_atom_idx'])
        
        center = [(begin_pos.x + end_pos.x) / 2, 
                 (begin_pos.y + end_pos.y) / 2, 
                 (begin_pos.z + end_pos.z) / 2]
        bond_centers.append(center)
        
        # Add bond as node
        viewer.addSphere({'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                         'radius': 0.3, 'color': 'gold', 'alpha': 0.9}, viewer=(0, 1))
        
        bond_desc = f"{bond_info['begin_atom_symbol']}-{bond_info['end_atom_symbol']}"
        viewer.addLabel(bond_desc,
                       {'position': {'x': float(center[0]), 'y': float(center[1] + 0.6), 'z': float(center[2])}, 
                        'backgroundColor': 'gold', 'fontColor': 'black', 'fontSize': 9}, viewer=(0, 1))
    
    # Connect bonds with angle edges
    for i in range(0, min(len(edge_list), 10), 2):
        bond_i_idx, bond_j_idx = edge_list[i]
        if bond_i_idx < len(bond_centers) and bond_j_idx < len(bond_centers):
            pos_i = bond_centers[bond_i_idx]
            pos_j = bond_centers[bond_j_idx]
            
            # Add angle edge
            viewer.addCylinder({'start': {'x': float(pos_i[0]), 'y': float(pos_i[1]), 'z': float(pos_i[2])},
                              'end': {'x': float(pos_j[0]), 'y': float(pos_j[1]), 'z': float(pos_j[2])},
                              'radius': 0.08, 'color': 'red', 'alpha': 0.7}, viewer=(0, 1))
    
    viewer.addLabel('Bond-Angle Graph (H)\nGold spheres = Bonds as nodes\nRed cylinders = Angle edges', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 
                    'backgroundColor': 'lightcoral',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(0, 1))
    
    viewer.zoomTo()
    
    print(f"✅ Bond angles visualization ready for {molecule_name}!")
    if angle_analysis.get('n_angles', 0) > 0:
        print(f"💡 Showing {min(5, angle_analysis['n_angles'])} bond angles")
        print(f"💡 Mean angle: {angle_analysis.get('mean_angle', 0):.1f}°")
    print("🔄 Rotate both panels to understand the angle relationships")
    
    return viewer

# Create bond angles visualization
angles_viz = visualize_bond_angles_3d("CCO", "Ethanol")
if angles_viz:
    angles_viz.show()

# + [markdown] id="qdNQD25Ky1uY"
# ## 6. Dihedral Angles and Torsional Information <a name="dihedral-angles-and-torsional-information"></a>
#
# Dihedral angles provide crucial information about molecular conformation and flexibility:

# + colab={"base_uri": "https://localhost:8080/"} id="dihedral-angles"
def calculate_dihedral_angle(mol, atom_indices):
    """
    Calculate dihedral angle for four atoms.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        atom_indices: List of 4 atom indices [i, j, k, l]
    
    Returns:
        float: Dihedral angle in degrees
    """
    if mol.GetNumConformers() == 0:
        return 0.0
    
    conf = mol.GetConformer()
    
    # Get atom positions
    positions = []
    for idx in atom_indices:
        pos = conf.GetAtomPosition(idx)
        positions.append(np.array([pos.x, pos.y, pos.z]))
    
    p1, p2, p3, p4 = positions
    
    # Calculate vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    
    # Calculate normal vectors to planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Normalize
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    # Calculate dihedral angle
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    
    # Determine sign using the scalar triple product
    if np.dot(np.cross(n1, n2), v2) < 0:
        angle = -angle
    
    return np.degrees(angle)

def find_dihedral_patterns(mol):
    """
    Find all possible dihedral angles in a molecule.
    """
    dihedrals = []
    
    # Find all paths of 4 connected atoms
    for bond1 in mol.GetBonds():
        for bond2 in mol.GetBonds():
            if bond1.GetIdx() == bond2.GetIdx():
                continue
                
            # Check if bonds are connected
            if (bond1.GetEndAtomIdx() == bond2.GetBeginAtomIdx() or
                bond1.GetEndAtomIdx() == bond2.GetEndAtomIdx() or
                bond1.GetBeginAtomIdx() == bond2.GetBeginAtomIdx() or
                bond1.GetBeginAtomIdx() == bond2.GetEndAtomIdx()):
                
                # Find the 4-atom path
                atoms1 = [bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()]
                atoms2 = [bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx()]
                
                # Find shared atom
                shared = set(atoms1) & set(atoms2)
                if len(shared) == 1:
                    shared_atom = list(shared)[0]
                    
                    # Get the other atoms
                    other1 = [a for a in atoms1 if a != shared_atom][0]
                    other2 = [a for a in atoms2 if a != shared_atom][0]
                    
                    # Look for a third bond connected to other2
                    for bond3 in mol.GetBonds():
                        if bond3.GetIdx() in [bond1.GetIdx(), bond2.GetIdx()]:
                            continue
                        
                        atoms3 = [bond3.GetBeginAtomIdx(), bond3.GetEndAtomIdx()]
                        if other2 in atoms3:
                            other3 = [a for a in atoms3 if a != other2][0]
                            
                            # Create 4-atom dihedral
                            dihedral = [other1, shared_atom, other2, other3]
                            
                            # Avoid duplicates
                            reverse_dihedral = dihedral[::-1]
                            if dihedral not in dihedrals and reverse_dihedral not in dihedrals:
                                dihedrals.append(dihedral)
    
    return dihedrals

def analyze_molecular_dihedrals(mol):
    """
    Analyze dihedral angles in a molecule.
    """
    dihedrals = find_dihedral_patterns(mol)
    dihedral_angles = []
    
    for dihedral in dihedrals:
        angle = calculate_dihedral_angle(mol, dihedral)
        dihedral_angles.append({
            'atoms': dihedral,
            'angle': angle,
            'atom_symbols': [mol.GetAtomWithIdx(i).GetSymbol() for i in dihedral]
        })
    
    return dihedral_angles

# Test dihedral angle calculations
print("Dihedral Angle Analysis:")
print("=" * 35)

# Test with molecules that have interesting dihedrals
test_molecules_dihedrals = {
    "Butane": "CCCC",
    "Ethanol": "CCO", 
    "Propanol": "CCCO",
    "Benzene": "c1ccccc1"
}

for name, smiles in test_molecules_dihedrals.items():
    # Create molecule with 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id != -1:
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Analyze dihedrals
        dihedrals = analyze_molecular_dihedrals(mol)
        
        print(f"\n{name}:")
        print(f"  Number of dihedral angles: {len(dihedrals)}")
        
        # Show first few dihedrals
        for i, dihedral in enumerate(dihedrals[:3]):
            atom_str = "-".join(dihedral['atom_symbols'])
            print(f"    Dihedral {i+1}: {atom_str} = {dihedral['angle']:.1f}°")
    else:
        print(f"\n{name}: Could not generate 3D coordinates")

# + [markdown] id="dihedral-3d-viz"
# ### 3D Visualization of Dihedral Angles
#
# Let's create an interactive visualization that shows dihedral angles and torsional flexibility:

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="dihedral-viz-3d"
def visualize_dihedral_angles_3d(smiles="CCCC", molecule_name="Butane"):
    """
    Create 3D visualization showing dihedral angles and torsional information.
    """
    if not PY3DMOL_AVAILABLE:
        print("⚠️ py3Dmol not available - cannot create dihedral angles visualization")
        return None
    
    print(f"🌀 Visualizing Dihedral Angles for {molecule_name}")
    print("=" * 50)
    
    # Create molecule with 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id == -1:
        print(f"❌ Could not generate 3D coordinates for {smiles}")
        return None
    
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Analyze dihedrals
    dihedrals = analyze_molecular_dihedrals(mol)
    
    if not dihedrals:
        print(f"⚠️ No dihedral angles found in {molecule_name}")
        return None
    
    # Create viewer
    viewer = py3Dmol.view(width=1000, height=500)
    mol_block = Chem.MolToMolBlock(mol)
    viewer.addModel(mol_block, 'mol')
    
    # Base molecular structure
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}, 
                                   'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}})
    
    # Visualize first few dihedral angles
    conf = mol.GetConformer()
    dihedral_colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, dihedral in enumerate(dihedrals[:3]):  # Show first 3 dihedrals
        atoms = dihedral['atoms']
        angle = dihedral['angle']
        symbols = dihedral['atom_symbols']
        
        # Get atom positions
        positions = []
        for atom_idx in atoms:
            pos = conf.GetAtomPosition(atom_idx)
            positions.append([pos.x, pos.y, pos.z])
        
        color = dihedral_colors[i]
        
        # Draw the four atoms involved in dihedral with special highlighting
        for j, (atom_idx, pos) in enumerate(zip(atoms, positions)):
            viewer.addSphere({'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                             'radius': 0.4, 'color': color, 'alpha': 0.6})
        
        # Draw the dihedral path
        for j in range(len(positions) - 1):
            viewer.addCylinder({'start': {'x': float(positions[j][0]), 'y': float(positions[j][1]), 'z': float(positions[j][2])},
                              'end': {'x': float(positions[j+1][0]), 'y': float(positions[j+1][1]), 'z': float(positions[j+1][2])},
                              'radius': 0.15, 'color': color, 'alpha': 0.8})
        
        # Add dihedral label
        center_pos = np.mean(positions, axis=0)
        dihedral_name = "-".join(symbols)
        label_text = f"Dihedral {i+1}\n{dihedral_name}\n{angle:.1f}°"
        
        viewer.addLabel(label_text,
                       {'position': {'x': float(center_pos[0]), 'y': float(center_pos[1] + 2.0), 'z': float(center_pos[2])}, 
                        'backgroundColor': color, 'fontColor': 'white', 'fontSize': 11})
        
        # Add planes to visualize the dihedral concept
        if len(positions) >= 4:
            # First plane (atoms 1-2-3)
            p1, p2, p3, p4 = positions
            
            # Calculate normal vectors for planes
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p2)
            v3 = np.array(p4) - np.array(p3)
            
            normal1 = np.cross(v1, v2)
            normal2 = np.cross(v2, v3)
            
            # Normalize
            if np.linalg.norm(normal1) > 0:
                normal1 = normal1 / np.linalg.norm(normal1)
            if np.linalg.norm(normal2) > 0:
                normal2 = normal2 / np.linalg.norm(normal2)
            
            # Add small plane indicators
            plane1_center = (np.array(p1) + np.array(p2) + np.array(p3)) / 3
            plane2_center = (np.array(p2) + np.array(p3) + np.array(p4)) / 3
            
            # Add small discs to represent planes
            viewer.addSphere({'center': {'x': float(plane1_center[0]), 'y': float(plane1_center[1]), 'z': float(plane1_center[2])},
                             'radius': 0.2, 'color': color, 'alpha': 0.3})
            viewer.addSphere({'center': {'x': float(plane2_center[0]), 'y': float(plane2_center[1]), 'z': float(plane2_center[2])},
                             'radius': 0.2, 'color': color, 'alpha': 0.3})
    
    # Add explanation
    viewer.addLabel(f'Dihedral Angles in {molecule_name}\n\nColored paths show 4-atom sequences\nDihedral = angle between two planes\nImportant for molecular flexibility', 
                   {'position': {'x': 0, 'y': 5, 'z': 0}, 
                    'backgroundColor': 'lightyellow',
                    'fontColor': 'black', 'fontSize': 12})
    
    viewer.zoomTo()
    
    print(f"✅ Dihedral angles visualization ready!")
    print(f"💡 Showing {min(3, len(dihedrals))} dihedral angles out of {len(dihedrals)} total")
    print("💡 Each colored path represents a 4-atom dihedral sequence")
    print("🔄 Rotate to see the spatial relationships between atoms")
    
    return viewer

# Create dihedral angles visualization
dihedral_viz = visualize_dihedral_angles_3d("CCCC", "Butane")
if dihedral_viz:
    dihedral_viz.show()

# + [markdown] id="RLkVqjjYy2te"
# ## 7. Implementing GeoGNN-Inspired Dual Graphs <a name="implementing-gemnet-inspired-dual-graphs"></a>
#
# Now let's combine everything into a complete dual graph implementation:

# + colab={"base_uri": "https://localhost:8080/"} id="gemnet-implementation"
class DualMolecularGraph:
    """
    Implementation of dual molecular graph representation inspired by GemNet.
    
    Creates two graphs:
    1. Traditional atom-bond graph (G)
    2. Bond-angle graph (H)
    """
    
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = None
        self.bonds = []
        
        # Traditional graph (G)
        self.atom_features = None
        self.atom_adjacency = None
        self.bond_edge_features = None
        
        # Bond-angle graph (H)
        self.bond_features = None
        self.bond_adjacency = None
        self.angle_edge_features = None
        self.edge_list = []
        self.shared_atoms = []
        
        # Additional information
        self.dihedral_angles = []
        
    def build_graphs(self, add_3d=True):
        """Build both traditional and bond-angle graphs."""
        # Create molecule
        self.mol = Chem.MolFromSmiles(self.smiles)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {self.smiles}")
        
        self.mol = Chem.AddHs(self.mol)
        
        # Add 3D coordinates if requested
        if add_3d:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            conf_id = AllChem.EmbedMolecule(self.mol, params)
            if conf_id != -1:
                AllChem.MMFFOptimizeMolecule(self.mol)
        
        # Build traditional graph (G)
        self._build_atom_graph()
        
        # Build bond-angle graph (H)
        self._build_bond_graph()
        
        # Calculate dihedral angles
        if add_3d:
            self.dihedral_angles = analyze_molecular_dihedrals(self.mol)
    
    def _build_atom_graph(self):
        """Build traditional atom-bond graph."""
        n_atoms = self.mol.GetNumAtoms()
        
        # Create atom features
        atom_features = []
        for atom in self.mol.GetAtoms():
            # Basic atom features
            atom_type = atom.GetSymbol()
            atomic_num = atom.GetAtomicNum()
            formal_charge = atom.GetFormalCharge()
            is_aromatic = int(atom.GetIsAromatic())
            is_in_ring = int(atom.IsInRing())
            
            # One-hot encoding for common atoms
            atom_types = ['C', 'O', 'N', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
            atom_type_onehot = [1 if atom_type == t else 0 for t in atom_types]
            if atom_type not in atom_types:
                atom_type_onehot.append(1)  # "Other"
            else:
                atom_type_onehot.append(0)
            
            features = atom_type_onehot + [
                formal_charge, is_aromatic, is_in_ring,
                atom.GetDegree(), atom.GetTotalNumHs()
            ]
            atom_features.append(features)
        
        self.atom_features = np.array(atom_features)
        
        # Create adjacency matrix
        self.atom_adjacency = GetAdjacencyMatrix(self.mol)
        
        # Create bond edge features (traditional edges)
        bond_edge_features = []
        for bond in self.mol.GetBonds():
            # Bond type one-hot
            bond_type_features = [0, 0, 0, 0]  # [single, double, triple, aromatic]
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                bond_type_features[0] = 1
            elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                bond_type_features[1] = 1
            elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                bond_type_features[2] = 1
            elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                bond_type_features[3] = 1
            
            features = bond_type_features + [
                float(bond.GetIsAromatic()),
                float(bond.IsInRing())
            ]
            bond_edge_features.append(features)
        
        self.bond_edge_features = np.array(bond_edge_features)
    
    def _build_bond_graph(self):
        """Build bond-angle graph."""
        # Extract bonds
        self.bonds = extract_bonds_from_molecule(self.mol)
        
        # Create bond features (nodes in H)
        self.bond_features = create_bond_features(self.bonds)
        
        # Create bond adjacency and edge features (H)
        self.bond_adjacency, self.edge_list, self.shared_atoms = create_bond_adjacency(self.bonds)
        
        # Create angle-based edge features
        if len(self.edge_list) > 0:
            self.angle_edge_features = create_angle_based_edges(
                self.bonds, self.edge_list, self.shared_atoms
            )
        else:
            self.angle_edge_features = np.array([]).reshape(0, 5)
    
    def get_graph_statistics(self):
        """Get comprehensive statistics for both graphs."""
        stats = {
            'molecule': {
                'smiles': self.smiles,
                'atoms': self.mol.GetNumAtoms() if self.mol else 0,
                'bonds': self.mol.GetNumBonds() if self.mol else 0,
                'rings': self.mol.GetRingInfo().NumRings() if self.mol else 0
            },
            'atom_graph': {
                'nodes': self.atom_features.shape[0] if self.atom_features is not None else 0,
                'node_features': self.atom_features.shape[1] if self.atom_features is not None else 0,
                'edges': np.sum(self.atom_adjacency) // 2 if self.atom_adjacency is not None else 0
            },
            'bond_graph': {
                'nodes': self.bond_features.shape[0] if self.bond_features is not None else 0,
                'node_features': self.bond_features.shape[1] if self.bond_features is not None else 0,
                'edges': len(self.edge_list) // 2 if self.edge_list else 0,
                'edge_features': self.angle_edge_features.shape[1] if self.angle_edge_features.size > 0 else 0
            },
            'dihedrals': len(self.dihedral_angles)
        }
        return stats
    
    def to_pytorch_geometric(self):
        """Convert to PyTorch Geometric format."""
        # Traditional graph
        atom_data = Data(
            x=torch.tensor(self.atom_features, dtype=torch.float),
            edge_index=torch.tensor(np.array(np.where(self.atom_adjacency)), dtype=torch.long),
            smiles=self.smiles
        )
        
        # Bond graph
        if len(self.edge_list) > 0:
            bond_edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous()
            bond_edge_attr = torch.tensor(self.angle_edge_features, dtype=torch.float)
        else:
            bond_edge_index = torch.zeros((2, 0), dtype=torch.long)
            bond_edge_attr = torch.zeros((0, 5), dtype=torch.float)
        
        bond_data = Data(
            x=torch.tensor(self.bond_features, dtype=torch.float),
            edge_index=bond_edge_index,
            edge_attr=bond_edge_attr,
            smiles=self.smiles
        )
        
        return atom_data, bond_data

# Test the dual graph implementation
print("Dual Graph Implementation Test:")
print("=" * 40)

test_molecules_dual = {
    "Methanol": "CO",
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O"
}

for name, smiles in test_molecules_dual.items():
    print(f"\n{name} ({smiles}):")
    
    try:
        # Create dual graph
        dual_graph = DualMolecularGraph(smiles)
        dual_graph.build_graphs(add_3d=True)
        
        # Get statistics
        stats = dual_graph.get_graph_statistics()
        
        print(f"  Molecule: {stats['molecule']['atoms']} atoms, {stats['molecule']['bonds']} bonds")
        print(f"  Atom graph (G): {stats['atom_graph']['nodes']} nodes, {stats['atom_graph']['edges']} edges")
        print(f"  Bond graph (H): {stats['bond_graph']['nodes']} nodes, {stats['bond_graph']['edges']} edges")
        print(f"  Dihedral angles: {stats['dihedrals']}")
        
        # Convert to PyG format
        atom_data, bond_data = dual_graph.to_pytorch_geometric()
        print(f"  PyG atom graph: {atom_data}")
        print(f"  PyG bond graph: {bond_data}")
        
    except Exception as e:
        print(f"  Error: {e}")

# + [markdown] id="dual-graph-complete-3d"
# ### Complete 3D Dual Graph Visualization
#
# Now let's create a comprehensive visualization that shows both graphs side by side:

# + colab={"base_uri": "https://localhost:8080/", "height": 500} id="complete-dual-viz"
def create_complete_dual_graph_3d(smiles="CC(=O)C", molecule_name="Acetone"):
    """
    Create a comprehensive 3D visualization of both traditional and bond-angle graphs.
    """
    if not PY3DMOL_AVAILABLE:
        print("⚠️ py3Dmol not available - cannot create complete dual graph visualization")
        return None
    
    print(f"🔄 Creating Complete Dual Graph Visualization for {molecule_name}")
    print("=" * 70)
    
    try:
        # Create dual graph
        dual_graph = DualMolecularGraph(smiles)
        dual_graph.build_graphs(add_3d=True)
        
        # Get statistics
        stats = dual_graph.get_graph_statistics()
        
        print(f"📊 Graph Statistics:")
        print(f"  Traditional graph (G): {stats['atom_graph']['nodes']} atoms, {stats['atom_graph']['edges']} bonds")
        print(f"  Bond-angle graph (H): {stats['bond_graph']['nodes']} bond-nodes, {stats['bond_graph']['edges']} angle-edges")
        print(f"  Dihedral angles: {stats['dihedrals']}")
        
        # Create 2x2 viewer grid
        viewer = py3Dmol.view(width=1200, height=800, viewergrid=(2, 2))
        mol_block = Chem.MolToMolBlock(dual_graph.mol)
        
        # TOP-LEFT: Traditional molecular view
        viewer.addModel(mol_block, 'mol', viewer=(0, 0))
        viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}, 
                                       'sphere': {'colorscheme': 'Jmol', 'scale': 0.35}}, viewer=(0, 0))
        
        viewer.addLabel(f'Traditional Graph (G)\n{molecule_name}\nAtoms as nodes, Bonds as edges', 
                       {'position': {'x': 0, 'y': 4, 'z': 0}, 
                        'backgroundColor': 'lightblue',
                        'fontColor': 'black', 'fontSize': 12}, viewer=(0, 0))
        
        # TOP-RIGHT: Bond-as-nodes visualization
        viewer.addModel(mol_block, 'mol', viewer=(0, 1))
        viewer.setStyle({'model': -1}, {'line': {'color': 'lightgray', 'width': 2}}, viewer=(0, 1))
        
        # Add bond nodes
        conf = dual_graph.mol.GetConformer()
        bond_colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink']
        
        for i, bond_info in enumerate(dual_graph.bonds):
            begin_pos = conf.GetAtomPosition(bond_info['begin_atom_idx'])
            end_pos = conf.GetAtomPosition(bond_info['end_atom_idx'])
            
            center = [(begin_pos.x + end_pos.x) / 2, 
                     (begin_pos.y + end_pos.y) / 2, 
                     (begin_pos.z + end_pos.z) / 2]
            
            color = bond_colors[i % len(bond_colors)]
            viewer.addSphere({'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                             'radius': 0.35, 'color': color, 'alpha': 0.9}, viewer=(0, 1))
            
            bond_desc = f"{bond_info['begin_atom_symbol']}-{bond_info['end_atom_symbol']}"
            viewer.addLabel(bond_desc,
                           {'position': {'x': float(center[0]), 'y': float(center[1] + 0.7), 'z': float(center[2])}, 
                            'backgroundColor': color, 'fontColor': 'white', 'fontSize': 9}, viewer=(0, 1))
        
        viewer.addLabel(f'Bonds as Nodes\nEach colored sphere = one bond\nReady to connect via angles', 
                       {'position': {'x': 0, 'y': 4, 'z': 0}, 
                        'backgroundColor': 'lightgreen',
                        'fontColor': 'black', 'fontSize': 12}, viewer=(0, 1))
        
        # BOTTOM-LEFT: Bond angles highlighted
        viewer.addModel(mol_block, 'mol', viewer=(1, 0))
        viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.15}, 
                                       'sphere': {'colorscheme': 'Jmol', 'scale': 0.25}}, viewer=(1, 0))
        
        # Show bond angles
        if len(dual_graph.edge_list) > 0:
            for i in range(0, min(len(dual_graph.edge_list), 8), 2):  # Show first 4 angles
                bond_i_idx, bond_j_idx = dual_graph.edge_list[i]
                shared_atom_idx = dual_graph.shared_atoms[i]
                
                bond_i = dual_graph.bonds[bond_i_idx]
                bond_j = dual_graph.bonds[bond_j_idx]
                
                angle = calculate_bond_angle(bond_i, bond_j, shared_atom_idx)
                if angle > 0:
                    # Highlight the angle
                    shared_pos = conf.GetAtomPosition(shared_atom_idx)
                    
                    other_atom_i = bond_i['end_atom_idx'] if bond_i['begin_atom_idx'] == shared_atom_idx else bond_i['begin_atom_idx']
                    other_atom_j = bond_j['end_atom_idx'] if bond_j['begin_atom_idx'] == shared_atom_idx else bond_j['begin_atom_idx']
                    
                    other_pos_i = conf.GetAtomPosition(other_atom_i)
                    other_pos_j = conf.GetAtomPosition(other_atom_j)
                    
                    # Add thick lines for the angle
                    angle_color = 'red'
                    viewer.addCylinder({'start': {'x': float(shared_pos.x), 'y': float(shared_pos.y), 'z': float(shared_pos.z)},
                                      'end': {'x': float(other_pos_i.x), 'y': float(other_pos_i.y), 'z': float(other_pos_i.z)},
                                      'radius': 0.12, 'color': angle_color, 'alpha': 0.8}, viewer=(1, 0))
                    
                    viewer.addCylinder({'start': {'x': float(shared_pos.x), 'y': float(shared_pos.y), 'z': float(shared_pos.z)},
                                      'end': {'x': float(other_pos_j.x), 'y': float(other_pos_j.y), 'z': float(other_pos_j.z)},
                                      'radius': 0.12, 'color': angle_color, 'alpha': 0.8}, viewer=(1, 0))
        
        viewer.addLabel(f'Bond Angles\nRed lines highlight angles\nAngles become edges in graph H', 
                       {'position': {'x': 0, 'y': 4, 'z': 0}, 
                        'backgroundColor': 'lightcoral',
                        'fontColor': 'black', 'fontSize': 12}, viewer=(1, 0))
        
        # BOTTOM-RIGHT: Complete bond-angle graph
        viewer.addModel(mol_block, 'mol', viewer=(1, 1))
        viewer.setStyle({'model': -1}, {'line': {'color': 'lightgray', 'width': 1}}, viewer=(1, 1))
        
        # Show complete bond-angle graph
        bond_positions = []
        for i, bond_info in enumerate(dual_graph.bonds):
            begin_pos = conf.GetAtomPosition(bond_info['begin_atom_idx'])
            end_pos = conf.GetAtomPosition(bond_info['end_atom_idx'])
            
            center = [(begin_pos.x + end_pos.x) / 2, 
                     (begin_pos.y + end_pos.y) / 2, 
                     (begin_pos.z + end_pos.z) / 2]
            bond_positions.append(center)
            
            # Bond node
            color = bond_colors[i % len(bond_colors)]
            viewer.addSphere({'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                             'radius': 0.3, 'color': color, 'alpha': 0.9}, viewer=(1, 1))
        
        # Add angle edges
        for i in range(0, len(dual_graph.edge_list), 2):
            bond_i_idx, bond_j_idx = dual_graph.edge_list[i]
            if bond_i_idx < len(bond_positions) and bond_j_idx < len(bond_positions):
                pos_i = bond_positions[bond_i_idx]
                pos_j = bond_positions[bond_j_idx]
                
                # Angle edge
                viewer.addCylinder({'start': {'x': float(pos_i[0]), 'y': float(pos_i[1]), 'z': float(pos_i[2])},
                                  'end': {'x': float(pos_j[0]), 'y': float(pos_j[1]), 'z': float(pos_j[2])},
                                  'radius': 0.08, 'color': 'black', 'alpha': 0.7}, viewer=(1, 1))
        
        viewer.addLabel(f'Complete Bond-Angle Graph (H)\nSpheres = Bond nodes\nBlack lines = Angle edges', 
                       {'position': {'x': 0, 'y': 4, 'z': 0}, 
                        'backgroundColor': 'lightyellow',
                        'fontColor': 'black', 'fontSize': 12}, viewer=(1, 1))
        
        viewer.zoomTo()
        
        print("✅ Complete dual graph visualization ready!")
        print("💡 Top panels show the transformation from atoms to bonds")
        print("💡 Bottom panels show angle extraction and final dual graph")
        print("🔄 Rotate all panels to understand the full concept")
        
        return viewer
        
    except Exception as e:
        print(f"❌ Error creating dual graph visualization: {e}")
        return None

# Create complete dual graph visualization
complete_viz = create_complete_dual_graph_3d("CC(=O)C", "Acetone")
if complete_viz:
    complete_viz.show()

# + [markdown] id="J8Zsc4zOxqJN"
# ## 8. Visualization of Alternative Representations <a name="visualization-of-alternative-representations"></a>
#
# Let's create visualizations to understand the different graph representations:

# + colab={"base_uri": "https://localhost:8080/", "height": 800} id="visualization-alternative"
def visualize_dual_graphs(dual_graph, show_3d=True):
    """
    Visualize both traditional and bond-angle graphs side by side.
    """
    if show_3d:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Molecular Structure", "Traditional Graph (G)",
                          "Bond Graph Nodes", "Bond-Angle Graph (H)"),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
    else:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Traditional Graph (G)", "Bond-Angle Graph (H)"),
            specs=[[{"type": "xy"}, {"type": "xy"}]]
        )
    
    # 1. Molecular structure (if 3D available)
    if show_3d and dual_graph.mol.GetNumConformers() > 0:
        conf = dual_graph.mol.GetConformer()
        
        # Get atom positions
        atom_positions = []
        atom_symbols = []
        for i, atom in enumerate(dual_graph.mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atom_positions.append([pos.x, pos.y, pos.z])
            atom_symbols.append(atom.GetSymbol())
        
        atom_positions = np.array(atom_positions)
        
        # Plot atoms
        fig.add_trace(
            go.Scatter(
                x=atom_positions[:, 0], y=atom_positions[:, 1],
                mode='markers+text',
                marker=dict(size=10, color='lightblue'),
                text=atom_symbols,
                textposition="middle center",
                name="Atoms"
            ),
            row=1, col=1
        )
        
        # Plot bonds
        for bond in dual_graph.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            fig.add_trace(
                go.Scatter(
                    x=[atom_positions[i, 0], atom_positions[j, 0]],
                    y=[atom_positions[i, 1], atom_positions[j, 1]],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # 2. Traditional atom-bond graph
    G_traditional = nx.Graph()
    
    # Add atoms as nodes
    for i in range(dual_graph.mol.GetNumAtoms()):
        atom = dual_graph.mol.GetAtomWithIdx(i)
        G_traditional.add_node(i, symbol=atom.GetSymbol())
    
    # Add bonds as edges
    for bond in dual_graph.mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G_traditional.add_edge(i, j)
    
    # Layout for traditional graph
    pos_trad = nx.spring_layout(G_traditional, seed=42)
    
    # Plot traditional graph nodes
    node_x = [pos_trad[node][0] for node in G_traditional.nodes()]
    node_y = [pos_trad[node][1] for node in G_traditional.nodes()]
    node_text = [dual_graph.mol.GetAtomWithIdx(node).GetSymbol() for node in G_traditional.nodes()]
    
    row_idx = 1 if show_3d else 1
    col_idx = 2 if show_3d else 1
    
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=15, color='lightcoral'),
            text=node_text,
            textposition="middle center",
            name="Atoms (G)"
        ),
        row=row_idx, col=col_idx
    )
    
    # Plot traditional graph edges
    for edge in G_traditional.edges():
        x0, y0 = pos_trad[edge[0]]
        x1, y1 = pos_trad[edge[1]]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ),
            row=row_idx, col=col_idx
        )
    
    # 3. Bond graph visualization
    if len(dual_graph.bonds) > 0:
        # Show bond nodes
        if show_3d:
            # Plot bond centers
            bond_centers = []
            bond_labels = []
            for bond in dual_graph.bonds:
                if 'begin_pos' in bond and 'end_pos' in bond:
                    center = (bond['begin_pos'] + bond['end_pos']) / 2
                    bond_centers.append(center[:2])  # Use only x, y for 2D plot
                    bond_labels.append(f"{bond['begin_atom_symbol']}-{bond['end_atom_symbol']}")
            
            if bond_centers:
                bond_centers = np.array(bond_centers)
                fig.add_trace(
                    go.Scatter(
                        x=bond_centers[:, 0], y=bond_centers[:, 1],
                        mode='markers+text',
                        marker=dict(size=8, color='lightgreen', symbol='square'),
                        text=bond_labels,
                        textposition="top center",
                        name="Bonds"
                    ),
                    row=2, col=1
                )
        
        # 4. Bond-angle graph
        if dual_graph.bond_adjacency.size > 0:
            G_bond = nx.Graph()
            
            # Add bonds as nodes
            for i, bond in enumerate(dual_graph.bonds):
                bond_label = f"{bond['begin_atom_symbol']}-{bond['end_atom_symbol']}"
                G_bond.add_node(i, label=bond_label)
            
            # Add angle-based edges
            for i in range(0, len(dual_graph.edge_list), 2):  # Skip every other (bidirectional)
                bond_i, bond_j = dual_graph.edge_list[i]
                G_bond.add_edge(bond_i, bond_j)
            
            if len(G_bond.nodes()) > 0:
                # Layout for bond graph
                pos_bond = nx.spring_layout(G_bond, seed=42)
                
                # Plot bond graph nodes
                bond_node_x = [pos_bond[node][0] for node in G_bond.nodes()]
                bond_node_y = [pos_bond[node][1] for node in G_bond.nodes()]
                bond_node_text = [G_bond.nodes[node]['label'] for node in G_bond.nodes()]
                
                row_idx = 2 if show_3d else 1
                col_idx = 2 if show_3d else 2
                
                fig.add_trace(
                    go.Scatter(
                        x=bond_node_x, y=bond_node_y,
                        mode='markers+text',
                        marker=dict(size=12, color='gold', symbol='square'),
                        text=bond_node_text,
                        textposition="top center",
                        name="Bonds (H)"
                    ),
                    row=row_idx, col=col_idx
                )
                
                # Plot bond graph edges (angles)
                for edge in G_bond.edges():
                    x0, y0 = pos_bond[edge[0]]
                    x1, y1 = pos_bond[edge[1]]
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1], y=[y0, y1],
                            mode='lines',
                            line=dict(color='red', width=2),
                            showlegend=False
                        ),
                        row=row_idx, col=col_idx
                    )
    
    # Update layout
    fig.update_layout(
        title=f"Dual Graph Representation: {dual_graph.smiles}",
        showlegend=True,
        height=600 if show_3d else 400
    )
    
    return fig

# Visualize dual graphs for selected molecules
molecules_to_visualize = ["Ethanol", "Benzene"]

for name in molecules_to_visualize:
    if name in test_molecules_dual:
        smiles = test_molecules_dual[name]
        
        print(f"Creating dual graph visualization for {name}...")
        dual_graph = DualMolecularGraph(smiles)
        dual_graph.build_graphs(add_3d=True)
        
        fig = visualize_dual_graphs(dual_graph, show_3d=True)
        fig.show()

# + [markdown] id="interactive-comparison"
# ### Interactive 3D Comparison of Graph Representations
#
# Let's create an interactive comparison tool that lets you explore different molecules:

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="interactive-comparison-3d"
def create_interactive_molecule_explorer():
    """
    Create an interactive explorer for different molecular representations.
    """
    if not PY3DMOL_AVAILABLE:
        print("⚠️ py3Dmol not available - cannot create interactive explorer")
        return
    
    print("🎮 Interactive Molecular Graph Explorer")
    print("=" * 45)
    
    # Selection of interesting molecules for comparison
    explorer_molecules = {
        "Water": "O",
        "Methane": "C",
        "Ethanol": "CCO",
        "Benzene": "c1ccccc1",
        "Acetone": "CC(=O)C",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    }
    
    print("Available molecules for exploration:")
    for name, smiles in explorer_molecules.items():
        print(f"  • {name}: {smiles}")
    
    # Create visualizations for each molecule type
    results = []
    
    for name, smiles in list(explorer_molecules.items())[:4]:  # Show first 4
        try:
            print(f"\n🔍 Analyzing {name}...")
            
            # Create dual graph
            dual_graph = DualMolecularGraph(smiles)
            dual_graph.build_graphs(add_3d=True)
            stats = dual_graph.get_graph_statistics()
            
            result = {
                'name': name,
                'smiles': smiles,
                'atoms': stats['molecule']['atoms'],
                'bonds': stats['molecule']['bonds'],
                'atom_graph_edges': stats['atom_graph']['edges'],
                'bond_graph_nodes': stats['bond_graph']['nodes'],
                'bond_graph_edges': stats['bond_graph']['edges'],
                'dihedrals': stats['dihedrals']
            }
            
            print(f"  Traditional graph: {result['atoms']} atoms, {result['atom_graph_edges']} edges")
            print(f"  Bond-angle graph: {result['bond_graph_nodes']} nodes, {result['bond_graph_edges']} edges")
            print(f"  Dihedral angles: {result['dihedrals']}")
            
            results.append(result)
            
            # Create visualization for this molecule
            viewer = create_complete_dual_graph_3d(smiles, name)
            if viewer:
                print(f"  ✅ 3D visualization created for {name}")
            
        except Exception as e:
            print(f"  ❌ Error with {name}: {e}")
    
    # Summary comparison
    if results:
        print("\n📊 COMPARATIVE ANALYSIS")
        print("=" * 30)
        
        print(f"{'Molecule':<12} {'Atoms':<6} {'Bonds':<6} {'Bond-Nodes':<11} {'Angle-Edges':<12} {'Dihedrals':<10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['name']:<12} {result['atoms']:<6} {result['bonds']:<6} {result['bond_graph_nodes']:<11} {result['bond_graph_edges']:<12} {result['dihedrals']:<10}")
        
        print("\n🎯 Key Insights:")
        print("• Larger molecules → More bond nodes and angle edges")
        print("• Ring structures → Different angle distributions")
        print("• Flexible molecules → More dihedral angles")
        print("• Dual graphs capture geometric relationships missed by traditional graphs")
    
    print("\n💡 Try exploring different molecules to see how graph complexity changes!")

# Run the interactive explorer
create_interactive_molecule_explorer()

# + [markdown] id="DTkyWf7BzFY1"
# ## 9. Comparing Different Graph Paradigms <a name="comparing-different-graph-paradigms"></a>
#
# Let's systematically compare traditional vs. alternative graph representations:

# + colab={"base_uri": "https://localhost:8080/", "height": 600} id="comparison-analysis"
def comprehensive_graph_comparison(smiles_list):
    """
    Compare different graph representations across multiple molecules.
    """
    results = []
    
    for smiles in smiles_list:
        try:
            dual_graph = DualMolecularGraph(smiles)
            dual_graph.build_graphs(add_3d=True)
            stats = dual_graph.get_graph_statistics()
            
            # Calculate additional metrics
            atom_graph_density = (2 * stats['atom_graph']['edges']) / (
                stats['atom_graph']['nodes'] * (stats['atom_graph']['nodes'] - 1)
            ) if stats['atom_graph']['nodes'] > 1 else 0
            
            bond_graph_density = (2 * stats['bond_graph']['edges']) / (
                stats['bond_graph']['nodes'] * (stats['bond_graph']['nodes'] - 1)
            ) if stats['bond_graph']['nodes'] > 1 else 0
            
            # Analyze angle distribution
            angles = []
            if dual_graph.angle_edge_features.size > 0:
                # Extract angles from edge features (3rd column is normalized angle)
                angles = dual_graph.angle_edge_features[::2, 2] * 180  # Convert back to degrees
            
            result = {
                'smiles': smiles,
                'n_atoms': stats['molecule']['atoms'],
                'n_bonds': stats['molecule']['bonds'],
                'n_rings': stats['molecule']['rings'],
                
                # Traditional graph
                'atom_nodes': stats['atom_graph']['nodes'],
                'atom_edges': stats['atom_graph']['edges'],
                'atom_density': atom_graph_density,
                
                # Bond graph
                'bond_nodes': stats['bond_graph']['nodes'],
                'bond_edges': stats['bond_graph']['edges'],
                'bond_density': bond_graph_density,
                
                # Geometric information
                'n_angles': len(angles),
                'mean_angle': np.mean(angles) if len(angles) > 0 else 0,
                'angle_std': np.std(angles) if len(angles) > 0 else 0,
                'n_dihedrals': stats['dihedrals'],
                
                # Information richness
                'atom_features': stats['atom_graph']['node_features'],
                'bond_features': stats['bond_graph']['node_features'],
                'angle_features': stats['bond_graph']['edge_features']
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
    
    return pd.DataFrame(results)

# Comprehensive comparison
comparison_molecules = [
    "CO",  # Methanol
    "CCO",  # Ethanol
    "CCC",  # Propane
    "CCCC",  # Butane
    "c1ccccc1",  # Benzene
    "C1CCC1",  # Cyclobutane
    "CC(=O)C",  # Acetone
    "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
]

print("Comprehensive Graph Comparison:")
print("=" * 50)

comparison_df = comprehensive_graph_comparison(comparison_molecules)

# Display results
print("\nGraph Statistics Summary:")
display_cols = ['smiles', 'n_atoms', 'n_bonds', 'atom_edges', 'bond_nodes', 
                'bond_edges', 'n_angles', 'mean_angle', 'n_dihedrals']
print(comparison_df[display_cols].to_string(index=False))

# Calculate information gain from dual representation
comparison_df['edge_ratio'] = comparison_df['bond_edges'] / comparison_df['atom_edges'].replace(0, 1)
comparison_df['information_gain'] = (
    comparison_df['bond_features'] + comparison_df['angle_features']
) / comparison_df['atom_features'].replace(0, 1)

print(f"\nInformation Analysis:")
print(f"Average edge ratio (bond_edges/atom_edges): {comparison_df['edge_ratio'].mean():.2f}")
print(f"Average information gain: {comparison_df['information_gain'].mean():.2f}")

# Visualization of comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Node count comparison
axes[0, 0].scatter(comparison_df['atom_nodes'], comparison_df['bond_nodes'], alpha=0.7)
axes[0, 0].plot([0, comparison_df['atom_nodes'].max()], [0, comparison_df['atom_nodes'].max()], 'r--', alpha=0.5)
axes[0, 0].set_xlabel('Atom Graph Nodes')
axes[0, 0].set_ylabel('Bond Graph Nodes')
axes[0, 0].set_title('Node Count Comparison')

# 2. Edge count comparison
axes[0, 1].scatter(comparison_df['atom_edges'], comparison_df['bond_edges'], alpha=0.7)
axes[0, 1].set_xlabel('Atom Graph Edges')
axes[0, 1].set_ylabel('Bond Graph Edges')
axes[0, 1].set_title('Edge Count Comparison')

# 3. Angle distribution
angles_data = []
for _, row in comparison_df.iterrows():
    if row['n_angles'] > 0:
        angles_data.extend([row['mean_angle']] * int(row['n_angles']))

if angles_data:
    axes[1, 0].hist(angles_data, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Bond Angle (degrees)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Bond Angle Distribution')

# 4. Information richness
comparison_df['total_atom_info'] = comparison_df['atom_nodes'] * comparison_df['atom_features']
comparison_df['total_bond_info'] = (comparison_df['bond_nodes'] * comparison_df['bond_features'] + 
                                  comparison_df['bond_edges'] * comparison_df['angle_features'])

axes[1, 1].scatter(comparison_df['total_atom_info'], comparison_df['total_bond_info'], alpha=0.7)
axes[1, 1].set_xlabel('Traditional Graph Information')
axes[1, 1].set_ylabel('Bond-Angle Graph Information')
axes[1, 1].set_title('Information Content Comparison')

plt.tight_layout()
plt.show()

# + [markdown] id="0TU76Wlv25Kt"
# ## 10. Advanced Features and Applications <a name="advanced-features-and-applications"></a>
#
# Let's explore advanced applications of dual graph representations:

# + colab={"base_uri": "https://localhost:8080/"} id="advanced-applications"
class EnhancedDualGraph(DualMolecularGraph):
    """
    Enhanced dual graph with additional geometric and chemical features.
    """
    
    def __init__(self, smiles):
        super().__init__(smiles)
        self.pharmacophore_features = {}
        self.geometric_descriptors = {}
    
    def add_pharmacophore_features(self):
        """Add pharmacophore-related features."""
        if self.mol is None:
            return
        
        # Identify pharmacophore features
        features = {
            'h_bond_donors': 0,
            'h_bond_acceptors': 0,
            'aromatic_atoms': 0,
            'hydrophobic_atoms': 0,
            'charged_atoms': 0
        }
        
        for atom in self.mol.GetAtoms():
            # Hydrogen bond donors (N-H, O-H)
            if atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0:
                features['h_bond_donors'] += 1
            
            # Hydrogen bond acceptors (N, O with lone pairs)
            if atom.GetSymbol() in ['N', 'O'] and atom.GetFormalCharge() <= 0:
                features['h_bond_acceptors'] += 1
            
            # Aromatic atoms
            if atom.GetIsAromatic():
                features['aromatic_atoms'] += 1
            
            # Hydrophobic atoms (C, not in aromatic ring)
            if atom.GetSymbol() == 'C' and not atom.GetIsAromatic():
                features['hydrophobic_atoms'] += 1
            
            # Charged atoms
            if atom.GetFormalCharge() != 0:
                features['charged_atoms'] += 1
        
        self.pharmacophore_features = features
    
    def calculate_geometric_descriptors(self):
        """Calculate advanced geometric descriptors."""
        if self.mol is None or self.mol.GetNumConformers() == 0:
            return
        
        conf = self.mol.GetConformer()
        
        # Get all atom positions
        positions = []
        for i in range(self.mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append(np.array([pos.x, pos.y, pos.z]))
        positions = np.array(positions)
        
        # Calculate descriptors
        descriptors = {}
        
        # Molecular volume (convex hull approximation)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(positions)
            descriptors['convex_hull_volume'] = hull.volume
            descriptors['surface_area'] = hull.area
        except:
            descriptors['convex_hull_volume'] = 0.0
            descriptors['surface_area'] = 0.0
        
        # Principal moments of inertia
        center_of_mass = np.mean(positions, axis=0)
        centered_coords = positions - center_of_mass
        
        # Inertia tensor
        I = np.zeros((3, 3))
        for coord in centered_coords:
            x, y, z = coord
            I[0, 0] += y*y + z*z
            I[1, 1] += x*x + z*z
            I[2, 2] += x*x + y*y
            I[0, 1] -= x*y
            I[0, 2] -= x*z
            I[1, 2] -= y*z
        
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
        
        eigenvalues = np.linalg.eigvals(I)
        eigenvalues = np.sort(eigenvalues)
        
        descriptors['principal_moments'] = eigenvalues
        descriptors['asphericity'] = eigenvalues[2] - 0.5*(eigenvalues[0] + eigenvalues[1])
        descriptors['acylindricity'] = eigenvalues[1] - eigenvalues[0]
        
        # Radius of gyration
        descriptors['radius_of_gyration'] = np.sqrt(np.mean(np.sum(centered_coords**2, axis=1)))
        
        # Molecular diameter
        pairwise_distances = []
        n_atoms = len(positions)
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                pairwise_distances.append(dist)
        
        if pairwise_distances:
            descriptors['diameter'] = max(pairwise_distances)
            descriptors['mean_distance'] = np.mean(pairwise_distances)
        else:
            descriptors['diameter'] = 0.0
            descriptors['mean_distance'] = 0.0
        
        self.geometric_descriptors = descriptors
    
    def build_enhanced_graphs(self, add_3d=True):
        """Build graphs with enhanced features."""
        self.build_graphs(add_3d=add_3d)
        self.add_pharmacophore_features()
        if add_3d:
            self.calculate_geometric_descriptors()
    
    def get_enhanced_statistics(self):
        """Get comprehensive statistics including new features."""
        stats = self.get_graph_statistics()
        
        stats['pharmacophore'] = self.pharmacophore_features
        stats['geometric'] = self.geometric_descriptors
        
        return stats

def analyze_enhanced_features(smiles_list):
    """Analyze enhanced features across molecules."""
    results = []
    
    for smiles in smiles_list:
        try:
            enhanced_graph = EnhancedDualGraph(smiles)
            enhanced_graph.build_enhanced_graphs(add_3d=True)
            stats = enhanced_graph.get_enhanced_statistics()
            
            result = {
                'smiles': smiles,
                'n_atoms': stats['molecule']['atoms'],
                'n_bonds': stats['molecule']['bonds']
            }
            
            # Add pharmacophore features
            result.update({f'pharm_{k}': v for k, v in stats['pharmacophore'].items()})
            
            # Add geometric features
            if stats['geometric']:
                result.update({
                    'volume': stats['geometric'].get('convex_hull_volume', 0),
                    'surface_area': stats['geometric'].get('surface_area', 0),
                    'radius_gyration': stats['geometric'].get('radius_of_gyration', 0),
                    'asphericity': stats['geometric'].get('asphericity', 0),
                    'diameter': stats['geometric'].get('diameter', 0)
                })
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
    
    return pd.DataFrame(results)

# Test enhanced features
print("Enhanced Dual Graph Analysis:")
print("=" * 40)

enhanced_molecules = [
    "CO",  # Methanol
    "CCO",  # Ethanol  
    "c1ccccc1",  # Benzene
    "CC(=O)N",  # Acetamide
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"  # Caffeine
]

enhanced_df = analyze_enhanced_features(enhanced_molecules)

if not enhanced_df.empty:
    print("\nEnhanced Features Summary:")
    
    # Pharmacophore features
    pharm_cols = [col for col in enhanced_df.columns if col.startswith('pharm_')]
    if pharm_cols:
        print("\nPharmacophore Features:")
        print(enhanced_df[['smiles'] + pharm_cols].to_string(index=False))
    
    # Geometric features
    geom_cols = ['volume', 'surface_area', 'radius_gyration', 'asphericity', 'diameter']
    available_geom_cols = [col for col in geom_cols if col in enhanced_df.columns]
    if available_geom_cols:
        print("\nGeometric Descriptors:")
        print(enhanced_df[['smiles'] + available_geom_cols].round(2).to_string(index=False))

# + [markdown] id="zMjXPTzUvnYk"
# ### ✅ Checkpoint: Understanding Alternative Representations
#
# To reinforce your understanding, try answering these questions:
#
# 1. **Question**: What is the main advantage of using bonds as nodes in a molecular graph?
#    - **Answer**: It allows explicit representation of bond angles and geometric relationships between bonds, capturing spatial information that traditional atom-bond graphs miss.
#
# 2. **Question**: How does the bond-angle graph capture more geometric information than traditional graphs?
#    - **Answer**: By making bonds the nodes and angles the edges, it explicitly encodes the 3D spatial relationships between chemical bonds, including bond angles and torsional information.
#
# 3. **Question**: When would you prefer a dual graph representation over a traditional one?
#    - **Answer**: When geometric properties are crucial (drug-receptor binding, catalysis, stereochemistry), when predicting properties that depend on molecular shape, or when working with conformationally flexible molecules.
#
# 4. **Question**: What information is lost when using only traditional atom-bond graphs?
#    - **Answer**: Bond angles, dihedral angles, spatial relationships between non-bonded atoms, and overall molecular geometry and shape information.

# + [markdown] id="zn8cSorXv7SI"
# ## 11. Conclusion <a name="conclusion"></a>
#
# This tutorial introduced alternative molecular graph representations that go beyond the traditional "atoms as nodes, bonds as edges" paradigm. Here are the key takeaways:
#
# ### Key Concepts Learned
#
# 1. **Dual Graph Representation**: The GemNet-inspired approach uses two complementary graphs:
#    - **Traditional Graph (G)**: Atoms as nodes, bonds as edges
#    - **Bond-Angle Graph (H)**: Bonds as nodes, angles as edges
#
# 2. **Enhanced Geometric Information**: Alternative representations capture:
#    - Bond angles and their distribution
#    - Dihedral angles and torsional flexibility  
#    - Spatial relationships between chemical bonds
#    - 3D molecular geometry explicitly
#
# 3. **Richer Feature Space**: Dual graphs provide:
#    - More detailed chemical information
#    - Explicit geometric features
#    - Better representation of molecular flexibility
#    - Enhanced property prediction capabilities
#
# 4. **Implementation Strategies**:
#    - Bond feature extraction and encoding
#    - Angle calculation and edge construction
#    - Dual graph visualization techniques
#    - PyTorch Geometric integration
#
# ### Advantages of Alternative Representations
#
# **Scientific Benefits**:
# - **Better chemical intuition**: Aligns with how chemists think about molecular structure
# - **Explicit geometry**: Captures spatial relationships crucial for many properties
# - **Conformational awareness**: Better handles molecular flexibility
# - **Multi-scale information**: Combines topological and geometric features
#
# **Machine Learning Benefits**:
# - **Improved performance**: Better property prediction accuracy
# - **Richer features**: More information for model training
# - **Geometric awareness**: Models understand molecular shape
# - **Flexibility**: Can be combined with traditional approaches
#
# ### When to Use Alternative Representations
#
# **Prefer dual/alternative graphs when**:
# - Molecular geometry is crucial (drug design, catalysis)
# - Working with conformationally flexible molecules
# - Predicting properties related to molecular shape/binding
# - Need explicit representation of non-covalent interactions
# - Working with stereochemically important molecules
#
# **Prefer traditional graphs when**:
# - Computational efficiency is critical
# - Working with very large datasets
# - Topology is more important than geometry
# - 3D coordinates are not available or reliable
#
# ### Future Directions
#
# Alternative molecular representations open up exciting possibilities:
# 1. **Multi-scale GNNs**: Combining multiple graph representations
# 2. **Dynamic graphs**: Handling conformational ensembles
# 3. **Hierarchical representations**: From atoms to bonds to functional groups
# 4. **Hybrid approaches**: Combining 2D and 3D information optimally
#
# ### Comparison Summary
#
# | Aspect | Traditional (Atom-Bond) | Dual (Bond-Angle) |
# |--------|-------------------------|-------------------|
# | **Nodes** | Atoms | Atoms + Bonds |
# | **Edges** | Bonds | Bonds + Angles |
# | **Geometry** | Implicit | Explicit |
# | **Features** | Atomic properties | Atomic + Geometric |
# | **Flexibility** | Limited | High |
# | **Computation** | Fast | Moderate |
# | **Applications** | General chemistry | Structure-property |
#
# ### Additional Resources
#
# 1. **"Geometry-enhanced molecular representation learning for property prediction"** - Fang et al., Nature Machine Intelligence (2022)
# 2. **"DimeNet: Directional Message Passing for Molecular Graphs"** - Klicpera et al.
# 3. **"PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments"** - Unke & Meuwly
# 4. **RDKit Documentation** - Comprehensive guide to molecular geometry calculations
#
# ### Final Thoughts
#
# Alternative molecular graph representations represent a significant advancement in molecular machine learning. By explicitly capturing geometric information that traditional graphs miss, they enable more accurate and chemically meaningful property predictions. As computational resources continue to improve, these richer representations will likely become standard tools in computational chemistry and drug discovery.
#
# The choice between different representations should be guided by your specific application, computational constraints, and the importance of geometric information for your problem. Often, the best approach may involve combining multiple representations to capture both topological and geometric aspects of molecular structure.

# + [markdown] id="zMjXPTzUvnYk_final"
# ### ✅ Final Challenge: Complete Alternative Representation Analysis
#
# Put your knowledge to the test with this comprehensive exercise:
#
# **Challenge**: Choose a biologically active molecule (e.g., a drug, natural product, or enzyme inhibitor) and perform a complete alternative representation analysis:
#
# 1. **Multi-representation construction**: Create traditional, bond-angle, and enhanced dual graphs
# 2. **Geometric analysis**: Calculate all bond angles, dihedral angles, and geometric descriptors
# 3. **Pharmacophore mapping**: Identify and encode pharmacophore features
# 4. **Conformational sampling**: Generate multiple conformers and analyze geometric variation
# 5. **Visualization**: Create compelling visualizations of all graph representations
# 6. **Comparative analysis**: Compare information content and potential ML performance
# 7. **PyG integration**: Convert all representations to PyTorch Geometric format
#
# **Bonus Tasks**:
# - Implement a graph neural network that uses both traditional and bond-angle graphs
# - Design a custom message passing scheme that leverages geometric information
# - Compare your approach with existing molecular GNN architectures
#
# This comprehensive exercise will solidify your understanding of alternative molecular representations and prepare you for advanced applications in molecular machine learning!
