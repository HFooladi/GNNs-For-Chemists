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
# <a href="https://colab.research.google.com/github/HFooladi/GNNs-For-Chemists/blob/main/notebooks/01.1_GNN_3D_representation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="bXCYDK0NkdpC"
# # 3D Molecular Representation for Graph Neural Networks: Tutorial
#
# ## Table of Contents
# 1. [Setup and Installation](#setup-and-installation)
# 2. [Introduction to 3D Molecular Representation](#introduction)
# 3. [From 2D to 3D: Understanding Conformers](#from-2d-to-3d-understanding-conformers)
# 4. [Generating 3D Molecular Conformations](#generating-3d-molecular-conformations)
# 5. [3D Graph Construction from Conformers](#3d-graph-construction-from-conformers)
# 6. [Distance-Based Edge Features](#distance-based-edge-features)
# 7. [3D Visualization of Molecular Graphs](#3d-visualization-of-molecular-graphs)
# 8. [Comparing 2D vs 3D Representations](#comparing-2d-vs-3d-representations)
# 9. [Advanced 3D Features](#advanced-3d-features)
# 10. [Conclusion](#conclusion)

# + [markdown] id="XHX_uMbYkkX7"
# ## 1. Setup and Installation <a name="setup-and-installation"></a>
#
# Building on the previous tutorial, we'll now explore 3D molecular representations. We'll need additional libraries for 3D visualization and conformer generation:
# - **Plotly**: For interactive 3D visualizations
# - **Py3Dmol**: For molecular 3D visualization
# - **Additional RDKit functions**: For conformer generation

# + colab={"base_uri": "https://localhost:8080/"} id="A7-zyVlhhvwk" outputId="11f359c0-1a20-4e60-c6ed-e5425d1df826"
#@title install required libraries
# !pip install -q rdkit
# !pip install -q torch_geometric
# !pip install -q plotly
# !pip install -q py3dmol

# + [markdown] id="cKHLDrQ1mwDo"
# Now let's import the libraries we'll need throughout this tutorial:

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
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# PyTorch and PyTorch Geometric
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import to_networkx

# 3D visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
    print("✅ py3Dmol available - professional molecular visualization enabled!")
except ImportError:
    PY3DMOL_AVAILABLE = False
    print("⚠️ py3Dmol not available - install with: pip install py3dmol")
    print("Some advanced 3D visualizations will use fallback methods")

# NetworkX for graph visualization
import networkx as nx

# Import enhanced visualization utilities
try:
    from utils.enhanced_3d_visualizations import (
        create_py3dmol_stereochemistry_example,
        create_py3dmol_conformer_explorer,
        create_py3dmol_binding_pocket_demo,
        create_py3dmol_graph_edges_demo,
        plot_3d_molecular_graph_py3dmol
    )
    ENHANCED_VIZ_AVAILABLE = True
    print("🚀 Enhanced 3D visualization module loaded!")
except ImportError:
    ENHANCED_VIZ_AVAILABLE = False
    print("📝 Enhanced visualizations not found - using standard functions")

# Set plotting style
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("Set2")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# + [markdown] id="r1xzRB8EkUtQ"
# ## 2. Introduction to 3D Molecular Representation <a name="introduction"></a>
#
# In the previous tutorial, we learned how to represent molecules as 2D graphs based on their connectivity (which atoms are bonded to which). However, **chemistry happens in 3D space**. The actual shape and spatial arrangement of atoms in a molecule profoundly affects its properties:
#
# - **Drug-receptor interactions** depend on the 3D shape complementarity
# - **Catalytic activity** is determined by the spatial arrangement of active sites
# - **Molecular properties** like boiling point and solubility are influenced by molecular volume and surface area
# - **Stereochemistry** cannot be captured without 3D information
#
# ### Why 3D Matters in Chemistry
#
# Consider these examples where 3D structure is crucial:
# 1. **Enantiomers**: Mirror-image molecules with identical connectivity but different biological activities
# 2. **Conformational flexibility**: The same molecule can adopt different 3D shapes
# 3. **Steric hindrance**: Bulky groups preventing reactions due to spatial constraints
# 4. **Binding pockets**: Drug molecules must fit precisely into protein binding sites
#
# ### Learning Objectives
#
# By the end of this tutorial, you will be able to:
# - **Generate** 3D molecular conformations from SMILES strings
# - **Construct** distance-based molecular graphs incorporating spatial information
# - **Visualize** molecules in 3D space with their corresponding graph representations
# - **Compare** 2D topology-based vs 3D geometry-based molecular graphs
# - **Understand** how 3D features enhance molecular property prediction

# + [markdown] id="VzsxoQxEm9dM"
# ## 3. From 2D to 3D: Understanding Conformers <a name="from-2d-to-3d-understanding-conformers"></a>

# + [markdown] id="zF4rEulu6gUI"
# ### What is a Molecular Conformer?
#
# A **conformer** (or conformation) is a specific 3D arrangement of atoms in a molecule that can be achieved by rotation around single bonds without breaking any covalent bonds. Unlike 2D structural representations that only show connectivity, conformers capture the actual spatial positions of atoms.
#
# Key concepts:
# - **Constitutional isomers**: Different connectivity (different molecules)
# - **Conformers**: Same connectivity, different 3D arrangements (same molecule, different shapes)  
# - **Conformational energy**: Energy required to adopt a specific shape
# - **Preferred conformations**: Low-energy, stable 3D arrangements

# + [markdown] id="8hNh4BnQnQYL"
# Let's start with a simple example to understand the difference between 2D connectivity and 3D geometry:

# + colab={"base_uri": "https://localhost:8080/", "height": 330} id="uM4e1dRlndV_" outputId="cbc17194-2921-4177-fd9c-5517983b637f"
def demonstrate_2d_vs_3d_concept():
    """
    Demonstrate the critical difference between 2D connectivity and 3D spatial arrangement
    using stereoisomers - molecules that have identical 2D graphs but different 3D structures.
    """
    print("🔬 Why 3D Matters: The Case of Stereoisomers")
    print("=" * 50)
    print("Demonstration: Two molecules with IDENTICAL 2D connectivity but DIFFERENT biological activities!")
    print()
    
    # Use stereoisomers that clearly show the 2D vs 3D difference
    # R and S enantiomers of 2-butanol - same connectivity, different 3D arrangement
    r_smiles = "CC[C@H](C)O"  # R-2-butanol  
    s_smiles = "CC[C@@H](C)O"  # S-2-butanol
    
    # Create molecules
    mol_r = Chem.MolFromSmiles(r_smiles)
    mol_s = Chem.MolFromSmiles(s_smiles)
    
    # Add explicit hydrogens for proper 3D representation
    mol_r = Chem.AddHs(mol_r)
    mol_s = Chem.AddHs(mol_s)
    
    # Generate 3D conformers
    try:
        AllChem.EmbedMolecule(mol_r, randomSeed=42)
        AllChem.EmbedMolecule(mol_s, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_r)
        AllChem.MMFFOptimizeMolecule(mol_s)
        
        conformers_generated = True
    except:
        conformers_generated = False
        print("⚠️ Could not generate 3D conformers, showing 2D analysis only")
    
    # Analysis of molecular properties
    print("Molecular Analysis:")
    print(f"  R-enantiomer SMILES: {r_smiles}")
    print(f"  S-enantiomer SMILES: {s_smiles}")
    print(f"  Molecular formula: {CalcMolFormula(mol_r)}")
    print(f"  Number of atoms: {mol_r.GetNumAtoms()}")
    print(f"  Number of bonds: {mol_r.GetNumBonds()}")
    print()
    
    # Create comprehensive visualization
    if PY3DMOL_AVAILABLE and conformers_generated:
        print("🧬 3D INTERACTIVE COMPARISON")
        print("-" * 30)
        
        # Create side-by-side py3Dmol viewer
        viewer = py3Dmol.view(width=1200, height=400, viewergrid=(1, 2))
        
        # Add R-enantiomer (left side)
        mol_block_r = Chem.MolToMolBlock(mol_r)
        viewer.addModel(mol_block_r, 'mol', viewer=(0, 0))
        viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}, 
                                       'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}}, viewer=(0, 0))
        
        # Highlight the chiral center
        chiral_atom_idx = None
        for atom in mol_r.GetAtoms():
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                chiral_atom_idx = atom.GetIdx()
                break
        
        if chiral_atom_idx is not None:
            pos = mol_r.GetConformer().GetAtomPosition(chiral_atom_idx)
            viewer.addSphere({'center': {'x': float(pos.x), 'y': float(pos.y), 'z': float(pos.z)},
                             'radius': 0.55, 'color': 'yellow', 'alpha': 0.8}, viewer=(0, 0))
        
        viewer.addLabel('R-2-butanol\n(Right-handed)\nSame 2D graph!', 
                       {'position': {'x': 0, 'y': 4, 'z': 0}, 
                        'backgroundColor': 'lightblue',
                        'fontColor': 'black', 'fontSize': 14}, viewer=(0, 0))
        
        # Add S-enantiomer (right side)
        mol_block_s = Chem.MolToMolBlock(mol_s)
        viewer.addModel(mol_block_s, 'mol', viewer=(0, 1))
        viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}, 
                                       'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}}, viewer=(0, 1))
        
        # Highlight the chiral center
        if chiral_atom_idx is not None:
            pos_s = mol_s.GetConformer().GetAtomPosition(chiral_atom_idx)
            viewer.addSphere({'center': {'x': float(pos_s.x), 'y': float(pos_s.y), 'z': float(pos_s.z)},
                             'radius': 0.55, 'color': 'yellow', 'alpha': 0.8}, viewer=(0, 1))
        
        viewer.addLabel('S-2-butanol\n(Left-handed)\nSame 2D graph!', 
                       {'position': {'x': 0, 'y': 4, 'z': 0}, 
                        'backgroundColor': 'lightcoral',
                        'fontColor': 'black', 'fontSize': 14}, viewer=(0, 1))
        
        viewer.zoomTo()
        
        print("✅ Interactive 3D stereoisomers ready!")
        print("💡 Notice the yellow spheres highlighting the chiral centers")
        print("🔄 Try rotating both molecules - they are non-superimposable mirror images!")
        
        viewer.show()
        
    else:
        print("📊 2D Structure Comparison (Static)")
        print("-" * 35)
    
    # Create matplotlib comparison showing the key point
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Why 2D Graphs Fail: The Stereoisomer Problem", fontsize=16, fontweight='bold')
    
    # Row 1: 2D structures (look identical in terms of connectivity)
    mol_r_2d = Chem.MolFromSmiles(r_smiles)  # Remove explicit H for 2D display
    mol_s_2d = Chem.MolFromSmiles(s_smiles)
    
    ax1 = axes[0, 0]
    img_r = Draw.MolToImage(mol_r_2d, size=(250, 200))
    ax1.imshow(img_r)
    ax1.set_title('R-2-butanol\n(2D Structure)', fontweight='bold')
    ax1.axis('off')
    
    ax2 = axes[0, 1] 
    img_s = Draw.MolToImage(mol_s_2d, size=(250, 200))
    ax2.imshow(img_s)
    ax2.set_title('S-2-butanol\n(2D Structure)', fontweight='bold')
    ax2.axis('off')
    
    # 2D Graph Analysis
    ax3 = axes[0, 2]
    ax3.text(0.1, 0.9, "2D Graph Analysis:", fontsize=14, fontweight='bold', color='red')
    ax3.text(0.1, 0.75, "IDENTICAL connectivity", fontsize=12)
    ax3.text(0.1, 0.65, "Same adjacency matrix", fontsize=12)
    ax3.text(0.1, 0.55, "Same bond counts", fontsize=12)  
    ax3.text(0.1, 0.45, "Same atom types", fontsize=12)
    ax3.text(0.1, 0.3, "Cannot distinguish", fontsize=12, color='red', fontweight='bold')
    ax3.text(0.1, 0.2, "Miss different properties", fontsize=12, color='red', fontweight='bold')
    ax3.text(0.1, 0.05, "Machine learning models\nwould predict identical properties!", fontsize=10, color='darkred', style='italic')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Row 2: Real-world consequences
    ax4 = axes[1, 0]
    ax4.text(0.1, 0.85, "Real-World Impact:", fontsize=14, fontweight='bold', color='darkgreen')
    ax4.text(0.1, 0.7, "Drug Examples:", fontsize=12, fontweight='bold')
    ax4.text(0.1, 0.6, "• Thalidomide:", fontsize=11)
    ax4.text(0.15, 0.5, "R = Sedative", fontsize=10)
    ax4.text(0.15, 0.4, "S = Birth defects", fontsize=10, color='red')
    ax4.text(0.1, 0.25, "• Ibuprofen:", fontsize=11)
    ax4.text(0.15, 0.15, "S = Active anti-inflammatory", fontsize=10)
    ax4.text(0.15, 0.05, "R = Inactive", fontsize=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    ax5 = axes[1, 1]
    ax5.text(0.1, 0.85, "3D Features Needed:", fontsize=14, fontweight='bold', color='green')
    ax5.text(0.1, 0.7, " Spatial coordinates", fontsize=12, color='green')
    ax5.text(0.1, 0.6, "Chirality information", fontsize=12, color='green')
    ax5.text(0.1, 0.5, "3D distances", fontsize=12, color='green')
    ax5.text(0.1, 0.4, "Conformational flexibility", fontsize=12, color='green')
    ax5.text(0.1, 0.25, "Shape descriptors", fontsize=12, color='green')
    ax5.text(0.1, 0.1, "Non-covalent interactions", fontsize=12, color='green')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    
    ax6 = axes[1, 2]
    ax6.text(0.1, 0.85, "3D GNN Benefits:", fontsize=14, fontweight='bold', color='purple')
    ax6.text(0.1, 0.7, "Distinguish enantiomers", fontsize=11, color='purple')
    ax6.text(0.1, 0.6, "Predict chirality effects", fontsize=11, color='purple')
    ax6.text(0.1, 0.5, "Model shape-property", fontsize=11, color='purple')
    ax6.text(0.1, 0.4, "relationships", fontsize=11, color='purple')
    ax6.text(0.1, 0.3, " Capture binding affinity", fontsize=11, color='purple')
    ax6.text(0.1, 0.2, " Enable drug design", fontsize=11, color='purple')
    ax6.text(0.1, 0.05, "Essential for many pharmaceutical\napplications!", fontsize=10, color='darkmagenta', style='italic', fontweight='bold')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n🎯 Key Takeaway:")
    print("2D molecular graphs treat these molecules as IDENTICAL")
    print("3D molecular graphs can distinguish them and predict different properties")
    print("This is why 3D representations are crucial for drug discovery and chemistry!")

# Run the enhanced demonstration
demonstrate_2d_vs_3d_concept()

# +
# Additional demonstration: Conformational Flexibility
def demonstrate_conformational_flexibility():
    """
    Show how the same molecule can have multiple 3D shapes (conformers)
    but identical 2D connectivity - another crucial 3D concept.
    """
    print("\n🔄 BONUS: Conformational Flexibility Demonstration")
    print("=" * 55)
    print("Same molecule, same 2D graph, but multiple 3D shapes!")
    print()
    
    # Use a flexible molecule that shows clear conformational differences
    flexible_smiles = "CCCCCO"  # Pentanol - has rotatable bonds
    molecule_name = "Pentanol"
    
    print(f"Molecule: {molecule_name} ({flexible_smiles})")
    print(f"This molecule has rotatable C-C bonds → multiple 3D conformations possible")
    print()
    
    if PY3DMOL_AVAILABLE:
        try:
            # Generate multiple conformers
            mol = Chem.MolFromSmiles(flexible_smiles)
            mol = Chem.AddHs(mol)
            
            # Generate multiple conformers using ETKDG
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=4, params=params)
            
            if conf_ids:
                # Optimize conformers
                energies = []
                for conf_id in conf_ids:
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
                        # Calculate energy (simplified)
                        props = AllChem.MMFFGetMoleculeProperties(mol)
                        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                        if ff:
                            energy = ff.CalcEnergy()
                        else:
                            energy = 0.0
                        energies.append(energy)
                    except:
                        energies.append(0.0)
                
                # Create 2x2 grid showing different conformers
                viewer = py3Dmol.view(width=800, height=600, viewergrid=(2, 2))
                
                colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
                conformer_names = ['Extended', 'Folded', 'Twisted', 'Compact']
                
                for i, conf_id in enumerate(conf_ids[:4]):
                    row = i // 2
                    col = i % 2
                    
                    # Extract this specific conformer
                    mol_copy = Chem.Mol(mol)
                    mol_copy.RemoveAllConformers()
                    conf = mol.GetConformer(conf_id)
                    mol_copy.AddConformer(conf, assignId=True)
                    
                    mol_block = Chem.MolToMolBlock(mol_copy)
                    viewer.addModel(mol_block, 'mol', viewer=(row, col))
                    
                    # Style the molecule
                    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.15}, 
                                                   'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}}, viewer=(row, col))
                    
                    # Add labels
                    rel_energy = energies[i] - min(energies) if energies else 0
                    viewer.addLabel(f'{conformer_names[i]} Conformer\nΔE: {rel_energy:.1f} kcal/mol\nSame 2D connectivity!', 
                                   {'position': {'x': 0, 'y': 4, 'z': 0}, 
                                    'backgroundColor': colors[i],
                                    'fontColor': 'black', 'fontSize': 11}, viewer=(row, col))
                
                viewer.zoomTo()
                
                print("✅ 3D Conformational flexibility visualization ready!")
                print("💡 All 4 conformers have IDENTICAL 2D graphs but different:")
                print("   • 3D shapes")
                print("   • Energies") 
                print("   • Surface areas")
                print("   • Binding properties")
                print("🔄 Try rotating each conformer to see the shape differences!")
                
                viewer.show()
                
            else:
                print("⚠️ Could not generate conformers for this molecule")
                
        except Exception as e:
            print(f"❌ Error in conformer demonstration: {e}")
    else:
        print("⚠️ py3Dmol not available - 3D conformer visualization not possible")
    
    print("\n🎯 Conformational Flexibility Takeaway:")
    print("• One molecule → One 2D graph → Multiple 3D shapes")
    print("• Each shape has different properties and binding behavior")
    print("• 3D GNNs can capture these shape-dependent effects")
    print("• Essential for modeling flexible drug molecules!")

# Run the conformational flexibility demonstration
demonstrate_conformational_flexibility()


# + [markdown] id="mLLG4lblpF3J"
# ### The Conformational Landscape
#
# Molecules exist in a **conformational landscape** - an energy surface where each point represents a different 3D arrangement. Understanding this landscape is crucial for:
# - Drug design (active conformations)
# - Reaction mechanisms (transition states)
# - Material properties (packing arrangements)

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="35Utwdu2q4Yh" outputId="af7bf734-5c1a-48f1-ab35-4bf97cdb2f40"
def visualize_conformational_concept():
    """
    Create a conceptual diagram of conformational space.
    """
    # Create a sample energy landscape
    x = np.linspace(0, 360, 100)
    # Simplified rotational energy profile for butane
    energy = 3 * np.cos(np.radians(2*x)) + 1.5 * np.cos(np.radians(3*x)) + 5
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, energy, 'b-', linewidth=2)
    plt.xlabel('Dihedral Angle (degrees)')
    plt.ylabel('Relative Energy (kcal/mol)')
    plt.title('Conformational Energy Profile')
    plt.grid(True, alpha=0.3)
    
    # Mark stable conformations
    stable_angles = [75, 285]
    stable_energies = [3 * np.cos(np.radians(2*angle)) + 1.5 * np.cos(np.radians(3*angle)) + 5 
                      for angle in stable_angles]
    plt.scatter(stable_angles, stable_energies, color='red', s=100, zorder=5)
    plt.annotate('Stable\nConformation', xy=(75, stable_energies[0]), 
                xytext=(50, stable_energies[0]+1),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.8, 'Key Concepts:', fontsize=14, weight='bold')
    plt.text(0.1, 0.7, '• Energy minima = stable conformers', fontsize=12)
    plt.text(0.1, 0.6, '• Energy barriers between conformers', fontsize=12)
    plt.text(0.1, 0.5, '• Room temperature accessible conformers', fontsize=12)
    plt.text(0.1, 0.4, '• Conformer populations follow Boltzmann distribution', fontsize=12)
    
    plt.text(0.1, 0.25, 'For GNNs:', fontsize=14, weight='bold', color='purple')
    plt.text(0.1, 0.15, '• Which conformer to use?', fontsize=12, color='purple')
    plt.text(0.1, 0.05, '• How to incorporate flexibility?', fontsize=12, color='purple')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Conformational Considerations for ML')
    
    plt.tight_layout()
    plt.show()

visualize_conformational_concept()

# + [markdown] id="esDZEn1lsUUr"
# ## 4. Generating 3D Molecular Conformations <a name="generating-3d-molecular-conformations"></a>
#
# RDKit provides several methods for generating 3D conformations from SMILES strings. The most common approach uses the **ETKDG** (Experimental-Torsional Knowledge Distance Geometry) algorithm.

# + [markdown] id="OYSKIWa5sXJv"
# ### Single Conformer Generation
#
# Let's start by generating a single, optimized 3D conformer:

# + colab={"base_uri": "https://localhost:8080/"} id="XL33mziBv859" outputId="28fe96e7-944c-4536-8301-1f30215b577d"
def generate_3d_conformer(smiles: str, optimize=True):
    """
    Generate a single 3D conformer from a SMILES string.
    
    Args:
        smiles (str): SMILES string of the molecule
        optimize (bool): Whether to optimize the conformer geometry
    
    Returns:
        rdkit.Chem.Mol: Molecule with 3D coordinates
    """
    # Step 1: Create molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Step 2: Add hydrogens (essential for realistic 3D geometry)
    mol = Chem.AddHs(mol)
    
    # Step 3: Generate 3D coordinates using ETKDG
    # ETKDG = Experimental-Torsional Knowledge Distance Geometry
    # This method uses experimental torsional angle preferences
    params = AllChem.ETKDGv3()
    params.randomSeed = 42  # For reproducible results
    
    # Generate the conformer
    conf_id = AllChem.EmbedMolecule(mol, params)
    
    if conf_id == -1:
        raise RuntimeError(f"Could not generate 3D coordinates for {smiles}")
    
    # Step 4: Optimize geometry using MMFF force field (optional but recommended)
    if optimize:
        # MMFF94 is a molecular mechanics force field for geometry optimization
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
    
    return mol

# Test the function with several molecules
test_molecules = {
    "Methanol": "CO",
    "Ethanol": "CCO", 
    "Cyclohexane": "C1CCCCC1",
    "Benzene": "c1ccccc1",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O"
}

print("Generating 3D conformers for test molecules:")
print("=" * 50)

conformers = {}
for name, smiles in test_molecules.items():
    try:
        mol_3d = generate_3d_conformer(smiles)
        conformers[name] = mol_3d
        
        # Get some basic 3D properties
        conf = mol_3d.GetConformer()
        n_atoms = mol_3d.GetNumAtoms()
        
        # Calculate molecular volume (approximate)
        positions = []
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        positions = np.array(positions)
        
        # Calculate bounding box volume as rough estimate
        ranges = np.ptp(positions, axis=0)  # peak-to-peak (max - min) for each dimension
        bbox_volume = np.prod(ranges)
        
        print(f"{name}:")
        print(f"  SMILES: {smiles}")
        print(f"  Atoms: {n_atoms}")
        print(f"  3D Bounding Box Volume: {bbox_volume:.2f} Ų")
        print(f"  Conformer generated successfully ✓")
        print()
        
    except Exception as e:
        print(f"{name}: Failed - {e}")
        print()

# + [markdown] id="JlzUrrQJxqJK"
# ### Multiple Conformer Generation
#
# For flexible molecules, it's often useful to generate multiple conformers to sample the conformational space:

# + colab={"base_uri": "https://localhost:8080/"} id="g4wAq8xMyRTu" outputId="ebdd985a-ac19-4ce0-e59c-45cb54a3d974"
def generate_multiple_conformers(smiles: str, n_conformers=10, optimize=True):
    """
    Generate multiple 3D conformers for a molecule.
    
    Args:
        smiles (str): SMILES string of the molecule
        n_conformers (int): Number of conformers to generate
        optimize (bool): Whether to optimize conformer geometries
    
    Returns:
        tuple: (mol with conformers, list of energies)
    """
    # Create molecule and add hydrogens
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Set up parameters for conformer generation
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0  # Use all available cores
    
    # Generate multiple conformers
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    
    if not conf_ids:
        raise RuntimeError(f"Could not generate conformers for {smiles}")
    
    # Optimize each conformer and calculate energies
    energies = []
    if optimize:
        for conf_id in conf_ids:
            # Optimize with MMFF94 force field
            props = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            if ff:
                ff.Minimize()
                energy = ff.CalcEnergy()
                energies.append(energy)
            else:
                energies.append(float('inf'))
    else:
        energies = [0.0] * len(conf_ids)
    
    return mol, energies

# Generate multiple conformers for a flexible molecule (aspirin)
aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
print(f"Generating multiple conformers for aspirin ({aspirin_smiles}):")
print("=" * 50)

try:
    aspirin_conformers, energies = generate_multiple_conformers(aspirin_smiles, n_conformers=20)
    
    # Sort conformers by energy
    sorted_indices = np.argsort(energies)
    sorted_energies = np.array(energies)[sorted_indices]
    
    print(f"Generated {len(energies)} conformers")
    print(f"Energy range: {min(energies):.2f} to {max(energies):.2f} kcal/mol")
    print(f"Energy spread: {max(energies) - min(energies):.2f} kcal/mol")
    print()
    
    # Show top 5 lowest energy conformers
    print("Top 5 lowest energy conformers:")
    for i, idx in enumerate(sorted_indices[:5]):
        rel_energy = sorted_energies[i] - sorted_energies[0]  # Relative to lowest
        print(f"  Conformer {idx}: {energies[idx]:.2f} kcal/mol (ΔE = {rel_energy:.2f})")
    
    # Calculate Boltzmann populations at room temperature
    kT = 0.593  # kcal/mol at 298K
    exp_factors = np.exp(-(sorted_energies - sorted_energies[0]) / kT)
    populations = exp_factors / np.sum(exp_factors)
    
    print(f"\nBoltzmann populations at 298K (top 3):")
    for i in range(min(3, len(populations))):
        print(f"  Conformer {sorted_indices[i]}: {populations[i]*100:.1f}%")
        
except Exception as e:
    print(f"Error: {e}")

# + [markdown] id="mXkhsu7myPUC"
# ## 5. 3D Graph Construction from Conformers <a name="3d-graph-construction-from-conformers"></a>
#
# Now that we can generate 3D conformers, let's create molecular graphs that incorporate spatial information. Unlike 2D graphs that only consider chemical bonds, 3D graphs can include:
# 1. **Covalent bonds** (traditional edges)
# 2. **Distance-based edges** (atoms within a certain distance)
# 3. **Spatial features** (coordinates, distances, angles)

# + [markdown] id="qdNQD25Ky1uY"
# ### Extracting 3D Coordinates
#
# First, let's create a function to extract 3D coordinates from conformers:

# + colab={"base_uri": "https://localhost:8080/"} id="RLkVqjjYy2te" outputId="451beacb-a611-4488-d67e-6459dad57b53"
def extract_3d_coordinates(mol, conf_id=0):
    """
    Extract 3D coordinates from a molecular conformer.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        conf_id: Conformer ID to use (default: 0)
    
    Returns:
        numpy.ndarray: Array of shape (n_atoms, 3) with x, y, z coordinates
    """
    conformer = mol.GetConformer(conf_id)
    coordinates = []
    
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conformer.GetAtomPosition(atom_idx)
        coordinates.append([pos.x, pos.y, pos.z])
    
    return np.array(coordinates)

def get_atomic_features_3d(mol):
    """
    Extract atomic features including 3D-specific properties.
    
    Args:
        mol: RDKit molecule with 3D coordinates
    
    Returns:
        numpy.ndarray: Extended feature matrix including 3D features
    """
    coordinates = extract_3d_coordinates(mol)
    n_atoms = mol.GetNumAtoms()
    
    # Basic features (from previous tutorial)
    basic_features = []
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())
        
        # One-hot encoding for atom types
        atom_types = ['C', 'O', 'N', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        atom_type_onehot = [1 if atom_type == t else 0 for t in atom_types]
        if atom_type not in atom_types:
            atom_type_onehot.append(1)  # "Other"
        else:
            atom_type_onehot.append(0)
        
        features = atom_type_onehot + [
            formal_charge,
            is_aromatic,
            is_in_ring,
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons()
        ]
        basic_features.append(features)
    
    basic_features = np.array(basic_features)
    
    # Add 3D coordinates as features
    enhanced_features = np.concatenate([basic_features, coordinates], axis=1)
    
    # Add distance-based features
    center_of_mass = np.mean(coordinates, axis=0)
    distances_to_com = np.linalg.norm(coordinates - center_of_mass, axis=1)
    
    # Add distances to center of mass as feature
    enhanced_features = np.concatenate([enhanced_features, distances_to_com.reshape(-1, 1)], axis=1)
    
    return enhanced_features, coordinates

# Test the coordinate extraction
print("Testing 3D coordinate extraction:")
print("=" * 40)

for name, mol in conformers.items():
    coords = extract_3d_coordinates(mol)
    features, _ = get_atomic_features_3d(mol)
    
    print(f"{name}:")
    print(f"  Shape of coordinates: {coords.shape}")
    print(f"  Shape of features: {features.shape}")
    print(f"  Coordinate range:")
    print(f"    X: {coords[:, 0].min():.2f} to {coords[:, 0].max():.2f}")
    print(f"    Y: {coords[:, 1].min():.2f} to {coords[:, 1].max():.2f}")
    print(f"    Z: {coords[:, 2].min():.2f} to {coords[:, 2].max():.2f}")
    print()

# + [markdown] id="J8Zsc4zOxqJN"
# ### Distance-Based Edge Construction
#
# In 3D molecular graphs, we can define edges not just based on chemical bonds, but also based on spatial proximity. This captures important non-covalent interactions:

# + colab={"base_uri": "https://localhost:8080/"} id="DTkyWf7BzFY1" outputId="0TU76Wlv25Kt"
def create_3d_molecular_graph(mol, cutoff_distance=5.0, include_bond_edges=True):
    """
    Create a 3D molecular graph with distance-based edges.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        cutoff_distance: Maximum distance for creating edges (Angstroms)
        include_bond_edges: Whether to include covalent bond edges
    
    Returns:
        tuple: (node_features, edge_indices, edge_features, coordinates)
    """
    # Get enhanced node features and coordinates
    node_features, coordinates = get_atomic_features_3d(mol)
    n_atoms = mol.GetNumAtoms()
    
    # Calculate all pairwise distances
    distance_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                distance_matrix[i, j] = distance
    
    # Create edge lists
    edge_indices = []
    edge_features = []
    
    # Add covalent bond edges (if requested)
    if include_bond_edges:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            distance = distance_matrix[i, j]
            
            # Edge features: [is_covalent, distance, 1/distance]
            edge_feature = [1.0, distance, 1.0/distance if distance > 0 else 0.0]
            
            # Add edge in both directions
            edge_indices.extend([(i, j), (j, i)])
            edge_features.extend([edge_feature, edge_feature])
    
    # Add distance-based edges
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = distance_matrix[i, j]
            
            # Create edge if within cutoff and not already a covalent bond
            if distance <= cutoff_distance:
                # Check if this is already a covalent bond
                is_covalent_bond = False
                if include_bond_edges:
                    bond = mol.GetBondBetweenAtoms(i, j)
                    is_covalent_bond = bond is not None
                
                if not is_covalent_bond:
                    # Edge features: [is_covalent, distance, 1/distance]
                    edge_feature = [0.0, distance, 1.0/distance if distance > 0 else 0.0]
                    
                    # Add edge in both directions
                    edge_indices.extend([(i, j), (j, i)])
                    edge_features.extend([edge_feature, edge_feature])
    
    return node_features, edge_indices, edge_features, coordinates


def calculate_geometric_features(mol, coordinates):
    """
    Calculate comprehensive geometric features for a molecule.
    
    These features capture the 3D shape and size of the molecule, which are
    critical for understanding molecular properties like binding affinity,
    solubility, and reactivity.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        coordinates: numpy array of shape (n_atoms, 3) with x, y, z positions
    
    Returns:
        dict: Dictionary containing geometric descriptors:
            - diameter: Maximum interatomic distance (molecular size)
            - radius_of_gyration: Measure of molecular compactness
            - asphericity: Deviation from spherical shape (0 = sphere)
            - acylindricity: Deviation from cylindrical shape
            - convex_hull_volume: Volume of convex hull enclosing all atoms
    """
    n_atoms = len(coordinates)
    
    # Calculate center of mass (geometric center)
    center_of_mass = np.mean(coordinates, axis=0)
    
    # Calculate radius of gyration (measure of molecular spread/compactness)
    # Smaller values = more compact molecule
    rg_squared = np.mean(np.sum((coordinates - center_of_mass)**2, axis=1))
    radius_of_gyration = np.sqrt(rg_squared)
    
    # Calculate molecular diameter (maximum pairwise distance between atoms)
    # This gives the "size" of the molecule
    max_distance = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            max_distance = max(max_distance, distance)
    diameter = max_distance
    
    # Calculate asphericity and acylindricity from inertia tensor
    # These describe molecular shape: asphericity = 0 for perfect sphere
    centered_coords = coordinates - center_of_mass
    
    # Build the gyration tensor (similar to inertia tensor)
    I = np.zeros((3, 3))
    for coord in centered_coords:
        x, y, z = coord
        I[0, 0] += y*y + z*z
        I[1, 1] += x*x + z*z
        I[2, 2] += x*x + y*y
        I[0, 1] -= x*y
        I[0, 2] -= x*z
        I[1, 2] -= y*z
    
    # Symmetrize the tensor
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    
    # Eigenvalues give principal moments (λ1 ≤ λ2 ≤ λ3)
    eigenvalues = np.sort(np.real(np.linalg.eigvals(I)))
    
    # Asphericity: 0 for sphere, positive for non-spherical
    asphericity = eigenvalues[2] - 0.5*(eigenvalues[0] + eigenvalues[1])
    
    # Acylindricity: difference between two smaller eigenvalues
    acylindricity = eigenvalues[1] - eigenvalues[0]
    
    # Calculate convex hull volume (if scipy available)
    # This is the volume of the smallest convex shape containing all atoms
    convex_hull_volume = 0.0
    try:
        from scipy.spatial import ConvexHull
        if n_atoms >= 4:  # Need at least 4 points for 3D hull
            hull = ConvexHull(coordinates)
            convex_hull_volume = hull.volume
    except ImportError:
        # scipy not available, use bounding box as approximation
        ranges = np.ptp(coordinates, axis=0)
        convex_hull_volume = np.prod(ranges)
    except Exception:
        # ConvexHull can fail for degenerate cases (e.g., planar molecules)
        ranges = np.ptp(coordinates, axis=0)
        convex_hull_volume = np.prod(ranges)
    
    return {
        'diameter': diameter,
        'radius_of_gyration': radius_of_gyration,
        'asphericity': asphericity,
        'acylindricity': acylindricity,
        'convex_hull_volume': convex_hull_volume,
        'center_of_mass': center_of_mass
    }


def create_advanced_3d_graph(mol, cutoff_distance=5.0):
    """
    Create an advanced 3D molecular graph with comprehensive features.
    
    This function combines the basic 3D graph construction with geometric
    feature calculation to provide a complete representation for GNNs.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        cutoff_distance: Maximum distance for creating non-covalent edges (Angstroms)
    
    Returns:
        tuple: (node_features, edge_indices, edge_features, coordinates, geometric_features)
            - node_features: Array of atom features including 3D coordinates
            - edge_indices: List of (source, target) tuples for edges
            - edge_features: Array of edge features (covalent/distance, distance, 1/distance)
            - coordinates: Raw 3D coordinates (n_atoms, 3)
            - geometric_features: Dictionary of molecular shape descriptors
    """
    # Get basic 3D graph components
    node_features, edge_indices, edge_features, coordinates = create_3d_molecular_graph(
        mol, cutoff_distance=cutoff_distance
    )
    
    # Calculate comprehensive geometric features
    geometric_features = calculate_geometric_features(mol, coordinates)
    
    return node_features, edge_indices, edge_features, coordinates, geometric_features


# Test 3D graph construction
print("Testing 3D molecular graph construction:")
print("=" * 45)

# Test with different cutoff distances
cutoff_distances = [3.0, 4.0, 5.0]
test_mol = conformers["Cyclohexane"]

for cutoff in cutoff_distances:
    node_feat, edge_idx, edge_feat, coords = create_3d_molecular_graph(
        test_mol, cutoff_distance=cutoff
    )
    
    # Count different edge types
    edge_feat_array = np.array(edge_feat)
    covalent_edges = np.sum(edge_feat_array[:, 0] == 1.0) // 2  # Divide by 2 (bidirectional)
    distance_edges = np.sum(edge_feat_array[:, 0] == 0.0) // 2
    
    print(f"Cutoff {cutoff} Å:")
    print(f"  Total edges: {len(edge_idx)}")
    print(f"  Covalent edges: {covalent_edges}")
    print(f"  Distance-based edges: {distance_edges}")
    print(f"  Average edge distance: {np.mean(edge_feat_array[:, 1]):.2f} Å")
    print()

# + [markdown] id="0TU76Wlv25Kt"
# ## 6. Distance-Based Edge Features <a name="distance-based-edge-features"></a>
#
# 3D molecular graphs can incorporate rich edge features based on spatial relationships:

# + colab={"base_uri": "https://localhost:8080/", "height": 600} id="ZhzvT4yarsSp" outputId="advanced-edge-features"
def calculate_advanced_3d_features(mol, coordinates):
    """
    Calculate advanced 3D features for molecular graphs.
    """
    n_atoms = mol.GetNumAtoms()
    
    # Calculate molecular descriptors
    features = {
        'molecular_weight': Descriptors.MolWt(mol),
        'volume_estimate': np.prod(np.ptp(coordinates, axis=0)),  # Bounding box volume
        'surface_area_estimate': 0.0,  # Would need more complex calculation
        'radius_of_gyration': 0.0
    }
    
    # Calculate radius of gyration
    center_of_mass = np.mean(coordinates, axis=0)
    rg_squared = np.mean(np.sum((coordinates - center_of_mass)**2, axis=1))
    features['radius_of_gyration'] = np.sqrt(rg_squared)
    
    return features

def analyze_3d_graph_properties(mol, cutoff_distance=5.0):
    """
    Analyze properties of 3D molecular graph.
    """
    node_feat, edge_idx, edge_feat, coords = create_3d_molecular_graph(
        mol, cutoff_distance=cutoff_distance
    )
    
    # Basic graph statistics
    n_nodes = len(node_feat)
    n_edges = len(edge_idx) // 2  # Bidirectional edges
    
    # Edge statistics
    edge_feat_array = np.array(edge_feat)
    distances = edge_feat_array[:, 1]
    covalent_mask = edge_feat_array[:, 0] == 1.0
    
    covalent_distances = distances[covalent_mask]
    noncovalent_distances = distances[~covalent_mask]
    
    results = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'avg_covalent_distance': np.mean(covalent_distances) if len(covalent_distances) > 0 else 0,
        'avg_noncovalent_distance': np.mean(noncovalent_distances) if len(noncovalent_distances) > 0 else 0,
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'coordinates': coords
    }
    
    return results

# Analyze different molecules
print("3D Graph Analysis for Different Molecules:")
print("=" * 50)

for name, mol in conformers.items():
    print(f"\n{name}:")
    analysis = analyze_3d_graph_properties(mol, cutoff_distance=4.0)
    
    print(f"  Nodes: {analysis['n_nodes']}")
    print(f"  Edges: {analysis['n_edges']}")
    print(f"  Avg covalent distance: {analysis['avg_covalent_distance']:.2f} Å")
    print(f"  Avg non-covalent distance: {analysis['avg_noncovalent_distance']:.2f} Å")
    print(f"  Distance range: {analysis['min_distance']:.2f} - {analysis['max_distance']:.2f} Å")
    
    # Calculate 3D features
    features_3d = calculate_advanced_3d_features(mol, analysis['coordinates'])
    print(f"  Molecular weight: {features_3d['molecular_weight']:.1f} g/mol")
    print(f"  Volume estimate: {features_3d['volume_estimate']:.1f} Ų")
    print(f"  Radius of gyration: {features_3d['radius_of_gyration']:.2f} Å")

# + [markdown] id="I15NypCO1bc6"
# ## 7. 3D Visualization of Molecular Graphs <a name="3d-visualization-of-molecular-graphs"></a>
#
# Visualizing 3D molecular graphs helps us understand the spatial relationships between atoms. We'll use **py3Dmol** for professional molecular rendering that provides:
#
# - **Ball-and-stick models** with proper atomic radii and bond representations
# - **Van der Waals surfaces** showing molecular shape and volume
# - **Interactive visualization** with rotation, zoom, and selection capabilities
# - **Chemical color schemes** (Jmol/CPK) familiar to chemists
# - **Distance-based interaction visualization** with chemical context
#
# ### Why py3Dmol is Superior for Chemistry Education
#
# | Feature | matplotlib/plotly | py3Dmol |
# |---------|------------------|----------|
# | Molecular representation | Colored dots and lines | Professional ball-and-stick, space-filling |
# | Chemical accuracy | Generic | Standard chemistry representations |
# | Interactivity | Limited | Full 3D rotation, zoom, selection |
# | Surface visualization | None | Van der Waals, electrostatic surfaces |
# | Performance | Slow for large molecules | Optimized for molecular data |
# | Chemical context | Minimal | Familiar to chemists |

# + colab={"base_uri": "https://localhost:8080/", "height": 800} id="sq2P6wCr1c6V"
def plot_3d_molecular_graph(mol, cutoff_distance=4.0, show_all_edges=False, use_py3dmol=True):
    """
    Create an interactive 3D visualization of a molecular graph.
    Now enhanced with py3Dmol for professional molecular rendering!
    
    Args:
        mol: RDKit molecule with 3D coordinates
        cutoff_distance: Distance cutoff for edges (Angstroms)
        show_all_edges: Whether to show all distance-based edges
        use_py3dmol: Whether to use py3Dmol (recommended) or fallback to plotly
    
    Returns:
        py3Dmol viewer (if available) or plotly figure
    """
    # Try py3Dmol first for professional molecular visualization
    if use_py3dmol and PY3DMOL_AVAILABLE and ENHANCED_VIZ_AVAILABLE:
        print("🧪 Creating professional molecular visualization with py3Dmol...")
        viewer = plot_3d_molecular_graph_py3dmol(mol, cutoff_distance, show_all_edges)
        if viewer:
            print("💡 Call viewer.show() to display the interactive 3D molecular model")
            return viewer
        else:
            print("⚠️ py3Dmol failed, falling back to plotly...")
    
    # Fallback to original plotly implementation
    print("📊 Creating basic 3D graph with plotly...")
    
    # Get graph data
    node_feat, edge_idx, edge_feat, coords = create_3d_molecular_graph(
        mol, cutoff_distance=cutoff_distance
    )
    
    # Get atom symbols
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # Color mapping for atoms (CPK colors)
    atom_colors = {
        'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red',
        'F': 'lightgreen', 'P': 'orange', 'S': 'yellow', 'Cl': 'green',
        'Br': 'darkred', 'I': 'purple'
    }
    
    colors = [atom_colors.get(symbol, 'gray') for symbol in atom_symbols]
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add atoms as scatter points with better styling
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers+text',
        marker=dict(
            size=12, 
            color=colors, 
            opacity=0.9,
            line=dict(width=2, color='black')
        ),
        text=atom_symbols,
        textposition="middle center",
        textfont=dict(size=10, color='white'),
        name="Atoms",
        hovertemplate="Atom: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
    ))
    
    # Add edges with different colors for covalent vs non-covalent
    edge_feat_array = np.array(edge_feat)
    
    # Separate covalent and non-covalent edges
    covalent_x, covalent_y, covalent_z = [], [], []
    noncovalent_x, noncovalent_y, noncovalent_z = [], [], []
    
    for i, (start, end) in enumerate(edge_idx):
        if i % 2 == 0:  # Only draw each edge once
            is_covalent = edge_feat_array[i, 0] == 1.0
            x_coords = [coords[start, 0], coords[end, 0], None]
            y_coords = [coords[start, 1], coords[end, 1], None]
            z_coords = [coords[start, 2], coords[end, 2], None]
            
            if is_covalent:
                covalent_x.extend(x_coords)
                covalent_y.extend(y_coords)
                covalent_z.extend(z_coords)
            else:
                noncovalent_x.extend(x_coords)
                noncovalent_y.extend(y_coords)
                noncovalent_z.extend(z_coords)
    
    # Add covalent bonds
    if covalent_x:
        fig.add_trace(go.Scatter3d(
            x=covalent_x, y=covalent_y, z=covalent_z,
            mode='lines',
            line=dict(color='black', width=4),
            name="Covalent Bonds",
            hoverinfo='skip'
        ))
    
    # Add non-covalent interactions
    if noncovalent_x:
        fig.add_trace(go.Scatter3d(
            x=noncovalent_x, y=noncovalent_y, z=noncovalent_z,
            mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
            name="Non-covalent Interactions",
            opacity=0.6,
            hoverinfo='skip'
        ))
    
    # Update layout with better styling
    fig.update_layout(
        title={
            'text': f"3D Molecular Graph (cutoff: {cutoff_distance} Å)<br><sub>💡 For better visualization, try py3Dmol version</sub>",
            'x': 0.5
        },
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode='cube',
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        width=800,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

# +
# Create 3D visualizations for selected molecules
molecules_to_visualize = ["Cyclohexane", "Benzene"]

print("🧪 Creating Enhanced 3D Molecular Visualizations")
print("=" * 50)
print()

for name in molecules_to_visualize:
    if name in conformers:
        print(f"📋 Analyzing {name}...")
        
        # First try py3Dmol for professional visualization
        result = plot_3d_molecular_graph(conformers[name], cutoff_distance=4.0, use_py3dmol=True)
        
        if hasattr(result, 'show'):  # It's a py3Dmol viewer
            print(f"✅ Professional py3Dmol visualization ready for {name}!")
            print(f"💡 Execute: result.show() to display interactive molecular model")
            result.show()  # This will display in Jupyter
        else:  # It's a plotly figure
            print(f"📊 Plotly visualization for {name}:")
            result.show()
        
        print(f"{'='*30}\n")

print("🎯 Key Benefits of py3Dmol Visualization:")
print("• Professional ball-and-stick molecular models")
print("• Standard chemical color schemes (CPK/Jmol)")
print("• Van der Waals surfaces showing molecular shape")
print("• Interactive rotation, zoom, and selection")
print("• Color-coded interaction strengths")
print("• Chemical context for graph neural networks")
print()
print("💡 If you see py3Dmol viewers above, they are fully interactive!")
print("   Try clicking and dragging to rotate the molecules.")
# -

# ### Advanced py3Dmol Demonstrations
#
# Let's explore some advanced visualization capabilities that showcase why 3D molecular representations are crucial for understanding chemistry:

# +
# 1. Stereochemistry Demonstration
print("🔬 STEREOCHEMISTRY: Why 2D Graphs Can't Tell the Whole Story")
print("=" * 60)

if PY3DMOL_AVAILABLE and ENHANCED_VIZ_AVAILABLE:
    print("Creating interactive stereochemistry example...")
    try:
        stereo_viewer = create_py3dmol_stereochemistry_example()
        if stereo_viewer:
            print("✅ Stereochemistry demonstration ready!")
            print("💡 You'll see L-alanine and D-alanine side-by-side")
            print("   Notice: Same 2D connectivity, different 3D arrangements!")
            stereo_viewer.show()
        else:
            print("⚠️ Stereochemistry demo not available")
    except Exception as e:
        print(f"❌ Error creating stereochemistry demo: {e}")
else:
    print("⚠️ py3Dmol or enhanced visualizations not available")
    print("   Install with: pip install py3dmol")
    
print("\n" + "="*40)

# +
# 2. Conformer Explorer for Flexible Molecules
print("🌀 CONFORMATIONAL FLEXIBILITY: Multiple Shapes, Same Molecule")
print("=" * 60)

if PY3DMOL_AVAILABLE and ENHANCED_VIZ_AVAILABLE:
    # Demonstrate with a flexible drug molecule
    flexible_molecules = [
        ("CCCCCCCC", "Octane", "Flexible hydrocarbon"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", "Drug molecule")
    ]
    
    for smiles, name, description in flexible_molecules:
        print(f"\n📋 Exploring {name} ({description})...")
        try:
            conformer_viewer = create_py3dmol_conformer_explorer(smiles, name, n_conformers=5)
            if conformer_viewer:
                print(f"✅ Conformer explorer for {name} ready!")
                print(f"💡 You'll see multiple conformations with energy rankings")
                conformer_viewer.show()
            else:
                print(f"⚠️ Could not create conformer explorer for {name}")
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
else:
    print("⚠️ Advanced conformer exploration requires py3Dmol")
    print("   Install with: pip install py3dmol")

print("\n" + "="*40)

# +
# 3. Distance-Based Edges Visualization
print("📊 DISTANCE-BASED EDGES: How Cutoffs Affect Graph Connectivity")
print("=" * 60)

if PY3DMOL_AVAILABLE and ENHANCED_VIZ_AVAILABLE:
    print("Demonstrating distance edge effects with aspirin...")
    try:
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        edge_viewer = create_py3dmol_graph_edges_demo(aspirin_smiles, "Aspirin", [3.0, 4.0, 5.0])
        if edge_viewer:
            print("✅ Distance edges demonstration ready!")
            print("💡 You'll see how different cutoffs capture different interactions:")
            print("   • Red lines: Strong interactions (< 3.0 Å)")
            print("   • Orange lines: Medium interactions (3.0-4.0 Å)")
            print("   • Yellow lines: Weak interactions (4.0-5.0 Å)")
            edge_viewer.show()
        else:
            print("⚠️ Could not create distance edges demo")
    except Exception as e:
        print(f"❌ Error creating distance edges demo: {e}")
else:
    print("⚠️ Distance edges visualization requires py3Dmol")
    
print("\n" + "="*40)

# +
# 4. Drug-Receptor Binding Demonstration
print("💊 DRUG-RECEPTOR BINDING: 3D Shape Complementarity")
print("=" * 60)

if PY3DMOL_AVAILABLE and ENHANCED_VIZ_AVAILABLE:
    print("Demonstrating molecular shape complementarity...")
    try:
        binding_viewer = create_py3dmol_binding_pocket_demo()
        if binding_viewer:
            print("✅ Binding demonstration ready!")
            print("💡 You'll see van der Waals surfaces showing molecular shapes")
            print("   This illustrates why 3D structure matters for drug design")
            binding_viewer.show()
        else:
            print("⚠️ Could not create binding demo")
    except Exception as e:
        print(f"❌ Error creating binding demo: {e}")
else:
    print("⚠️ Binding demonstration requires py3Dmol")
    
print("\n" + "="*40)


# + [markdown] id="y6nDaKVY3dCN"
# ## 8. Comparing 2D vs 3D Representations <a name="comparing-2d-vs-3d-representations"></a>
#
# Let's directly compare 2D connectivity-based graphs with 3D distance-based graphs:

# + colab={"base_uri": "https://localhost:8080/", "height": 800} id="Wc-kQZ-k1O00"
def compare_2d_vs_3d_graphs(mol, cutoff_distance=4.0):
    """
    Compare 2D (connectivity-based) vs 3D (distance-based) molecular graphs.
    """
    # Get 2D graph (connectivity only)
    adjacency_2d = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
    edges_2d = []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency_2d[i, j] = 1
        adjacency_2d[j, i] = 1
        edges_2d.extend([(i, j), (j, i)])
    
    # Get 3D graph
    node_feat_3d, edges_3d, edge_feat_3d, coords = create_3d_molecular_graph(
        mol, cutoff_distance=cutoff_distance
    )
    
    # Analysis
    n_atoms = mol.GetNumAtoms()
    n_edges_2d = len(edges_2d) // 2
    n_edges_3d = len(edges_3d) // 2
    
    # Count covalent vs non-covalent edges in 3D
    edge_feat_array = np.array(edge_feat_3d)
    covalent_edges_3d = np.sum(edge_feat_array[:, 0] == 1.0) // 2
    noncovalent_edges_3d = np.sum(edge_feat_array[:, 0] == 0.0) // 2
    
    print("2D vs 3D Graph Comparison:")
    print("=" * 30)
    print(f"Number of atoms: {n_atoms}")
    print(f"2D edges (bonds only): {n_edges_2d}")
    print(f"3D total edges: {n_edges_3d}")
    print(f"  - Covalent: {covalent_edges_3d}")
    print(f"  - Non-covalent: {noncovalent_edges_3d}")
    print(f"Edge ratio (3D/2D): {n_edges_3d/n_edges_2d:.2f}")
    
    return {
        '2d_edges': n_edges_2d,
        '3d_edges': n_edges_3d,
        'covalent_3d': covalent_edges_3d,
        'noncovalent_3d': noncovalent_edges_3d,
        'coordinates': coords
    }

# Compare for different molecules
print("Comparison Results for Different Molecules:")
print("=" * 50)

comparison_results = {}
for name, mol in conformers.items():
    print(f"\n{name}:")
    results = compare_2d_vs_3d_graphs(mol, cutoff_distance=4.0)
    comparison_results[name] = results

# + [markdown] id="zn8cSorXv7SI"
# ## 9. Advanced 3D Features <a name="advanced-3d-features"></a>
#
# 3D molecular representations can include sophisticated geometric features:
# -

# ### Enhanced 3D Visualization of Advanced Features
#
# Let's visualize these advanced 3D features using py3Dmol to better understand their chemical significance:

# +
# Enhanced visualization of advanced 3D features with py3Dmol
print("🔬 ADVANCED 3D FEATURES VISUALIZATION")
print("=" * 50)

def visualize_advanced_features_py3dmol(mol, name="Molecule"):
    """
    Create comprehensive py3Dmol visualization showing advanced 3D molecular features.
    """
    if not PY3DMOL_AVAILABLE:
        print("⚠️ py3Dmol not available for advanced feature visualization")
        return None
    
    # Calculate advanced features
    coords = extract_3d_coordinates(mol)
    geom_features = calculate_geometric_features(mol, coords)
    
    # Create 2x2 grid viewer for different representations
    viewer = py3Dmol.view(width=800, height=600, viewergrid=(2, 2))
    mol_block = Chem.MolToMolBlock(mol)
    
    # 1. Ball-and-stick with molecular diameter
    viewer.addModel(mol_block, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.15}, 
                                   'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}}, viewer=(0, 0))
    
    # Add sphere representing molecular diameter
    center = coords.mean(axis=0)
    diameter = geom_features['diameter']
    viewer.addSphere({'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                     'radius': float(diameter/2), 'color': 'blue', 'alpha': 0.1}, viewer=(0, 0))
    
    viewer.addLabel(f'Molecular Diameter\n{diameter:.2f} Å\n(Blue sphere)', 
                   {'position': {'x': float(center[0]), 'y': float(center[1] + 4), 'z': float(center[2])}, 
                    'backgroundColor': 'lightblue',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(0, 0))
    
    # 2. Van der Waals surface showing shape
    viewer.addModel(mol_block, 'mol', viewer=(0, 1))
    viewer.setStyle({'model': -1}, {'line': {'color': 'black', 'width': 2}}, viewer=(0, 1))
    viewer.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'orange'}, viewer=(0, 1))
    
    rg = geom_features['radius_of_gyration']
    viewer.addLabel(f'Van der Waals Surface\nRadius of Gyration: {rg:.2f} Å\n(Molecular compactness)', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 
                    'backgroundColor': 'lightyellow',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(0, 1))
    
    # 3. Wireframe showing asphericity
    viewer.addModel(mol_block, 'mol', viewer=(1, 0))
    viewer.setStyle({'model': -1}, {'line': {'colorscheme': 'Jmol', 'width': 3}}, viewer=(1, 0))
    
    asphericity = geom_features['asphericity']
    shape_desc = "Spherical" if asphericity < 0.1 else "Rod-like" if asphericity > 0.5 else "Intermediate"
    viewer.addLabel(f'Molecular Shape\nAsphericity: {asphericity:.3f}\nShape: {shape_desc}', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 
                    'backgroundColor': 'lightgreen',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(1, 0))
    
    # 4. Space-filling with convex hull info
    viewer.addModel(mol_block, 'mol', viewer=(1, 1))
    viewer.setStyle({'model': -1}, {'sphere': {'colorscheme': 'Jmol', 'scale': 1.0}}, viewer=(1, 1))
    
    hull_vol = geom_features.get('convex_hull_volume', 0)
    acylind = geom_features['acylindricity']
    viewer.addLabel(f'Space-Filling Model\nConvex Hull Vol: {hull_vol:.1f} ų\nAcylindricity: {acylind:.3f}', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 
                    'backgroundColor': 'lightcoral',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(1, 1))
    
    viewer.zoomTo()
    return viewer

# Visualize advanced features for selected molecules
advanced_demo_molecules = ["Aspirin", "Benzene", "Cyclohexane"]

for name in advanced_demo_molecules:
    if name in conformers:
        print(f"\n📊 Advanced 3D Analysis: {name}")
        print("-" * 30)
        
        # Calculate features first
        mol = conformers[name]
        coords = extract_3d_coordinates(mol)
        geom_features = calculate_geometric_features(mol, coords)
        
        print(f"Molecular Properties:")
        print(f"  • Diameter: {geom_features['diameter']:.2f} Å")
        print(f"  • Radius of gyration: {geom_features['radius_of_gyration']:.2f} Å")
        print(f"  • Asphericity: {geom_features['asphericity']:.3f}")
        print(f"  • Acylindricity: {geom_features['acylindricity']:.3f}")
        if geom_features.get('convex_hull_volume', 0) > 0:
            print(f"  • Convex hull volume: {geom_features['convex_hull_volume']:.1f} ų")
        
        # Shape interpretation
        asphericity = geom_features['asphericity']
        if asphericity < 0.1:
            shape_desc = "Nearly spherical (compact)"
        elif asphericity > 0.5:
            shape_desc = "Rod-like (elongated)"
        else:
            shape_desc = "Intermediate shape"
        print(f"  • Shape classification: {shape_desc}")
        
        # Create py3Dmol visualization
        if PY3DMOL_AVAILABLE:
            try:
                viewer = visualize_advanced_features_py3dmol(mol, name)
                if viewer:
                    print(f"\n✅ Interactive 3D visualization ready for {name}!")
                    print("💡 The visualization shows:")
                    print("   - Top-left: Molecular diameter (blue sphere)")
                    print("   - Top-right: Van der Waals surface (molecular shape)")
                    print("   - Bottom-left: Wireframe (asphericity visualization)")
                    print("   - Bottom-right: Space-filling model")
                    viewer.show()
                else:
                    print("⚠️ Could not create py3Dmol visualization")
            except Exception as e:
                print(f"❌ Error creating visualization: {e}")
        else:
            print("⚠️ py3Dmol not available for advanced visualization")
        
        print(f"{'='*50}\n")

# +
# Comparative visualization showing molecular property relationships
print("🔍 MOLECULAR PROPERTY VISUALIZATION")
print("=" * 50)

def create_molecular_property_comparison():
    """
    Create a comprehensive comparison of molecular properties using py3Dmol.
    """
    if not PY3DMOL_AVAILABLE:
        print("⚠️ py3Dmol not available for property comparison")
        return
    
    # Test molecules with different shapes
    comparison_molecules = [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", "Rigid drug molecule"),
        ("c1ccccc1", "Benzene", "Rigid aromatic ring"),
        ("CCCCCCCC", "Octane", "Flexible chain"),
        ("C1CCCCC1", "Cyclohexane", "Flexible ring")
    ]
    
    print("Analyzing molecular shape diversity...")
    
    # Create molecules and calculate properties
    molecules_data = []
    for smiles, name, description in comparison_molecules:
        try:
            mol = generate_3d_conformer(smiles)
            coords = extract_3d_coordinates(mol)
            geom_features = calculate_geometric_features(mol, coords)
            
            molecules_data.append({
                'name': name,
                'description': description,
                'mol': mol,
                'features': geom_features,
                'smiles': smiles
            })
        except Exception as e:
            print(f"❌ Could not process {name}: {e}")
    
    if not molecules_data:
        print("❌ No molecules could be processed")
        return
    
    # Create comparison visualization
    n_molecules = len(molecules_data)
    viewer = py3Dmol.view(width=300*n_molecules, height=500, viewergrid=(2, n_molecules))
    
    for i, mol_data in enumerate(molecules_data):
        mol = mol_data['mol']
        name = mol_data['name']
        features = mol_data['features']
        
        mol_block = Chem.MolToMolBlock(mol)
        
        # Top row: Ball-and-stick models
        viewer.addModel(mol_block, 'mol', viewer=(0, i))
        viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.15}, 
                                       'sphere': {'colorscheme': 'Jmol', 'scale': 0.25}}, viewer=(0, i))
        
        # Add molecular property labels
        viewer.addLabel(f'{name}\nDiameter: {features["diameter"]:.1f} Å\nR_g: {features["radius_of_gyration"]:.1f} Å', 
                       {'position': {'x': 0, 'y': 3, 'z': 0}, 
                        'backgroundColor': 'lightblue',
                        'fontColor': 'black', 'fontSize': 10}, viewer=(0, i))
        
        # Bottom row: Surface models showing shape
        viewer.addModel(mol_block, 'mol', viewer=(1, i))
        viewer.setStyle({'model': -1}, {'line': {'color': 'gray', 'width': 1}}, viewer=(1, i))
        viewer.addSurface(py3Dmol.VDW, {'opacity': 0.6, 'color': 'orange'}, viewer=(1, i))
        
        # Shape classification
        asphericity = features['asphericity']
        if asphericity < 0.1:
            shape_class = "Spherical"
            color = 'lightgreen'
        elif asphericity > 0.5:
            shape_class = "Rod-like"
            color = 'lightcoral'
        else:
            shape_class = "Intermediate"
            color = 'lightyellow'
        
        viewer.addLabel(f'Shape: {shape_class}\nAsphericity: {asphericity:.3f}', 
                       {'position': {'x': 0, 'y': 3, 'z': 0}, 
                        'backgroundColor': color,
                        'fontColor': 'black', 'fontSize': 10}, viewer=(1, i))
    
    viewer.zoomTo()
    
    print("✅ Molecular property comparison ready!")
    print("\n💡 The visualization shows:")
    print("   - Top row: Ball-and-stick models with size metrics")
    print("   - Bottom row: Surface models with shape classification")
    print("   - Color coding: Green = Spherical, Yellow = Intermediate, Red = Rod-like")
    print("\n🎯 Notice how different molecular architectures affect 3D properties:")
    
    for mol_data in molecules_data:
        name = mol_data['name']
        desc = mol_data['description']
        features = mol_data['features']
        print(f"   • {name} ({desc}): Diameter = {features['diameter']:.1f} Å, Asphericity = {features['asphericity']:.3f}")
    
    return viewer

# Create the comparative visualization
try:
    comparison_viewer = create_molecular_property_comparison()
    if comparison_viewer:
        comparison_viewer.show()
except Exception as e:
    print(f"❌ Error creating comparison visualization: {e}")

print("\n" + "="*60)


# + [markdown] id="zMjXPTzUvnYk"
# ### ✅ Checkpoint: Understanding 3D Molecular Graphs
#
# To reinforce your understanding, try answering these questions:
#
# 1. **Question**: What is the main difference between 2D and 3D molecular graphs?
#    - **Answer**: 2D graphs only consider chemical connectivity (bonds), while 3D graphs incorporate spatial positions and can include distance-based edges representing non-covalent interactions.
#
# 2. **Question**: Why might a 3D molecular graph have more edges than a 2D graph?
#    - **Answer**: 3D graphs can include non-covalent interactions (hydrogen bonds, van der Waals forces, etc.) as edges based on spatial proximity, in addition to covalent bonds.
#
# 3. **Question**: What role does the cutoff distance play in 3D graph construction?
#    - **Answer**: The cutoff distance determines which atom pairs are connected by distance-based edges. Larger cutoffs create denser graphs but may include irrelevant long-range interactions.
#
# 4. **Question**: How does conformational flexibility affect 3D molecular graphs?
#    - **Answer**: Different conformers of the same molecule will have different 3D coordinates and potentially different distance-based edges, leading to different graph representations for the same chemical structure.

# + [markdown] id="zn8cSorXv7SI_checkpoint"
# ### ✅ Checkpoint Exercise: Build Your Own 3D Molecular Graph
#
# Try these exercises to practice what you've learned:
#
# 1. **Basic Exercise**: Choose a molecule with rotatable bonds (e.g., "CCCCCO" - pentanol) and:
#    - Generate multiple conformers
#    - Create 3D graphs for each conformer
#    - Compare the edge counts and average distances
#
# 2. **Intermediate Exercise**: For caffeine ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C"):
#    - Create 3D graphs with cutoffs of 3, 4, and 5 Å
#    - Identify which interactions (beyond covalent bonds) are captured at each cutoff
#    - Calculate and compare geometric descriptors
#
# 3. **Advanced Exercise**: Design a function that automatically selects an optimal cutoff distance based on molecular size (e.g., using radius of gyration as a guide).

# + [markdown] id="mXkhsu7myPUC_conversion"
# ### Converting to PyTorch Geometric Format
#
# Let's convert our 3D molecular graphs to PyG format for use with graph neural networks:

# + colab={"base_uri": "https://localhost:8080/"} id="g4wAq8xMyRTu_pyg"
def smiles_to_3d_pyg(smiles: str, cutoff_distance=4.0, optimize=True):
    """
    Convert a SMILES string to a 3D PyTorch Geometric Data object.
    
    Args:
        smiles (str): SMILES string of the molecule
        cutoff_distance (float): Distance cutoff for creating edges
        optimize (bool): Whether to optimize the 3D geometry
    
    Returns:
        torch_geometric.data.Data: PyG Data object with 3D features
    """
    # Generate 3D conformer
    mol = generate_3d_conformer(smiles, optimize=optimize)
    
    # Create 3D graph
    enhanced_feat, edge_idx, edge_feat, coords, geom_feat = create_advanced_3d_graph(
        mol, cutoff_distance=cutoff_distance
    )
    
    # Convert to PyTorch tensors
    x = torch.tensor(enhanced_feat, dtype=torch.float)
    edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_feat, dtype=torch.float)
    pos = torch.tensor(coords, dtype=torch.float)  # 3D coordinates
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,  # 3D coordinates as 'pos' attribute
        smiles=smiles
    )
    
    # Add molecular-level features as graph attributes
    data.molecular_diameter = torch.tensor([geom_feat['diameter']], dtype=torch.float)
    data.radius_of_gyration = torch.tensor([geom_feat['radius_of_gyration']], dtype=torch.float)
    data.asphericity = torch.tensor([geom_feat['asphericity']], dtype=torch.float)
    
    return data

# Test 3D PyG conversion
print("Converting molecules to 3D PyG format:")
print("=" * 40)

test_molecules_3d = ["CO", "CCO", "c1ccccc1"]  # methanol, ethanol, benzene

for smiles in test_molecules_3d:
    try:
        data_3d = smiles_to_3d_pyg(smiles, cutoff_distance=4.0)
        
        print(f"SMILES: {smiles}")
        print(f"  Node features: {data_3d.x.shape}")
        print(f"  Edge index: {data_3d.edge_index.shape}")
        print(f"  Edge features: {data_3d.edge_attr.shape}")
        print(f"  3D coordinates: {data_3d.pos.shape}")
        print(f"  Molecular diameter: {data_3d.molecular_diameter.item():.2f} Å")
        print(f"  Radius of gyration: {data_3d.radius_of_gyration.item():.2f} Å")
        print()
        
    except Exception as e:
        print(f"SMILES: {smiles} - Error: {e}")
        print()

# + [markdown] id="4xmsaEc01LZT_conclusion"
# ## 10. Conclusion <a name="conclusion"></a>
#
# This tutorial introduced you to 3D molecular representation for graph neural networks. Here are the key takeaways:
#
# ### Key Concepts Learned
#
# 1. **3D Conformers**: Molecules exist in 3D space with specific geometric arrangements that profoundly affect their properties and behavior.
#
# 2. **Conformational Flexibility**: Many molecules can adopt multiple 3D shapes, each with different energies and properties.
#
# 3. **Distance-Based Graphs**: 3D molecular graphs can include both covalent bonds and non-covalent interactions based on spatial proximity.
#
# 4. **Geometric Features**: 3D representations enable calculation of sophisticated molecular descriptors like asphericity, molecular volume, and surface area.
#
# 5. **Visualization**: 3D visualization helps understand the spatial relationships and interactions within molecules.
#
# ### Advantages of 3D Representations
#
# - **Captures stereochemistry** and conformational effects
# - **Includes non-covalent interactions** (H-bonds, π-π stacking, etc.)
# - **Enables geometric descriptors** that correlate with molecular properties
# - **Better represents drug-target interactions** and binding
#
# ### Challenges and Considerations
#
# - **Conformational sampling**: Which conformer(s) to use?
# - **Computational cost**: 3D coordinates and distance calculations are expensive
# - **Cutoff distance selection**: Balance between capturing interactions and avoiding noise
# - **Dynamic nature**: Molecules are flexible, but we represent static snapshots
#
# ### When to Use 3D vs 2D
#
# **Use 3D representations when:**
# - Stereochemistry matters (drug design, catalysis)
# - Non-covalent interactions are important
# - Predicting properties related to molecular shape/size
# - Working with conformationally flexible molecules
#
# **Use 2D representations when:**
# - Computational efficiency is critical
# - Chemical connectivity is the primary concern
# - Working with large datasets where 3D generation is impractical
# - Focusing on chemical reaction prediction
#
# ### Next Steps
#
# Now that you understand both 2D and 3D molecular representations, you're ready to:
# 1. Build graph neural networks that can handle both representation types
# 2. Explore how different representations affect model performance
# 3. Learn about advanced GNN architectures designed for 3D molecular data
# 4. Investigate ensemble methods that combine multiple conformers
#
# ### Additional Resources
#
# 1. **"Geometric Deep Learning"** by Bronstein et al. - Comprehensive overview of GNNs for 3D data
# 2. **"DimeNet: Directional Message Passing for Molecular Graphs"** - Advanced 3D molecular GNN architecture
# 3. **RDKit Documentation** - Comprehensive guide to conformer generation and 3D molecular descriptors
# 4. **PyTorch Geometric Tutorials** - Implementation examples for 3D molecular GNNs

# + [markdown] id="zMjXPTzUvnYk_final"
# ### ✅ Final Challenge: Comprehensive 3D Analysis
#
# Put your knowledge to the test with this comprehensive exercise:
#
# **Challenge**: Choose a drug molecule (e.g., aspirin, ibuprofen, or caffeine) and perform a complete 3D analysis:
#
# 1. **Multi-conformer analysis**: Generate 10-20 conformers and analyze their energy distribution
# 2. **Graph comparison**: Compare 2D vs 3D graph representations across different conformers
# 3. **Feature analysis**: Calculate and compare geometric descriptors across conformers
# 4. **Visualization**: Create compelling 3D visualizations showing conformational diversity
# 5. **PyG conversion**: Convert all conformers to PyG format for ML applications
#
# This exercise will reinforce all concepts from this tutorial and prepare you for real-world applications of 3D molecular GNNs.
#
# **Bonus**: Investigate how conformational flexibility might affect molecular property prediction by comparing the variance in calculated descriptors across conformers.
