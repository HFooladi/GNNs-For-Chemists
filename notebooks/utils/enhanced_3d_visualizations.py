#!/usr/bin/env python3
"""
Enhanced 3D Molecular Visualizations for Chemists using py3Dmol
==============================================================

This module provides improved visualizations and chemical intuition 
for the 01.1_GNN_3D_representation notebook using py3Dmol for 
professional molecular rendering.

Key improvements:
1. Interactive conformer comparison with chemical properties using py3Dmol
2. Drug-receptor binding pocket visualization with proper molecular rendering
3. Stereochemistry examples with 3D implications using ball-and-stick models
4. Distance-based graph construction with chemical context
5. Conformational energy landscapes with molecular interpretations
6. Professional molecular visualization using py3Dmol instead of matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.Descriptors import MolLogP, TPSA, NumHDonors, NumHAcceptors
import networkx as nx

# Import py3Dmol for molecular visualization
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    print("py3Dmol not available - falling back to matplotlib visualizations")

# Set style for better chemistry visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_stereochemistry_3d_example():
    """
    Create a compelling example showing why 3D matters using stereoisomers.
    Demonstrates how mirror-image molecules have identical 2D graphs but different 3D properties.
    
    This addresses the key chemical intuition gap in the current notebook.
    """
    # Example: Thalidomide-like molecule (simplified for educational purposes)
    # R and S enantiomers have same connectivity but different biological activity
    
    fig = plt.figure(figsize=(16, 10))
    
    # Title with chemical context
    fig.suptitle("Why 3D Matters: Stereochemistry and Drug Activity", fontsize=16, fontweight='bold')
    
    # Create example molecules - chiral center example
    chiral_smiles = "C[C@H](N)C(=O)O"  # L-alanine
    chiral_smiles_mirror = "C[C@@H](N)C(=O)O"  # D-alanine
    
    mol_l = Chem.MolFromSmiles(chiral_smiles)
    mol_d = Chem.MolFromSmiles(chiral_smiles_mirror)
    
    # Add explicit hydrogens for proper 3D representation
    mol_l = Chem.AddHs(mol_l)
    mol_d = Chem.AddHs(mol_d)
    
    # Generate 3D conformers
    AllChem.EmbedMolecule(mol_l, randomSeed=42)
    AllChem.EmbedMolecule(mol_d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_l)
    AllChem.MMFFOptimizeMolecule(mol_d)
    
    # Plot 1: 2D structures (identical!)
    ax1 = plt.subplot(2, 3, 1)
    img_l = Draw.MolToImage(mol_l, size=(250, 250))
    ax1.imshow(img_l)
    ax1.set_title("L-Alanine (2D)", fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    img_d = Draw.MolToImage(mol_d, size=(250, 250))
    ax2.imshow(img_d)
    ax2.set_title("D-Alanine (2D)", fontweight='bold')
    ax2.axis('off')
    
    # Plot 3: Graph analysis
    ax3 = plt.subplot(2, 3, 3)
    ax3.text(0.1, 0.8, "2D Graph Analysis:", fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.65, "• Same connectivity", fontsize=12)
    ax3.text(0.1, 0.55, "• Identical adjacency matrices", fontsize=12)
    ax3.text(0.1, 0.45, "• Same node/edge features", fontsize=12)
    ax3.text(0.1, 0.3, "❌ Cannot distinguish enantiomers", fontsize=12, color='red', fontweight='bold')
    ax3.text(0.1, 0.15, "❌ Miss biological differences", fontsize=12, color='red', fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Plot 4-5: 3D coordinates comparison
    coords_l = []
    coords_d = []
    
    # Create py3Dmol viewers for L and D alanine
    viewer_l = py3Dmol.view(width=400, height=400)
    viewer_d = py3Dmol.view(width=400, height=400)
    
    # Add molecules to viewers
    mol_block_l = Chem.MolToMolBlock(mol_l)
    mol_block_d = Chem.MolToMolBlock(mol_d)
    
    viewer_l.addModel(mol_block_l, "mol")
    viewer_d.addModel(mol_block_d, "mol")
    
    # Style the molecules
    style = {
        'stick': {'colorscheme': 'Jmol', 'radius': 0.2},
        'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}
    }
    
    viewer_l.setStyle({'model': -1}, style)
    viewer_d.setStyle({'model': -1}, style)
    
    # Set camera angle and zoom
    viewer_l.zoomTo()
    viewer_d.zoomTo()
    
    # Create matplotlib subplots to host the py3Dmol viewers
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("L-Alanine (3D)", fontweight='bold')
    ax4.axis('off')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("D-Alanine (3D)", fontweight='bold') 
    ax5.axis('off')
    
    # Display the viewers in the matplotlib subplots
    viewer_l.show()
    viewer_d.show()
    
    ax5.set_title("D-Alanine (3D)", fontweight='bold')
    ax5.set_xlabel('X (Å)')
    ax5.set_ylabel('Y (Å)')
    ax5.set_zlabel('Z (Å)')
    
    # Plot 6: Chemical consequences
    ax6 = plt.subplot(2, 3, 6)
    ax6.text(0.1, 0.85, "3D Graph Benefits:", fontsize=14, fontweight='bold', color='green')
    ax6.text(0.1, 0.7, "✓ Captures spatial arrangement", fontsize=12, color='green')
    ax6.text(0.1, 0.6, "✓ Distinguishes enantiomers", fontsize=12, color='green')
    ax6.text(0.1, 0.5, "✓ Predicts binding affinity", fontsize=12, color='green')
    ax6.text(0.1, 0.4, "✓ Explains selectivity", fontsize=12, color='green')
    
    ax6.text(0.1, 0.25, "Real Example:", fontsize=12, fontweight='bold')
    ax6.text(0.1, 0.15, "• Thalidomide: R = sedative", fontsize=10)
    ax6.text(0.1, 0.08, "• Thalidomide: S = teratogenic", fontsize=10)
    ax6.text(0.1, 0.01, "Same 2D graph, opposite effects!", fontsize=10, style='italic')
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return mol_l, mol_d

def create_interactive_conformer_explorer(smiles, molecule_name, n_conformers=10):
    """
    Create an interactive plotly visualization to explore conformational space.
    Shows energy landscape, molecular properties, and 3D structures side-by-side.
    
    This addresses the lack of interactive visualization in the current notebook.
    """
    print(f"Creating interactive conformer explorer for {molecule_name}...")
    
    # Generate molecule and conformers
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate multiple conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    
    # Optimize and calculate properties for each conformer
    conformer_data = []
    
    for i, conf_id in enumerate(conf_ids):
        # Optimize geometry
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        
        if ff:
            ff.Minimize()
            energy = ff.CalcEnergy()
        else:
            energy = 0.0
        
        # Get 3D coordinates
        conf = mol.GetConformer(conf_id)
        coords = []
        for atom_idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            coords.append([pos.x, pos.y, pos.z])
        coords = np.array(coords)
        
        # Calculate 3D descriptors
        volume = calculate_molecular_volume(coords)
        radius_gyration = calculate_radius_of_gyration(coords)
        asphericity = calculate_asphericity(coords)
        
        conformer_data.append({
            'conformer_id': i,
            'energy': energy,
            'volume': volume,
            'radius_gyration': radius_gyration,
            'asphericity': asphericity,
            'coordinates': coords
        })
    
    # Create interactive dashboard
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            f"{molecule_name}: Energy Landscape",
            "Volume vs Energy",
            "3D Shape Analysis",
            "Conformer Properties",
            "Distance Matrix",
            "Chemical Interpretation"
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
               [{"type": "table"}, {"type": "heatmap"}, {"type": "xy"}]]
    )
    
    # Plot 1: Energy landscape
    energies = [conf['energy'] for conf in conformer_data]
    fig.add_trace(
        go.Scatter(
            x=list(range(len(energies))),
            y=energies,
            mode='markers+lines',
            name='Conformer Energy',
            marker=dict(size=10, color=energies, colorscale='Viridis'),
            hovertemplate="Conformer %{x}<br>Energy: %{y:.2f} kcal/mol<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Plot 2: Volume vs Energy correlation
    volumes = [conf['volume'] for conf in conformer_data]
    fig.add_trace(
        go.Scatter(
            x=energies,
            y=volumes,
            mode='markers',
            name='Volume-Energy',
            marker=dict(size=8, color='blue'),
            hovertemplate="Energy: %{x:.2f} kcal/mol<br>Volume: %{y:.2f} Ų<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Plot 3: 3D shape analysis
    rg_values = [conf['radius_gyration'] for conf in conformer_data]
    asp_values = [conf['asphericity'] for conf in conformer_data]
    
    fig.add_trace(
        go.Scatter(
            x=rg_values,
            y=asp_values,
            mode='markers',
            name='Shape Analysis',
            marker=dict(size=8, color=energies, colorscale='Plasma'),
            hovertemplate="Radius of Gyration: %{x:.2f} Å<br>Asphericity: %{y:.2f}<extra></extra>"
        ),
        row=1, col=3
    )
    
    # Plot 4: Properties table
    df_props = pd.DataFrame(conformer_data).round(3)
    fig.add_trace(
        go.Table(
            header=dict(values=['Conformer', 'Energy', 'Volume', 'R_g', 'Asphericity']),
            cells=dict(values=[
                df_props['conformer_id'],
                df_props['energy'],
                df_props['volume'],
                df_props['radius_gyration'],
                df_props['asphericity']
            ])
        ),
        row=2, col=1
    )
    
    # Plot 5: Distance matrix for lowest energy conformer
    lowest_energy_conf = min(conformer_data, key=lambda x: x['energy'])
    coords = lowest_energy_conf['coordinates']
    dist_matrix = calculate_distance_matrix(coords)
    
    fig.add_trace(
        go.Heatmap(
            z=dist_matrix,
            colorscale='Blues',
            name='Distance Matrix'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Interactive Conformer Analysis: {molecule_name}",
        height=800,
        showlegend=False
    )
    
    fig.show()
    
    return conformer_data

def create_binding_pocket_visualization():
    """
    Create a visualization showing how 3D molecular shape affects drug-receptor binding.
    This provides concrete chemical context for why 3D representations matter.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("3D Shape Complementarity in Drug Design", fontsize=16, fontweight='bold')
    
    # Simulate a binding pocket (simplified)
    # Create a "lock and key" visualization
    
    # Plot 1: 2D representation (misleading)
    ax1 = axes[0, 0]
    ax1.set_title("2D Drug Design (Misleading)", fontweight='bold', color='red')
    
    # Draw simple 2D shapes
    drug_2d = plt.Circle((0.3, 0.5), 0.15, color='blue', alpha=0.7, label='Drug')
    pocket_2d = plt.Rectangle((0.1, 0.3), 0.8, 0.4, fill=False, 
                             edgecolor='red', linewidth=3, label='Binding Site')
    
    ax1.add_patch(drug_2d)
    ax1.add_patch(pocket_2d)
    ax1.text(0.5, 0.1, "2D suggests good fit", ha='center', fontsize=12)
    ax1.text(0.5, 0.05, "❌ Actually poor binding", ha='center', fontsize=12, color='red')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.axis('off')
    
    # Plot 2: 3D reality (accurate)
    ax2 = axes[0, 1]
    ax2.set_title("3D Shape Complementarity (Accurate)", fontweight='bold', color='green')
    
    # Create 3D-like visualization
    x = np.linspace(0, 1, 100)
    y1 = 0.5 + 0.2 * np.sin(8 * np.pi * x)  # Drug surface
    y2 = 0.5 - 0.2 * np.sin(8 * np.pi * x)  # Receptor surface
    
    ax2.fill_between(x, y2, y1, alpha=0.3, color='green', label='Good Complementarity')
    ax2.plot(x, y1, 'b-', linewidth=3, label='Drug Surface')
    ax2.plot(x, y2, 'r-', linewidth=3, label='Receptor Surface')
    ax2.text(0.5, 0.8, "Perfect shape match", ha='center', fontsize=12)
    ax2.text(0.5, 0.2, "✓ Strong binding", ha='center', fontsize=12, color='green')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.axis('off')
    
    # Plot 3: Distance-based features importance
    ax3 = axes[1, 0]
    ax3.set_title("Why Distance-Based Edges Matter", fontweight='bold')
    
    # Create example showing non-covalent interactions
    atom_positions = np.array([[0.2, 0.3], [0.8, 0.3], [0.5, 0.7], [0.5, 0.1]])
    atom_labels = ['H-bond\nDonor', 'H-bond\nAcceptor', 'Hydrophobic', 'Charged']
    colors = ['lightblue', 'lightcoral', 'yellow', 'lightgreen']
    
    for i, (pos, label, color) in enumerate(zip(atom_positions, atom_labels, colors)):
        circle = plt.Circle(pos, 0.08, color=color, alpha=0.7)
        ax3.add_patch(circle)
        ax3.text(pos[0], pos[1]-0.15, label, ha='center', fontsize=9)
    
    # Draw distance-based interactions
    ax3.plot([0.2, 0.8], [0.3, 0.3], 'g--', linewidth=2, label='H-bond (2.8 Å)')
    ax3.plot([0.5, 0.5], [0.7, 0.1], 'b--', linewidth=2, label='Electrostatic (4.5 Å)')
    
    ax3.text(0.5, 0.9, "Non-covalent interactions", ha='center', fontsize=12, fontweight='bold')
    ax3.text(0.5, 0.05, "Captured by distance cutoffs", ha='center', fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper right')
    ax3.axis('off')
    
    # Plot 4: GNN advantage
    ax4 = axes[1, 1]
    ax4.set_title("3D GNN Advantages", fontweight='bold')
    
    advantages = [
        "✓ Captures shape complementarity",
        "✓ Models non-covalent interactions", 
        "✓ Predicts binding affinity",
        "✓ Explains selectivity",
        "✓ Guides optimization"
    ]
    
    for i, advantage in enumerate(advantages):
        ax4.text(0.1, 0.9 - i*0.15, advantage, fontsize=12, color='green')
    
    # Add example applications
    ax4.text(0.1, 0.2, "Applications:", fontsize=12, fontweight='bold')
    ax4.text(0.1, 0.1, "• Drug discovery", fontsize=10)
    ax4.text(0.1, 0.05, "• Enzyme design", fontsize=10)
    ax4.text(0.1, 0.0, "• Protein folding", fontsize=10)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_conformer_energy_landscape_with_chemistry(smiles, molecule_name):
    """
    Create a chemically-informed energy landscape visualization showing
    how molecular properties change across conformational space.
    """
    print(f"Analyzing conformational landscape for {molecule_name}...")
    
    # Generate conformers
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate many conformers for smooth landscape
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=50, params=params)
    
    # Calculate properties for each conformer
    energies = []
    torsions = []  # Main chain torsion angle
    volumes = []
    dipole_moments = []
    
    for conf_id in conf_ids:
        # Optimize and get energy
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        
        if ff:
            ff.Minimize()
            energy = ff.CalcEnergy()
        else:
            energy = 0.0
        
        energies.append(energy)
        
        # Calculate torsion angle (for flexible molecules)
        if mol.GetNumAtoms() >= 4:
            torsion = calculate_main_torsion(mol, conf_id)
            torsions.append(torsion)
        else:
            torsions.append(0.0)
        
        # Calculate volume
        coords = get_conformer_coordinates(mol, conf_id)
        volume = calculate_molecular_volume(coords)
        volumes.append(volume)
        
        # Approximate dipole moment (simplified)
        dipole = calculate_approximate_dipole(mol, conf_id)
        dipole_moments.append(dipole)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Conformational Analysis: {molecule_name}", fontsize=16, fontweight='bold')
    
    # Plot 1: Energy vs Torsion
    ax1 = axes[0, 0]
    scatter = ax1.scatter(torsions, energies, c=volumes, cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('Main Torsion Angle (degrees)')
    ax1.set_ylabel('Energy (kcal/mol)')
    ax1.set_title('Energy vs Conformation')
    plt.colorbar(scatter, ax=ax1, label='Volume (ų)')
    
    # Add chemical interpretation
    min_energy_idx = np.argmin(energies)
    ax1.scatter(torsions[min_energy_idx], energies[min_energy_idx], 
               c='red', s=100, marker='*', label='Global Minimum')
    ax1.legend()
    
    # Plot 2: Volume distribution
    ax2 = axes[0, 1]
    ax2.hist(volumes, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(volumes), color='red', linestyle='--', 
               label=f'Mean: {np.mean(volumes):.1f} ų')
    ax2.set_xlabel('Molecular Volume (ų)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Volume Distribution')
    ax2.legend()
    
    # Plot 3: Energy-Volume correlation
    ax3 = axes[0, 2]
    ax3.scatter(energies, volumes, alpha=0.7, c='green')
    correlation = np.corrcoef(energies, volumes)[0, 1]
    ax3.set_xlabel('Energy (kcal/mol)')
    ax3.set_ylabel('Volume (ų)')
    ax3.set_title(f'Energy-Volume Correlation (r={correlation:.3f})')
    
    # Add trendline
    z = np.polyfit(energies, volumes, 1)
    p = np.poly1d(z)
    ax3.plot(energies, p(energies), "r--", alpha=0.8)
    
    # Plot 4: Ramachandran-style plot
    ax4 = axes[1, 0]
    hist, xedges, yedges = np.histogram2d(torsions, energies, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax4.imshow(hist.T, extent=extent, origin='lower', cmap='Blues', alpha=0.7)
    ax4.contour(hist.T, extent=extent, colors='black', alpha=0.5)
    ax4.set_xlabel('Torsion Angle (degrees)')
    ax4.set_ylabel('Energy (kcal/mol)')
    ax4.set_title('Conformational Density Map')
    plt.colorbar(im, ax=ax4, label='Conformer Count')
    
    # Plot 5: Dipole moment analysis
    ax5 = axes[1, 1]
    ax5.scatter(dipole_moments, energies, c=torsions, cmap='plasma', alpha=0.7)
    ax5.set_xlabel('Dipole Moment (approx.)')
    ax5.set_ylabel('Energy (kcal/mol)')
    ax5.set_title('Polarity vs Energy')
    
    # Plot 6: Chemical interpretation
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Statistical summary
    energy_range = max(energies) - min(energies)
    volume_range = max(volumes) - min(volumes)
    
    interpretation_text = f"""
Chemical Interpretation:

Energy Analysis:
• Range: {energy_range:.2f} kcal/mol
• Accessible conformers (kT = 0.6): {sum(1 for e in energies if e - min(energies) < 2)}/50
• Flexibility: {'High' if energy_range > 5 else 'Moderate' if energy_range > 2 else 'Low'}

Volume Analysis:
• Range: {volume_range:.1f} ų
• Shape flexibility: {'High' if volume_range > 20 else 'Moderate' if volume_range > 10 else 'Low'}

Implications for GNNs:
• Distance cutoffs matter more for flexible molecules
• Conformer selection affects predictions
• Multiple conformers may be needed
    """
    
    ax6.text(0.05, 0.95, interpretation_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'energies': energies,
        'torsions': torsions,
        'volumes': volumes,
        'dipole_moments': dipole_moments,
        'min_energy_conformer': min_energy_idx
    }

# Helper functions for calculations
def calculate_molecular_volume(coords):
    """Calculate approximate molecular volume using convex hull."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords)
        return hull.volume
    except:
        # Fallback: bounding box volume
        ranges = np.ptp(coords, axis=0)
        return np.prod(ranges)

def calculate_radius_of_gyration(coords):
    """Calculate radius of gyration."""
    center_of_mass = np.mean(coords, axis=0)
    rg_squared = np.mean(np.sum((coords - center_of_mass)**2, axis=1))
    return np.sqrt(rg_squared)

def calculate_asphericity(coords):
    """Calculate asphericity (measure of non-spherical shape)."""
    center_of_mass = np.mean(coords, axis=0)
    centered_coords = coords - center_of_mass
    
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
    
    return eigenvalues[2] - 0.5*(eigenvalues[0] + eigenvalues[1])

def calculate_distance_matrix(coords):
    """Calculate pairwise distance matrix."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

def calculate_main_torsion(mol, conf_id):
    """Calculate main chain torsion angle for analysis."""
    conf = mol.GetConformer(conf_id)
    
    # Find first 4 connected atoms for torsion
    atoms = list(range(min(4, mol.GetNumAtoms())))
    
    if len(atoms) < 4:
        return 0.0
    
    # Get positions
    positions = []
    for atom_idx in atoms:
        pos = conf.GetAtomPosition(atom_idx)
        positions.append(np.array([pos.x, pos.y, pos.z]))
    
    # Calculate torsion angle
    try:
        return calculate_dihedral_angle(positions[0], positions[1], positions[2], positions[3])
    except:
        return 0.0

def calculate_dihedral_angle(p1, p2, p3, p4):
    """Calculate dihedral angle between four points."""
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def get_conformer_coordinates(mol, conf_id):
    """Get coordinates for a specific conformer."""
    conf = mol.GetConformer(conf_id)
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    return np.array(coords)

def calculate_approximate_dipole(mol, conf_id):
    """Calculate approximate dipole moment based on electronegativity."""
    conf = mol.GetConformer(conf_id)
    
    # Electronegativity values (Pauling scale)
    electronegativity = {'H': 2.1, 'C': 2.5, 'N': 3.0, 'O': 3.5, 'F': 4.0, 'S': 2.5, 'P': 2.1}
    
    dipole_vector = np.zeros(3)
    
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        pos = conf.GetAtomPosition(i)
        
        electronegativity_val = electronegativity.get(symbol, 2.5)
        charge_approximation = electronegativity_val - 2.5  # Relative to carbon
        
        dipole_vector += charge_approximation * np.array([pos.x, pos.y, pos.z])
    
    return np.linalg.norm(dipole_vector)

# =============================================================================
# PY3DMOL-BASED VISUALIZATIONS
# =============================================================================

def create_py3dmol_stereochemistry_example():
    """
    Create py3Dmol visualization showing stereoisomers side by side.
    This provides much better molecular representation than matplotlib.
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available. Install with: pip install py3dmol")
        return None
    
    print("🧬 3D STEREOCHEMISTRY WITH PY3DMOL")
    print("=" * 45)
    
    # Create example molecules - chiral center example
    chiral_smiles_l = "C[C@H](N)C(=O)O"  # L-alanine
    chiral_smiles_d = "C[C@@H](N)C(=O)O"  # D-alanine
    
    mol_l = Chem.MolFromSmiles(chiral_smiles_l)
    mol_d = Chem.MolFromSmiles(chiral_smiles_d)
    
    # Add explicit hydrogens for proper 3D representation
    mol_l = Chem.AddHs(mol_l)
    mol_d = Chem.AddHs(mol_d)
    
    # Generate 3D conformers
    AllChem.EmbedMolecule(mol_l, randomSeed=42)
    AllChem.EmbedMolecule(mol_d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_l)
    AllChem.MMFFOptimizeMolecule(mol_d)
    
    # Create py3Dmol viewer
    viewer = py3Dmol.view(width=800, height=400, viewergrid=(1, 2))
    
    # Add L-alanine to left panel
    mol_block_l = Chem.MolToMolBlock(mol_l)
    viewer.addModel(mol_block_l, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.15}}, viewer=(0, 0))
    viewer.addLabel('L-Alanine\n(Natural amino acid)', 
                   {'position': {'x': 0, 'y': 3, 'z': 0}, 'backgroundColor': 'lightblue', 
                    'fontColor': 'black', 'fontSize': 14}, viewer=(0, 0))
    
    # Add D-alanine to right panel
    mol_block_d = Chem.MolToMolBlock(mol_d)
    viewer.addModel(mol_block_d, 'mol', viewer=(0, 1))
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.15}}, viewer=(0, 1))
    viewer.addLabel('D-Alanine\n(Synthetic enantiomer)', 
                   {'position': {'x': 0, 'y': 3, 'z': 0}, 'backgroundColor': 'lightcoral', 
                    'fontColor': 'black', 'fontSize': 14}, viewer=(0, 1))
    
    # Center and zoom
    viewer.zoomTo()
    
    print("Key insights:")
    print("• Same connectivity, different 3D arrangements")
    print("• Different biological activities")
    print("• 2D graphs cannot distinguish them")
    print("• 3D coordinates are essential")
    
    return viewer

def create_py3dmol_conformer_explorer(smiles, molecule_name, n_conformers=5):
    """
    Create interactive py3Dmol conformer explorer with multiple conformations.
    Much better than matplotlib for showing molecular flexibility.
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available. Install with: pip install py3dmol")
        return None
    
    print(f"🔄 CONFORMER EXPLORER: {molecule_name}")
    print("=" * 45)
    
    # Generate molecule and conformers
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate multiple conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    
    # Optimize conformers and calculate energies
    energies = []
    for conf_id in conf_ids:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff:
            ff.Minimize()
            energy = ff.CalcEnergy()
        else:
            energy = 0.0
        energies.append(energy)
    
    # Sort by energy
    sorted_conformers = sorted(zip(conf_ids, energies), key=lambda x: x[1])
    
    # Create viewer grid based on number of conformers
    cols = min(3, n_conformers)
    rows = (n_conformers + cols - 1) // cols
    viewer = py3Dmol.view(width=300*cols, height=300*rows, viewergrid=(rows, cols))
    
    # Display conformers
    for i, (conf_id, energy) in enumerate(sorted_conformers):
        row = i // cols
        col = i % cols
        
        # Extract conformer
        mol_copy = Chem.Mol(mol)
        mol_copy.RemoveAllConformers()
        conf = mol.GetConformer(conf_id)
        mol_copy.AddConformer(conf, assignId=True)
        
        mol_block = Chem.MolToMolBlock(mol_copy)
        viewer.addModel(mol_block, 'mol', viewer=(row, col))
        
        # Style based on energy rank
        if i == 0:  # Lowest energy - highlight
            style = {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.2}}
            bg_color = 'lightgreen'
            label_text = f'Global Min\n{energy:.2f} kcal/mol'
        else:
            style = {'stick': {'colorscheme': 'Jmol', 'radius': 0.15}}
            bg_color = 'lightyellow'
            label_text = f'Conformer {i+1}\n{energy:.2f} kcal/mol'
        
        viewer.setStyle({'model': -1}, style, viewer=(row, col))
        viewer.addLabel(label_text, 
                       {'position': {'x': 0, 'y': 3, 'z': 0}, 'backgroundColor': bg_color,
                        'fontColor': 'black', 'fontSize': 12}, viewer=(row, col))
    
    viewer.zoomTo()
    
    # Print analysis
    energy_range = max(energies) - min(energies)
    print(f"Energy range: {energy_range:.2f} kcal/mol")
    print(f"Flexibility: {'High' if energy_range > 5 else 'Moderate' if energy_range > 2 else 'Low'}")
    print(f"Accessible conformers (ΔE < 3 kcal/mol): {sum(1 for e in energies if e - min(energies) < 3)}/{n_conformers}")
    
    return viewer

def create_py3dmol_binding_pocket_demo():
    """
    Create a drug-receptor binding demonstration using py3Dmol.
    Shows complementary shapes and binding interactions.
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available. Install with: pip install py3dmol")
        return None
    
    print("💊 DRUG-RECEPTOR BINDING DEMO")
    print("=" * 45)
    
    # Example: Aspirin (drug) and a simplified binding site representation
    drug_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    substrate_smiles = "CC(=O)C"  # Simple substrate for comparison
    
    mol_drug = Chem.MolFromSmiles(drug_smiles)
    mol_substrate = Chem.MolFromSmiles(substrate_smiles)
    
    mol_drug = Chem.AddHs(mol_drug)
    mol_substrate = Chem.AddHs(mol_substrate)
    
    # Generate 3D conformers
    AllChem.EmbedMolecule(mol_drug, randomSeed=42)
    AllChem.EmbedMolecule(mol_substrate, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_drug)
    AllChem.MMFFOptimizeMolecule(mol_substrate)
    
    # Create side-by-side comparison
    viewer = py3Dmol.view(width=800, height=400, viewergrid=(1, 2))
    
    # Drug molecule (left)
    mol_block_drug = Chem.MolToMolBlock(mol_drug)
    viewer.addModel(mol_block_drug, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}}, viewer=(0, 0))
    viewer.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'blue'}, viewer=(0, 0))
    viewer.addLabel('Aspirin (Drug)\nComplex 3D shape\nMultiple interaction sites', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 'backgroundColor': 'lightblue',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(0, 0))
    
    # Simple substrate (right)
    mol_block_substrate = Chem.MolToMolBlock(mol_substrate)
    viewer.addModel(mol_block_substrate, 'mol', viewer=(0, 1))
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}}, viewer=(0, 1))
    viewer.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'orange'}, viewer=(0, 1))
    viewer.addLabel('Simple Substrate\nSimple shape\nFewer interactions', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 'backgroundColor': 'lightyellow',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(0, 1))
    
    viewer.zoomTo()
    
    print("Key concepts demonstrated:")
    print("• Van der Waals surfaces show molecular shape")
    print("• Complex shapes → selective binding")
    print("• 3D complementarity essential for activity")
    print("• Distance-based edges capture binding interactions")
    
    return viewer

def create_py3dmol_graph_edges_demo(smiles, molecule_name, cutoff_distances=[3.0, 4.0, 5.0]):
    """
    Demonstrate how distance cutoffs affect molecular graph connectivity using py3Dmol.
    Shows the same molecule with different edge definitions.
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available. Install with: pip install py3dmol")
        return None
    
    print(f"📊 DISTANCE-BASED EDGES DEMO: {molecule_name}")
    print("=" * 45)
    
    # Generate molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Create viewer for different cutoffs
    n_cutoffs = len(cutoff_distances)
    viewer = py3Dmol.view(width=300*n_cutoffs, height=400, viewergrid=(1, n_cutoffs))
    
    for i, cutoff in enumerate(cutoff_distances):
        # Add molecule
        mol_block = Chem.MolToMolBlock(mol)
        viewer.addModel(mol_block, 'mol', viewer=(0, i))
        
        # Base molecular structure
        viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.1}}, viewer=(0, i))
        
        # Calculate and visualize distance-based edges
        coords = mol.GetConformer().GetPositions()
        n_distance_edges = 0
        
        for atom1 in range(mol.GetNumAtoms()):
            for atom2 in range(atom1 + 1, mol.GetNumAtoms()):
                distance = np.linalg.norm(coords[atom1] - coords[atom2])
                
                # Only show non-covalent interactions
                bond = mol.GetBondBetweenAtoms(atom1, atom2)
                if bond is None and distance <= cutoff:
                    n_distance_edges += 1
                    # Add distance edge as a line
                    pos1 = coords[atom1]
                    pos2 = coords[atom2]
                    
                    # Color based on distance
                    if distance <= 3.0:
                        color = 'red'  # Strong interaction
                    elif distance <= 4.0:
                        color = 'orange'  # Medium interaction
                    else:
                        color = 'yellow'  # Weak interaction
                    
                    viewer.addCylinder({
                        'start': {'x': pos1[0], 'y': pos1[1], 'z': pos1[2]},
                        'end': {'x': pos2[0], 'y': pos2[1], 'z': pos2[2]},
                        'radius': 0.05,
                        'color': color,
                        'alpha': 0.5
                    }, viewer=(0, i))
        
        # Add label with statistics
        covalent_edges = mol.GetNumBonds()
        total_edges = covalent_edges + n_distance_edges
        
        viewer.addLabel(f'Cutoff: {cutoff} Å\nCovalent: {covalent_edges}\nDistance: {n_distance_edges}\nTotal: {total_edges}', 
                       {'position': {'x': 0, 'y': 4, 'z': 0}, 'backgroundColor': 'lightgray',
                        'fontColor': 'black', 'fontSize': 12}, viewer=(0, i))
    
    viewer.zoomTo()
    
    print(f"Covalent bonds: {mol.GetNumBonds()}")
    for cutoff in cutoff_distances:
        coords = mol.GetConformer().GetPositions()
        n_distance_edges = 0
        for atom1 in range(mol.GetNumAtoms()):
            for atom2 in range(atom1 + 1, mol.GetNumAtoms()):
                distance = np.linalg.norm(coords[atom1] - coords[atom2])
                bond = mol.GetBondBetweenAtoms(atom1, atom2)
                if bond is None and distance <= cutoff:
                    n_distance_edges += 1
        print(f"Distance edges ({cutoff} Å): {n_distance_edges}")
    
    return viewer

def create_py3dmol_multiple_conformers_overlay(smiles, molecule_name, n_conformers=3):
    """
    Create an overlay of multiple conformers to show conformational flexibility.
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available. Install with: pip install py3dmol")
        return None
    
    print(f"🌀 CONFORMATIONAL OVERLAY: {molecule_name}")
    print("=" * 45)
    
    # Generate molecule and conformers
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate multiple conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    
    # Optimize conformers
    for conf_id in conf_ids:
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
    
    # Create viewer
    viewer = py3Dmol.view(width=600, height=400)
    
    # Color scheme for different conformers
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Add each conformer with different colors
    for i, conf_id in enumerate(conf_ids):
        mol_copy = Chem.Mol(mol)
        mol_copy.RemoveAllConformers()
        conf = mol.GetConformer(conf_id)
        mol_copy.AddConformer(conf, assignId=True)
        
        mol_block = Chem.MolToMolBlock(mol_copy)
        viewer.addModel(mol_block, 'mol')
        
        color = colors[i % len(colors)]
        viewer.setStyle({'model': i}, {'stick': {'color': color, 'radius': 0.1 + i*0.05}})
    
    # Add legend
    legend_text = f'{molecule_name} Conformers:\n'
    for i in range(min(n_conformers, len(colors))):
        legend_text += f'● Conformer {i+1}\n'
    
    viewer.addLabel(legend_text, 
                   {'position': {'x': 5, 'y': 0, 'z': 0}, 'backgroundColor': 'white',
                    'fontColor': 'black', 'fontSize': 14})
    
    viewer.zoomTo()
    
    print(f"Overlaying {n_conformers} conformers")
    print("Different colors show conformational flexibility")
    print("Rigid parts overlap, flexible parts diverge")
    
    return viewer

def visualize_py3dmol_molecular_properties(smiles, molecule_name):
    """
    Create a comprehensive py3Dmol visualization showing multiple molecular representations.
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available. Install with: pip install py3dmol")
        return None
    
    print(f"🔬 MOLECULAR PROPERTIES: {molecule_name}")
    print("=" * 45)
    
    # Generate molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Create 2x2 grid viewer
    viewer = py3Dmol.view(width=800, height=600, viewergrid=(2, 2))
    
    mol_block = Chem.MolToMolBlock(mol)
    
    # 1. Ball and stick model
    viewer.addModel(mol_block, 'mol', viewer=(0, 0))
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}, 
                                   'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}}, viewer=(0, 0))
    viewer.addLabel('Ball & Stick\n(Bond representation)', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 'backgroundColor': 'lightblue',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(0, 0))
    
    # 2. Space-filling model
    viewer.addModel(mol_block, 'mol', viewer=(0, 1))
    viewer.setStyle({'model': -1}, {'sphere': {'colorscheme': 'Jmol', 'scale': 1.0}}, viewer=(0, 1))
    viewer.addLabel('Space-Filling\n(Actual molecular size)', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 'backgroundColor': 'lightgreen',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(0, 1))
    
    # 3. Wireframe with surface
    viewer.addModel(mol_block, 'mol', viewer=(1, 0))
    viewer.setStyle({'model': -1}, {'line': {'colorscheme': 'Jmol'}}, viewer=(1, 0))
    viewer.addSurface(py3Dmol.VDW, {'opacity': 0.4, 'color': 'yellow'}, viewer=(1, 0))
    viewer.addLabel('Wireframe + Surface\n(Shape & connectivity)', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 'backgroundColor': 'lightyellow',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(1, 0))
    
    # 4. Cartoon with electrostatic surface (simplified)
    viewer.addModel(mol_block, 'mol', viewer=(1, 1))
    viewer.setStyle({'model': -1}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.1}}, viewer=(1, 1))
    viewer.addSurface(py3Dmol.SAS, {'opacity': 0.6, 'colorscheme': 'RWB'}, viewer=(1, 1))
    viewer.addLabel('Solvent Surface\n(Electrostatic properties)', 
                   {'position': {'x': 0, 'y': 4, 'z': 0}, 'backgroundColor': 'lightcoral',
                    'fontColor': 'black', 'fontSize': 12}, viewer=(1, 1))
    
    viewer.zoomTo()
    
    # Calculate and display molecular properties
    mol_2d = Chem.MolFromSmiles(smiles)  # Remove hydrogens for property calculation
    mw = Descriptors.MolWt(mol_2d)
    logp = Descriptors.MolLogP(mol_2d)
    tpsa = Descriptors.TPSA(mol_2d)
    hbd = Descriptors.NumHDonors(mol_2d)
    hba = Descriptors.NumHAcceptors(mol_2d)
    
    print(f"Molecular Properties:")
    print(f"• Molecular Weight: {mw:.2f} g/mol")
    print(f"• LogP: {logp:.2f}")
    print(f"• TPSA: {tpsa:.2f} ų")
    print(f"• H-bond Donors: {hbd}")
    print(f"• H-bond Acceptors: {hba}")
    
    return viewer

# Enhanced main function to demonstrate all py3Dmol capabilities
def demonstrate_py3dmol_capabilities():
    """
    Comprehensive demonstration of all py3Dmol molecular visualization capabilities.
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available. Please install with: pip install py3dmol")
        print("Falling back to matplotlib visualizations...")
        return
    
    print("🚀 PY3DMOL MOLECULAR VISUALIZATION SHOWCASE")
    print("=" * 50)
    print("Professional molecular rendering for chemistry education")
    print()
    
    # Test molecules
    molecules = {
        "Flexible": ("CCCCCCCC", "Octane"),
        "Aromatic": ("c1ccccc1", "Benzene"), 
        "Drug": ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
        "Chiral": ("C[C@H](N)C(=O)O", "L-Alanine")
    }
    
    print("Available demonstrations:")
    print("1. Stereochemistry examples")
    print("2. Conformer exploration")
    print("3. Binding pocket demonstration")
    print("4. Distance-based edges")
    print("5. Conformational overlays")
    print("6. Molecular property visualization")
    print()
    
    # Return all available functions for use
    return {
        'stereochemistry': create_py3dmol_stereochemistry_example,
        'conformer_explorer': create_py3dmol_conformer_explorer,
        'binding_demo': create_py3dmol_binding_pocket_demo,
        'edges_demo': create_py3dmol_graph_edges_demo,
        'overlay': create_py3dmol_multiple_conformers_overlay,
        'properties': visualize_py3dmol_molecular_properties,
        'molecules': molecules
    }

def plot_3d_molecular_graph_py3dmol(mol, cutoff_distance=4.0, show_all_edges=False):
    """
    DROP-IN REPLACEMENT for the existing plotly-based plot_3d_molecular_graph function.
    
    Create a py3Dmol visualization of a molecular graph with distance-based edges.
    This provides much better molecular representation than the original plotly version.
    
    Args:
        mol: RDKit molecule object with 3D coordinates
        cutoff_distance (float): Distance cutoff for adding edges
        show_all_edges (bool): Whether to show all distance edges or just highlight them
    
    Returns:
        py3Dmol viewer object (call .show() to display in Jupyter)
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available. Please install with: pip install py3dmol")
        print("Falling back to original plotly visualization...")
        return None
    
    # Get molecule name if possible
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Molecule"
    
    # Create py3Dmol viewer
    viewer = py3Dmol.view(width=800, height=600)
    
    # Add the main molecular structure
    mol_block = Chem.MolToMolBlock(mol)
    viewer.addModel(mol_block, 'mol')
    
    # Style the molecule with standard chemical representation
    viewer.setStyle({'model': -1}, {
        'stick': {'colorscheme': 'Jmol', 'radius': 0.2}, 
        'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}
    })
    
    # Calculate and visualize distance-based edges
    coords = mol.GetConformer().GetPositions()
    n_atoms = mol.GetNumAtoms()
    n_distance_edges = 0
    
    for atom1 in range(n_atoms):
        for atom2 in range(atom1 + 1, n_atoms):
            distance = np.linalg.norm(coords[atom1] - coords[atom2])
            
            # Check if there's already a covalent bond
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            
            # Add distance-based edge if within cutoff and not covalently bonded
            if bond is None and distance <= cutoff_distance:
                n_distance_edges += 1
                pos1 = coords[atom1]
                pos2 = coords[atom2]
                
                # Color based on distance (chemical intuition)
                if distance <= 3.0:
                    color = 'red'      # Strong interaction (H-bond, etc.)
                    alpha = 0.8
                elif distance <= 4.0:
                    color = 'orange'   # Medium interaction
                    alpha = 0.6
                else:
                    color = 'yellow'   # Weak interaction (van der Waals)
                    alpha = 0.4
                
                # Add distance edge as a cylinder
                if show_all_edges or distance <= 3.5:  # Only show stronger interactions by default
                    viewer.addCylinder({
                        'start': {'x': pos1[0], 'y': pos1[1], 'z': pos1[2]},
                        'end': {'x': pos2[0], 'y': pos2[1], 'z': pos2[2]},
                        'radius': 0.05,
                        'color': color,
                        'alpha': alpha
                    })
    
    # Add informative labels
    n_covalent = mol.GetNumBonds()
    total_edges = n_covalent + n_distance_edges
    
    label_text = f'{mol_name}\nCovalent bonds: {n_covalent}\nDistance edges: {n_distance_edges}\nCutoff: {cutoff_distance} Å'
    viewer.addLabel(label_text, {
        'position': {'x': 5, 'y': 5, 'z': 0}, 
        'backgroundColor': 'white',
        'fontColor': 'black', 
        'fontSize': 14,
        'showBackground': True
    })
    
    # Add surface representation for better shape understanding
    viewer.addSurface(py3Dmol.VDW, {'opacity': 0.2, 'color': 'blue'})
    
    # Center and zoom
    viewer.zoomTo()
    
    # Print analysis
    print(f"📊 3D Molecular Graph Analysis:")
    print(f"   Molecule: {mol_name}")
    print(f"   Atoms: {n_atoms}")
    print(f"   Covalent bonds: {n_covalent}")
    print(f"   Distance edges: {n_distance_edges} (cutoff: {cutoff_distance} Å)")
    print(f"   Total edges: {total_edges}")
    print(f"   Edge density: {total_edges / (n_atoms * (n_atoms - 1) / 2):.3f}")
    print(f"   💡 Call viewer.show() to display in Jupyter notebook")
    
    return viewer

def enhanced_3d_molecular_demo_py3dmol():
    """
    Enhanced demonstration replacing all matplotlib/plotly visualizations with py3Dmol.
    This is a comprehensive upgrade to the existing tutorial demonstrations.
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol not available. Please install with: pip install py3dmol")
        print("Run: pip install py3dmol")
        return None
    
    print("🧬 ENHANCED 3D MOLECULAR DEMO WITH PY3DMOL")
    print("=" * 55)
    print("Professional molecular visualization for GNN education")
    print()
    
    # Example molecules covering different chemical scenarios
    demo_molecules = [
        ("CCO", "Ethanol", "Simple alcohol"),
        ("CCCCCCCC", "Octane", "Flexible hydrocarbon"),
        ("c1ccccc1", "Benzene", "Rigid aromatic"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", "Drug molecule"),
        ("C[C@H](N)C(=O)O", "L-Alanine", "Chiral amino acid")
    ]
    
    results = {}
    
    print("🔬 Generating molecules and visualizations...")
    print("-" * 50)
    
    for smiles, name, description in demo_molecules:
        print(f"\n📋 Analyzing {name} ({description})")
        
        try:
            # Generate 3D conformer
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol.SetProp("_Name", name)
            
            # Create py3Dmol visualization
            viewer = plot_3d_molecular_graph_py3dmol(mol, cutoff_distance=4.0)
            
            if viewer:
                results[name] = {
                    'molecule': mol,
                    'viewer': viewer,
                    'smiles': smiles,
                    'description': description
                }
                print(f"✅ {name} visualization ready!")
            else:
                print(f"❌ Failed to create visualization for {name}")
                
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
    
    print(f"\n🎉 Generated {len(results)} molecular visualizations!")
    print("\nTo display in Jupyter notebook:")
    print("for name, data in results.items():")
    print("    print(f'Displaying {name}...')")
    print("    data['viewer'].show()")
    
    return results

if __name__ == "__main__":
    print("Enhanced 3D Visualization Module with py3Dmol Loaded!")
    print("Available functions:")
    print("- create_stereochemistry_3d_example() [matplotlib version]")
    print("- create_py3dmol_stereochemistry_example() [py3Dmol version]")
    print("- create_interactive_conformer_explorer(smiles, name) [plotly version]")
    print("- create_py3dmol_conformer_explorer(smiles, name) [py3Dmol version]")
    print("- create_binding_pocket_visualization() [matplotlib version]")
    print("- create_py3dmol_binding_pocket_demo() [py3Dmol version]")
    print("- visualize_conformer_energy_landscape_with_chemistry(smiles, name)")
    print("- create_py3dmol_graph_edges_demo(smiles, name, cutoffs)")
    print("- create_py3dmol_multiple_conformers_overlay(smiles, name)")
    print("- visualize_py3dmol_molecular_properties(smiles, name)")
    print("- demonstrate_py3dmol_capabilities()")