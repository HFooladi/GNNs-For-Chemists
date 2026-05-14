"""Atom / bond featurization and SMILES → PyG graph conversion.

These helpers are introduced and explained in detail in
``04_GNN_GCN.ipynb``. From notebook 05 onward we import them rather
than redefine them — see each architecture notebook's helper-import
cell. The function bodies here are intentionally byte-identical to
the canonical inline definitions in notebook 04, so chemists who
read 04 can recognise them here without surprise.
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data


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
