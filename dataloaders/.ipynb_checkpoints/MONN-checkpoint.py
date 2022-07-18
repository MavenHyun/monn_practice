from .base import *

BLOSUM_DICT = {
	'A': [4,0,-2,-1,-2,0,-2,-1,-1,-1,-1,-2,-1,-1,-1,1,0,0,-3,-2],
	'C': [0,9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2],
	'D': [-2,-3,6,2,-3,-1,-1,-3,-1,-4,-3,1,-1,0,-2,0,-1,-3,-4,-3],
	'E': [-1,-4,2,5,-3,-2,0,-3,1,-3,-2,0,-1,2,0,0,-1,-2,-3,-2],
	'F': [-2,-2,-3,-3,6,-3,-1,0,-3,0,0,-3,-4,-3,-3,-2,-2,-1,1,3],
	'G': [0,-3,-1,-2,-3,6,-2,-4,-2,-4,-3,0,-2,-2,-2,0,-2,-3,-2,-3],
	'H': [-2,-3,-1,0,-1,-2,8,-3,-1,-3,-2,1,-2,0,0,-1,-2,-3,-2,2],
	'I': [-1,-1,-3,-3,0,-4,-3,4,-3,2,1,-3,-3,-3,-3,-2,-1,3,-3,-1],
	'K': [-1,-3,-1,1,-3,-2,-1,-3,5,-2,-1,0,-1,1,2,0,-1,-2,-3,-2],
	'L': [-1,-1,-4,-3,0,-4,-3,2,-2,4,2,-3,-3,-2,-2,-2,-1,1,-2,-1],
	'M': [-1,-1,-3,-2,0,-3,-2,1,-1,2,5,-2,-2,0,-1,-1,-1,1,-1,-1],
	'N': [-2,-3,1,0,-3,0,1,-3,0,-3,-2,6,-2,0,0,1,0,-3,-4,-2],
	'P': [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2,7,-1,-2,-1,-1,-2,-4,-3],
	'Q': [-1,-3,0,2,-3,-2,0,-3,1,-2,0,0,-1,5,1,0,-1,-2,-2,-1],
	'R': [-1,-3,-2,0,-3,-2,0,-3,2,-2,-1,0,-2,1,5,-1,-1,-3,-3,-2],
	'S': [1,-1,0,0,-2,0,-1,-2,0,-2,-1,1,-1,0,-1,4,1,-2,-3,-2],
	'T': [0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1,0,-1,-1,-1,1,5,0,-2,-2],
	'V': [0,-1,-3,-2,-1,-3,-3,3,-2,1,1,-3,-2,-2,-3,-2,0,4,-3,-1],
	'W': [-3,-2,-4,-3,1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11,2],
	'Y': [-2,-2,-3,-2,3,-3,2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1,2,7],
    'X': [0 for _ in range(20)],
	'unk':[0 for _ in range(20)]}

ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 
             'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 
             'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 
             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 
             'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 
             'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd',
             'Ga','Cs', 'unk']


class DtiDataset(DtiDatasetBase):
    def __init__(self, args):
        super().__init__(args)
        self.default_batch_size = 32
        max_nb = 10
        for complex_idx in tqdm(self.complex_indices):
            try:
                ligand_idx = self.complex_dataframe.loc[complex_idx, 'ligand_id']
                protein_idx = self.complex_dataframe.loc[complex_idx, 'protein_id']
                ba_value = self.complex_dataframe.loc[complex_idx, 'ba_value']  
                self.check_protein(protein_idx)

                # Ligand / Atom / Features
                atomwise_features = []
                smiles = self.ligand_dataframe.loc[ligand_idx, 'smiles']
                mol = Chem.MolFromSmiles(smiles)
                for atom in mol.GetAtoms():
                    try: atomwise_features.append(self.atom_features(atom).reshape(1, -1))
                    except: atomwise_features.append(np.zeros(82).reshape(1, -1))
                atomwise_features = np.vstack(atomwise_features)

                idxfunc = lambda x: x.GetIdx()
                n_atoms = mol.GetNumAtoms()
                assert mol.GetNumBonds() >= 0
                n_bonds = max(mol.GetNumBonds(), 1)
                atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
                bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
                num_nbs = np.zeros((n_atoms,), dtype=np.int32)
                num_nbs_mat = np.zeros((n_atoms, max_nb), dtype=np.int32)

                # Ligand / Bond / Features
                bondwise_features = ['null' for _ in range(n_bonds)]
                for bond in mol.GetBonds():
                    a1, a2 = idxfunc(bond.GetBeginAtom()), idxfunc(bond.GetEndAtom())
                    bondwise_features[bond.GetIdx()] = self.bond_features(bond).reshape(1, -1)
                    # IndexError: index 6 is out of bounds for axis 1 with size 6
                    # bond_nb[a1, num_nbs[a1]]=bond.GetIdx()
                    atom_nb[a1, num_nbs[a1]] = a2
                    atom_nb[a2, num_nbs[a2]] = a1
                    bond_nb[a1, num_nbs[a1]] = bond.GetIdx()
                    bond_nb[a2, num_nbs[a2]] = bond.GetIdx()
                    num_nbs[a1] += 1
                    num_nbs[a2] += 1
                bondwise_features = np.vstack(bondwise_features)
                for i in range(len(num_nbs)):
                    num_nbs_mat[i, :num_nbs[i]] = 1

                # Protein / Residue / Features
                resiwise_features = []
                fasta = self.protein_dataframe.loc[protein_idx, 'fasta']
                for resi in fasta:
                    resiwise_features.append(np.array(self.resi_features(resi)).reshape(1, -1))
                resiwise_features = np.vstack(resiwise_features)

                # Complex / Residue / 2D Graph
                atomatom_graph = Chem.rdmolops.GetAdjacencyMatrix(mol)
                plip_path = f'{self.data_path}complexes/{complex_idx}/{complex_idx}.plip.npy'
                if check_exists(plip_path):
                    atomresi_graph = np.load(plip_path)[:,:,:-1]
                    atomresi_label = np.ones((atomwise_features.shape[0], resiwise_features.shape[0], 1))
                else:
                    atomresi_graph = np.zeros((atomwise_features.shape[0], resiwise_features.shape[0], 1))
                    atomresi_label = np.zeros((atomwise_features.shape[0], resiwise_features.shape[0], 1))
                smiles = ''.join(list(filter(str.isalpha, smiles)))

                metadata = (complex_idx, ligand_idx, protein_idx, smiles, fasta, ba_value)
                pytrdata = (atomwise_features, bondwise_features, atom_nb, bond_nb, num_nbs_mat, resiwise_features, atomresi_graph, ba_value)

                self.data_instances.append(pytrdata)
                self.meta_instances.append(metadata)

            except Exception as e:
                pass

        print("Number of data samples for MONN: ", len(self.data_instances))
        self.indices = [i for i in range(len(self.data_instances))]

    def check_protein(self, protein_idx):
        if self.protein_dataframe.loc[protein_idx, 'fasta_length'] >= 1000:
            raise FastaLengthException(self.protein_dataframe.loc[protein_idx, 'fasta_length'])

    def check_complex(self, complex_idx):
        return
        # if not check_exists(f'{self.data_path}complexes/{complex_idx}/{complex_idx}.arpeggio.npy'):
        #     import pdb;
        #     pdb.set_trace()
        #     raise NoComplexGraphException(complex_idx)  

    def atom_features(self, atom):
        return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST)
                        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                        + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
                        + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                        + [atom.GetIsAromatic()], dtype=np.float32)

    def bond_features(self, bond):
        bt = bond.GetBondType()
        return np.array([bt == Chem.rdchem.BondType.SINGLE,
                        bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE,
                        bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()], dtype=np.float32)

    def resi_features(self, resi):
        return BLOSUM_DICT[resi]


def collate_fn(batch):
    tensor_list = []
    list_atomwise_features = [x[0] for x in batch]
    list_bondwise_features = [x[1] for x in batch]
    list_atom_neighbors = [x[2] for x in batch]
    list_bond_neighbors = [x[3] for x in batch]
    list_neighbor_matrices = [x[4] for x in batch]
    list_resiwise_features = [x[5] for x in batch]
    list_atomresi_graphs = [(x[6] > 0.).astype(np.int_) for x in batch]
    list_ba_values = [x[7] for x in batch]

    x, y, _ = stack_and_pad(list_atomwise_features)
    tensor_list.append(torch.cuda.FloatTensor(x))
    tensor_list.append(torch.cuda.FloatTensor(y))
    x, _, _ = stack_and_pad(list_bondwise_features)
    tensor_list.append(torch.cuda.FloatTensor(x))
    x, _, _ = stack_and_pad(list_atom_neighbors)
    tensor_list.append(torch.cuda.LongTensor(add_index(x, x.shape[1])))
    x, _, _ = stack_and_pad(list_bond_neighbors)
    tensor_list.append(torch.cuda.LongTensor(add_index(x, x.shape[1])))
    x, _, _ = stack_and_pad(list_neighbor_matrices)
    tensor_list.append(torch.cuda.FloatTensor(x))
    x, y, _ = stack_and_pad(list_resiwise_features)
    tensor_list.append(torch.cuda.FloatTensor(x))
    tensor_list.append(torch.cuda.FloatTensor(y))
    tensor_list.append(torch.cuda.FloatTensor(list_ba_values).view(-1, 1))
    x, _, z = stack_and_pad(list_atomresi_graphs)
    tensor_list.append(torch.cuda.FloatTensor(x))
    tensor_list.append(torch.cuda.FloatTensor(z))

    return tensor_list