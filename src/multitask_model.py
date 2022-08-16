from gcn import GCN

import json
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class MultiTaskGraphNeuralNetModel:
    def __init__(self, model_params={}):
        self._training_set_molecules = None
        self._training_set_logps = None
        self._test_set_molecules = None
        self._test_set_logps = None
        self._encodings = None

        self._model = None
        self._model_params = model_params

    def save_model(self, path):
        torch.save(self._model.state_dict(), path + '/model.pt')
        with open(path + '/encodings.json', 'w') as f:
            json.dump(self._encodings, f)
        
    def load_model(self, path):
        with open(path + '/encodings.json') as f:
            self._encodings = json.load(f)
        
        self._model = GCN(len(self._encodings) + 1, **self._model_params).to('cpu')
        self._model.load_state_dict(torch.load(path + '/model.pt'))
        self._model.eval()

    def load_training_dataset(self):
        dataset = pd.read_csv('./datasets/opera_logp_multitask.csv')
        smiles_list, props_list = self.process_dataset(dataset)
        self._training_set_molecules, self._test_set_molecules, self._training_set_props, self._test_set_props = train_test_split(smiles_list, props_list, test_size=0.2, random_state=42)

    @staticmethod
    def process_dataset(dataset):
        smiles_list = []
        props_list = []
        for index, row in dataset.iterrows():
            smiles = row['smiles']
            props = [row[column] for column in dataset.columns if column != 'smiles']

            if '.' in smiles:
                continue

            try:
                smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
                props_list.append(props)
            except:
                continue
        
        return smiles_list, props_list

    def fit_model(self, epochs=2000, lr=0.0001, lr_decay=0.999, batch_size=32, mode='both'):
        mols = [Chem.MolFromSmiles(smiles) for smiles in self._training_set_molecules]
        if self._encodings is None:
            self._encodings = self.preprocess_smiles(mols)

        data_list = [self.mol_to_data(mol, self._training_set_props[i], self._encodings) for i, mol in enumerate(mols)]
        loader = DataLoader(data_list, batch_size=batch_size)

        if self._model is None:
            self._model = GCN(len(self._encodings) + 1, **self._model_params).to('cpu')

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._model.train()
        for epoch in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                out = self._model.forward(batch, mode=mode)
                loss = F.mse_loss(out, batch.y)
                loss.backward()
                optimizer.step()
            
            if lr_decay is not None:
                optimizer.param_groups[0]['lr'] *= lr_decay

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, RMSE train {self.compute_rmse(test=False, mode=mode)}, RMSE test {self.compute_rmse(mode=mode)}')

    def predict(self, smiles_list, mode='both'):
        with torch.no_grad():
            self._model.eval()
            data_list = [self.mol_to_data(Chem.MolFromSmiles(smiles), 0, self._encodings) for smiles in smiles_list]
            val_loader = DataLoader(data_list, batch_size=16)
            predictions = torch.cat([self._model.forward(batch, mode=mode) for batch in val_loader], 0)

            return predictions
    
    def compute_rmse(self, test=True, mode='both'):
        mols = self._test_set_molecules if test else self._training_set_molecules
        props = self._test_set_props if test else self._training_set_props

        return np.sqrt(F.mse_loss(self.predict(mols, mode=mode), torch.FloatTensor(props)))

    def encode_atom(self, mol, atom):
        atom_index = atom.GetIdx()
        atom_number = atom.GetAtomicNum()
        neighbors = []
        for neighbor in atom.GetNeighbors():
            neighbor_index = neighbor.GetIdx() 
            neighbor_number = neighbor.GetAtomicNum()
            bond_type = mol.GetBondBetweenAtoms(atom_index, neighbor_index).GetBondType()

            neighbors.append(f'{neighbor_number}-{bond_type}')

        encoded_neighbors = ';'.join(sorted(neighbors))
        encoded_atom = f'{atom_number}:{encoded_neighbors}'

        return encoded_atom

    def preprocess_smiles(self, mol_list): 
        atom_encodings = []
        for mol in mol_list:
            for atom in mol.GetAtoms():
                encoded_atom = self.encode_atom(mol, atom)
                if encoded_atom not in atom_encodings:
                    atom_encodings.append(encoded_atom)

        return atom_encodings

    def mol_to_data(self, mol, y, encodings):
        edge_index = [[], []]
        x = []
        for i, atom in enumerate(mol.GetAtoms()):
            encoding = self.encode_atom(mol, atom)
            x.append(encodings.index(encoding) if encoding in encodings else len(encodings))
    
            atom_idx = atom.GetIdx()
            
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()

                edge_index[0].append(atom_idx)
                edge_index[1].append(neighbor_idx)

                edge_index[1].append(atom_idx)
                edge_index[0].append(neighbor_idx)

        return Data(
            x=torch.tensor(x, dtype=torch.int),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor([y], dtype=torch.float)
        )

