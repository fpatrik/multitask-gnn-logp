import torch

import numpy as np
import pandas as pd

from multitask_model import MultiTaskGraphNeuralNetModel

class MultiTaskGraphNeuralNetEnsemble:
    def __init__(self, n_models=5):
        self._n_models = n_models
        self._models = []

    def load_models(self):
        self._models = []
        for i in range(self._n_models):
            model = MultiTaskGraphNeuralNetModel()
            model.load_model(f'./models/model_{i}/')
            self._models.append(model)

    def train_models(self):
        for i in range(self._n_models):
            model = MultiTaskGraphNeuralNetModel()
            model.load_training_dataset()
            model.fit_model()
            model.save_model(f'./models/model_{i}/')

    
    def predict(self, smiles_list):
        return np.mean([torch.flatten(model.predict(smiles_list, mode='logp')).cpu().detach().numpy() for model in self._models], axis=0)

if __name__ == '__main__':
    ensemble = MultiTaskGraphNeuralNetEnsemble()
    ensemble.load_models()

    martel_dataset = pd.read_csv('./datasets/martel_logp.csv')
    martel_smiles = martel_dataset['smiles'].to_list()
    
    predicted_logps = ensemble.predict(martel_smiles)
    actual_logps = martel_dataset['log_p'].to_list()

    print('RMSE:', np.sqrt(np.mean(np.square([prediction - actual_logps[i] for i, prediction in enumerate(predicted_logps)]))))