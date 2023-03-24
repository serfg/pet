from utilities import get_all_species, get_compositional_features
import os

import torch
import ase.io
import numpy as np
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from utilities import ModelKeeper
import time
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import LambdaLR
import sys
import copy
import inspect
import yaml
from torch_geometric.nn import DataParallel

from molecule import Molecule, batch_to_dict
from hypers import Hypers
from pet import PET
from utilities import FullLogger
from utilities import get_rmse, get_mae, get_relative_rmse, get_loss
from analysis import get_structural_batch_size, convert_atomic_throughput


STRUCTURES_PATH = sys.argv[1]
PATH_TO_CALC_FOLDER = sys.argv[2]
CHECKPOINT = sys.argv[3]
N_AUG = int(sys.argv[4])
DEFAULT_HYPERS_PATH = sys.argv[5]
BATCH_SIZE = sys.argv[6]

HYPERS_PATH = PATH_TO_CALC_FOLDER + '/hypers_used.yaml'
PATH_TO_MODEL_STATE_DICT = PATH_TO_CALC_FOLDER + '/' + CHECKPOINT + '_state_dict'
ALL_SPECIES_PATH = PATH_TO_CALC_FOLDER + '/all_species.npy'
SELF_CONTRIBUTIONS_PATH = PATH_TO_CALC_FOLDER + '/self_contributions.npy'


'''STRUCTURES_PATH = 'small_data/test_small.xyz'
HYPERS_PATH = 'results/test_calc_continuation_0/hypers_used.yaml'
ALL_SPECIES_PATH = 'results/test_calc_continuation_0/all_species.npy'
SELF_CONTRIBUTIONS_PATH = 'results/test_calc_continuation_0/self_contributions.npy'
PATH_TO_MODEL_STATE_DICT = 'results/test_calc_continuation_0/best_val_mae_energies_model_state_dict' '''

hypers = Hypers()
# loading default values for the new hypers potentially added into the codebase after the calculation is done
# assuming that the default values do not change the logic
hypers.set_from_files(HYPERS_PATH, DEFAULT_HYPERS_PATH, check_dublicated = False)

if BATCH_SIZE == 'None':
    BATCH_SIZE = hypers.STRUCTURAL_BATCH_SIZE
else:
    BATCH_SIZE = int(BATCH_SIZE)
    
structures = ase.io.read(STRUCTURES_PATH, index = ':')

all_species = np.load(ALL_SPECIES_PATH)
if hypers.USE_ENERGIES:
    self_contributions = np.load(SELF_CONTRIBUTIONS_PATH)

molecules = [Molecule(structure, hypers.R_CUT, hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES, hypers.USE_FORCES) for structure in tqdm(structures)]
max_nums = [molecule.get_max_num() for molecule in molecules]
max_num = np.max(max_nums)
graphs = [molecule.get_graph(max_num, all_species) for molecule in tqdm(molecules)]

if hypers.MULTI_GPU:
    loader = DataListLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
else:        
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)

add_tokens = []
for _ in range(hypers.N_GNN_LAYERS - 1):
    add_tokens.append(hypers.ADD_TOKEN_FIRST)
add_tokens.append(hypers.ADD_TOKEN_SECOND)

model = PET(hypers, hypers.TRANSFORMER_D_MODEL, hypers.TRANSFORMER_N_HEAD,
                       hypers.TRANSFORMER_DIM_FEEDFORWARD, hypers.N_TRANS_LAYERS, 
                       0.0, len(all_species), 
                       hypers.N_GNN_LAYERS, hypers.HEAD_N_NEURONS, hypers.TRANSFORMERS_CENTRAL_SPECIFIC, hypers.HEADS_CENTRAL_SPECIFIC, 
                       add_tokens).cuda()
if hypers.MULTI_GPU:
    model = DataParallel(model)
    device = torch.device('cuda:0')
    model = model.to(device)
    
model.load_state_dict(torch.load(PATH_TO_MODEL_STATE_DICT))
model.eval()

if hypers.USE_ENERGIES:
    energies_ground_truth = np.array([struc.info['energy'] for struc in structures])
    
if hypers.USE_FORCES:
    forces_ground_truth = [struc.arrays['forces'] for struc in structures]
    forces_ground_truth = np.concatenate(forces_ground_truth, axis = 0)
    
    

if hypers.USE_ENERGIES:
    all_energies_predicted = []
    
if hypers.USE_FORCES:
    all_forces_predicted = []
    
for _ in tqdm(range(N_AUG)):
    if hypers.USE_ENERGIES:
        energies_predicted = []
    if hypers.USE_FORCES:
        forces_predicted = []
    
    for batch in loader:
        if not hypers.MULTI_GPU:
            batch.cuda()
            model.augmentation = True
        else:
            model.module.augmentation = True
            
        predictions_energies, targets_energies, predictions_forces, targets_forces = model(batch)
        if hypers.USE_ENERGIES:
            energies_predicted.append(predictions_energies.data.cpu().numpy())
        if hypers.USE_FORCES:
            forces_predicted.append(predictions_forces.data.cpu().numpy())
            
    if hypers.USE_ENERGIES:
        energies_predicted = np.concatenate(energies_predicted, axis = 0)
        all_energies_predicted.append(energies_predicted)
        
    if hypers.USE_FORCES:
        forces_predicted = np.concatenate(forces_predicted, axis = 0)
        all_forces_predicted.append(forces_predicted)
        
 
if hypers.USE_ENERGIES:
    all_energies_predicted = [el[np.newaxis] for el in all_energies_predicted]
    all_energies_predicted = np.concatenate(all_energies_predicted, axis = 0)
    energies_predicted_mean = np.mean(all_energies_predicted, axis = 0)
    
if hypers.USE_FORCES:
    all_forces_predicted = [el[np.newaxis] for el in all_forces_predicted]
    all_forces_predicted = np.concatenate(all_forces_predicted, axis = 0)
    forces_predicted_mean = np.mean(all_forces_predicted, axis = 0)

if hypers.USE_ENERGIES:
    
    compositional_features = get_compositional_features(structures, all_species)
    self_contributions_energies = []
    for i in range(len(structures)):
        self_contributions_energies.append(np.dot(compositional_features[i], self_contributions))
    self_contributions_energies = np.array(self_contributions_energies)
    
    energies_predicted_mean = energies_predicted_mean + self_contributions_energies
    
    print(f"energies mae: {get_mae(energies_ground_truth, energies_predicted_mean)}")
    print(f"energies rmse: {get_rmse(energies_ground_truth, energies_predicted_mean)}")
    
if hypers.USE_FORCES:
    print(f"forces mae per component: {get_mae(forces_ground_truth, forces_predicted_mean)}")
    print(f"forces rmse per component: {get_rmse(forces_ground_truth, forces_predicted_mean)}")
    


