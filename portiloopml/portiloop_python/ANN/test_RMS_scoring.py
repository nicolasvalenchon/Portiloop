# Test the RMS scoring function on compute canada

import numpy as np

from portiloopml.portiloop_python.ANN.data.mass_data_new import MassDataset

# Parse arguments from command line
import argparse

from portiloopml.portiloop_python.ANN.wamsley_utils import RMS_score_all
parser = argparse.ArgumentParser(description='Test the RMS scoring function on compute canada')
parser.add_argument('--data_path', type=str, help='Path to the data folder', default='/project/MASS/mass_spindles_dataset')

args = parser.parse_args()
data_path = args.data_path

subject = 'PN_05_EC_Night4'

# Load the signal using dataset
dataset = MassDataset(data_path, 30, 30, 30, subjects=[subject], use_filtered=False, sampleable='spindles', compute_spindle_labels=False)

signal = dataset.data[subject]['signal']

# Get a random number of indexes from the signal
indexes = np.random.randint(0, len(signal), 1000) 

# Get the RMS of the signal
all_scores = np.array(RMS_score_all(signal, indexes))

# Get the mean and std of the scores
mean = np.mean(all_scores)
std = np.std(all_scores)

print(f'Mean: {mean}, Std: {std}')