#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch
from torch import nn

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import check_random_state

from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset

from braindecode.models import SleepStagerChambon2018


import mne
from mne.datasets.sleep_physionet.age import fetch_data

from cmmn.data import load_sleep_physionet, extract_epochs
from cmmn import CMMN

# their tutorial imports above, my imports below

from scipy.io import loadmat

# emotion only missing subj 22
emotion_subj_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']

cue_subj_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

