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

from cmmn.data import load_sleep_physionet, extract_epochs
from cmmn import CMMN

# their tutorial imports above, my imports below

from scipy.io import loadmat

# emotion only missing subj 22
emotion_subj_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']

cue_subj_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


# load in the first five minutes of each IC from each emotion subject

emotion_five_min = []

for subj in emotion_subj_list:
    subj_data = np.load(f'../data/emotion/emotion_subj_{subj}_chunked.npz')
    icaact = subj_data['icaact']
    #icaact = icaact[:, :7680*10] # 30 seconds times ten = 5 minutes

    # actually icaact is currently of shape (K, C, T) where K is # of chunks, C is # of ICs, and T is time which is 7680
    # reconstruct to (C, T). Then take first 5 minutes and chunk into separate source domains
    chunks, ics, time = icaact.shape
    icaact = icaact.reshape(ics, -1)
    assert icaact.shape == (ics, chunks * time)

    # now take first 5 minutes
    icaact = icaact[:, :7680*10]

    # now chunk into separate source domains
    icaact = icaact.reshape(-1, ics, 7680)
    assert icaact.shape == (10, ics, 7680)

    emotion_five_min.append(icaact)

# now load all of cue data

# check conceptual understanding here first before moving on.
# CMMN learns a filter for each of the emotion subjects.
# How does it know which one to use for cue? diff numb of subjects there.

# Dr. B and I discussed this yesterday, but review it again to fully internalize.