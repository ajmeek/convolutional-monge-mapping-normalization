#imports
import pyrootutils

pyrootutils.set_root(path='/work/cniel/ajmeek/bowaves_cmmn/convolutional-monge-mapping-normalization',
                     pythonpath=True)

import numpy as np
from cmmn import CMMN

# their tutorial imports above, my imports below

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
cue = []

for subj in cue_subj_list:
    subj_data = np.load(f'../data/cue_resampled_to_emotion/cue_subj_{subj}_chunked.npz') # check file path.
    icaact = subj_data['icaact']
    chunks, channels, time = icaact.shape
    #icaact = icaact[:, :7680*10] # 30 seconds times ten = 5 minutes, actually let's map all of them
    icaact = icaact.reshape(-1, 63, 7680)
    assert icaact.shape == (time // 7680, 63, 7680)

    cue.append(icaact)

# check conceptual understanding here first before moving on.
# CMMN learns a filter for each of the emotion subjects.
# How does it know which one to use for cue? diff numb of subjects there.

# Dr. B and I discussed this yesterday, but review it again to fully internalize.

# now fit the CMMN model to the emotion data
cmmn = CMMN()
cmmn.fit(emotion_five_min)
cue_chunked_transformed = cmmn.transform(cue)

# now that cue has been transformed, put it each subj back into original shape and save
for i, subj in enumerate(cue_subj_list):
    chunks, channels, time = cue_chunked_transformed[i].shape
    icaact_reformed = cue_chunked_transformed[i].reshape(63, -1)
    assert icaact_reformed.shape == (63, chunks * time)

    np.savez(f'../data/cue_mapped/cue_subj_{subj}_cmmn.npz', icaact=icaact_reformed)
