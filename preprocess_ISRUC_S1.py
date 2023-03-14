import numpy as np
import scipy.io as scio
import os
from scipy import signal

path_Extracted = 'data/ISRUC_S1/ExtractedChannels/'
path_RawData = 'data/ISRUC_S1/RawData/'
path_output = 'data/ISRUC_S1/'

def read_psg(path_Extracted, sub_id, fs=100):
    psg = scio.loadmat(os.path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = psg['F3_A2'] # 'C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1', 'LOC_A2', 'ROC_A1','X1', 'X2'
    psg_resample = signal.resample(psg_use, fs*30, axis=-1)
    return psg_resample


def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(os.path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])

for sub in range(1, 101):
    print('Read subject', sub)
    psg = read_psg(path_Extracted, sub)
    label = read_label(path_RawData, sub)
    
    print('Subject', sub, ':', label.shape, psg.shape)
    assert len(label) == len(psg)

    # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
    label[label==5] = 4  # make 4 correspond to REM
    
    filename = os.path.join(path_output, 'ISRUC_S1_%d.npz' % (sub))
    save_dict = {'x': psg, 'y': label}
        
    np.savez(filename, **save_dict)
    