"""
Dataset to load NMM data based on the metadata saved during dataset_generation

"""
import sys
import os 
from pathlib import Path 
import matplotlib.pyplot as plt 
import numpy as np 
import json 
import torch 
from scipy.io import loadmat, matlab
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D
import mne
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path

from scipy.signal import find_peaks, convolve
from scipy.ndimage import gaussian_filter1d
import argparse
import time
from torch.utils.data import Dataset
import pytorch_lightning as pl

pl.seed_everything(333)

def load_nmm_data_from_meta( dataset_meta, idx, nmm_wvf_folder, n_times, n_sources =994, verbose=False ): 
    # Extract information from the dataset metadata:
    n_patch = dataset_meta['n_patches'][idx][0]
    ## New waveform: 
    # wvf_new = 0.1*np.random.randn(  n_sources, n_times )
    wvf_new = np.zeros( (n_sources, n_times) )
        
    for p in range(n_patch): 
        region_active = dataset_meta['seed'][idx][p]
        region = dataset_meta['region'][idx][p]
        offset_time = dataset_meta['time_offset'][idx][p]
        ampl_decrease = np.array( dataset_meta['mag_change'][idx][p] )
        ampl_scale_factor = dataset_meta['ampl_scale_factor'][idx][p]
        clip_name = dataset_meta['clip_name'][idx][p]

        ## Waveform
        # nmm_wvf_folder = Path(f'{nmm_wvf_folder}/a{region_active}')
        if verbose: 
            print(f"{clip_name=}")
        ## load the waveform clip: 
        wvf_clip = np.load( Path(nmm_wvf_folder,f"a{region_active}", clip_name ) )
        wvf_clip = wvf_clip - wvf_clip.mean() # center the waveform
        if verbose: 
            print(f"{wvf_clip.shape=}")
        wvf_length = wvf_clip.shape[1]

        wvf_new[region_active,:] = 0.1*np.random.randn( 1, n_times)
        wvf_new[region, :] = 0.1*np.random.randn( *wvf_new[region, :].shape )
        # print( f"{offset_time}" )
        # print(f"{wvf_length}")
        # print(f"{wvf_clip[region_active,:].shape}")
        # wvf_new[region_active,offset_time-time_peak:offset_time-time_peak+wvf_length] = ampl_scale_factor * wvf_clip[region_active,:]
        if offset_time+wvf_length > n_times: 
            wvf_new[region, offset_time:offset_time+wvf_length] = ampl_decrease[:, None] * ampl_scale_factor * wvf_clip[region_active,:n_times-(offset_time+wvf_length)]
        else:  
            wvf_new[region, offset_time:offset_time+wvf_length] = ampl_decrease[:, None] * ampl_scale_factor * wvf_clip[region_active,:]

    return wvf_new

def add_noise_snr(snr_db, signal, noise_type = {"white":1.}, return_noise=False): 
    """  
    Return a signal which is a linear combination of signal and noise with a ponderation to have a given snr
    noise_type : dict key = type of noise, value = ponderation of the noise (for example use white and pink noise)
    """

    snr     = 10**(snr_db/10)
    noise = 0
    dims = [signal.shape[i] for i in range(len(signal.shape))]
    for n_type, pond in noise_type.items() : 
        if n_type=="white": 
            noise = noise + pond * np.random.randn( *dims )
        if n_type=="pink" : 
            beta = (1.8-0.2)*np.random.rand(1) + 0.2
            noise = noise + pond * cn.powerlaw_psd_gaussian(beta, signal.shape)

    if len(signal.shape)==2: 
        # 2D signal
        x = signal + (noise/np.linalg.norm(noise))*(np.linalg.norm(signal)/np.sqrt(snr))
    elif len(signal.shape)==3:
        # batch data 
        noise_norm = np.expand_dims( np.linalg.norm(noise, axis=(1,2)), (1,2) )
        sig_norm = np.expand_dims( np.linalg.norm(signal, axis=(1,2)), (1,2) )
        x = signal + (noise/noise_norm)*(sig_norm/np.sqrt(snr))
        #x = signal + (noise/np.linalg.norm(noise, axis=(1,2)))*(np.linalg.norm(signal,axis=(1,2))/np.sqrt(snr))
    else: 
        x = None
        sys.exit('Signal must be of dimension 2 (unbatched data) or 3 (batched data)')
        
    if return_noise : 
        return x, noise
    else: 
        return x

def array_range_scaling(x, inf, sup): 
    """
    rescale x so that it max(x) == sup and min(x) == inf.
    !! works for x a tensor of a single tensor (not a batch)
    returns rescaled_x, the rescaled tensor
    """
    x_maxs = np.max(x)
    x_mins = np.min(x)

    scale_factor = (sup - inf) / (x_maxs - x_mins) 
    rescaled_x = (x - x_mins) * scale_factor + x_mins
    return rescaled_x
import json


class NMMDataset( Dataset ): 
    """
    Neural Mass Model dataset based on the python, handmade simulation (inspired from the code from Sun et al., but adapted)

    """
    def __init__( self, datafolder,
        dataset_meta_file, 
        clip_folder, 
        n_times, 
        subject_name='fsaverage',
        source_sampling='fsav_994',
        electrode_montage="standard_1020",
        to_load=10,
        snr_db=10,
        scaler_type="linear",
        orientation="constrained",
        load_lf = True,
        subset_file = None ):
        super().__init__()

        self.to_load = 10 # number of samples to load
        self.snr = snr_db
        self.clip_folder = clip_folder

        self.datafolder = datafolder
        try:
            home
        except:
            if self.datafolder.startswith("/"):
                home = ""
            else:
                home = os.path.expanduser('~')
        
        self.to_load = to_load
        self.load_lf = load_lf
        self.snr_db = snr_db
        if self.load_lf: 
            model_path = Path( home, datafolder, subject_name, orientation, electrode_montage, source_sampling, "model" )
            # ## Load forward matrix 
            fwd = mne.convert_forward_solution(
                mne.read_forward_solution( Path(model_path, "fwd_regionsfsav_994-fwd.fif" ) ), force_fixed=True )
            self.leadfield = fwd['sol']['data']
        ## build path to simulation ##
        self.simu_path = Path(  
            self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "simu"
        )
        self.config = {
            "metadata_file": dataset_meta_file,
            "subject_name": subject_name,
            "orientation": orientation, 
            "electrode_montage": electrode_montage, 
            "source_sampling": source_sampling, 
            "n_times": n_times
            # "n_electrodes": , 
            # "n_sources": config_dict['source_space']['n_sources'] , 
            # "fs": config_dict['rec_info']['fs'],
            # "n_times": config_dict['rec_info']['n_times'], 
        }
        ## load dataset_meta
        with open(dataset_meta_file, 'r') as file:
            self.dataset_meta = json.load(file)

        # samples to keep #
        n_samples = len(self.dataset_meta['seed'])
        if self.to_load >= n_samples : 
            self.to_load = n_samples
            print(f"--- only {n_samples} available ---")
            self.data_ids = np.arange(n_samples)
        if self.to_load < n_samples : 
            self.data_ids = np.random.choice(np.arange(n_samples), self.to_load, replace=False)


    def __len__(self):
        return self.to_load

    def __getitem__(self, index):
        index = self.data_ids[index]
        wvf_new = load_nmm_data_from_meta( self.dataset_meta, idx=index, nmm_wvf_folder=self.clip_folder, n_times=self.config['n_times'], verbose=False )
        eeg = self.leadfield @ wvf_new

        ## add noise ##
        mins_clean = eeg.min()
        maxs_clean = eeg.max()
        
        eeg = add_noise_snr(
                snr_db=self.snr_db, signal=eeg
            )
        ### rescale
        eeg = array_range_scaling(
            eeg, inf=mins_clean, sup=maxs_clean
        )

        return torch.from_numpy( eeg ), torch.from_numpy( wvf_new )


class DatamoduleNMM(pl.LightningDataModule):
    """ 
    Lightning datamodule associated with the NMMDataset above.
    
    """
    def __init__(self, dataset_kw, dl_kw, per_valid=0.2, config_file=None, subset_name=None, time_window=False, noise_fixed=False, di=False, src_noise=False):
        super().__init__()
        self.dl_kw = dl_kw
        self.per_valid = per_valid
        self.train_ds = None
        self.val_ds = None
        self.noise_fixed = noise_fixed
        self.di = di # direct inverstion
        self.src_noise = src_noise

        self.dataset_kw =  dataset_kw
        self.dataset_kw.update({'dataset_meta_file': dataset_meta_file})
        self.dataset_kw.update({'clip_folder': nmm_clips_folder})
        
        # if subset_name.lower() != 'none': 
        #     subset_file = Path(
        #         dataset_kw['datafolder'], 
        #         dataset_kw['subject_name'], dataset_kw['orientation'], dataset_kw['electrode_montage'], dataset_kw['source_sampling'], "simu",
        #         dataset_kw['simu_name'], f"{subset_name}.txt"
        #     )
        #     self.dataset_kw.update({'subset_file': subset_file})
        # else : 
        #     self.dataset_kw.update({'subset_file': None})

    def setup(self, stage):
        if stage == "test": 
            self.test_ds = NMMDataset(
                **self.dataset_kw
            )
        else : 
            ds_dataset = NMMDataset(
                **self.dataset_kw
            )
            self.dataset_kw['to_load'] = len(ds_dataset) #ds_dataset.to_load
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                ds_dataset,
                [int(self.dataset_kw['to_load'] * (1 - self.per_valid)), 
                 int(self.dataset_kw['to_load']) - int(self.dataset_kw['to_load']*(1 - self.per_valid))],
            )
        

    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

####################################### MAIN #####################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-clip_folder', type=str, default="./nmm_clips", help="Folder in which to save the extracted clips")
    parser.add_argument('-n_source_max', default=100, type=int, help="Number of regions to consider")
    ## Head model parameters : 
    parser.add_argument('-sn', '--subject_name', type=str, default="fsav_994", help="Name of the subject")
    parser.add_argument('-o', '--orientation', type=str, default='constrained', help="Orientation of the sources (constrained/unconstrained)")
    parser.add_argument('-em', '--electrode_montage', type=str, default="standard_1020", help="Name of the electrode montage")
    parser.add_argument('-ss', '--source_sampling', type=str, default='fsav_994', help="Name of the source space/sampling used")
    args = parser.parse_args()

    home = os.path.expanduser('~')
    nmm_clips_folder = Path("./nmm_clips")
    dataset_meta_file = Path("./nmm_dataset_metas/test_meta.json")


    datafolder = Path("Documents", "Data", "simulation")
    subject_name = args.subject_name
    orientation = args.orientation
    electrode_montage = args.electrode_montage
    source_sampling = args.source_sampling

    ## load dataset_meta
    with open(dataset_meta_file, 'r') as file:
        dataset_meta = json.load(file)
    
    dataset_kw = {
        "datafolder": datafolder, 
        "dataset_meta_file": dataset_meta_file, 
        "clip_folder": nmm_clips_folder, 
        "n_times": 500, 
        "subject_name": subject_name,
        "source_sampling": source_sampling,
        "electrode_montage": electrode_montage,
        "to_load": 10,
        "snr_db": 10,
        "scaler_type": "linear",
        "orientation": "constrained",
        "load_lf": True,
        "subset_file": None 
    }
    ds = NMMDataset( 
        datafolder=datafolder, 
        dataset_meta_file=dataset_meta_file, 
        clip_folder=nmm_clips_folder, 
        n_times=500, 
        subject_name=subject_name,
        source_sampling=source_sampling,
        electrode_montage=electrode_montage,
        to_load=10,
        snr_db=10,
        scaler_type="linear",
        orientation="constrained",
        load_lf = True,
        subset_file = None 
      )
    
    dm = DatamoduleNMM( dataset_kw=dataset_kw, dl_kw={"batch_size":8} )
    eeg, src = ds[0]
    n_electrodes = eeg.shape[0]
    n_sources = src.shape[0]
    plt.figure()
    plt.subplot(121)
    for e in range(n_electrodes): 
        plt.plot( eeg[e,:] )
    plt.subplot(122)
    for s in range(n_sources): 
        plt.plot(src[s,:])
    plt.xlabel('time samples')
    plt.ylabel("Amplitude")
    plt.show(block=False)



    
