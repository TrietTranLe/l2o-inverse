### EEG
import functools as ft
import itertools
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch.utils.data
import tqdm
import xarray as xr
from torch.utils.data import Dataset
from data.eeg.utils_eeg import find_exp

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class EsiDataset(Dataset):
    """
    Dataset for ESI data of SEREEGA based simulations
    **inputs**
        - root_simu:str , root path of simulation data (ex: home/user/Data/simulation)
        - config_file: str , name of the configuration file of the simulation
        - simu_name: str , name of the simulation
        - subject_name: str, name of the subject
        - source_sampling: str, subsampling of the source space used
        - electrode_montage: str, name of the electrode montage used
        - to_load: int, number of samples to load from the dataset
        - snr_db: int, snr of the EEG data
        - noise_type={"white":1.}: dict, type of noise to add (dict with the color of the noise and a ratio of the total noise 
                                that corresponds to this component)
        - scaler_type="linear":str, type of scaling to use for normalisation (linear or max-max)
    """

    def __init__(
        self,
        datafolder,
        config_file,
        simu_name,
        subject_name,
        source_sampling,
        electrode_montage,
        to_load,
        snr_db,
        noise_type={"white":1.},
        scaler_type="linear",
        orientation="constrained",
        replace_root=False, 
        load_lf = True,
        subset_file = None
    ):
        super().__init__()
        #home = expanduser('~')
        self.datafolder = datafolder
        
        self.to_load = to_load
        self.load_lf = load_lf
        
        ## build path to simulation ##
        self.simu_path = Path(  
            self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "simu"
        )

        ### load confi ###
        # load config 
        with open(config_file) as f : 
            config_dict = json.load(f)

        self.config = {
            "subject_name": subject_name,
            "simu_name": simu_name, 
            "orientation": orientation, 
            "electrode_montage": electrode_montage, 
            "source_sampling": source_sampling, 
            "n_electrodes": config_dict['electrode_space']['n_electrodes'], 
            "n_sources": config_dict['source_space']['n_sources'] , 
            "fs": config_dict['rec_info']['fs'],
            "n_times": config_dict['rec_info']['n_times'], 
        }
        
        self.replace_root = replace_root
        self.snr = snr_db
        self.scaler_type = scaler_type

        info_file = Path( self.simu_path, simu_name ,f"{simu_name}{source_sampling}_match_json_file.json" )
        # load info to match the files
        with open(info_file) as f : 
            self.match_info_dict = json.load(f)
        if subset_file is None : 
            self.data_ids = np.array( 
                list( self.match_info_dict.keys() )
            )
        else : 
            with open(subset_file, 'r') as file:
                self.data_ids = np.array(
                    [line.rstrip() for line in file]
                )
            
        # samples to keep #
        n_samples = len(self.data_ids)
        if self.to_load > n_samples : 
            self.to_load = n_samples
            print(f"--- only {n_samples} available ---")

        if self.to_load != n_samples : 
            self.data_ids = self.data_ids[
                np.random.choice(np.arange(n_samples), self.to_load, replace=False)
            ]
        
        # metadata #
        self.md = [{}] * self.to_load #dict( zip(self.data_ids, {} ) )
        self.act_src = [[]] * self.to_load #dict( zip(self.data_ids, {} ) )
        
        for k in range(self.to_load) : 
            if self.replace_root : 
                md_json_file_name = self.replace_root_fn( 
                    self.match_info_dict[self.data_ids[k]]['md_json_file_name']
                )
                with open( Path(self.datafolder, md_json_file_name) ) as f : 
                    self.md[k] = json.load(f)
            else : 
                with open( Path(self.datafolder, self.match_info_dict[self.data_ids[k]]['md_json_file_name']) ) as f : 
                    self.md[k] = json.load(f)

            act_src = []
            for p in range(self.md[k]['n_patch']) : 
                act_src += self.md[k]['act_src'][f'patch_{p+1}']
            self.act_src[k] = act_src
        #self.act_src = torch.from_numpy( np.array(self.act_src) )

     
        self.max_eeg = torch.zeros([self.to_load, 1])
        self.max_src = torch.zeros([self.to_load, 1])
        self.alpha_lf = None

        if self.load_lf: 
            from data.eeg.utils_eeg import load_mat
            model_path = Path(  
               self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "model"
        ) 
            self.leadfield = load_mat(Path(model_path, f"LF_{source_sampling}.mat"))['G']
        if self.scaler_type == "linear_bis":
            self.alpha_lf = 10**(find_exp(self.leadfield.max()) + 1)

    def replace_root_fn(self, string): 
        """ 
        replace "root" of file name (handle differences in simulations)
        """
        simu_name = self.config['simu_name']
        split = string.split(simu_name)
        mod_string = Path( 
            self.simu_path, simu_name, *split[1].split('/') 
            )
        return mod_string
    
    def __len__(self):
        return self.to_load

    def __getitem__(self, index):
        from data.eeg.utils_eeg import (add_noise_snr, array_range_scaling,
                                           load_mat)
        root_simu = self.datafolder
        md = self.md[index]
        if self.replace_root : 
            eeg_file_name = self.replace_root_fn(self.match_info_dict[self.data_ids[index]]['eeg_file_name'] )
            act_src_file_name = self.replace_root_fn( self.match_info_dict[self.data_ids[index]]['act_src_file_name'] )
            eeg = load_mat( Path( root_simu, eeg_file_name ) )['eeg_data']['EEG'] 
            src = load_mat(Path( root_simu, act_src_file_name) )['Jact']['Jact']
        else : 
            eeg = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['eeg_file_name'] ) )['eeg_data']['EEG'] 
            src = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['act_src_file_name']) )['Jact']['Jact']

        #reconstruct source data
        src_tot = np.zeros([self.config['n_sources'], self.config['n_times']])
        src_tot[self.act_src[index], :] = src.copy()

        
        # add noise to EEG data
        inf = np.min(eeg)
        sup = np.max(eeg)
        if self.snr < 50:
            eeg = array_range_scaling(
                add_noise_snr( self.snr,eeg ), inf, sup
            )       
        
        # scale data
        eeg, src_tot, _, alpha_eeg, alpha_src, _ = scaled_data( 
            eeg, src_tot, scaling_type=self.scaler_type, leadfield=self.leadfield, alpha_L=self.alpha_lf)
        self.max_eeg[index] = alpha_eeg
        self.max_src[index] = alpha_src
        # self.alpha_lf = alpha_lf 

        return TrainingItem(input=torch.from_numpy(eeg).float(), tgt=torch.from_numpy(src_tot).float()) 

class EsiDatasetNoise(Dataset):
    """
    Dataset for ESI data of SEREEGA based simulations
    **inputs**
        - root_simu:str , root path of simulation data (ex: home/user/Data/simulation)
        - config_file: str , name of the configuration file of the simulation
        - simu_name: str , name of the simulation
        - subject_name: str, name of the subject
        - source_sampling: str, subsampling of the source space used
        - electrode_montage: str, name of the electrode montage used
        - to_load: int, number of samples to load from the dataset
        - snr_db: int, snr of the EEG data
        - noise_type={"white":1.}: dict, type of noise to add (dict with the color of the noise and a ratio of the total noise 
                                that corresponds to this component)
        - scaler_type="linear":str, type of scaling to use for normalisation (linear or max-max)
    """

    def __init__(
        self,
        datafolder,
        config_file,
        simu_name,
        subject_name,
        source_sampling,
        electrode_montage,
        to_load,
        snr_db,
        noise_type={"white":1.},
        scaler_type="linear",
        orientation="constrained",
        replace_root=False, 
        load_lf = True,
        subset_file = None
    ):
        super().__init__()
        #home = expanduser('~')
        self.datafolder = datafolder
        
        self.to_load = to_load
        self.load_lf = load_lf
        
        ## build path to simulation ##
        self.simu_path = Path(  
            self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "simu"
        )

        ### load confi ###
        # load config 
        with open(config_file) as f : 
            config_dict = json.load(f)

        self.config = {
            "subject_name": subject_name,
            "simu_name": simu_name, 
            "orientation": orientation, 
            "electrode_montage": electrode_montage, 
            "source_sampling": source_sampling, 
            "n_electrodes": config_dict['electrode_space']['n_electrodes'], 
            "n_sources": config_dict['source_space']['n_sources'] , 
            "fs": config_dict['rec_info']['fs'],
            "n_times": config_dict['rec_info']['n_times'], 
        }
        
        self.replace_root = replace_root
        self.snr = snr_db
        self.scaler_type = scaler_type

        info_file = Path( self.simu_path, simu_name ,f"{simu_name}{source_sampling}_match_json_file.json" )
        # load info to match the files
        with open(info_file) as f : 
            self.match_info_dict = json.load(f)
        if subset_file is None : 
            self.data_ids = np.array( 
                list( self.match_info_dict.keys() )
            )
        else : 
            with open(subset_file, 'r') as file:
                self.data_ids = np.array(
                    [line.rstrip() for line in file]
                )
        ## change match_info_dict pathes...
        for i in range(len(self.data_ids)) :
            self.match_info_dict[self.data_ids[i]]['eeg_file_name'] = Path(
                datafolder, subject_name, orientation, electrode_montage, source_sampling, "simu", simu_name, "eeg", f"{self.snr}db", f"{self.data_ids[i].split('_')[-1]}_eeg.mat"
                )
            self.match_info_dict[self.data_ids[i]]['act_src_file_name'] = Path(
                datafolder, subject_name, orientation, electrode_montage, source_sampling, "simu", simu_name, "sources", "Jact", f"{self.data_ids[i].split('_')[-1]}_src_act.mat"
                )
            self.match_info_dict[self.data_ids[i]]['md_json_file_name'] = Path(
                datafolder, subject_name, orientation, electrode_montage, source_sampling, "simu", simu_name, "md", f"{self.data_ids[i].split('_')[-1]}_md_json_flie.json"
                )
            # noise_src_file_name 
            
        # samples to keep #
        n_samples = len(self.data_ids)
        if self.to_load > n_samples : 
            self.to_load = n_samples
            print(f"--- only {n_samples} available ---")

        if self.to_load != n_samples : 
            self.data_ids = self.data_ids[
                np.random.choice(np.arange(n_samples), self.to_load, replace=False)
            ]
        
        # metadata #
        self.md = [{}] * self.to_load #dict( zip(self.data_ids, {} ) )
        self.act_src = [[]] * self.to_load #dict( zip(self.data_ids, {} ) )
        
        for k in range(self.to_load) : 
            # if self.replace_root : 
                # md_json_file_name = self.replace_root_fn( 
                    # self.match_info_dict[self.data_ids[k]]['md_json_file_name']
                # )
                # with open( Path(self.datafolder, md_json_file_name) ) as f : 
                    # self.md[k] = json.load(f)
            # else : 
            with open( Path(self.datafolder, self.match_info_dict[self.data_ids[k]]['md_json_file_name']) ) as f : 
                self.md[k] = json.load(f)

            act_src = []
            for p in range(self.md[k]['n_patch']) : 
                act_src += self.md[k]['act_src'][f'patch_{p+1}']
            self.act_src[k] = act_src
        #self.act_src = torch.from_numpy( np.array(self.act_src) )

     
        self.max_eeg = torch.zeros([self.to_load, 1])
        self.max_src = torch.zeros([self.to_load, 1])
        self.alpha_lf = None

        if self.load_lf: 
            from data.eeg.utils_eeg import load_mat
            model_path = Path(  
               self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "model"
        ) 
            self.leadfield = load_mat(Path(model_path, f"LF_{source_sampling}.mat"))['G']

    def replace_root_fn(self, string): 
        """ 
        replace "root" of file name (handle differences in simulations)
        """
        simu_name = self.config['simu_name']
        split = string.split(simu_name)
        mod_string = Path( 
            self.simu_path, simu_name, *split[1].split('/') 
            )
        return mod_string
    
    def __len__(self):
        return self.to_load

    def __getitem__(self, index):
        from data.eeg.utils_eeg import (add_noise_snr, array_range_scaling,
                                           load_mat)
        root_simu = self.datafolder
        md = self.md[index]
        # if self.replace_root : 
        #     eeg_file_name = self.replace_root_fn(self.match_info_dict[self.data_ids[index]]['eeg_file_name'] )
        #     act_src_file_name = self.replace_root_fn( self.match_info_dict[self.data_ids[index]]['act_src_file_name'] )
        #     eeg = load_mat( Path( root_simu, eeg_file_name ) )['eeg_data']['EEG'] 
        #     src = load_mat(Path( root_simu, act_src_file_name) )['Jact']['Jact']
        # else : 
        eeg = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['eeg_file_name'] ) )['eeg_data']['EEG'] 
        src = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['act_src_file_name']) )['Jact']['Jact']

        #reconstruct source data
        src_tot = np.zeros([self.config['n_sources'], self.config['n_times']])
        src_tot[self.act_src[index], :] = src.copy()

        # scale data
        eeg, src_tot, _, alpha_eeg, alpha_src, alpha_lf = scaled_data( 
            eeg, src_tot, scaling_type=self.scaler_type, leadfield=self.leadfield )
        self.max_eeg[index] = alpha_eeg
        self.max_src[index] = alpha_src
        self.alpha_lf = alpha_lf 

        return TrainingItem(input=torch.from_numpy(eeg).float(), tgt=torch.from_numpy(src_tot).float()) 

import pytorch_lightning as pl

class EsiDatamodule(pl.LightningDataModule):
    def __init__(self, dataset_kw, dl_kw, per_valid=0.2, config_file=None, subset_name=None, time_window=False, noise_fixed=False):
        super().__init__()
        self.dl_kw = dl_kw
        self.time_window = time_window
        self.per_valid = per_valid
        self.train_ds = None
        self.val_ds = None
        self.noise_fixed = noise_fixed

        self.dataset_kw =  dataset_kw
        if config_file is None : 
            config_file = Path(
                dataset_kw['datafolder'], 
                dataset_kw['subject_name'], dataset_kw['orientation'], dataset_kw['electrode_montage'], dataset_kw['source_sampling'], "simu",
                dataset_kw['simu_name'], f"{dataset_kw['simu_name']}{dataset_kw['source_sampling']}_config.json"
            )
        self.dataset_kw.update({'config_file': config_file})

        if subset_name.lower() != 'none':
            subset_file = Path(
                dataset_kw['datafolder'], 
                dataset_kw['subject_name'], dataset_kw['orientation'], dataset_kw['electrode_montage'], dataset_kw['source_sampling'], "simu",
                dataset_kw['simu_name'], f"{subset_name}.txt"
            )
            self.dataset_kw.update({'subset_file': subset_file})
        else : 
            self.dataset_kw.update({'subset_file': None})

    def setup(self, stage):
        if stage == "test": 
            if self.time_window: 
                self.test_ds = TWEsiDataset(
                    **self.dataset_kw
                )
            elif self.noise_fixed: 
                self.test_ds = EsiDatasetNoise(
                    **self.dataset_kw
                )
            else : 
                self.test_ds = EsiDataset(
                    **self.dataset_kw
                )
        else : 
            if self.time_window: 
                ds_dataset = TWEsiDataset(
                    **self.dataset_kw
                )
            elif self.noise_fixed: 
                self.test_ds = EsiDatasetNoise(
                    **self.dataset_kw
                )
            else :
                ds_dataset = EsiDataset(
                    **self.dataset_kw
                )
            self.dataset_kw['to_load'] = len(ds_dataset) #ds_dataset.to_load
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                ds_dataset,
                [int(self.dataset_kw['to_load'] * (1 - self.per_valid)), 
                 int(self.dataset_kw['to_load']) - int(self.dataset_kw['to_load']*(1 - self.per_valid))],
            ) 

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

import sys


def scaled_data(y, x, scaling_type=None, leadfield=None, alpha_L=1e3): 
    """
    Inputs :
        x : EEG data - numpy array - shape (Ne,T)
        y : source data - numpy array - shape (Ns,T)
        scaling_type : type of scaling -> None (raw data), linear, nlinear or linear bis
        leadfield : leadfield matrix (required for linear_bis scaling)
        alpha_L : scaling factor for the leadfield matrix + source data if scaling_type=linear bis

    Outputs : 
        scaled data + scaling factor(s) in the order :
        scaled_y, scaled_x, scaled_lf, y_scale_factor, x_scale_factor, leadfield_scale_factor
    """
    if scaling_type.lower()=='raw' : 
        return y, x, leadfield, 1, 1, 1
    elif scaling_type.lower()=="linear": 
        max_y = np.max(np.abs(y))
        return y / max_y, x / max_y, leadfield, max_y, max_y, 1
    elif scaling_type.lower()=="nlinear":
        max_y = np.max(np.abs(y))
        max_x = np.max(np.abs(x))
        return y / max_y, x / max_x, leadfield, max_y, max_x, 1
    elif scaling_type.lower()=="linear_bis": 
        if leadfield is None : 
            sys.exit(f"leadfield required for scaling {scaling_type}")
        # alpha_L = 1e3
        max_y = np.max(np.abs(y))   
        # return y / max_y, x / (1e-3*max_y), leadfield / alpha_L, max_y, 1e-3*max_y, alpha_L
        return y / max_y, (alpha_L *x) / (max_y), leadfield / alpha_L, max_y, max_y / alpha_L, alpha_L
    elif scaling_type.lower() =="leadfield": 
        if leadfield is None : 
            sys.exit(f"leadfield required for scaling {scaling_type}")
        alpha_L = 10*np.max(leadfield)
        max_y = np.max(np.abs(y)) 
        
        return y / max_y, alpha_L*x / max_y, leadfield / alpha_L, max_y, alpha_L/max_y, alpha_L

    else : 
        sys.exit(f"unsupported scaling type {scaling_type}")

class EsiDatasetAE(EsiDataset):
    def __init__(self, datafolder, config_file, simu_name, subject_name, source_sampling, electrode_montage, to_load, snr_db, noise_type={ "white": 1 }, scaler_type="linear", orientation="constrained", replace_root=False, load_lf=True, subset_file=None, add_noise=True):
        super().__init__(datafolder, config_file, simu_name, subject_name, source_sampling, electrode_montage, to_load, snr_db, noise_type, scaler_type, orientation, replace_root, load_lf, subset_file)
        self.add_noise = add_noise

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.add_noise:
            # add noise to data - random snr in a range
            noise = torch.randn_like(data.tgt) 
            snr_db_range = np.arange(-15, 0, 1)
            snr_db = np.random.choice(snr_db_range)
            snr = 10**(snr_db/10)

            alpha_snr = (1/np.sqrt(snr))*(data.tgt.norm() / noise.norm())
            data_noisy = data.tgt + alpha_snr*noise
            return data_noisy, data.tgt
        else : 
            return data.tgt, data.tgt 
        
class EsiDatamoduleAE(EsiDatamodule):
    def __init__(self, dataset_kw, dl_kw, per_valid=0.2, config_file=None, subset_name=None):
        super().__init__(dataset_kw, dl_kw, per_valid, config_file, subset_name)

    def setup(self, stage) : 
        if stage == "test": 
            self.test_ds = EsiDatasetAE(
                **self.dataset_kw
            )
        else : 
            ds_dataset = EsiDatasetAE(
                **self.dataset_kw
            )
            self.dataset_kw['to_load'] = len(ds_dataset) #ds_dataset.to_load
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                ds_dataset,
                [int(self.dataset_kw['to_load'] * (1 - self.per_valid)), 
                 int(self.dataset_kw['to_load']) - int(self.dataset_kw['to_load']*(1 - self.per_valid))],
            )

############## time window dataset
class TWEsiDataset(EsiDataset): 
    def __init__(self, win_length = 16, **kwargs):
        super().__init__(**kwargs)
        self.win_length = win_length
        self.win_centers = [None] * self.to_load

    def __getitem__(self, index):  
        from data.eeg.utils_eeg import (add_noise_snr, array_range_scaling,
                                           load_mat)     
        from data.eeg.data import scaled_data
        root_simu = self.datafolder
        md = self.md[index]
        if self.replace_root : 
            eeg_file_name = self.replace_root_fn(self.match_info_dict[self.data_ids[index]]['eeg_file_name'] )
            act_src_file_name = self.replace_root_fn( self.match_info_dict[self.data_ids[index]]['act_src_file_name'] )
            eeg = load_mat( Path( root_simu, eeg_file_name ) )['eeg_data']['EEG'] 
            src = load_mat(Path( root_simu, act_src_file_name) )['Jact']['Jact']
        else : 
            eeg = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['eeg_file_name'] ) )['eeg_data']['EEG'] 
            src = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['act_src_file_name']) )['Jact']['Jact']

        #reconstruct source data
        src_tot = np.zeros([self.config['n_sources'], self.config['n_times']])
        src_tot[self.act_src[index], :] = src.copy()
        
        if self.win_centers[index] is None :
            src_tot = torch.from_numpy(src_tot).float()
            # t_max = torch.argmax( src_tot.sum(axis=0).abs() )
            t_act = torch.where( src_tot.sum(axis=0).abs() > 0.1*src_tot.sum(axis=0).abs().max() )[0]

            window_center = t_act[torch.randint(0,len(t_act),(1,1)).item()]
            n_times = eeg.shape[1]
            ## check if window is not out of bounds
            if window_center - self.win_length//2 < 0: 
                window_center = self.win_length//2
            if window_center + self.win_length//2 >= n_times:
                window_center = n_times - self.win_length//2
            self.win_centers[index] = window_center
        else : 
            window_center = self.win_centers[index]

        eeg = eeg[:, window_center-self.win_length//2 : window_center+self.win_length//2]
        src_tot = src_tot[:, window_center-self.win_length//2 : window_center+self.win_length//2]
        # print(f"window center : {window_center}")
        # print(f"window range: {window_center-self.win_length//2},{window_center+self.win_length//2}")
        # add noise to EEG data
        inf = np.min(eeg)
        sup = np.max(eeg)
        if self.snr < 50:
            eeg = array_range_scaling(
                add_noise_snr( self.snr,eeg ), inf, sup
            )       
        
        # scale data
        eeg, src_tot, _, alpha_eeg, alpha_src, alpha_lf = scaled_data( 
            eeg, src_tot, scaling_type=self.scaler_type, leadfield=self.leadfield )
        self.max_eeg[index] = alpha_eeg
        self.max_src[index] = alpha_src
        self.alpha_lf = alpha_lf 
        # print(eeg.shape, src_tot.shape)
        if not torch.is_tensor(eeg): 
            eeg = torch.from_numpy(eeg).float()
        if not torch.is_tensor(src_tot):
            src_tot = torch.from_numpy(src_tot).float()
        return TrainingItem(
            input=eeg, 
            tgt=src_tot )

# NMM dataset
from scipy.io import loadmat
import os

def ispadding(x):
    # identify the padding in array
    return np.abs(x - 15213) < 1e-6


def add_white_noise(sig, snr, args_params=None):
    """
    :param sig: np.array; num_electrode * num_time
    :param snr: int; signal to noise level in dB
    :param args_params: optional parameters, could be
                        ratio: np.array; ratio between white Gaussian noise and pre-set realistic noise
                        rndata: np.array; realistic noise data; num_sample * num_electrode * num_time
                        rnpower: np.array; pre-calculated power for rndata; num_sample * num_electrode

    :return: noise_sig: np.array; num_electrode * num_time
    """

    num_elec, num_time = sig.shape
    noise_sig = np.zeros((num_elec, num_time))
    sig_power = np.square(np.linalg.norm(sig, axis=1))/num_time
    if args_params is None:
        # Only add Gaussian noise
        for i in range(num_elec):
            noise_power = 10 ** (-(snr / 10)) * sig_power[i] / 2
            noise_std = np.sqrt(noise_power)
            noise_sig[i, :] = sig[i, :] + np.random.normal(0, noise_std, (num_time,))
    else:
        # Add realistic and Gaussian noise
        rnpower = args_params['rnpower']/num_time
        rndata = args_params['rndata']
        select_id = np.random.randint(0, rndata.shape[0])
        for i in range(num_elec):
            noise_power = 10 ** (-(snr / 10)) * sig_power[i]
            rpower = args_params['ratio']*noise_power                                 # realistic noise power
            noise_std = np.sqrt(noise_power - rpower)
            noise_sig[i, :] = sig[i, :] + np.random.normal(0, noise_std, (num_time,)) + np.sqrt(rpower/rnpower[select_id][i])*rndata[select_id][:, i]
    return noise_sig


class ModSpikeEEGBuild(Dataset):
    """Dataset, generate input/output on the run

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    dataset_meta : dict
        Information needed to generate data
        selected_region: spatial model for the sources; num_examples * num_sources * max_size
                         num_examples: num_examples in this dataset
                         num_sources: num_sources in one example
                         max_size: cortical regions in one source patch; first value is the center region id; variable length, padded to max_size
                            (set to 70, an arbitrary number)
        nmm_idx:         num_examples * num_sources: index of the TVB data to use as the source
        scale_ratio:     scale the waveform maginitude in source region; num_examples * num_sources * num_scale_ratio (num_snr_level)
        mag_change:      magnitude changes inside a source patch; num_examples * num_sources * max_size
                         weight decay inside a patch; equals to 1 in the center region; variable length; padded to max_size
        sensor_snr:      the Gaussian noise added to the sensor space; num_examples * 1;

    dataset_len : int
        size of the dataset, can be set as a small value during debugging
    """

    def __init__(
        self,
        datafolder,
        subject_name,
        orientation,
        electrode_montage,
        source_sampling,
        spike_simu_name,
        metadata_name,
        fwd,
        spos=None,
        n_times=500,
        transform=None,
        args_params=None,
        scaler_type="linear",
        to_load = 10000,
    ):
        # args_params: optional parameters; can be dataset_len
        self.spike_data_path = Path(
            datafolder, subject_name, orientation, electrode_montage, source_sampling,
            "simu", spike_simu_name
        )

        self.metadata_file_path = Path(
            datafolder, subject_name, orientation, electrode_montage, source_sampling,
            "simu", metadata_name, f"{metadata_name}.mat"
        )

        # fwd_file_path = Path(
        #     datafolder, subject_name, orientation, electrode_montage, source_sampling,
        #     "model", fwd_regions_name
        # )

        # self.fwd = mne.convert_forward_solution(mne.read_forward_solution(fwd_file_path, verbose=False ), force_fixed=True, verbose=False )
        self.fwd = fwd
        self.scaler_type = scaler_type
        self.transform = transform
        self.n_times = n_times
        # self.spos = spos  # source positions, used to compute weight decay
        self.spos = self.fwd['source_rr']
        self.to_load = to_load

        self.dataset_meta = loadmat(self.metadata_file_path)

        if to_load:
            self.dataset_len = to_load
        else:  # use the whole dataset
            self.dataset_len = self.dataset_meta["selected_region"].shape[0]
        self.num_scale_ratio = self.dataset_meta["scale_ratio"].shape[2]

        ### QUICK FIX OF PROBLEMATIC DATASET
        # self.shitty_results = [
        #    387, 910, 7, 419, 936,
        #    938, 417, 325, 411, 921,
        #    356, 923, 915, 949, 917,
        #    418, 940, 920, 922, 415,
        #    993 ]
        self.shitty_results = []
        self.good_regions = np.setdiff1d(np.arange(0, 994, 1), self.shitty_results)

        self.max_eeg = torch.zeros((self.dataset_len), 1)
        self.max_src = torch.zeros((self.dataset_len), 1)

        ### select nmm index
        for k in range(self.dataset_meta["random_samples"].shape[0]):
            raw_lb = self.dataset_meta["selected_region"][k].astype(
                int
            )  # labels with padding
            #lb = raw_lb[np.logical_not(ispadding(raw_lb))]  # labels without padding

            for kk in range(raw_lb.shape[0]):  # iterate through number of sources
                curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
                a_center_kk = curr_lb[0]

                n_nmm_clips = len(os.listdir(f"{self.spike_data_path}/a{a_center_kk}"))
                if n_nmm_clips == 1:
                    self.dataset_meta["random_samples"][k][kk] = 1
                else:
                    self.dataset_meta["random_samples"][k][kk] = np.random.randint(
                        1, n_nmm_clips
                    )

    def __getitem__(self, index):

        raw_lb = self.dataset_meta["selected_region"][index].astype(
            int
        )  # labels with padding
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]  # labels without padding

        raw_nmm     = np.zeros((self.n_times, self.fwd['sol']['data'].shape[1]))
        noise_nmm   = np.zeros((self.n_times, self.fwd['sol']['data'].shape[1])) #

        for kk in range(raw_lb.shape[0]):  # iterate through number of sources
            curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
            a_center_kk = curr_lb[0]

            current_nmm = loadmat(
                f"{self.spike_data_path}/a{a_center_kk}/nmm_{self.dataset_meta['random_samples'][index][kk]}.mat"
            )["data"]

            ssig = current_nmm[:, [curr_lb[0]]]  # waveform in the center region
            # set source space SNR
            ssig = (
                ssig
                / np.max(ssig)
                * self.dataset_meta["scale_ratio"][index][kk][
                    np.random.randint(0, self.num_scale_ratio - 1)
                ]
            )
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            # set weight decay inside one source patch
            ### ------- change in the computation of weight decay for amplitude decay ---------###
            d_in_patch = np.sqrt(
                np.sum((self.spos[curr_lb[0], :] - self.spos[curr_lb, :]) ** 2, 1)
            )
            sig = (np.max(d_in_patch)) / np.sqrt(2 * np.log(2))
            weight_decay = np.exp(-0.5 * (d_in_patch / sig) ** 2)
            
            # inters = exctract_spike(current_nmm.transpose(-1, -2), a_center_kk)
            # current_nmm = remove_redundant_spike(current_nmm.transpose(-1, -2), inters)
            noise_nmm = noise_nmm + current_nmm
            raw_nmm[:,curr_lb] = ssig.reshape(-1, 1) * weight_decay

        noise_nmm /= len(raw_lb)
        unnoisy_sources = np.setdiff1d(np.arange(994), lb)
        raw_nmm[:,unnoisy_sources] = noise_nmm[:, unnoisy_sources]

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]

        eeg = np.matmul(
            self.fwd['sol']['data'], empty_nmm.transpose()
        )  # project data to sensor space; num_electrode * num_time
        # csnr = self.dataset_meta["current_snr"][index]
        csnr=np.random.randint(5,15,1).squeeze()
        noisy_eeg = add_white_noise(eeg, csnr).transpose()
        # noisy_eeg = add_noise_snr(5, eeg.transpose(-1, -2))

        ## why?
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel

        # normalize data
        self.max_eeg[index] = np.max(np.abs(noisy_eeg))
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))
        if self.scaler_type == "max-max":
            self.max_src[index] = np.max(np.abs(empty_nmm))
        else:
            self.max_src[index] = self.max_eeg[index]

        # Each data sample
        sample = {
            "data": noisy_eeg.astype("float32"),
            "nmm": empty_nmm.astype("float32"),
            "label": raw_lb,
            "snr": csnr,
        }
        # if self.transform:
        #     sample = self.transform(sample)

        # savemat('{}/data{}.mat'.format(self.file_path[0][:-4],index),{'data':noisy_eeg,'label':raw_lb,'nmm':empty_nmm[:,lb]})
        return TrainingItem(
            input=torch.from_numpy(sample["data"].transpose()), 
            tgt=torch.from_numpy(sample["nmm"].transpose())
        )

    def __len__(self):
        return self.dataset_len
    
class NMMDatamodule(pl.LightningDataModule):
    def __init__(self, dataset_kw, dl_kw, per_valid=0.2):
        super().__init__()
        self.dl_kw = dl_kw
        self.per_valid = per_valid
        self.train_ds = None
        self.val_ds = None

        self.dataset_kw = dataset_kw
        print(self.dataset_kw)

    def setup(self, stage):
        if stage == "test": 
            self.test_ds = ModSpikeEEGBuild(
                **self.dataset_kw
            )
        else:
            ds_dataset = ModSpikeEEGBuild(
                **self.dataset_kw
            )
            self.dataset_kw['to_load'] = len(ds_dataset) #ds_dataset.to_load
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                ds_dataset,
                [int(self.dataset_kw['to_load'] * (1 - self.per_valid)), 
                 int(self.dataset_kw['to_load']) - int(self.dataset_kw['to_load']*(1 - self.per_valid))],
            ) 

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)