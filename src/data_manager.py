import logging
import time
from os.path import dirname, join, realpath
import itertools

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import comb
import scipy as sc

from utils.conf import Params

PARAMS_PATH = join(dirname(realpath(__file__)), "..", "conf", "conf.yaml")
logger_name = Params(PARAMS_PATH, 'Logger').get_dict_params()['name']
class_sep_index = Params(PARAMS_PATH, 'Data_Settings').get_dict_params()['class_sep_index']
path_to_data_obj = Params(PARAMS_PATH, 'Paths').get_dict_params()['path_to_model_input']

class DataManager():
    """
    Class to create dataset and apply feature selection using PCA.
    """
    def __init__(
        self, 
        path_to_raw,
    ):
        """
        Create data manager object
            Params
            ------       
            Returns
            -------                
            Raises
            ------
        """
        self.logger = logging.getLogger(logger_name)
        self.path_to_raw = path_to_raw
        self.df_raw, self.df_sref = self.set_raw_data(class_sep_index)
        self.df_phost, self.df_ster = self.set_phost_ster_df()
        self.pca_phost, self.n_pc_phost, self.egval_phost_kaiser, self.pca_ster, self.n_pc_ster, self.egval_ster_kaiser = self.run_pca_with_kaiser()
        self.flas_subsets, self.flwas_subsets = self.set_subsets_from_pca_loadings()
        self.combinations_flas_top10, self.combinations_flwas_top10 = self.compute_combinations_top10()
        self.combinations_flas_mean, self.combinations_flwas_mean = self.compute_combinations_mean()
        self.combinations_flas_median, self.combinations_flwas_median = self.compute_combinations_median()
        self.save_data_object()

    def set_raw_data(
        self, 
        separator_index, 
    ):
        """
        Params
        ------
            separator_index :   int, row index st : rows before separator_index are class 1, rows 
                                after are class 0
            augment_dat :       bool, if true, add data augmentation step

        Returns
        -------
            Raw dataframe, with label columns, and dataframe containing Sref column. These dfs have
            same index.
        """
        self.logger.info(f"Reading data from {self.path_to_raw}...")
        df_raw = pd.read_excel(self.path_to_raw)
        
        self.logger.info(f"Extracting Sref column...")
        df_sref = pd.DataFrame(df_raw["Unnamed: 0"]).rename(columns={"Unnamed: 0" : "Sref"})
        df_raw = df_raw.drop("Unnamed: 0", axis=1)
        
        self.logger.info(f"Adding Label column (Phosphate: 1, Sterile: 0)")
        df_raw["Label"] = 0
        df_raw.loc[0:separator_index,"Label"] = 1
        
        self.logger.info("Setting raw data completed")
        return df_raw, df_sref
    
    def set_phost_ster_df(self):
        """
        """
        self.logger.info(f"Setting phosphate and sterile dataframes (Phosphate: 1, Sterile: 0)...")
        df_phost = self.df_raw[self.df_raw["Label"]==1]
        df_ster  = self.df_raw[self.df_raw["Label"]==0]
        self.logger.info(f"Shape of phosphate class dataframe : {df_phost.shape}")
        self.logger.info(f"Shape of sterile class dataframe : {df_ster.shape}")
        return df_phost, df_ster

    def run_pca_with_kaiser(self):
        """
        Apply pca and Kaiser threshold on both phosphate and sterile dataframes.
        
        Returns
        -------
            pca_phost, pca_ster : 
                pca object from sklearn.PCA() class, fit on phosphate/sterile data
            n_pc_phost, n_pc_ster : 
                number of principal components which have eigenvalues > 1 (Kaiser criterion), 
                for both phosphate and sterile PCAs
            egval_phost_kaiser, egval_ster_kaiser : 
                eigenvalues that are > 1 for both PCAs
        """
        self.logger.info("Applying PCA on phosphate dataframe...")
        x_phost = StandardScaler().fit_transform(self.df_phost.drop("Label", axis=1))
        pca_phost = PCA()
        pca_phost.fit(x_phost)
        self.logger.info(
            "Applying Kaiser criterion (looking for principal components with eigenvalues > 1)..."
            )
        egval_phost_kaiser = [v for v in pca_phost.explained_variance_ if v>1]
        n_pc_phost = len(egval_phost_kaiser)
        self.logger.info(
            f"Number of PCs with eigenvalues > 1 : {n_pc_phost}.\n\
            Eigenvalues : {egval_phost_kaiser}\n\
            Explained variance of each PC : {pca_phost.explained_variance_ratio_[:n_pc_phost]}\n\
            Cumulative explained variance : {sum(pca_phost.explained_variance_ratio_[:n_pc_phost])}"
            )

        self.logger.info("Applying PCA on sterile dataframe...")
        x_ster = StandardScaler().fit_transform(self.df_ster.drop("Label", axis=1))
        pca_ster = PCA()
        pca_ster.fit(x_ster)
        self.logger.info(
            "Applying Kaiser criterion (looking for principal components with eigenvalues > 1)..."
            )
        egval_ster_kaiser = [v for v in pca_ster.explained_variance_ if v>1]
        n_pc_ster = len(egval_ster_kaiser)
        self.logger.info(
            f"Number of PCs with eigenvalues > 1 : {n_pc_ster}.\n\
            Eigenvalues : {egval_ster_kaiser}\n\
            Explained variance of each PC : {pca_ster.explained_variance_ratio_[:n_pc_ster]}\n\
            Cumulative explained variance : {sum(pca_ster.explained_variance_ratio_[:n_pc_ster])}"
            )
        
        return pca_phost, n_pc_phost, egval_phost_kaiser, pca_ster, n_pc_ster, egval_ster_kaiser

    def set_subsets_from_pca_loadings(self):
        """
        Define 6 subsets of features for each class, based on pca loadings and FLAS/FLAWS values.
        
        Returns
        -------
            flas_subsets : dictionary
                With 3 elements : 'mean', 'median', and 'top10'. Each corresponds to the threshold 
                used to select subset of features, based on the FLAS values computed for each feature
            flwas_subsets : dictionary
                With 3 elements : 'mean', 'median', and 'top10'. Each corresponds to the threshold 
                used to select subset of features, based on the FLWAS values computed for each feature
        """
        
        self.logger.info("Computing FLAS for PHOSPHATE dataset")
        loadings_phost = pd.DataFrame(self.pca_phost.components_).loc[0:self.n_pc_phost-1,:]
        flas_phost = loadings_phost.apply(np.abs).sum()
        self.logger.info("Getting subset of features using 3 thresholds (mean, median, top10%)")
        flas_phost_ts_1 = list(flas_phost[flas_phost>flas_phost.mean()].index)
        flas_phost_ts_2 = list(flas_phost[flas_phost>flas_phost.median()].index)
        flas_phost_ts_3 = list(flas_phost.nlargest(45).index)
        
        self.logger.info("Computing FLAS for STERILE dataset")
        loadings_ster = pd.DataFrame(self.pca_ster.components_).loc[0:self.n_pc_ster-1,:]
        flas_ster = loadings_ster.apply(np.abs).sum()
        self.logger.info("Getting subset of features using 3 thresholds (mean, median, top10%)")
        flas_ster_ts_1 = list(flas_ster[flas_ster>flas_ster.mean()].index)
        flas_ster_ts_2 = list(flas_ster[flas_ster>flas_ster.median()].index)
        flas_ster_ts_3 = list(flas_ster.nlargest(45).index)
        
        flas_subsets = {}
        self.logger.info("Computing union of subsets (phost and ster) obtained with FLAS...")
        flas_subsets["mean"] = sorted(list(set(flas_phost_ts_1) | set(flas_ster_ts_1)))
        self.logger.info(f"Number of features using FLAS with mean threshold : {len(flas_subsets['mean'])}")
        flas_subsets["median"] = sorted(list(set(flas_phost_ts_2) | set(flas_ster_ts_2)))
        self.logger.info(f"Number of features using FLAS with median threshold : {len(flas_subsets['median'])}")
        flas_subsets["top10"] = sorted(list(set(flas_phost_ts_3) | set(flas_ster_ts_3)))
        self.logger.info(f"Number of features using FLAS with top10% threshold : {len(flas_subsets['top10'])}")

        self.logger.info("Computing FLWAS for PHOSPHATE dataset")
        flwas_phost = loadings_phost.mul(self.pca_phost.explained_variance_ratio_[0:self.n_pc_phost],axis=0).apply(np.abs).sum()
        self.logger.info("Getting subset of features using 3 thresholds (mean, median, top10%)")
        flwas_phost_ts_1 = list(flwas_phost[flwas_phost>flwas_phost.mean()].index)
        flwas_phost_ts_2 = list(flwas_phost[flwas_phost>flwas_phost.median()].index)
        flwas_phost_ts_3 = list(flwas_phost.nlargest(45).index)
        
        self.logger.info("Computing FLWAS for STERILE dataset")
        flwas_ster = loadings_ster.mul(self.pca_ster.explained_variance_ratio_[0:self.n_pc_ster],axis=0).apply(np.abs).sum()
        self.logger.info("Getting subset of features using 3 thresholds (mean, median, top10%)")
        flwas_ster_ts_1 = list(flwas_ster[flwas_ster>flwas_ster.mean()].index)
        flwas_ster_ts_2 = list(flwas_ster[flwas_ster>flwas_ster.median()].index)
        flwas_ster_ts_3 = list(flwas_ster.nlargest(45).index)
        
        flaws_subsets = {}
        self.logger.info("Computing union of subsets (phost and ster) obtained with FLWAS...")
        flaws_subsets["mean"] = sorted(list(set(flwas_phost_ts_1) | set(flwas_ster_ts_1)))
        self.logger.info(f"Number of features using FLWAS with mean threshold : {len(flaws_subsets['mean'])}")
        flaws_subsets["median"] = sorted(list(set(flwas_phost_ts_2) | set(flwas_ster_ts_2)))
        self.logger.info(f"Number of features using FLWAS with median threshold : {len(flaws_subsets['median'])}")
        flaws_subsets["top10"] = sorted(list(set(flwas_phost_ts_3) | set(flwas_ster_ts_3)))
        self.logger.info(f"Number of features using FLWAS with top10% threshold : {len(flaws_subsets['top10'])}")

        return flas_subsets, flaws_subsets

    @staticmethod
    def B_dist(
        df_phost,
        df_ster, 
        indices
    ):
        """
        Compute B-distance between df_phost[indices] and df_ster[indices].
        Params
        ------
            df_phost : dataframe 
                contains phosphate samples
            df_ster : dataframe
                contains sterile samples
            indices : list 
                feature names to keep for the computation of B-distance between the 2 classes
        
        Returns
        -------
            B_dist : float
        """

        logger = logging.getLogger(logger_name)
        logging.info(f"Computing B_dist between phosphate and sterile samples, with indices {indices}")
        try:
            assert(all(i in df_phost.columns for i in indices))
            assert(all(i in df_ster.columns for i in indices))
        except AssertionError as e:
            logging.error(
                "Given features to computes B_dist were not found in dataframes.\
                Details: " + str(e)
                )

        phost = df_phost[indices]
        ster  = df_ster[indices]
        #For the selected intermediate phosphate and sterile classes we measure the B-distance
        mean1 = phost.mean()
        mean2 = ster.mean()
        cov1 = phost.cov()
        cov2 = ster.cov()
        covaverage = 1/2*(cov1+cov2)
        Cholesky_compute = np.transpose(np.linalg.cholesky(covaverage))
        meandiff = np.linalg.lstsq(Cholesky_compute.T,(np.subtract(mean1,mean2)).T, rcond=None)[0].T
        B_dist1 = 0.125*np.dot(meandiff,np.transpose(meandiff))
        B_dist2 = np.linalg.lstsq(sc.linalg.sqrtm(np.dot(cov1,cov2)).T,covaverage.T, rcond=None)[0].T
        B_dist3 = np.linalg.det(B_dist2)
        B_dist = B_dist1 + 0.5*np.log(B_dist3)

        return B_dist

    def compute_combinations_and_B_dist(
        self, 
        indices, 
        k=3, 
        return_top_=5
    ):
        """
        Compute all combinations of k features from len(indices), then compute B_dist with each 
        combination, and return 'return_top_' combinations with biggest B_dist between the 2 classes

        Params
        ------
            indices : list 
                contains values from 0 to 450. Will be transformed to features first, 
                ie from 250 to 2500.
            k : int, optional
                Number of elements in the combinations to compute. Default is 3.
                if k>3 : program might be interrupted because of MemoryError
            return_top_ : int, optional
                Number of top combinations to return (ordered by decreasing B-distance value). 
                Must be <= than total number of combinations. Default is 5.
        
        Returns
        -------
            combination_B_dist : list
                list of tuple, each tuple contain : combination of features (list), B_dist(float)
                eg : [([630, 2015, 2355], 1.39)]
        """
        self.logger.info("Converting list of indices (0-450) to list of features (250-2500)")
        features = [e*5 + 250 for e in indices]
        
        self.logger.info(
            f"Begin computing combinations of {k} features from {len(features)} features.\n\
            (Total nmber of combinations : {comb(len(features), k, repetition=False)})"
            )
        l_combinations = list(itertools.combinations(features, k))

        self.logger.info(f"Computing B_distance for each of the {len(l_combinations)} combinations...")
        combination_B_dist = [(list(l), self.B_dist(self.df_phost, self.df_ster, list(l))) for l in l_combinations]
        self.logger.info(f"Sorting combinations by decreasing order of B_distance...")
        combination_B_dist = sorted(combination_B_dist, key = lambda x: x[1], reverse=True)
        self.logger.info(f"Extracting top {return_top_} combinations (with highest B_dist)...")
        combination_B_dist = combination_B_dist[0:return_top_]
        self.logger.info(
            f"Result combinations and corresponding B_distance :\n\
            {combination_B_dist}"
            )

        return combination_B_dist

    def compute_combinations_top10(self):
        """
        Compute combinations that score highest B_distance, from flas and flaws subsets that
        were obtained with top10% threshold.

        Returns
        -------
            combinations_flaws_top10, combinations_flaws_top10 : lists
        """
        self.logger.info("Computing combinations for FLAS Top10\%\ subset :")
        combinations_flas_top10 = self.compute_combinations_and_B_dist(self.flas_subsets['top10'])
        self.logger.info("Computing combinations for FLWAS Top10\%\ subset :")
        combinations_flwas_top10 = self.compute_combinations_and_B_dist(self.flwas_subsets['top10'])

        return combinations_flas_top10, combinations_flwas_top10
    
    def compute_combinations_mean(self):
        """
        Compute combinations that score highest B_distance, from flas and flaws subsets that
        were obtained with mean threshold.

        Returns
        -------
            combinations_flaws_mean, combinations_flaws_mean : lists
        """
        self.logger.info("Computing combinations for FLAS mean subset :")
        combinations_flas_mean = self.compute_combinations_and_B_dist(self.flas_subsets['mean'])
        self.logger.info("Computing combinations for FLWAS mean subset :")
        combinations_flwas_mean = self.compute_combinations_and_B_dist(self.flwas_subsets['mean'])

        return combinations_flas_mean, combinations_flwas_mean
    
    def compute_combinations_median(self):
        """
        Compute combinations that score highest B_distance, from flas and flaws subsets that
        were obtained with median threshold.

        Returns
        -------
            combinations_flaws_median, combinations_flaws_median : lists
        """
        self.logger.info("Computing combinations for FLAS median subset :")
        combinations_flas_median = self.compute_combinations_and_B_dist(self.flas_subsets['median'])
        self.logger.info("Computing combinations for FLWAS median subset :")
        combinations_flwas_median = self.compute_combinations_and_B_dist(self.flwas_subsets['median'])

        return combinations_flas_median, combinations_flwas_median

    def save_data_object(self):
        """
        Saves data object with pickle, after initialisation is completed.
        """
        self. logger.info(f"Saving data object in {path_to_data_obj}...")
        file_name = time.strftime("%Y-%m-%d_%H-%M-%S_DataObject")
        with open(path_to_data_obj + file_name, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        self.logger.info("Saving completed.")

        ## to load save data object with python :
        ## with open(path_to_data_obj + file_name, "rb") as saved_data:
        ##      data = pickle.load(saved_data)