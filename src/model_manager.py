import datetime
import logging
from os.path import dirname, join, realpath

import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.utils import shuffle

from utils.conf import Params
from data_manager import DataManager

PARAMS_PATH = join(dirname(realpath(__file__)), "..", "conf", "conf.yaml")
logger_name = Params(PARAMS_PATH, 'Logger').get_dict_params()['name']

class ModelManager():
    """
    """

    def __init__(
        self,
        data,
        augment_data
    ):
        self.data = data
        self.eval_metrics = {}
        self.X_train, self.y_train, self.X_test, self.y_test = self.set_train_test_data(with_aug_data=False)
        if augment_data:
            self.df_aug = self.augment_data()
            self.X_train_aug, self.y_train_aug, self.X_test_aug, self.y_test_aug = self.set_train_test_data(with_aug_data=True)      
            self.knn_flas_aug, self.knn_flwas_aug = self.run_knn_on_selected_ft(with_aug_data=True)
        self.knn_flas, self.knn_flaws = self.run_knn_on_selected_ft(with_aug_data=False)
        self.svm_flas, self.svm_flaws = self.run_svm_on_selected_ft()

    
    def set_train_test_data(
        self,
        with_aug_data,
        test_size=0.33,
        random_state=33
    ):
        """
        """
        logger = logging.getLogger(logger_name)
        if not with_aug_data :
            logger.info("Splitting non augmented data into train and test datasets for models ...")
            y = self.data.df_raw["Label"]
            X = self.data.df_raw.drop("Label",axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
            logger.info(f"Shape of X_train : {X_train.shape}")
            logger.info(f"Shape of y_train : {y_train.shape}")
            logger.info(f"Shape of X_test : {X_test.shape}")
            logger.info(f"Shape of y_test : {y_test.shape}")
            return X_train, y_train, X_test, y_test
        else:
            logger.info("Splitting AUGMENTED data into train and test datasets for models ...")
            y = self.df_aug["Label"]
            X = self.df_aug.drop("Label",axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
            logger.info(f"Shape of X_train (augmented) : {X_train.shape}")
            logger.info(f"Shape of y_train (augmented) : {y_train.shape}")
            logger.info(f"Shape of X_test (augmented) : {X_test.shape}")
            logger.info(f"Shape of y_test (augmented) : {y_test.shape}")
            return X_train, y_train, X_test, y_test

    def run_knn_on_selected_ft(
        self, 
        with_aug_data, 
        n=3
    ):
        """
        """
        
        logger = logging.getLogger(logger_name)

        logger.info("Running KNN algorithm on best combination of features from FLAS subsets (no data aug)...")
        features_flas = self.data.combinations_flas_top10[0][0]
        logger.info(f"These features are : {features_flas}")
        x_train = self.X_train[features_flas].values
        y_train = self.y_train.values
        x_test = self.X_test[features_flas].values
        y_test = self.y_test.values
        knn_flas = KNeighborsClassifier(n_neighbors=n)
        logger.info("Fitting train data on KNN model...")
        knn_flas.fit(x_train, y_train)
        y_pred_flas = knn_flas.predict(x_test)
        logger.info(f"Train accuracy : {accuracy_score(y_train, knn_flas.predict(x_train))}")
        logger.info(f"Test accuracy : {accuracy_score(y_test, y_pred_flas)}")
        
        logger.info("Saving results for knn with best combination of features from FLAS subsets...")
        self.eval_metrics["knn_flas"] = {}
        self.eval_metrics["knn_flas"]["Features"] = features_flas
        self.eval_metrics["knn_flas"]["Train accuracy"] = accuracy_score(y_train, knn_flas.predict(x_train))
        self.eval_metrics["knn_flas"]["Test accuracy"] = accuracy_score(y_test, y_pred_flas)

        logger.info("Running KNN algorithm on best combination of features from FLWAS subsets (no data aug)...")
        features_flwas = self.data.combinations_flwas_top10[0][0]
        logger.info(f"These features are : {features_flwas}")
        x_train = self.X_train[features_flwas].values
        y_train = self.y_train.values
        x_test = self.X_test[features_flwas].values
        y_test = self.y_test.values
        knn_flwas = KNeighborsClassifier(n_neighbors=n)
        logger.info("Fitting train data on KNN model...")
        knn_flwas.fit(x_train, y_train)
        y_pred_flwas = knn_flwas.predict(x_test)
        logger.info(f"Train accuracy : {accuracy_score(y_train, knn_flwas.predict(x_train))}")
        logger.info(f"Test accuracy : {accuracy_score(y_test, y_pred_flwas)}")
        
        logger.info("Saving results for knn with best combination of features from FLAS subsets...")
        self.eval_metrics["knn_flwas"] = {}
        self.eval_metrics["knn_flwas"]["Features"] = features_flwas
        self.eval_metrics["knn_flwas"]["Train accuracy"] = accuracy_score(y_train, knn_flwas.predict(x_train))
        self.eval_metrics["knn_flwas"]["Test accuracy"] = accuracy_score(y_test, y_pred_flwas)

        if with_aug_data:
            logger.info("Running KNN algorithm on best combination of features from FLAS subsets (with data aug)...")
            x_train = self.X_train_aug[features_flas].values
            y_train = self.y_train_aug.values
            x_test = self.X_test_aug[features_flas].values
            y_test = self.y_test_aug.values
            knn_flas_aug = KNeighborsClassifier(n_neighbors=n)
            logger.info("Fitting train data on KNN model...")
            knn_flas_aug.fit(x_train, y_train)
            y_pred_flas = knn_flas_aug.predict(x_test)
            logger.info(f"Train accuracy : {accuracy_score(y_train, knn_flas_aug.predict(x_train))}")
            logger.info(f"Test accuracy : {accuracy_score(y_test, y_pred_flas)}")
            
            logger.info("Saving results for knn with best combination of features from FLAS subsets (with data aug)...")
            self.eval_metrics["knn_flas_aug"] = {}
            self.eval_metrics["knn_flas_aug"]["Features"] = features_flas
            self.eval_metrics["knn_flas_aug"]["Train accuracy"] = accuracy_score(y_train, knn_flas_aug.predict(x_train))
            self.eval_metrics["knn_flas_aug"]["Test accuracy"] = accuracy_score(y_test, y_pred_flas)

            logger.info("Running KNN algorithm on best combination of features from FLWAS subsets (with data aug)...")
            x_train = self.X_train_aug[features_flwas].values
            y_train = self.y_train_aug.values
            x_test = self.X_test_aug[features_flwas].values
            y_test = self.y_test_aug.values
            knn_flwas_aug = KNeighborsClassifier(n_neighbors=n)
            logger.info("Fitting train data on KNN model...")
            knn_flwas_aug.fit(x_train, y_train)
            y_pred_flwas = knn_flwas_aug.predict(x_test)
            logger.info(f"Train accuracy : {accuracy_score(y_train, knn_flwas_aug.predict(x_train))}")
            logger.info(f"Test accuracy : {accuracy_score(y_test, y_pred_flwas)}")
            
            logger.info("Saving results for knn with best combination of features from FLAS subsets...")
            self.eval_metrics["knn_flwas_aug"] = {}
            self.eval_metrics["knn_flwas_aug"]["Features"] = features_flwas
            self.eval_metrics["knn_flwas_aug"]["Train accuracy"] = accuracy_score(y_train, knn_flwas.predict(x_train))
            self.eval_metrics["knn_flwas_aug"]["Test accuracy"] = accuracy_score(y_test, y_pred_flwas)

            return knn_flas_aug, knn_flwas_aug

        else:
            return knn_flas, knn_flwas

    def run_svm_on_selected_ft(self, kernel_choice='rbf'):
        """
        """

        logger = logging.getLogger(logger_name)

        logger.info("Running SVM algorithm on best combination of features from FLAS subsets...")
        features_flas = self.data.combinations_flas_top10[0][0]
        logger.info(f"These features are : {features_flas}")
        x_train = self.X_train[features_flas].values
        y_train = self.y_train.values
        x_test = self.X_test[features_flas].values
        y_test = self.y_test.values
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        svm_flas = svm.SVC(kernel=kernel_choice)
        logger.info("Fitting train data on SVM model...")
        svm_flas.fit(x_train, y_train)
        y_pred_flas = svm_flas.predict(x_test)
        logger.info(f"Train accuracy : {accuracy_score(y_train, svm_flas.predict(x_train))}")
        logger.info(f"Test accuracy : {accuracy_score(y_test, y_pred_flas)}")
        
        logger.info("Saving results for knn with best combination of features from FLAS subsets...")
        self.eval_metrics["svm_flas"] = {}
        self.eval_metrics["svm_flas"]["Features"] = features_flas
        self.eval_metrics["svm_flas"]["Train accuracy"] = accuracy_score(y_train, svm_flas.predict(x_train))
        self.eval_metrics["svm_flas"]["Test accuracy"] = accuracy_score(y_test, y_pred_flas)

        logger.info("Running SVM algorithm on best combination of features from FLWAS subsets...")
        features_flwas = self.data.combinations_flwas_top10[0][0]
        logger.info(f"These features are : {features_flwas}")
        x_train = self.X_train[features_flwas].values
        y_train = self.y_train.values
        x_test = self.X_test[features_flwas].values
        y_test = self.y_test.values
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        svm_flwas = svm.SVC(kernel=kernel_choice)
        logger.info("Fitting train data on SVM model...")
        svm_flwas.fit(x_train, y_train)
        y_pred_flwas = svm_flwas.predict(x_test)
        logger.info(f"Train accuracy : {accuracy_score(y_train, svm_flwas.predict(x_train))}")
        logger.info(f"Test accuracy : {accuracy_score(y_test, y_pred_flwas)}")
        
        logger.info("Saving results for knn with best combination of features from FLAS subsets...")
        self.eval_metrics["svm_flwas"] = {}
        self.eval_metrics["svm_flwas"]["Features"] = features_flwas
        self.eval_metrics["svm_flwas"]["Train accuracy"] = accuracy_score(y_train, svm_flwas.predict(x_train))
        self.eval_metrics["svm_flwas"]["Test accuracy"] = accuracy_score(y_test, y_pred_flwas)

        return svm_flas, svm_flwas

    def augment_data(
        self,
        nbr_aug=1, #Defines the number of time youwish to augment your data set
        ecartype=0.1, #Can be set to any value you wish to try and verify with your data set
        nbr_pca_comp=1 #Defines the number of principle components to use for the augmentation process
    ):
        """
        """
        
        logger = logging.getLogger(logger_name)
        logger.info(f"Augmenting data for models. Initial data shape is : {self.data.df_raw.shape}")

        X_phos = self.data.df_phost.drop("Label",axis=1)
        y_phos = self.data.df_phost["Label"]
        X_ster = self.data.df_ster.drop("Label", axis=1)
        y_ster = self.data.df_ster["Label"]

        pca_phos = self.data.pca_phost 
        pca_ster = self.data.pca_ster

        X_new=[]
        y_new=[]

        # Phosphate
        XX_train_new=np.array(X_phos)
        yy_train_new=np.array(y_phos)

        alpha_phos =np.random.RandomState(0).normal(0, ecartype, len(X_phos)*len(pca_phos.components_[0:nbr_pca_comp])*nbr_aug)
        a=0
        for k in range(0,nbr_aug):
            for j in range(0,len(X_phos)):
                for i in range(0,nbr_pca_comp):
                    XX_train_new[j]=XX_train_new[j]+pca_phos.components_[i]*pca_phos.explained_variance_[i]*alpha_phos[a]
                a+=1
                X_new.append(XX_train_new[j])
                y_new.append(yy_train_new[j])
                XX_train_new=np.array(X_phos)

        for j in range(0,len(X_phos)):
            X_new.append(np.array(X_phos)[j])
            y_new.append(np.array(y_phos)[j])
        # Sterile
        XX_train_new=np.array(X_ster)
        yy_train_new=np.array(y_ster)

        alpha_ster =np.random.RandomState(0).normal(0, ecartype, len(X_ster)*len(pca_ster.components_[0:nbr_pca_comp])*nbr_aug)
        a=0
        for k in range(0,nbr_aug):
            for j in range(0,len(X_ster)):
                for i in range(0,nbr_pca_comp):
                    XX_train_new[j]=XX_train_new[j]+pca_ster.components_[i]*pca_ster.explained_variance_[i]*alpha_ster[a]
                a+=1
                X_new.append(XX_train_new[j])
                y_new.append(yy_train_new[j])
                XX_train_new=np.array(X_ster)

        for j in range(0,len(X_ster)):
            X_new.append(np.array(X_ster)[j])
            y_new.append(np.array(y_ster)[j])

        # Creating final df
        X_new, y_new = shuffle(X_new, y_new, random_state=0)
        df_aug = pd.DataFrame(np.array(X_new), columns=self.data.df_phost.drop("Label",axis=1).columns)
        df_aug["Label"] = y_new
        logger.info(f"Final dataset shape after augmentation : {df_aug.shape}")

        return df_aug










