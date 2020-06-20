import time

def write_report_from_run(
    model,
    augment_data
):
    """
    Params
    ------
        model : ModelManager instance
    """
    report = "Run summary - " + time.strftime("%Y-%m-%d_%H-%M-%S") + "\n\n"
    ## DataManager info
    report += f"Raw data path : {model.data.path_to_raw}\n"
    report += f"Raw dataframe shape : {model.data.df_raw.shape}\n"
    report += f"Phosphate dataframe shape : {model.data.df_phost.shape}\n"
    report += f"Sterile dataframe shape : {model.data.df_ster.shape}\n\n"
    ## PCA info
    report += f"PCA on phosphate dataframe: \n"
    report += f"  Number of principal components with eigen values > 1: {model.data.n_pc_phost}\n"
    report += f"  Their eigenvalues are : "
    for e in model.data.egval_phost_kaiser :
        report += str(e) + "; "
    report += "\n"
    report += f"  Their explained variance ratio : "
    for e in model.data.pca_phost.explained_variance_ratio_[:model.data.n_pc_phost] :
        report += str(e) + "; "
    report += "\n"
    report += f"PCA on sterile dataframe: \n"
    report += f"  Number of principal components with eigen values > 1: {model.data.n_pc_ster}\n"
    report += f"  Their eigenvalues are : "
    for e in model.data.egval_ster_kaiser :
        report += str(e) + "; "
    report += "\n"
    report += f"  Their explained variance ratio : "
    for e in model.data.pca_ster.explained_variance_ratio_[:model.data.n_pc_ster] :
        report += str(e) + "; "
    report += "\n\n"
    ##FLAS/FLWAS subspaces
    report += "Feature subsets : \n"
    report += "  Using FLAS :\n"
    report += f"    Number of features in subset using 'mean' threshold : {len(model.data.flas_subsets['mean'])}\n"
    report += f"    Number of features in subset using 'median' threshold : {len(model.data.flas_subsets['median'])}\n"
    report += f"    Number of features in subset using 'top10pct' threshold : {len(model.data.flas_subsets['top10'])}\n"
    report += "  Using FLWAS :\n"
    report += f"    Number of features in subset using 'mean' threshold : {len(model.data.flwas_subsets['mean'])}\n"
    report += f"    Number of features in subset using 'median' threshold : {len(model.data.flwas_subsets['median'])}\n"
    report += f"    Number of features in subset using 'top10pct' threshold : {len(model.data.flwas_subsets['top10'])}\n\n"
    ##Top combinations of features
    report += f"Top combinations of features (highest B-distance) :\n"
    report += f"  From FLAS_Top10_Subset : \n"
    for e in model.data.combinations_flas_top10 :
        report += f"    {e[0]};    B-distance : {e[1]}\n"
    report +="\n"
    report += f"  From FLWAS_Top10_Subset : \n"
    for e in model.data.combinations_flwas_top10 :
        report += f"    {e[0]};    B-distance : {e[1]}\n"
    report +="\n\n"
    ##Models
    report += f"Train data shape : {model.X_train.shape}\n"
    report += f"Test data shape : {model.X_test.shape}\n"
    report += "KNN results : \n"
    report += f"  Using data without data augmentation (shape : {model.data.df_raw.shape} :\n"
    report += "    On top combination of features from FLAS : \n"
    report += f"      Features : {model.eval_metrics['knn_flas']['Features']}\n"
    report += f"      Train accuracy score : {model.eval_metrics['knn_flas']['Train accuracy']}\n"
    report += f"      Test accuracy score : {model.eval_metrics['knn_flas']['Test accuracy']}\n"
    report += "    On top combination of features from FLWAS : \n"
    report += f"      Features : {model.eval_metrics['knn_flwas']['Features']}\n"
    report += f"      Train accuracy score : {model.eval_metrics['knn_flwas']['Train accuracy']}\n"
    report += f"      Test accuracy score : {model.eval_metrics['knn_flwas']['Test accuracy']}\n"
    if augment_data :
        report += f"  Using data with data augmentation (shape : {model.df_aug.shape} :\n"
        report += "    On top combination of features from FLAS : \n"
        report += f"      Features : {model.eval_metrics['knn_flas_aug']['Features']}\n"
        report += f"      Train accuracy score : {model.eval_metrics['knn_flas_aug']['Train accuracy']}\n"
        report += f"      Test accuracy score : {model.eval_metrics['knn_flas_aug']['Test accuracy']}\n"
        report += "    On top combination of features from FLWAS : \n"
        report += f"      Features : {model.eval_metrics['knn_flwas_aug']['Features']}\n"
        report += f"      Train accuracy score : {model.eval_metrics['knn_flwas_aug']['Train accuracy']}\n"
        report += f"      Test accuracy score : {model.eval_metrics['knn_flwas_aug']['Test accuracy']}\n"

    report += "SVM results : \n"
    report += "  On top combination of features from FLAS : \n"
    report += f"    Features : {model.eval_metrics['svm_flas']['Features']}\n"
    report += f"    Train accuracy score : {model.eval_metrics['svm_flas']['Train accuracy']}\n"
    report += f"    Test accuracy score : {model.eval_metrics['svm_flas']['Test accuracy']}\n"
    report += "  On top combination of features from FLWAS : \n"
    report += f"    Features : {model.eval_metrics['svm_flwas']['Features']}\n"
    report += f"    Train accuracy score : {model.eval_metrics['svm_flwas']['Train accuracy']}\n"
    report += f"    Test accuracy score : {model.eval_metrics['svm_flwas']['Test accuracy']}\n"

    return report