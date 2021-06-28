import matplotlib.pyplot as plt
import numpy as np
import os
from classification.utils_nlp import load_list_from_disk


def find_average_no_skill(load_dir, nb_random_realizations, embedding, date):
    """This function finds the no-skill value of the PR curve across the random realizations.
    Args:
        load_dir (str): path to folder from which we load the values
        nb_random_realizations (int): number of cross-validation runs that were performed
        embedding (str): embedding technique that was used
        date (str): date on which the output values were saved
    Returns:
        average_no_skill (float): no skill value of PR curves computed across the random realizations
    """
    no_skill_across_realizations = []
    for iteration in range(1, nb_random_realizations+1):
        y_test = load_list_from_disk(os.path.join(load_dir, "y_test_{}_random_realization_{}_{}.pkl".format(embedding, iteration, date)))
        y_test_np = np.asarray(y_test)
        no_skill = len(y_test_np[y_test_np == 1]) / len(y_test_np)
        no_skill_across_realizations.append(no_skill)

    average_no_skill = np.mean(no_skill_across_realizations)

    return average_no_skill


def load_values(load_dir, nb_random_realizations, date, embedding="tf_idf"):
    """This function loads from disk the values of the ROC and PR curves
    Args:
        load_dir (str): path to folder from which we load the values
        nb_random_realizations (int): number of cross-validation runs that were performed
        date (str): date on which the output values were saved
        embedding (str): embedding technique that was used; defaults to "tf_idf"
    Returns:
        fpr (list): FPR values across random realizations
        tpr (list): TPR values across random realizations
        auc (list): AUC values across random realizations
        recall (list): recall values across random realizations
        prec (list): precision values across random realizations
        aupr (list): AUPR values across random realizations
        avg_no_skill_across_realizations (float): no skill value of PR curves computed across the random realizations
    """
    # load curve values
    fpr = load_list_from_disk(os.path.join(load_dir, "fpr_values_{}_{}.pkl".format(embedding, date)))
    tpr = load_list_from_disk(os.path.join(load_dir, "tpr_values_{}_{}.pkl".format(embedding, date)))
    auc = load_list_from_disk(os.path.join(load_dir, "auc_values_{}_{}.pkl".format(embedding, date)))
    recall = load_list_from_disk(os.path.join(load_dir, "recall_values_{}_{}.pkl".format(embedding, date)))
    prec = load_list_from_disk(os.path.join(load_dir, "prec_values_{}_{}.pkl".format(embedding, date)))
    aupr = load_list_from_disk(os.path.join(load_dir, "aupr_values_{}_{}.pkl".format(embedding, date)))

    avg_no_skill_across_realizations = find_average_no_skill(load_dir, nb_random_realizations, embedding, date)

    return fpr, tpr, auc, recall, prec, aupr, avg_no_skill_across_realizations


def extract_average_values_roc(nb_random_realizations, fpr_list, tpr_list, auc_values):
    """This function extracts the average values for the ROC curve across the random realizations
        Args:
            nb_random_realizations (int): number of cross-validation runs that were performed
            fpr_list (list): it contains the FPR values across the random realizations
            tpr_list (list): it contains the TPR values across the random realizations
            auc_values (list): it contains the AUC values across the random realizations
        Returns:
            mean_tpr (float): mean TPR value across random realizations
            std_tpr (float): standard deviation TPR value across random realizations
            tprs_upper (float): upper bound (+1 std)
            tprs_lower (float): lower bound (+1 std)
            avg_auc (float): mean AUC value across random realizations
            std_auc (float): standard deviation AUC value across random realizations
        """
    tprs = []  # type: list
    mean_fpr = np.linspace(0, 1, 100)

    # since across random iterations the fpr and tpr vectors have different length, we must interpolate
    for iteration in range(nb_random_realizations):
        interp_tpr = np.interp(mean_fpr, fpr_list[iteration], tpr_list[iteration])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # average also the AUC
    avg_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)

    return mean_tpr, std_tpr, tprs_upper, tprs_lower, avg_auc, std_auc


def plot_roc_curve(fpr_tf_idf, tpr_tf_idf, auc_tf_idf, fpr_d2v, tpr_d2v, auc_d2v, nb_random_realizations):
    """This function plots the average ROC curve across the random realizations
    Args:
        fpr_tf_idf (list): FPR values of the TF-IDF algorithm
        tpr_tf_idf (list): TPR values of the TF-IDF algorithm
        auc_tf_idf (list): AUC values of the TF-IDF algorithm
        fpr_d2v (list): FPR values of the Doc2Vec algorithm
        tpr_d2v (list): TPR values of the Doc2Vec algorithm
        auc_d2v (list): AUC values of the Doc2Vec algorithm
        nb_random_realizations (int): number of cross-validation runs that were performed
    """
    mean_tpr_tf_idf, std_tpr_tf_idf, tpr_upper_tf_idf, tpr_lower_tf_idf, mean_auc_tf_idf, std_auc_tf_idf = extract_average_values_roc(nb_random_realizations, fpr_tf_idf, tpr_tf_idf, auc_tf_idf)

    mean_tpr_d2v, std_tpr_d2v, tpr_upper_d2v, tpr_lower_d2v, mean_auc_d2v, std_auc_d2v = extract_average_values_roc(nb_random_realizations, fpr_d2v, tpr_d2v, auc_d2v)

    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)
    # TF-IDF
    ax.plot(mean_fpr, mean_tpr_tf_idf, color="b", label=r'TF-IDF (mean AUC = {:.2f})'.format(mean_auc_tf_idf), lw=2, alpha=.8)
    ax.fill_between(mean_fpr, tpr_lower_tf_idf, tpr_upper_tf_idf, color='b', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # Dov2Vec
    ax.plot(mean_fpr, mean_tpr_d2v, color="g", label=r'Doc2Vec (mean AUC = {:.2f})'.format(mean_auc_d2v), lw=2, alpha=.8)
    ax.fill_between(mean_fpr, tpr_lower_d2v, tpr_upper_d2v, color='g', alpha=.2, label=r'$\pm$ 1 std. dev.')

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='No Skill', alpha=.8)  # draw chance line
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_title("ROC curves; {} realizations".format(nb_random_realizations), weight="bold", fontsize=15)
    ax.set_xlabel('FPR (1- specificity)', fontsize=12)
    ax.set_ylabel('TPR (sensitivity)', fontsize=12)
    ax.legend(loc="lower right")


def extract_average_values_pr(nb_random_realizations, recall_list, precision_list, aupr_values):
    """This function extracts the average values for the PR curve across the random realizations
    Args:
        nb_random_realizations (int): number of cross-validation runs that were performed
        recall_list (list): it contains the recall values across the random realizations
        precision_list (list): it contains the precision values across the random realizations
        aupr_values (list): it contains the AUPR values across the random realizations
    Returns:
        mean_prec (float): mean precision value across random realizations
        std_prec (float): standard deviation precision value across random realizations
        precisions_upper (float): upper bound (+1 std)
        precisions_lower (float): lower bound (+1 std)
        avg_aupr (float): mean AUPR value across random realizations
        std_aupr (float): standard deviation AUPR value across random realizations
    """
    precisions = []  # type: list
    mean_recall = np.linspace(0, 1, 100)

    # since across random iterations the recall and precision vectors have different length, we must interpolate
    for iteration in range(nb_random_realizations):
        # flip the vectors because np.interp expects increasing values of the x-axis
        rec_flip = np.flip(recall_list[iteration])
        prec_flip = np.flip(precision_list[iteration])

        interp_prec = np.interp(mean_recall, rec_flip, prec_flip)
        interp_prec[0] = 1.0
        precisions.append(interp_prec)

    mean_prec = np.mean(precisions, axis=0)
    std_prec = np.std(precisions, axis=0)
    precisions_upper = np.minimum(mean_prec + std_prec, 1)
    precisions_lower = np.maximum(mean_prec - std_prec, 0)

    # average also the AUPR
    avg_aupr = np.mean(aupr_values)
    std_aupr = np.std(aupr_values)

    return mean_prec, std_prec, precisions_upper, precisions_lower, avg_aupr, std_aupr


def plot_pr_curve(recall_tf_idf, prec_tf_idf, aupr_tf_idf, avg_no_skill_tf_idf, recall_d2v, prec_d2v, aupr_d2v, avg_no_skill_d2v, nb_random_realizations):
    """This function plots the average ROC curve across the random realizations
    Args:
        recall_tf_idf (list): recall values of the TF-IDF algorithm
        prec_tf_idf (list): precision values of the TF-IDF algorithm
        aupr_tf_idf (list): AUPR values of the TF-IDF algorithm
        avg_no_skill_tf_idf (float): no-skill value for the TF-IDF algorithm
        recall_d2v (list): recall values of the Doc2Vec algorithm
        prec_d2v (list): precision values of the Doc2Vec algorithm
        aupr_d2v (list): AUPR values of the Doc2Vec algorithm
        avg_no_skill_d2v (float): no-skill value for the Doc2Vec algorithm
        nb_random_realizations (int): number of cross-validation runs that were performed
    """
    assert avg_no_skill_tf_idf == avg_no_skill_d2v, "The no-skill values should be identical between the two embeddings"

    mean_prec_tf_idf, std_prec_tf_idf, precisions_upper_tf_idf, precisions_lower_tf_idf, avg_aupr_tf_idf, std_aupr_tf_idf = extract_average_values_pr(nb_random_realizations,
                                                                                                                                                      recall_tf_idf,
                                                                                                                                                      prec_tf_idf,
                                                                                                                                                      aupr_tf_idf)

    mean_prec_d2v, std_prec_d2v, precisions_upper_d2v, precisions_lower_d2v, avg_aupr_d2v, std_aupr_d2v = extract_average_values_pr(nb_random_realizations,
                                                                                                                                    recall_d2v,
                                                                                                                                    prec_d2v,
                                                                                                                                    aupr_d2v)

    fig, ax = plt.subplots()
    mean_recall = np.linspace(0, 1, 100)
    # TF-IDF
    ax.plot(mean_recall, mean_prec_tf_idf, color="b", label=r'TF-IDF (mean AUPR = {:.2f})'.format(avg_aupr_tf_idf), lw=2, alpha=.8)
    ax.fill_between(mean_recall, precisions_lower_tf_idf, precisions_upper_tf_idf, color='b', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # Dov2Vec
    ax.plot(mean_recall, mean_prec_d2v, color="g", label=r'Doc2Vec (mean AUPR = {:.2f})'.format(avg_aupr_d2v), lw=2, alpha=.8)
    ax.fill_between(mean_recall, precisions_lower_d2v, precisions_upper_d2v, color='g', alpha=.2, label=r'$\pm$ 1 std. dev.')

    ax.plot([0, 1], [avg_no_skill_tf_idf, avg_no_skill_tf_idf], linestyle='--', lw=2, color="r", label='No Skill', alpha=.8)
    ax.set_title("PR curves; {} realizations".format(nb_random_realizations), weight="bold", fontsize=15)
    ax.set_xlabel('Recall (sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (PPV)', fontsize=12)
    ax.legend(loc="center left")


def main():
    # INPUT ARGUMENTS THAT MUST BE SET BY THE USER
    load_dir = "/path/to/output/dir/out_curve_values/"  # type: str # folder where output files were saved
    date = "Jun_19_2021"  # type: str # date on which output files were saved

    # FIXED ARG
    nb_random_realizations = 10

    fpr_tf_idf, tpr_tf_idf, auc_tf_idf, recall_tf_idf, prec_tf_idf, aupr_tf_idf, avg_no_skill_tf_idf = load_values(load_dir, nb_random_realizations, date, embedding="tf_idf")

    fpr_d2v, tpr_d2v, auc_d2v, recall_d2v, prec_d2v, aupr_d2v, avg_no_skill_d2v = load_values(load_dir, nb_random_realizations, date, embedding="d2v")

    plot_roc_curve(fpr_tf_idf, tpr_tf_idf, auc_tf_idf, fpr_d2v, tpr_d2v, auc_d2v, nb_random_realizations)
    plot_pr_curve(recall_tf_idf, prec_tf_idf, aupr_tf_idf, avg_no_skill_tf_idf, recall_d2v, prec_d2v, aupr_d2v, avg_no_skill_d2v, nb_random_realizations)
    plt.show()


if __name__ == '__main__':
    main()
