import pandas as pd
import os
import re
import csv
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pickle
from nltk import wordpunct_tokenize
import spacy
from collections import Counter
from dateutil.parser import parse
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing
from gensim.models import Doc2Vec
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from typing import List, Any


def print_running_time(start_time, end_time, process_name):
    """This function takes as input the start and the end time of a process and prints to console the time elapsed for this process
    Args:
        start_time (float): instant when the timer is started
        end_time (float): instant when the timer was stopped
        process_name (string): name of the process
    Returns:
        None
    """
    sentence = str(process_name)  # convert to string whatever the user inputs as third argument
    temp = end_time - start_time  # compute time difference
    hours = temp // 3600  # compute hours
    temp = temp - 3600 * hours  # if hours is not zero, remove equivalent amount of seconds
    minutes = temp // 60  # compute minutes
    seconds = temp - 60 * minutes  # compute minutes
    print('\n%s time: %d hh %d mm %d ss' % (sentence, hours, minutes, seconds))
    return


def round_half_up(n: float, decimals: float = 0) -> float:
    """This function rounds to the nearest integer number (e.g 2.4 becomes 2.0 and 2.6 becomes 3);
     in case of tie, it rounds up (e.g. 1.5 becomes 2.0 and not 1.0)
    Args:
        n (float): number to round
        decimals (int): number of decimal figures that we want to keep; defaults to zero
    Returns:
        rounded_number (float): input number rounded with the desired decimals
    """
    multiplier = 10 ** decimals

    rounded_number = math.floor(n * multiplier + 0.5) / multiplier

    return rounded_number


def load_data_and_merge(*args):
    """This function loads two chunks of annotated reports as pd.Dataframes, merges them, and returns the merged dataframe.
    The data was split in two json files because the radiologist made two distinct annotation sessions.
    Args:
        *args: arbitrary number of input arguments
    Returns:
        df_merged (pd.Dataframe): concatenation of the two input dataframes
    Raises:
        AssertionError: if any of the input paths does not exist
        AssertionError: if any of the input paths does not have json extension
    """
    df_merged = pd.DataFrame()  # type: pd.DataFrame # initialize empty dataframe
    for json_path in args:
        assert os.path.exists(json_path), "Path {0} does not exist"
        ext = os.path.splitext(json_path)[-1].lower()  # get the file extension
        assert ext == ".json", "{} must be the path to a json file"
        df_chunk = pd.read_json(json_path, lines=True)  # type: pd.DataFrame  # build pandas dataframe from json
        df_merged = pd.concat([df_merged, df_chunk], ignore_index=True)  # ignore_index makes sure that indexes are labeled from 0 to n-1

    return df_merged


def most_frequent(input_list: list) -> Any:
    """This function is given a list as input and it returns its most frequent element
    Args:
        input_list (list)
    Returns:
        most_frequent_item (*): most frequent item in the list; can be of Any type
    """
    occurence_count = Counter(input_list)  # type: Counter
    most_frequent_item = occurence_count.most_common(1)[0][0]
    return most_frequent_item


def write_to_csv(input_file, out_dir, out_filename, delimiter='\t'):
    """This function saves the input_file as a csv row file
    Args:
        input_file (*): input file to save as csv row file
        out_dir (str): path to output folder
        out_filename (str): output filename
        delimiter (str): delimiter to use when saving the csv file
    Returns:
        None
    """
    if not os.path.exists(out_dir):  # if dir does not exist
        os.makedirs(out_dir)  # create it
    with open(os.path.join(out_dir, out_filename), "w") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(input_file)


def plot_roc_curve(flat_y_test, flat_y_pred_proba, cv_folds, embedding_label="doc2vec", plot=True):
    """This function computes FPR, TPR and AUC. Then, it plots the ROC curve
    Args:
        flat_y_test (list): labels
        flat_y_pred_proba (list): predictions
        cv_folds (int): number of folds in the cross-validation
        embedding_label (str): embedding algorithm that was used
        plot (bool): if True, the ROC curve will be displayed
    """
    fpr, tpr, _ = roc_curve(flat_y_test, flat_y_pred_proba, pos_label=1)
    tpr[0] = 0.0  # ensure that first element is 0
    tpr[-1] = 1.0  # ensure that last element is 1
    auc_roc = auc(fpr, tpr)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="b", label=r'{} (AUC = {:.2f})'.format(embedding_label, auc_roc), lw=2, alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_title("ROC curve; {}-fold CV".format(cv_folds), weight="bold", fontsize=15)
        ax.set_xlabel('FPR (1- specificity)', fontsize=12)
        ax.set_ylabel('TPR (sensitivity)', fontsize=12)
        ax.legend(loc="lower right")
    return fpr, tpr, auc_roc


def plot_pr_curve(flat_y_test, flat_y_pred_proba, cv_folds, embedding_label="doc2vec", plot=True):
    """This function computes precision, recall and AUPR. Then, it plots the PR curve
    Args:
        flat_y_test (list): labels
        flat_y_pred_proba (list): predictions
        cv_folds (int): number of folds in the cross-validation
        embedding_label (str): embedding algorithm that was used
        plot (bool): if True, the ROC curve will be displayed
    """
    precision, recall, _ = precision_recall_curve(flat_y_test, flat_y_pred_proba)
    aupr = auc(recall, precision)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color="g", label=r'{} (AUPR = {:.2f})'.format(embedding_label, aupr))
        ax.set_title("PR curve; {}-fold CV".format(cv_folds), weight="bold", fontsize=15)
        ax.set_xlabel('Recall (sensitivity)', fontsize=12)
        ax.set_ylabel('Precision (PPV)', fontsize=12)
        ax.legend(loc="lower left")
    return recall, precision, aupr


def classification_metrics(y_true, y_pred, binary=True):
    """This function computes some standard classification metrics for a binary problem
    Args:
        y_true (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels
        binary (bool): indicates whether the classification is binary (i.e. two classes) or not (i.e. more classes)
    Returns:
        conf_mat (np.ndarray): confusion matrix
        acc (float): accuracy
        rec (float): recall (i.e. sensitivity, or true positive rate)
        spec (float): specificity (i.e. true negative rate)
        prec (float): precision (i.e. positive predictive value)
        npv (float): negative predictive value
        f1 (float): F1-score (i.e. harmonic mean of precision and recall)
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    acc, rec, spec, prec, npv, f1 = None, None, None, None, None, None  # initialize all metrics to None
    if binary:
        assert conf_mat.shape == (2, 2), "Confusion Matrix does not correspond to a binary task"
        tn = conf_mat[0][0]
        fn = conf_mat[1][0]
        # tp = conf_mat[1][1]
        fp = conf_mat[0][1]

        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        spec = tn / (tn + fp)
        prec = precision_score(y_true, y_pred)
        npv = tn / (tn + fn)
        f1 = f1_score(y_true, y_pred)

    else:
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, average="weighted")
        prec = precision_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average='weighted')

    return conf_mat, acc, rec, spec, prec, npv, f1


def save_pickle_list_to_disk(list_to_save: list, out_dir: str, out_filename: str) -> None:
    """This function saves a list to disk
    Args:
        list_to_save (list): list that we want to save
        out_dir (str): path to output folder; will be created if not present
        out_filename (str): output filename
    Returns:
        None
    """
    if not os.path.exists(out_dir):  # if output folder does not exist
        os.makedirs(out_dir)  # create it
    open_file = open(os.path.join(out_dir, out_filename), "wb")
    pickle.dump(list_to_save, open_file)  # save list with pickle
    open_file.close()


def is_date(input_string: str, fuzzy: bool = False) -> bool:
    """This function checks whether in the input string there is a date
    Args:
        input_string: str, string to check for date
        fuzzy: bool, ignore unknown tokens in string if True
    Returns:
         True: whether the string can be interpreted as a date.
         False: otherwise
    """
    try:
        parse(input_string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def extract_global_label(label_list):
    """This function extracts the global label of the report from the input annotation list
    Args:
        label_list (list): it contains all the classification annotations
    Returns:
        global_label (int): value corresponding to the global class (0 = stable, 1 = response, 2 = progression, 3 = unknown)
    Raises:
        AssertionError: if more than one value is found for the global class
        ValueError: if the global label was not found
    """
    only_global = [x for x in label_list if "Global" in x]
    assert len(only_global) == 1, "Only one value should be retrieved"

    if only_global[0] == " Global_stable":
        global_label = 0
    elif only_global[0] == " Global_response":
        global_label = 1
    elif only_global[0] == " Global_progression":
        global_label = 2
    elif only_global[0] == " Global_unknown":
        global_label = 3
    else:
        raise ValueError("global label not found")

    return global_label


def preprocess_and_extract_labels(df, remove_proper_nouns=True, remove_custom_words=True, lowercase=True, remove_punctuation=True,
                                  remove_stopwords=True, from_indication_onward=True, from_description_onward=True, lemmatize=True, print_info=True):
    """This function takes as input a dataframe containing documents (i.e. reports), and preprocess each of them according to the specified input args.
    Also, it extracts the label associated with each report.
    Args:
        df (pd.Dataframe): dataframe containing all the documents
        remove_proper_nouns (bool): if set to True, it removes all proper nouns (e.g. physician's name, patient's names, etc.) with a trained POS tagger
        remove_custom_words (bool): if set to True, it removes some specific useless words that are recurrent in the reports
        lowercase (bool): if set to True, it converts all words to lowercase
        remove_punctuation (bool): if set to True, it removes all punctuation (but preserves dates)
        remove_stopwords (bool): if set to True, it removes most French stopwords (we keep the word "pas" though, since it's relevant for determining the meaning)
        from_indication_onward (bool): if set to True, everything before the Indication section is discarded
        from_description_onward (bool): if set to True, everything before the Description section is discarded
        lemmatize (bool): if set to True, it lemmatizes every word of each document
        print_info (bool): verbose; if set to True, it prints useful info and creates informative plots
    Returns:
        all_preprocessed_reports_tokenized (list): it contains all the preprocessed reports as lists of tokens
        all_preprocessed_reports (list): it contains all the preprocessed reports as free text
        all_global_labels (list): it contains the global label associated with each report
        all_original_reports (list): it contains the original (non-preprocessed) reports
        vocabulary_size_preprocessed (int): size of vocabulary after preprocessing
    """
    # check if there are missing values in the input dataframe's columns
    important_columns = ["annotation", "content"]  # type: list # set names of columns of interest
    for col in df.columns:  # loop over columns
        if col in important_columns:
            assert df[col].isnull().sum() == 0, "There are some missing values in the column {0}".format(col)

    # extract sub-dataframe only with the columns of interest (i.e. text and labels)
    sub_df = df.loc[:, important_columns]  # type: pd.DataFrame

    original_reports_len = []  # type: list # will contain the length of the original report
    preprocessed_reports_len = []  # type: list # will contain the length of the preprocessed report
    all_global_labels = []  # type: list # will contain the global labels
    all_original_reports = []  # type: list # will contain the original (i.e. non-prepocessed) reports
    all_original_reports_tokenized = []  # type: list # will contain the original (i.e. non-preprocessed) reports tokenized
    all_preprocessed_reports = []  # type: list # will contain the preprocessed reports as free text
    all_preprocessed_reports_tokenized = []  # type: list # will contain the preprocessed reports tokenized
    if remove_proper_nouns:
        nlp_french = spacy.load("fr_core_news_lg")  # load a french NLP trained model from spacy

    for idx, row in sub_df.iterrows():  # loop over dataframe's rows
        # --------------------------------- first deal with the TEXT -----------------------------------
        report = row["content"]  # type: str # extract report
        all_original_reports.append(report)
        orig_tokens = wordpunct_tokenize(report)  # type: list # tokenize report
        original_reports_len.append(len(orig_tokens))
        all_original_reports_tokenized.append(orig_tokens)

        # create hard copy of original report; this will be modified (i.e. preprocessed) according to the input args
        preprocessed_report = (report + '.')[:-1]  # type: str

        # remove proper nouns
        if remove_proper_nouns:
            preprocessed_report_list = []  # type: list
            tokens_to_keep = ["Pas", 'réséqué', "Comparatif", "oligodendrogliome", "Astrocytome", "punctiforme", "Gadolinium", "mastoïdiennes",
                              "pseudophake", "ethmoïdal", "néoangiogénèse", "MPRAGE"]  # type: list
            doc = nlp_french(preprocessed_report)  # analyze report with a trained French model. It computes, among other things, the POS tags
            for token in doc:
                # if the token is not a proper name or it contains even one digit --> we keep it (cause we discard names, but we want to keep dates)
                if token.pos_ != "PROPN" or any(chr.isdigit() for chr in token.text) or token.text in tokens_to_keep:
                    # append token (it will be kept)
                    preprocessed_report_list.append(token.text)

            # convert back from list to string using whitespace as separator
            preprocessed_report = ' '.join(preprocessed_report_list)  # type: str

        # lowercase report
        if lowercase:
            preprocessed_report = preprocessed_report.lower()  # type: str

        # tokenization
        preprocessed_tokens = wordpunct_tokenize(preprocessed_report)  # type: list

        if lemmatize:
            # TODO: consider adding this if performances are not satisfactory
            pass

        if remove_punctuation:
            # only keep alphanumeric tokens and dates
            preprocessed_tokens = [word for word in preprocessed_tokens if word.isalnum() or is_date(word, fuzzy=True)]  # type: list

        if remove_stopwords:
            stopwords = ['de', 'ce', 'cet', 'cette', 'la', 'en', 'et', 'du', 'd', 'le', 'l', 'un', 'une', 'les', 'des', 'ces', 'à', 'au', 'aux']  # type: list
            # keep all words except those in the list
            preprocessed_tokens = [word for word in preprocessed_tokens if word not in stopwords]  # type: list

        if remove_custom_words:
            # define list with some specific words to remove
            custom_words_to_remove = ["chuv", "onm1", "lausanne", "dre", "dr", "réf", "destinataire",
                                      "docteur", "preliminary", "1011", "n", "professeur", "assistante",
                                      "leonel", "cheffe", "page", "chef", "médecin", "v", "prof", "assistant"]  # type: list
            # keep all words except those in the list
            preprocessed_tokens = [word for word in preprocessed_tokens if word not in custom_words_to_remove]  # type: list

        if from_indication_onward:
            indication = ["indications", "Indications", "indication", "Indication"]  # type: list
            # find index of word "indication" (or equivalent)
            idxs = [i for i, x in enumerate(preprocessed_tokens) if x in indication]
            # if list is not empty
            if idxs:
                last_idx = max(idxs)  # type: int # take highest index which is most likely the last one corresponding to the "description" section
                preprocessed_tokens = preprocessed_tokens[last_idx + 1:]  # only retain tokens inside the "description" section
            else:  # if instead list is empty
                print("WARNING: indication section not found; take full report")
                # raise ValueError("Description section not found")

        if from_description_onward:
            description = ["descriptions", "Descriptions", "description", "Description"]  # type: list
            # find index of word "description" (or equivalent)
            idxs = [i for i, x in enumerate(preprocessed_tokens) if x in description]
            # if list is not empty
            if idxs:
                last_idx = max(idxs)  # type: int # take highest index which is most likely the last one corresponding to the "description" section
                preprocessed_tokens = preprocessed_tokens[last_idx+1:]  # only retain tokens inside the "description" section
            else:  # if instead list is empty
                print("WARNING: description section not found; take full report")
                # raise ValueError("Description section not found")

        all_preprocessed_reports_tokenized.append(preprocessed_tokens)
        preprocessed_reports_len.append(len(preprocessed_tokens))

        # convert back from list to string using whitespace as separator
        tokens_clean_text = ' '.join(preprocessed_tokens)  # type: str
        all_preprocessed_reports.append(tokens_clean_text)

        # --------------------------------- now deal with the labels -----------------------------------
        all_labels = row["annotation"]  # type: dict # extract report
        labels_dict = {}  # type: dict # initialize as empty
        for dict_name in all_labels.keys():  # loop over annotation dictionaries
            if dict_name == "classificationResult":
                for dicts in all_labels[dict_name]:
                    if "Classification" in dicts.values():
                        labels_dict = dicts

        assert labels_dict, "Dictionary is empty or not found"
        labels = labels_dict["classes"]  # type: list
        global_label = extract_global_label(labels)
        all_global_labels.append(global_label)

    all_reports_preprocessed_flat = [item for sublist in all_preprocessed_reports_tokenized for item in sublist]  # type: list # flatten list of lists
    vocabulary_size_preprocessed = len(list(set(all_reports_preprocessed_flat)))  # type: int # extract unique words and compute len of list

    if print_info:
        # check vocabulary size before and after preprocessing
        all_reports_original_flat = [item for sublist in all_original_reports_tokenized for item in sublist]  # type: list # flatten list of lists
        vocabulary_size_original = len(list(set(all_reports_original_flat)))  # type: int # extract unique words and compute len of list

        # check report length before and after preprocessing
        original_rep_len = {"original": np.asarray(original_reports_len)}
        preprocessed_rep_len = {"preprocessed": np.asarray(preprocessed_reports_len)}
        original_rep_len = pd.DataFrame(original_rep_len)
        preprocessed_rep_len = pd.DataFrame(preprocessed_rep_len)
        merged_df = pd.concat([original_rep_len, preprocessed_rep_len], ignore_index=True, axis=1)
        merged_df.columns = ["original", "preprocessed"]  # assign names to columns
        plt.figure(1)
        # create boxplots
        _ = sns.boxplot(data=merged_df)
        # set ylabel
        plt.ylabel("# tokens", fontsize=12)
        # set bold title
        plt.title("Report length difference", weight="bold", fontsize=15)
        # show horizontal gridlines
        plt.grid(axis='y', linestyle='-')

        # check most frequent words
        plt.figure(2)
        df = pd.DataFrame({'freq': all_reports_preprocessed_flat})
        sub_df = df['freq'].value_counts().nlargest(10)
        ax2 = sub_df.plot(kind='bar')
        ax2.set_title("10 most frequent words", fontsize=15, weight="bold")
        ax2.set_ylabel("Counts", fontsize=12)
        ax2.set_xticklabels(sub_df.keys(), rotation=45)

        # check global_label class distribution
        count = Counter(all_global_labels)
        df = pd.DataFrame.from_dict(count, orient='index')
        ax3 = df.plot(kind='bar')
        ax3.set_ylabel("Counts", fontsize=12)
        ax3.set_title('Global Label distribution', fontsize=15, weight="bold")
        xticks = ["stable", "response", "progression", "unknown"]
        ax3.set_xticklabels(xticks, rotation=45)
        ax3.get_legend().remove()
        plt.show()

        # print out vocabulary size before and after preprocessing
        print("Vocabulary size before preprocessing: {}".format(vocabulary_size_original))
        print("Vocabulary size after preprocessing: {}".format(vocabulary_size_preprocessed))

    return all_preprocessed_reports_tokenized, all_preprocessed_reports, all_global_labels, all_original_reports, vocabulary_size_preprocessed


def find_common_elements(list1: list, list2: list) -> List:
    """This function takes as input two lists and returns a list with the common elements
    Args:
        list1 (list): first list
        list2 (list): second list
    Returns:
        intersection_as_list (list): list containing the common elements between the two input lists
    """
    list1_as_set = set(list1)  # type: set
    intersection = list1_as_set.intersection(list2)  # type: set
    intersection_as_list = list(intersection)  # type: list

    return intersection_as_list


def adjust_cv_splits_for_multiple_sessions(all_reports, train_idxs, test_idxs, df_comparative_dates_and_reports, x_train, x_test, y_train, y_test):
    """This function ensures that multiple sessions of the same subjects are not present some in the training set and some in test set since
    this would lead to upwardly biased results; instead it moves these overlapping sessions from test to training.
    Args:
        all_reports (list): it contains the original (non-preprocessed) reports
        train_idxs (np.ndarray): indexes of train samples
        test_idxs (np.ndarray): indexes of test samples
        df_comparative_dates_and_reports (pd.Dataframe): it contains all the dates (current and previous), labels and corresponding report
        x_train (list): it contains the training free-text reports
        x_test (list): it contains the test free-text reports
        y_train (list): it contains the training labels
        y_test (list): it contains the test labels
    Returns:
        x_train (list): it contains the training free-text reports plus the test-free text reports that were from overlapping sessions
        y_train (list): it contains the corresponding adjusted labels
        x_test (list): it contains the test free-text reports without those that were from overlapping sessions
        y_test (list): it contains the corresponding adjusted labels
        train_idxs_after_adjustment (list): it contains the adjusted train indexes
        test_idxs_after_adjustment (list): it contains the adjusted test indexes
    """
    x_train_sanity_check, x_test_sanity_check = [all_reports[i] for i in train_idxs], [all_reports[j] for j in test_idxs]
    ipps_x_ext_train = df_comparative_dates_and_reports.loc[df_comparative_dates_and_reports["report"].isin(x_train_sanity_check)]["ipp"].tolist()
    ipps_x_test = df_comparative_dates_and_reports.loc[df_comparative_dates_and_reports["report"].isin(x_test_sanity_check)]["ipp"].tolist()
    intersection = find_common_elements(ipps_x_ext_train, ipps_x_test)

    # if the intersection list is not empty (i.e. if different sessions of the same subject are present some in training and some in test splits
    if intersection:
        # this is not a good split (it would lead to upwardly-biased results)  --> we move overlapping sessions from test to training
        idxs_to_remove = []
        for idx, value in enumerate(ipps_x_test):
            if value in intersection:
                x_train.append(x_test[idx])  # move problematic sub_ses from test to training
                y_train.append(y_test[idx])  # move problematic sub_ses from test to training
                idxs_to_remove.append(idx)  # save problematic indexes

        # remove problematic sub_ses from test set
        x_test = [value for idx, value in enumerate(x_test) if idx not in idxs_to_remove]
        y_test = [value for idx, value in enumerate(y_test) if idx not in idxs_to_remove]

        train_idxs_after_adjustment = list(train_idxs) + [value for idx, value in enumerate(test_idxs) if idx in idxs_to_remove]
        test_idxs_after_adjustment = [value for idx, value in enumerate(test_idxs) if idx not in idxs_to_remove]

    # if there is no intersection, the indexes don't change
    else:
        train_idxs_after_adjustment = train_idxs
        test_idxs_after_adjustment = test_idxs

    return x_train, y_train, x_test, y_test, train_idxs_after_adjustment, test_idxs_after_adjustment


def find_best_hyperparams_tf_idf(list_of_dicts, allowed_ngrams, percentage_vocab_size, allowed_max_features):
    """This function finds the best hyperparameters for the tf-idf embedding.
    Args:
        list_of_dicts (list): it contains several dicts. Each dict has the hyperparm combination as key and the a corresponding validation metric
        allowed_ngrams (list): ngrams that are being tuned
        percentage_vocab_size (list): vocabulary sizes that are being tuned
        allowed_max_features (list): values of max_features of the random forest that are being tuned
    Returns:
        ngram (int): best ngram across validation folds
        pvs (*): best percentage of vocabulary size; either float or None across validation folds
        mf (float): best max_features allowed across validation folds
    """
    hyperparams_combinations = []
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            hyperparams_combinations.append(key)

    counter_keys_list = list(Counter(hyperparams_combinations).values())

    # if list contains all unique elements
    if len(set(counter_keys_list)) == len(counter_keys_list):
        best_key, _ = Counter(hyperparams_combinations).most_common(1)[0]

    # if instead list contains duplicates
    else:
        counts = {}
        for n in hyperparams_combinations:
            counts[n] = counts.get(n, 0) + 1

        max_frequency = max(counter_keys_list)
        most_frequent_keys = [k for k, v in counts.items() if v == max_frequency]
        highest_f1 = 0
        for dictionary in list_of_dicts:
            for key, value in dictionary.items():
                if key in most_frequent_keys:
                    f1 = value
                    if f1 > highest_f1:
                        highest_f1 = f1
                        best_key = key

    ngram = re.findall(r"ngram\d+", best_key)[0]
    ngram = int(re.findall(r"\d+", ngram)[0])

    max_features = re.findall(r"mf\d+\.\d+", best_key)[0]
    mf = float(re.findall(r"\d+\.\d+", max_features)[0])

    if "pvsNone" not in best_key:
        perc_vocab_size = re.findall(r"pvs\d+\.\d+", best_key)[0]
        pvs = float(re.findall(r"\d+\.\d+", perc_vocab_size)[0])
    elif "pvsNone" in best_key:
        pvs = None
    else:
        raise ValueError("Percentage of vocabulary size can either be None or a float number")

    assert ngram in allowed_ngrams, "ngram has unexpected value"
    assert pvs in percentage_vocab_size, "Unknown vector size"
    assert mf in allowed_max_features, "Unknown max features parameter"

    return ngram, pvs, mf


def find_best_hyperparams_d2v(list_of_dicts, allowed_vector_sizes, allowed_max_features):
    """This function finds the best hyperparameters (across validation folds) for the Doc2Vec algorithm
    Args:
        list_of_dicts (list): it contains several dicts. Each dict has the hyperparm combination as key and the a corresponding validation metric
        allowed_vector_sizes (list): it contains the allowed values for the vector size (i.e. the ones that are being tuned)
        allowed_max_features (list): values of max_features of the random forest that are being tuned
    Returns:
        d2v_type (int): best Doc2Vec model across validation folds
        vs (int): best vector size across validation folds
        mf (float): best value for max_features of the random forest across validation folds
    """
    hyperparams_combinations = []
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            hyperparams_combinations.append(key)

    counter_keys_list = list(Counter(hyperparams_combinations).values())

    # if list contains all unique elements
    if len(set(counter_keys_list)) == len(counter_keys_list):
        best_key, _ = Counter(hyperparams_combinations).most_common(1)[0]

    # if instead list contains duplicates
    else:
        counts = {}
        for n in hyperparams_combinations:
            counts[n] = counts.get(n, 0) + 1

        max_frequency = max(counter_keys_list)
        most_frequent_keys = [k for k, v in counts.items() if v == max_frequency]
        highest_f1 = 0
        for dictionary in list_of_dicts:
            for key, value in dictionary.items():
                if key in most_frequent_keys:
                    f1 = value
                    if f1 > highest_f1:
                        highest_f1 = f1
                        best_key = key

    d2v_type = re.findall(r"type\d+", best_key)[0]
    vector_size = re.findall(r"vs\d+", best_key)[0]
    max_features = re.findall(r"\d+\.\d+", best_key)[0]

    if "1" in d2v_type:
        d2v_type = 1
    elif "0" in d2v_type:
        d2v_type = 0
    else:
        raise ValueError("Unknown value for doc2vec type. Only 0 and 1 allowed")

    vs = int(re.findall(r"\d+", vector_size)[0])
    assert vs in allowed_vector_sizes, "Unknown vector size"

    mf = float(max_features)
    assert mf in allowed_max_features, "Unknown max features parameter"

    return d2v_type, vs, mf


def tf_idf_embedding(document_list, vocabulary_size, percentage_vocab_size, ngram_range, min_doc_freq, max_doc_freq):
    """This function computes the tf-idf (term-frequency, inverse-document-frequency) embedding for the input document list.
    Args:
        document_list (list): it contains the free-text documents to embed
        vocabulary_size (int): size of vocabulary (i.e. unique words)
        percentage_vocab_size (float): percentage of vocabulary size; used to retain the most frequent words
        ngram_range (tuple): range (inclusive) of n-gram sizes for tokenizing text
        min_doc_freq (int): min number of documents where a word has to appear in order to be retained
        max_doc_freq (float): max document frequency. If a word appears in more than this proportion of documents, it's very common, thus it's removed
    Returns:
        embedded_reports (np.ndarray): documents embedded with TF-IDF
        tfidfconverter (sklearn.feature_extraction.text.TfidfVectorizer): trained TF-IDF vectorizer
    """
    if percentage_vocab_size:  # if percentage_vocab_size is not None
        # we only want to retain the "percentage_vocab_size" most frequent words. If "percentage_vocab_size" < 1, it means we remove very rare words
        top_max_features_by_term_frequency = int(round_half_up(vocabulary_size*percentage_vocab_size))
    else:  # if instead percentage_vocab_size is None
        top_max_features_by_term_frequency = None

    # create keyword arguments to pass to the 'tf-idf' vectorizer
    kwargs = {
        'ngram_range': ngram_range,  # set N-grams to retain
        'analyzer': 'word',  # split text into word tokens
        'max_features': top_max_features_by_term_frequency,  # remove rare infrequent words
        'min_df': min_doc_freq,
        'max_df': max_doc_freq
    }

    tfidfconverter = TfidfVectorizer(**kwargs)  # create vectorizer object
    embedded_reports = tfidfconverter.fit_transform(document_list).toarray()  # embed documents

    return embedded_reports, tfidfconverter


def create_doc2vec_model(alg_type, vector_size, window, neg_words, min_count, sample, epochs):
    """This function creates the Doc2Vec model
    Args:
        alg_type (int): it defines the training algorithm; if 1 --> PV-DM, if 0 --> PV-DBOW
        vector_size (int): dimensionality of feature vectors
        window (int): max distance between the current and predicted word within a sentence
        neg_words (int): "noise" words to draw
        min_count (int): minimum word frequency
        sample (float): threshold for configuring which higher-frequency words are randomly downsampled
        epochs (int): number of training iterations over the corpus
    Returns:
        model_d2v (gensim.models.doc2vec.Doc2Vec): initialized Doc2Vec model (not trained yet)
    """
    cores = multiprocessing.cpu_count()  # save number of available CPUs (threads)
    model_d2v = Doc2Vec(dm=alg_type,  # use distributed bag of words (PV-DBOW)
                        vector_size=vector_size,  # set dimensionality of feature vectors
                        window=window,  # set max distance between the current and predicted word within a sentence
                        hs=0,  # flag used to enable negative sampling
                        negative=neg_words,  # specify how many "noise" words to draw
                        min_count=min_count,  # ignore all words with total frequency lower than this
                        sample=sample,  # threshold for configuring which higher-frequency words are randomly downsampled
                        workers=cores,  # use these many worker threads to train the model faster
                        epochs=epochs)  # number of iterations (epochs) over the corpus
    return model_d2v


def create_and_train_doc2vec(doc2vec_type, vs, train_tagged):
    """This function creates the Doc2Vec model, builds the vocabulary and trains the model
    Args:
        doc2vec_type (int): type of Doc2Vec model to use (1 --> PV-DM, 0 --> PV-DBOW)
        vs (int): vector size (i.e. length of vectors that will embed the documents)
        train_tagged (pandas.core.series.Series): training reports and labels
    Returns:
        model_d2v (gensim.models.doc2vec.Doc2Vec): trained Doc2Vec model
    """
    model_d2v = create_doc2vec_model(alg_type=doc2vec_type,  # choose whether to use PV-DM or PV-DBOW
                                     vector_size=vs,  # set dimensionality of feature vectors
                                     window=5,  # set max distance between the current and predicted word within a sentence
                                     neg_words=5,  # specify how many "noise" words to draw
                                     min_count=2,  # ignore all words with total frequency lower than this
                                     sample=0,  # threshold for configuring which higher-frequency words are randomly downsampled
                                     epochs=100)  # number of iterations (epochs) over the corpus

    # build vocabulary from the sequence of train documents
    model_d2v.build_vocab([x for x in train_tagged.values])

    # train Doc2Vec model
    model_d2v.train(shuffle([x for x in train_tagged.values]),
                    total_examples=len(train_tagged.values),  # count of documents
                    epochs=model_d2v.epochs)  # use number of epochs specified when creating the model

    return model_d2v


def vec_for_learning(model, tagged_docs):
    """This function creates the final numerical vectors that will be fed to the following classifier
    Args:
        model (gensim.models.doc2vec.Doc2Vec): (trained) Doc2Vec model
        tagged_docs (gensim.models.doc2vec.TaggedDocument): tagged report organized as TaggedDocument objects
    Returns:
        regressors_np (np.ndarray): it contains the numerical vectors (np arrays) encoding the input tagged documents
        targets_np (np.ndarray): it contains the ground truth labels
        doc_2_embedding_mapping (dict): mapping between the tokenized reports and the corresponding numerical embeddings
    """
    documents = tagged_docs.values  # type: np.ndarray
    doc_2_embedding_mapping = {}  # type: dict
    targets_list = []  # type: list
    regressors_list = []  # type: list

    # infer vector representation from trained model
    for doc in documents:
        targets_list.append(doc.tags[0])
        embedding_vector = model.infer_vector(doc.words, steps=20)  # type: np.ndarray
        regressors_list.append(embedding_vector)
        doc_as_string = " ".join(item for item in doc.words)
        doc_2_embedding_mapping[doc_as_string] = embedding_vector

    targets_np = np.asarray(targets_list)
    regressors_np = np.asarray(regressors_list)
    return regressors_np, targets_np, doc_2_embedding_mapping


def plot_conf_matrix_and_results(conf_mat, acc, rec, spec, prec, npv, f1, plot=True):
    """This function plots the confusion matrix and the classification results
    Args:
        conf_mat (np.ndarray): confusion matrix
        acc (float): accuracy
        rec (float): recall (i.e. sensitivity, or true positive rate)
        spec (float): specificity (i.e. true negative rate)
        prec (float): precision (i.e. positive predictive value)
        npv (float): negative predictive value
        f1 (float): F1-score (i.e. harmonic mean of precision and recall)
        plot (bool): whether to plot the confusion matrix; defaults to True
    """
    df_conf_matrix = pd.DataFrame(conf_mat)  # type: pd.DataFrame
    if df_conf_matrix.shape == (2, 2):
        df_conf_matrix.columns = ["stable", "unstable"]
        df_conf_matrix.index = ["stable", "unstable"]
    elif df_conf_matrix.shape == (4, 4):
        df_conf_matrix.columns = ["stable", "response", "progression", "unknown"]
        df_conf_matrix.index = ["stable", "response", "progression", "unknown"]
    else:
        raise ValueError("Check number of labels")

    print("\nTest set results over {} documents: ".format(int(np.sum(conf_mat))))
    print("Accuracy = {:.2f}".format(acc))
    print("Sensitivity (recall) = {:.2f}".format(rec))
    print("Specificity = {:.2f}".format(spec))
    print("Precision (PPV) = {:.2f}".format(prec))
    print("NPV = {:.2f}".format(npv))
    print("f1-score = {:.2f}".format(f1))

    # plot CONFUSION MATRIX
    if plot:
        fig, ax = plt.subplots()
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_conf_matrix, annot=True, annot_kws={"size": 16}, fmt='d')  # font size
        ax.set_title("Conf. Matrix ({} samples)".format(int(np.sum(conf_mat))), weight="bold", fontsize=15)
        ax.set_ylabel("True", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12)
        plt.show()


def extract_lime_explanation_tf_idf(idx_doc_to_investigate, vectorizer, random_forest, x_test, y_test, out_dir, cnt_document, prediction, embedding, save=True):
    """This function computes the LIME explanation for one report
    Args:
        idx_doc_to_investigate (int): index of report to explain
        vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): trained TF-IDF vectorizer
        random_forest (sklearn.ensemble.RandomForestClassifier): trained random forest classifier
        x_test (list): it contains the free-text reports of the test set
        y_test (list): it contains the labels of the test set
        out_dir (str): folder where we want to save the explanation .html files
        cnt_document (int): dummy counter used to name each document differently
        prediction (str): type of example being explained (e.g. TN, TP, FP, FN)
        embedding (str): embedding used to compute the explanation
        save (bool): if True, the .html explanations are saved to disk
    Returns:
        explanation_list (list): it contains the most frequent words associated with the explanation
    """
    class_names = ["stable", "unstable"]
    explainer = LimeTextExplainer(class_names=class_names)
    if embedding == "tf_idf":
        c = make_pipeline(vectorizer, random_forest)
        exp = explainer.explain_instance(x_test[idx_doc_to_investigate], c.predict_proba, num_features=6)
    else:
        raise ValueError('Only "tf_idf" allowed as embedding; got {} instead'.format(embedding))

    # print("Preprocessed report:\n{}".format(x_test[idx_doc_to_investigate]))
    # print('\nProbability (unstable) =', c.predict_proba([x_test[idx_doc_to_investigate]])[0, 1])
    # print("\nTrue class: {}".format(class_names[y_test[idx_doc_to_investigate]]))
    explanation_list = exp.as_list()
    # exp.as_pyplot_figure()
    if save:
        out_dir = os.path.join(out_dir, "LIME", prediction, embedding)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_filepath = os.path.join(out_dir, "{}_lime_explanation_{}_{}.html".format(cnt_document, prediction, embedding))
        exp.save_to_file(out_filepath)

    return explanation_list


def print_most_common_features(explanation_list, predictions, nb_most_common_features=6):
    """This function prints the most common features chosen by LIME for the two classes (stable vs. unstable)
    Args:
        explanation_list (list): it contains the most common features for a certain category of samples (e.g. TPs)
        predictions (str): samples being explained (e.g. FPs)
        nb_most_common_features (int): number of most common features to print per category of samples
    """
    explanation_list_flat = [item for sublist in explanation_list for item in sublist]  # flatten list
    all_explanations_np = np.asarray(explanation_list_flat)
    numerical_values = all_explanations_np[:, 1].astype(np.float32)
    positive_values = np.argwhere(numerical_values > 0)  # the features indicating instability have positive values
    negative_values = np.argwhere(numerical_values < 0)  # the features indicating stability have negative values
    all_explanations_np_unstable = all_explanations_np[positive_values, 0].flatten()
    all_explanations_np_stable = all_explanations_np[negative_values, 0].flatten()
    most_common_unstable_features = Counter(all_explanations_np_unstable).most_common(nb_most_common_features)
    most_common_stable_features = Counter(all_explanations_np_stable).most_common(nb_most_common_features)

    print("\nMost common features for {} reports:".format(predictions))
    print("Unstable: {}".format(most_common_unstable_features))
    print("Stable: {}".format(most_common_stable_features))
