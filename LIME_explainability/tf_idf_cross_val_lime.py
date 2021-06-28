import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from classification.utils_nlp import print_running_time, load_data_and_merge, plot_conf_matrix_and_results,\
    classification_metrics, preprocess_and_extract_labels, adjust_cv_splits_for_multiple_sessions,\
    tf_idf_embedding, extract_lime_explanation_tf_idf, print_most_common_features


def tf_idf_classification_lime(df_reports, df_comparative_dates_and_reports, max_features_random_forest, n_grams, percentage_vocab_size, cv_folds, output_folder,
                               remove_proper_nouns, remove_custom_words, lowercase, remove_punctuation, from_indication_onward, from_description_onward,
                               binary_classification=True, embedding="tf_idf"):
    """This function executes the classification of the reports using the TF-IDF model for embedding and the random forest classifier
    Args:
        df_reports (pd.Dataframe): it contains all the information in the json file created during the manual annotation processing
        df_comparative_dates_and_reports (pd.Dataframe): it contains all the dates (current and previous), labels and corresponding report
        max_features_random_forest (float): it contains the percentage of features to be retained by the random forest classifier
        n_grams (int): n-grams to use
        percentage_vocab_size (float): percentages of vocabulary size to use
        cv_folds (int): number of cross-validation folds
        output_folder (str): path where we save the embedding files
        remove_proper_nouns (bool): if set to True, proper nouns in the reports will be removed
        remove_custom_words (bool): if set to True, specific words related to the reports will be removed
        lowercase (bool): if set to True, all words of the reports will be set to lowercase
        remove_punctuation (bool): if set to True, punctuation will be removed from the reports
        from_indication_onward (bool): if set to True, we will only use the words from the "Indication" section onwards; everything before will be discarded
        from_description_onward (bool): if set to True, we will only use the words from the "Description" section onwards; everything before will be discarded
        binary_classification (bool): if set to True, it means that we will perform a binary classification; defaults to True
        embedding (str): embedding algorithm used
    """
    start = time.time()  # start timer; used to compute the time needed to run this script
    # ---------------------------------------------------------------------------------------------

    _, all_preprocessed_reports, global_labels, all_reports, preprocessed_vocabulary_size = preprocess_and_extract_labels(df_reports,
                                                                                                                          remove_proper_nouns=remove_proper_nouns,
                                                                                                                          remove_custom_words=remove_custom_words,
                                                                                                                          lowercase=lowercase,
                                                                                                                          remove_punctuation=remove_punctuation,
                                                                                                                          from_indication_onward=from_indication_onward,
                                                                                                                          from_description_onward=from_description_onward,
                                                                                                                          lemmatize=False,
                                                                                                                          print_info=False)

    y_test_across_folds = []  # type: list
    y_pred_test_across_folds = []  # type: list

    # error analysis lists
    all_explanations_fp = []
    all_explanations_fn = []
    all_explanations_tp = []
    all_explanations_tn = []
    cnt_fp = 0
    cnt_fn = 0
    cnt_tp = 0
    cnt_tn = 0
    # ---------------------------------------- BEGIN EXTERNAL CROSS-VALIDATION ------------------------------------
    # since the dataset is very imbalanced, apply stratified cross validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=123)
    test_fold = 0  # type: int # counter to keep track of cross-validation fold
    for ext_train_idxs, test_idxs in skf.split(all_preprocessed_reports, global_labels):
        test_fold += 1
        print("\n\n------------------------------------------------- Test fold {}".format(test_fold))

        # create ext_train and test datasets
        x_ext_train, x_test = [all_preprocessed_reports[i] for i in ext_train_idxs], [all_preprocessed_reports[j] for j in test_idxs]
        y_ext_train, y_test = [global_labels[i] for i in ext_train_idxs], [global_labels[j] for j in test_idxs]

        # ensure that multiple sessions of the same sub are not some in ext_train and some in test set
        x_ext_train, y_ext_train, x_test, y_test, ext_train_idxs_after_adjust, test_idxs_after_adjust = adjust_cv_splits_for_multiple_sessions(all_reports,
                                                                                                                                               ext_train_idxs,
                                                                                                                                               test_idxs,
                                                                                                                                               df_comparative_dates_and_reports,
                                                                                                                                               x_ext_train,
                                                                                                                                               x_test,
                                                                                                                                               y_ext_train,
                                                                                                                                               y_test)

        # merge lists into one dataframe
        ext_train_data = pd.DataFrame({'reports': x_ext_train, 'global_labels': y_ext_train})
        test_data = pd.DataFrame({'reports': x_test, 'global_labels': y_test})

        # if we only want to perform a binary classification (i.e. stable vs. unstable)
        if binary_classification:
            # remove rows with unknown/not mentioned global label (i.e. where global_label = 3)
            ext_train_data = ext_train_data[ext_train_data.global_labels != 3]
            test_data = test_data[test_data.global_labels != 3]
            # convert all non-stable labels to "unstable" (i.e. merge progression and response into one unique class) to make the task binary
            ext_train_data['global_labels'].replace({2: 1}, inplace=True)
            test_data['global_labels'].replace({2: 1}, inplace=True)

        # unzip pd.Dataframe again
        x_ext_train, y_ext_train = ext_train_data["reports"].tolist(), ext_train_data["global_labels"].tolist()
        x_test, y_test = test_data["reports"].tolist(), test_data["global_labels"].tolist()

        # create and train TF-IDF model
        x_ext_train_embedded, tf_idf_converter = tf_idf_embedding(x_ext_train,
                                                                  preprocessed_vocabulary_size,
                                                                  ngram_range=(1, n_grams),  # choose which n-grams to include
                                                                  percentage_vocab_size=percentage_vocab_size,  # only retain "percentage_vocab_size" most frequent words
                                                                  min_doc_freq=2,
                                                                  max_doc_freq=0.9)

        # use trained tf-idf to embed validation data
        x_test_embedded = tf_idf_converter.transform(x_test).toarray()

        # ------------- begin CLASSIFICATION for test folder-------------
        # train a classification model using best hyperparams from internal CV
        random_forest = RandomForestClassifier(n_estimators=501, max_features=max_features_random_forest)  # define classifier
        random_forest.fit(x_ext_train_embedded, y_ext_train)  # train
        y_pred_test = random_forest.predict(x_test_embedded)  # compute (thresholded) predictions

        # append to external lists
        y_test_across_folds.append(y_test)
        y_pred_test_across_folds.append(y_pred_test)

        # ERROR ANALYSIS
        if binary_classification:
            # ---------------------------- FALSE POSITIVES ----------------------------
            false_pos_idxs_bool = list(np.logical_and(np.asarray(y_test) != np.asarray(y_pred_test), np.asarray(y_pred_test) == 1))
            false_pos_idxs = [i for i, x in enumerate(false_pos_idxs_bool) if x]
            # if list is not empty
            if false_pos_idxs:
                for idx in range(len(false_pos_idxs)):
                    cnt_fp += 1
                    one_fp_idx = false_pos_idxs[idx]  # take one sample
                    explanation_list_fp = extract_lime_explanation_tf_idf(one_fp_idx, tf_idf_converter, random_forest, x_test, y_test, output_folder, cnt_fp, prediction="FP", embedding=embedding, save=True)
                    all_explanations_fp.append(explanation_list_fp)

            # ---------------------------- FALSE NEGATIVES ----------------------------
            false_neg_idxs_bool = list(np.logical_and(np.asarray(y_test) != np.asarray(y_pred_test), np.asarray(y_pred_test) == 0))
            false_neg_idxs = [i for i, x in enumerate(false_neg_idxs_bool) if x]
            # if list is not empty
            if false_neg_idxs:
                for idx in range(len(false_neg_idxs)):
                    cnt_fn += 1
                    one_fn_idx = false_neg_idxs[idx]  # take one sample
                    explanation_list_fn = extract_lime_explanation_tf_idf(one_fn_idx, tf_idf_converter, random_forest, x_test, y_test, output_folder, cnt_fn, prediction="FN", embedding=embedding, save=True)
                    all_explanations_fn.append(explanation_list_fn)
            # ---------------------------- TRUE POSITIVES ----------------------------
            true_pos_idxs_bool = list(np.logical_and(np.asarray(y_test) == np.asarray(y_pred_test), np.asarray(y_pred_test) == 1))
            true_pos_idxs = [i for i, x in enumerate(true_pos_idxs_bool) if x]
            # if list is not empty
            if true_pos_idxs:
                for idx in range(len(true_pos_idxs)):
                    cnt_tp += 1
                    one_tp_idx = true_pos_idxs[idx]  # take one sample
                    explanation_list_tp = extract_lime_explanation_tf_idf(one_tp_idx, tf_idf_converter, random_forest, x_test, y_test, output_folder, cnt_tp, prediction="TP", embedding=embedding, save=True)
                    all_explanations_tp.append(explanation_list_tp)
                    
            # ---------------------------- TRUE NEGATIVES ----------------------------
            true_neg_idxs_bool = list(np.logical_and(np.asarray(y_test) == np.asarray(y_pred_test), np.asarray(y_pred_test) == 0))
            true_neg_idxs = [i for i, x in enumerate(true_neg_idxs_bool) if x]
            # if list is not empty
            if true_neg_idxs:
                for idx in range(len(true_neg_idxs)):
                    cnt_tn += 1
                    one_tn_idx = true_neg_idxs[idx]  # take one sample
                    explanation_list_tn = extract_lime_explanation_tf_idf(one_tn_idx, tf_idf_converter, random_forest, x_test, y_test, output_folder, cnt_tn, prediction="TN", embedding=embedding, save=True)
                    all_explanations_tn.append(explanation_list_tn)

    # ------------------------------ END OF EXTERNAL CROSS-VALIDATION ------------------------------

    # ----------------------------- ERROR ANALYSIS -----------------------------
    print_most_common_features(all_explanations_fn, predictions="FN", nb_most_common_features=6)
    print_most_common_features(all_explanations_fp, predictions="FP", nb_most_common_features=6)
    print_most_common_features(all_explanations_tp, predictions="TP", nb_most_common_features=6)
    print_most_common_features(all_explanations_tn, predictions="TN", nb_most_common_features=6)
    # --------------------------------------------------------------------------
    # flatten lists
    flat_y_test = [item for sublist in y_test_across_folds for item in sublist]
    flat_y_pred_test = [item for sublist in y_pred_test_across_folds for item in sublist]

    if binary_classification:
        # ------- plot confusion matrix and save results -------
        conf_mat, acc, rec, spec, prec, npv, f1 = classification_metrics(np.asarray(flat_y_test), np.asarray(flat_y_pred_test), binary=binary_classification)
        plot_conf_matrix_and_results(conf_mat, acc, rec, spec, prec, npv, f1)
    else:
        # ------- save general metrics to external dict -------
        conf_mat, acc, rec, spec, prec, npv, f1 = classification_metrics(np.asarray(flat_y_test), np.asarray(flat_y_pred_test), binary=binary_classification)
        plot_conf_matrix_and_results(conf_mat, acc, rec, spec, prec, npv, f1)

    # ---------------------------------------------------------------------------------------------
    end = time.time()  # stop timer
    print_running_time(start, end, "Running")

    # make a beep when script finishes
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))  # outputs a beep at the end of the script


def main():
    # INPUT ARGUMENTS THAT MUST BE SET BY THE USER
    output_folder = "/path/to/out/.../folder/"  # type: str # path where we save output files
    df_comparative_dates_and_reports_path = "/path/to/csvfile/with/comparative/dates/df_comparative_dates_and_reports.csv" # type: str
    annotated_reports_chunk_1_path = "/path/to/annotation/jsonfile/created/with/dataturks/annotations.json" # type: str
    
    # TUNABLE ARGS
    # n_grams = [3, 4, 5]  # type: list # n_grams to be included during the embedding
    # percentage_vocab_size = [0.9, None]  # type: list # percentage of vocabulary size to retain; if None, all words are retained
    # max_features_random_forest = [0.8, 1.0]  # type: list # it corresponds to the percentage of features retained by the random forest classifier

    # SET BEST HYPERPARAMS OBTAINED FROM INTERNAL CV
    n_grams = 3  # n_grams to be included during the embedding
    percentage_vocab_size = 0.9  # percentage of vocabulary size to retain; if None, all words are retained
    max_features_random_forest = 0.8  # it corresponds to the percentage of features retained by the random forest classifier

    # FIXED ARGS
    cv_folds = 5  # number of cross-validation folds
    embedding_label = "tf_idf"
    remove_proper_nouns = True
    remove_custom_words = True
    lowercase = True
    remove_punctuation = True
    from_indication_onward = True
    from_description_onward = False
    binary_classification = True
    
    # load reports and dataframe with info about the sessions
    df_reports = load_data_and_merge(annotated_reports_chunk_1_path)
    df_comparative_dates_and_reports = pd.read_csv(df_comparative_dates_and_reports_path)  # type: pd.DataFrame
    
    tf_idf_classification_lime(df_reports,
                               df_comparative_dates_and_reports,
                               max_features_random_forest,
                               n_grams,
                               percentage_vocab_size,
                               cv_folds,
                               output_folder,
                               remove_proper_nouns,
                               remove_custom_words,
                               lowercase,
                               remove_punctuation,
                               from_indication_onward,
                               from_description_onward,
                               binary_classification,
                               embedding=embedding_label)


if __name__ == '__main__':
    main()
