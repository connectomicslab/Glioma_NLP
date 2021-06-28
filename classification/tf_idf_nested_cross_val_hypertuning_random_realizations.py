import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from utils_nlp import print_running_time, load_data_and_merge, write_to_csv, plot_roc_curve, plot_pr_curve,\
    classification_metrics, preprocess_and_extract_labels, adjust_cv_splits_for_multiple_sessions,\
    find_best_hyperparams_tf_idf, tf_idf_embedding, most_frequent, save_pickle_list_to_disk


def tf_idf_classification(df_reports, df_comparative_dates_and_reports, max_features_random_forest, n_grams, percentage_vocab_size, cv_folds, random_realizations,
                          output_folder, remove_proper_nouns, remove_custom_words, lowercase, remove_punctuation, from_indication_onward, from_description_onward, binary_classification=True):
    """This function executes the classification of the reports using the TF-IDF model for embedding and the random forest classifier
    Args:
        df_reports (pd.Dataframe): it contains all the information in the json file created during the manual annotation processing
        df_comparative_dates_and_reports (pd.Dataframe): it contains all the dates (current and previous), labels and corresponding report
        n_grams (list): n-grams to try (tune) in the internal cross-validation
        percentage_vocab_size (list): percentages of vocabulary size to try (tune) in the internal cross-validation
        max_features_random_forest (list): it contains the percentage of features to be retained by the random forest classifier
        cv_folds (int): number of cross-validation folds
        random_realizations (int): number of random realizations (used to average results and compute statistics)
        output_folder (str): path where we save the embedding files
        remove_proper_nouns (bool): if set to True, proper nouns in the reports will be removed
        remove_custom_words (bool): if set to True, specific words related to the reports will be removed
        lowercase (bool): if set to True, all words of the reports will be set to lowercase
        remove_punctuation (bool): if set to True, punctuation will be removed from the reports
        from_indication_onward (bool): if set to True, we will only use the words from the "Indication" section onwards; everything before will be discarded
        from_description_onward (bool): if set to True, we will only use the words from the "Description" section onwards; everything before will be discarded
        binary_classification (bool): if set to True, it means that we will perform a binary classification; defaults to True
    """
    start = time.time()  # start timer; used to compute the time needed to run this script
    date = (datetime.today().strftime('%b_%d_%Y'))  # save today's date
    most_frequent_hyperparams_across_realizations = []
    acc_across_realizations = []
    sens_across_realizations = []
    spec_across_realizations = []
    ppv_across_realizations = []
    npv_across_realizations = []
    f1_across_realizations = []
    fpr_across_realizations = []
    tpr_across_realizations = []
    auc_across_realizations = []
    prec_across_realizations = []
    rec_across_realizations = []
    aupr_across_realizations = []
    for seed in range(random_realizations):
        print("\n--------------------------------------------------------- Random Realization {}".format(seed + 1))

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
        y_pred_proba_test_across_folds = []  # type: list
        best_ngram_across_test_folds = []  # type: list
        best_pvs_across_test_folds = []  # type: list
        best_mf_across_test_folds = []  # type: list
        # ---------------------------------------- BEGIN EXTERNAL CROSS-VALIDATION ------------------------------------
        # since the dataset is very imbalanced, apply stratified cross validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        test_fold = 0  # type: int # counter to keep track of cross-validation fold
        for ext_train_idxs, test_idxs in skf.split(all_preprocessed_reports, global_labels):
            test_fold += 1
            print("\n------------------------------------ Test fold {}".format(test_fold))

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

            # ------------------------------------ BEGIN INTERNAL CROSS-VALIDATION --------------------------------------
            val_fold = 0
            dict_highest_f1 = []
            for int_train_idxs, val_idxs in skf.split(x_ext_train, y_ext_train):
                val_fold += 1
                print("------------------------------ Val fold {}".format(val_fold))

                # create int_train and val datasets
                x_int_train, x_val = [x_ext_train[i] for i in int_train_idxs], [x_ext_train[j] for j in val_idxs]
                y_int_train, y_val = [y_ext_train[i] for i in int_train_idxs], [y_ext_train[j] for j in val_idxs]

                # ensure that multiple sessions of the same sub are not some in int_train and some in val set
                all_reports_internal = [all_reports[i] for i in ext_train_idxs_after_adjust]
                x_int_train, y_int_train, x_val, y_val, _, _ = adjust_cv_splits_for_multiple_sessions(all_reports_internal,
                                                                                                      int_train_idxs,
                                                                                                      val_idxs,
                                                                                                      df_comparative_dates_and_reports,
                                                                                                      x_int_train,
                                                                                                      x_val,
                                                                                                      y_int_train,
                                                                                                      y_val)

                # merge lists into one dataframe (to remove some elements in case we do binary classification)
                int_train_data = pd.DataFrame({'reports': x_int_train, 'global_labels': y_int_train})
                val_data = pd.DataFrame({'reports': x_val, 'global_labels': y_val})

                # if we only want to perform a binary classification (i.e. stable vs. unstable)
                if binary_classification:
                    # remove rows with unknown/not mentioned global label (i.e. where global_label = 3)
                    int_train_data = int_train_data[int_train_data.global_labels != 3]
                    val_data = val_data[val_data.global_labels != 3]
                    # convert all non-stable labels to "unstable" (i.e. merge progression and response into one unique class) to make the task binary
                    int_train_data['global_labels'].replace({2: 1}, inplace=True)
                    val_data['global_labels'].replace({2: 1}, inplace=True)

                # unzip pd.Dataframe again
                x_int_train, y_int_train = int_train_data["reports"].tolist(), int_train_data["global_labels"].tolist()
                x_val, y_val = val_data["reports"].tolist(), val_data["global_labels"].tolist()

                tot_num_combinations = len(n_grams) * len(percentage_vocab_size) * len(max_features_random_forest)  # type: int # compute all hyperparameter combinations
                hyperparam_combinations = 0  # type: int # counter to keep track of hyperparams combinations
                f1_dict_hypertuning = {}
                for n_gram in n_grams:
                    for pvs in percentage_vocab_size:
                        for mf in max_features_random_forest:
                            hyperparam_combinations += 1  # increment counter
                            print("---- Hyperp. combination {}/{}: n_grams={}, percentage_vocab_size={}, max_features={}".format(hyperparam_combinations, tot_num_combinations, n_gram, pvs, mf))
                            # create and train TF-IDF model
                            x_int_train_embedded, tf_idf_converter = tf_idf_embedding(x_int_train,
                                                                                      preprocessed_vocabulary_size,
                                                                                      ngram_range=(1, n_gram),  # choose which n-grams to include
                                                                                      percentage_vocab_size=pvs,  # only retain "percentage_vocab_size" most frequent words
                                                                                      min_doc_freq=2,
                                                                                      max_doc_freq=0.9)

                            # use trained tf-idf to embed validation data
                            x_val_embedded = tf_idf_converter.transform(x_val)

                            # ------------- begin CLASSIFICATION for validation folder -------------
                            # train a classification model
                            random_forest = RandomForestClassifier(n_estimators=501, max_features=mf)  # define classifier
                            random_forest.fit(x_int_train_embedded, y_int_train)  # train
                            y_pred_val = random_forest.predict(x_val_embedded)  # compute (thresholded) predictions

                            if binary_classification:
                                # ------- save general metrics to external dict -------
                                _, _, _, _, _, _, f1 = classification_metrics(np.asarray(y_val), np.asarray(y_pred_val), binary=True)
                                f1_dict_hypertuning["ngram{}_pvs{}_mf{}".format(n_gram, pvs, mf)] = f1
                            else:
                                # ------- save general metrics to external dict -------
                                _, _, rec, _, prec, _, f1 = classification_metrics(np.asarray(y_val), np.asarray(y_pred_val), binary=False)
                                f1_dict_hypertuning["ngram{}_pvs{}_mf{}".format(n_gram, pvs, mf)] = f1

                # --------------------- END OF HYPERTUNING FOR ONE VALIDATION FOLD ----------------
                # extract key corresponding to highest f1 score
                f1_key_max = max(f1_dict_hypertuning, key=f1_dict_hypertuning.get)  # type: str
                dict_highest_f1.append({f1_key_max: f1_dict_hypertuning[f1_key_max]})  # append sub-dict corresponding to highest f1 score

            # ---------------- END OF INTERNAL CV; BEGIN CLASSIFICATION FOR TEST FOLD ---------------
            best_n_gram, best_pvs, best_mf = find_best_hyperparams_tf_idf(dict_highest_f1, n_grams, percentage_vocab_size, max_features_random_forest)
            best_ngram_across_test_folds.append(best_n_gram)
            best_pvs_across_test_folds.append(best_pvs)
            best_mf_across_test_folds.append(best_mf)
            print("\nBest hyperparams: n_gram = {}, percentage_vocab_size = {}, max_features = {}".format(best_n_gram, best_pvs, best_mf))

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
                                                                      ngram_range=(1, best_n_gram),  # choose which n-grams to include
                                                                      percentage_vocab_size=best_pvs,  # only retain "percentage_vocab_size" most frequent words
                                                                      min_doc_freq=2,
                                                                      max_doc_freq=0.9)

            # use trained tf-idf to embed validation data
            x_test_embedded = tf_idf_converter.transform(x_test).toarray()

            # --------- SAVE embeddings to disk for visualization with the "Embedding Projector" -------------
            if seed == 0:  # only save the embeddings for one random iteration otherwise they are too many
                x_test_list = [list(x) for x in x_test_embedded]  # type: list # convert to list for saving the embedding to disk
                y_test_list = [list(str(y)) for y in y_test]  # type: list # convert to list for saving the embedding to disk
                write_to_csv(x_test_list, os.path.join(output_folder, "embeddings", "test_fold_{}".format(test_fold)), "test_vectors_tf_idf_ngrams_{}_pvs_{}_{}.csv".format(best_n_gram, best_pvs, date))
                write_to_csv(y_test_list, os.path.join(output_folder, "embeddings", "test_fold_{}".format(test_fold)), "test_labels_tf_idf_ngrams_{}_pvs_{}_{}.csv".format(best_n_gram, best_pvs, date))

            # ------------- begin CLASSIFICATION for test folder-------------
            # train a classification model using best hyperparams from internal CV
            random_forest = RandomForestClassifier(n_estimators=501, max_features=best_mf)  # define classifier
            random_forest.fit(x_ext_train_embedded, y_ext_train)  # train
            y_pred_test = random_forest.predict(x_test_embedded)  # compute (thresholded) predictions

            if binary_classification:
                y_pred_proba_test = random_forest.predict_proba(x_test_embedded)  # compute predictions
                # keep probabilities for the change class only
                y_pred_proba_test = y_pred_proba_test[:, 1]
                y_pred_proba_test_across_folds.append(y_pred_proba_test)
            # append to external lists
            y_test_across_folds.append(y_test)
            y_pred_test_across_folds.append(y_pred_test)

        # ------------------------------ END OF EXTERNAL CROSS-VALIDATION ------------------------------
        # find most frequent hyperparameters combination across test folds
        most_frequent_ngram = most_frequent(best_ngram_across_test_folds)
        most_frequent_pvs = most_frequent(best_pvs_across_test_folds)
        most_frequent_mf = most_frequent(best_mf_across_test_folds)
        print("\n--------------------------- END OF CROSS-VALIDATION ---------------------------")
        print("Most frequent hyperparams across test folds:\nngram {}, percentage_vocabulary_size {}, max_features {}".format(most_frequent_ngram,
                                                                                                                              most_frequent_pvs,
                                                                                                                              most_frequent_mf))

        most_frequent_hyperparams_across_realizations.append([most_frequent_ngram, most_frequent_pvs, most_frequent_mf])

        # flatten lists
        flat_y_test = [item for sublist in y_test_across_folds for item in sublist]
        save_pickle_list_to_disk(flat_y_test, out_dir=os.path.join(os.path.join(output_folder, "out_curve_values")), out_filename="y_test_tf_idf_random_realization_{}_{}.pkl".format(seed + 1, date))
        flat_y_pred_test = [item for sublist in y_pred_test_across_folds for item in sublist]
        flat_y_pred_proba = [item for sublist in y_pred_proba_test_across_folds for item in sublist]

        if binary_classification:
            # ------- plot ROC curve -------
            fpr, tpr, auc = plot_roc_curve(flat_y_test, flat_y_pred_proba, cv_folds, embedding_label="tf-idf", plot=False)

            # append to external lists
            fpr_across_realizations.append(fpr)
            tpr_across_realizations.append(tpr)
            auc_across_realizations.append(auc)

            # ------- plot PR curve -------
            rec_curve_values, prec_curve_values, aupr = plot_pr_curve(flat_y_test, flat_y_pred_proba, cv_folds, embedding_label="tf-idf", plot=False)

            # append to external lists
            rec_across_realizations.append(rec_curve_values)
            prec_across_realizations.append(prec_curve_values)
            aupr_across_realizations.append(aupr)

            # ------- plot confusion matrix and save results -------
            conf_mat, acc, rec, spec, prec, npv, f1 = classification_metrics(np.asarray(flat_y_test), np.asarray(flat_y_pred_test), binary=binary_classification)

            # append to external lists
            acc_across_realizations.append(acc)
            sens_across_realizations.append(rec)
            spec_across_realizations.append(spec)
            ppv_across_realizations.append(prec)
            npv_across_realizations.append(npv)
            f1_across_realizations.append(f1)

        else:
            # ------- save general metrics to external dict -------
            conf_mat, acc, rec, spec, prec, npv, f1 = classification_metrics(np.asarray(flat_y_test), np.asarray(flat_y_pred_test), binary=binary_classification)

            # append to external lists
            acc_across_realizations.append(acc)
            sens_across_realizations.append(rec)
            spec_across_realizations.append(spec)
            ppv_across_realizations.append(prec)
            npv_across_realizations.append(npv)
            f1_across_realizations.append(f1)

    print("\n------------------------------ END OF ALL RANDOM REALIZATIONS ---------------------------")
    # print most frequent hyperparams across random realizations
    most_frequent_hyperparams_across_realizations_np = np.asarray(most_frequent_hyperparams_across_realizations)
    most_frequent_ngram = most_frequent(list(most_frequent_hyperparams_across_realizations_np[:, 0]))
    most_frequent_pvs = most_frequent(list(most_frequent_hyperparams_across_realizations_np[:, 1]))
    most_frequent_mf = most_frequent(list(most_frequent_hyperparams_across_realizations_np[:, 2]))
    print("\nMost frequent hyperparams across random realizations:\nngram {}, percentage_vocabulary_size {}, max_features {}".format(most_frequent_ngram,
                                                                                                                                     most_frequent_pvs,
                                                                                                                                     most_frequent_mf))
    if binary_classification:
        # save values to disk; we will use them to plot the ROC and PR curves
        save_pickle_list_to_disk(fpr_across_realizations, out_dir=os.path.join(os.path.join(output_folder, "out_curve_values")), out_filename="fpr_values_tf_idf_{}.pkl".format(date))
        save_pickle_list_to_disk(tpr_across_realizations, out_dir=os.path.join(os.path.join(output_folder, "out_curve_values")), out_filename="tpr_values_tf_idf_{}.pkl".format(date))
        save_pickle_list_to_disk(auc_across_realizations, out_dir=os.path.join(os.path.join(output_folder, "out_curve_values")), out_filename="auc_values_tf_idf_{}.pkl".format(date))

        save_pickle_list_to_disk(rec_across_realizations, out_dir=os.path.join(os.path.join(output_folder, "out_curve_values")), out_filename="recall_values_tf_idf_{}.pkl".format(date))
        save_pickle_list_to_disk(prec_across_realizations, out_dir=os.path.join(os.path.join(output_folder, "out_curve_values")), out_filename="prec_values_tf_idf_{}.pkl".format(date))
        save_pickle_list_to_disk(aupr_across_realizations, out_dir=os.path.join(os.path.join(output_folder, "out_curve_values")), out_filename="aupr_values_tf_idf_{}.pkl".format(date))

    print("\n\n-----------------------------------------------------------")
    print("Average test set results over {} realizations:".format(random_realizations))
    print("Accuracy = {:.2f} ± {:.2f}".format(np.mean(acc_across_realizations), np.std(acc_across_realizations)))
    print("Sensitivity (recall) = {:.2f} ± {:.2f}".format(np.mean(sens_across_realizations), np.std(sens_across_realizations)))
    print("Specificity = {:.2f} ± {:.2f}".format(np.mean(spec_across_realizations), np.std(spec_across_realizations)))
    print("Precision (PPV) = {:.2f} ± {:.2f}".format(np.mean(ppv_across_realizations), np.std(ppv_across_realizations)))
    print("NPV = {:.2f} ± {:.2f}".format(np.mean(npv_across_realizations), np.std(npv_across_realizations)))
    print("f1-score = {:.2f} ± {:.2f}".format(np.mean(f1_across_realizations), np.std(f1_across_realizations)))
    print("AUC = {:.2f} ± {:.2f}".format(np.mean(auc_across_realizations), np.std(auc_across_realizations)))
    print("AUPR = {:.2f} ± {:.2f}".format(np.mean(aupr_across_realizations), np.std(aupr_across_realizations)))
    # --------------------------------------------------------------------
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
    n_grams = [3, 4, 5]  # type: list # n_grams to be included during the embedding
    percentage_vocab_size = [0.9, None]  # type: list # percentage of vocabulary size to retain; if None, all words are retained
    max_features_random_forest = [0.8, 1.0]  # type: list # it corresponds to the percentage of features retained by the random forest classifier

    # FIXED ARGS
    random_realizations = 10  # number of random realizations; at each iteration we change the seed of the CV split
    cv_folds = 5  # number of cross-validation folds
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

    tf_idf_classification(df_reports,
                          df_comparative_dates_and_reports,
                          max_features_random_forest,
                          n_grams,
                          percentage_vocab_size,
                          cv_folds,
                          random_realizations,
                          output_folder,
                          remove_proper_nouns,
                          remove_custom_words,
                          lowercase,
                          remove_punctuation,
                          from_indication_onward,
                          from_description_onward,
                          binary_classification)


if __name__ == '__main__':
    main()
