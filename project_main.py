import csv
import numpy as np

import preprocess
import model_naive_bayes
import model_svm_linear
import results_evaluator
import plot_generator

def get_labels(labels_path, num_topics) :

    all_labels = []
    for i in range(num_topics) :

        labels = []
        with open(labels_path) as labels_file :
            reader = csv.reader(labels_file, delimiter=',')
            for row in reader :
                labels.append(int((f'{row[i]}')))
        all_labels.append(labels)

    return np.asarray(all_labels, dtype=int)

def main() :

    train_label_path = 'dataset/train_labels.csv'
    test_label_path = 'dataset/test_labels.csv'

    nb_train_report_path = 'output/nb_report_train.txt'
    nb_test_report_path = 'output/nb_report_test.txt'
    svm_train_report_path = 'output/svm_report_train.txt'
    svm_test_report_path = 'output/svm_report_test.txt'

    nb_plot_path = 'output/nb_plot.jpg'
    svm_plot_path = 'output/svm_plot.jpg'
    test_plot_path = 'output/test_plot.jpg'

    topics = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative_Biology", "Quantitative_Finance"]

    #the topic frequencies in the dataset were acquired by running the code located at 'helper_programs/count_labels.py'
    topic_freq = np.array([41,29,27,25,3,1])

    train_labels = get_labels(train_label_path, len(topics))
    test_labels = get_labels(test_label_path, len(topics))

    # generating the training and testing matrices
    train_matrix_count, train_matrix_tfidf, test_matrix_count, test_matrix_tfidf = preprocess.get_matrices()

    # fitting and predicting with the naive bayes model on the train data
    nb_pred_count, nb_pred_tfidf = model_naive_bayes.run_model(train_matrix_count, train_matrix_tfidf, train_matrix_count, train_matrix_tfidf, train_labels)

    # generating report for the naive bayes model on the train data
    acc_nb_train, fnr_nb_train = results_evaluator.generate_report(nb_train_report_path, train_labels, nb_pred_count, nb_pred_tfidf)

    # fitting and predicting with the naive bayes model on the test data
    nb_pred_count, nb_pred_tfidf = model_naive_bayes.run_model(train_matrix_count, train_matrix_tfidf, test_matrix_count, test_matrix_tfidf, train_labels)

    # generating report for the naive bayes model on the test data
    acc_nb_test, fnr_nb_test = results_evaluator.generate_report(nb_test_report_path, test_labels, nb_pred_count, nb_pred_tfidf)

    # fitting and predicting with the linear support vector classifier on the train data
    svm_pred_count, svm_pred_tfidf = model_svm_linear.run_model(train_matrix_count, train_matrix_tfidf, train_matrix_count, train_matrix_tfidf, train_labels)

    # generating report for the linear support vector model on the train data
    acc_svm_train, fnr_svm_train = results_evaluator.generate_report(svm_train_report_path, train_labels, svm_pred_count, svm_pred_tfidf)

    # fitting and predicting with the linear support vector classifier on the test data
    svm_pred_count, svm_pred_tfidf = model_svm_linear.run_model(train_matrix_count, train_matrix_tfidf, test_matrix_count, test_matrix_tfidf, train_labels)

    # generating report for the linear support vector model on the test data
    acc_svm_test, fnr_svm_test = results_evaluator.generate_report(svm_test_report_path, test_labels, svm_pred_count, svm_pred_tfidf)

    plot_generator.gen_plot(topic_freq, acc_nb_train, fnr_nb_train, acc_nb_test, fnr_nb_test, nb_plot_path)
    plot_generator.gen_plot(topic_freq, acc_svm_train, fnr_svm_train, acc_svm_test, fnr_svm_test, svm_plot_path)
    plot_generator.gen_plot(topic_freq, acc_nb_test, fnr_nb_test, acc_svm_test, fnr_svm_test, test_plot_path)

if __name__ == "__main__":
    main()