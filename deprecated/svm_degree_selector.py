import numpy as np
import csv
from sklearn.svm import SVC

import preprocess

def fit_svm_polynomial(train_matrix, train_labels, degree) :
    # train_matrix shape = number of abstracts, number of words
    # train labels size = number of abstracts
    model = SVC(kernel = 'poly', degree=degree)
    model.fit(train_matrix, train_labels)

    return model

def predict_svm(model, test_matrix) :

    pred = model.predict(test_matrix)

    return pred

def get_labels(labels_path, label_row) :
    labels = []
    with open(labels_path) as labels_file :
        reader = csv.reader(labels_file, delimiter=',')
        for row in reader :
            labels.append(int((f'{row[label_row]}')))
    return np.asarray(labels, dtype=int)

def main() :

    test_label_path = 'dataset/test_labels.csv'
    train_label_path = 'dataset/train_labels.csv'
    results_path = 'output/svm_degree_results.txt'
    topics = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative_Biology", "Quantitative_Finance"]
    
    results = []

    train_matrix, _, test_matrix, _ = preprocess.get_matrices()

    for j in range(2,11) :
        s = "".join(['SVM with polynomial degree ', str(j)])
        print(s)
        results.append(s)

        for i in range(len(topics)) :
            
            train_labels = get_labels(train_label_path, i)
            test_labels = get_labels(test_label_path, i)

            svm_model = fit_svm_polynomial(train_matrix, train_labels, j)
            pred_labels = predict_svm(svm_model, test_matrix)

            accuracy = np.mean(test_labels == pred_labels) * 100
            s = "".join(["    ", topics[i], ": ", str(accuracy), "%"])
            print(s)
            results.append(s)

    with open(results_path, 'w') as f:
        for line in results:
            f.write("%s\n" % line)

def test_main() :

    tiny = False;

    train_data_path = 'dataset/train_data.csv'
    train_label_path = 'dataset/train_labels.csv'
    test_data_path = 'dataset/test_data.csv'
    test_label_path = 'dataset/test_labels.csv'

    dictionary_path = 'output/dictionary.csv'
    train_matrix_count_path = 'output/train_matrix_count.txt'
    test_matrix_count_path = 'output/test_matrix_count.txt'

    results_path = 'output/svm_degree_results.txt'

    if tiny == True :
        train_data_path = 'tinydataset/train_data.csv'
        train_label_path = 'tinydataset/train_labels.csv'
        test_data_path = 'tinydataset/test_data.csv'
        test_label_path = 'tinydataset/test_labels.csv'

        dictionary_path = 'tinyoutput/dictionary.csv'
        train_matrix_count_path = 'tinyoutput/train_matrix_count.txt'
        test_matrix_count_path = 'tinyoutput/test_matrix_count.txt'

        results_path = 'tinyoutput/svm_degree_results.txt'

    topics = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative_Biology", "Quantitative_Finance"]

    results = []

    for j in range(2,11) :
        s = "".join(['SVM with polynomial degree ', str(j)])
        print(s)
        results.append(s)

        for i in range(len(topics)) :
            
            train_labels = get_labels(train_label_path, i)
            test_labels = get_labels(test_label_path, i)
            train_matrix_count = np.genfromtxt(train_matrix_count_path, dtype=int, delimiter=',')
            test_matrix_count = np.genfromtxt(test_matrix_count_path, dtype=int, delimiter=',')

            svm_model = fit_svm_polynomial(train_matrix_count, train_labels, j)
            pred_labels = predict_svm(svm_model, test_matrix_count)

            accuracy = np.mean(test_labels == pred_labels) *100
            s = "".join(["    ", topics[i], ": ", str(accuracy), "%"])
            print(s)
            results.append(s)

    np_results = np.asarray(results)
    np.savetxt(results_path, np_results)

if __name__ == "__main__":
    main()
