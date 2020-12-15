import numpy as np
import csv
from sklearn.svm import SVC

import preprocess

def fit_svm_linear(train_matrix, train_labels) :
    # train_matrix shape = number of abstracts, number of words
    # train labels size = number of abstracts
    model = SVC(kernel = 'linear')
    model.fit(train_matrix, train_labels)

    return model

def fit_svm_polynomial(train_matrix, train_labels) :
    # train_matrix shape = number of abstracts, number of words
    # train labels size = number of abstracts
    model = SVC(kernel = 'poly', degree=2)
    model.fit(train_matrix, train_labels)

    return model

def fit_svm_gaussian(train_matrix, train_labels) :
    # train_matrix shape = number of abstracts, number of words
    # train labels size = number of abstracts
    model = SVC(kernel = 'rbf')
    model.fit(train_matrix, train_labels)

    return model

def fit_svm_sigmoid(train_matrix, train_labels) :
    # train_matrix shape = number of abstracts, number of words
    # train labels size = number of abstracts
    model = SVC(kernel = 'sigmoid')
    model.fit(train_matrix, train_labels)

    return model

def predict_svm(model, test_matrix) :

    pred = model.predict(test_matrix)

    return pred

def get_labels(labels_path, label_row) :
    print("GETTING LABELS")
    labels = []
    with open(labels_path) as labels_file :
        reader = csv.reader(labels_file, delimiter=',')
        for row in reader :
            labels.append(int((f'{row[label_row]}')))
    print("RETURNING LABELS AS NUMPY ARRAY")
    return np.asarray(labels, dtype=int)

def main() :

    test_label_path = 'dataset/test_labels.csv'
    train_label_path = 'dataset/train_labels.csv'
  
    results = []

    train_matrix_count, train_matrix_tfidf, test_matrix_count, test_matrix_tfidf = preprocess.get_matrices()

    pred_paths_start = ['output/svm_lin_pred_', 'output/svm_gauss_pred_', 'output/svm_sig_pred_']
    pred_count_path_end = 'count.txt'
    pred_tfidf_path_end = 'tfidf.txt'

    svm_functions = [fit_svm_linear, fit_svm_gaussian, fit_svm_sigmoid]
    topics = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative_Biology", "Quantitative_Finance"]

    for j in range(len(svm_functions)) :
        print("svm function: ", j)

        pred_all_count = []
        pred_all_tfidf = []

        for i in range(len(topics)) :
            print("topic: ", topics[i])

            train_labels = get_labels(train_label_path, i)

            print("modelling with count")
            svm_model_count = svm_functions[j](train_matrix_count, train_labels)
            print("predicting with count")
            pred_count = predict_svm(svm_model_count, test_matrix_count)
            print("predictions to array")
            pred_all_count.append(pred_count)

            print("modelling with tfidf")
            svm_model_tfidf = svm_functions[j](train_matrix_tfidf, train_labels)
            print("predicting with tfidf")
            pred_tfidf = predict_svm(svm_model_tfidf, test_matrix_tfidf)
            print("predictions to array")
            pred_all_tfidf.append(pred_tfidf)

        print("making count numpy array")
        np_pred_count = (np.asarray(pred_all_count, dtype=int)).T
        print("making count pred path")
        pred_count_path = pred_paths_start[j] + pred_count_path_end
        print("saving pred_count with numpy")
        np.savetxt(pred_count_path, np_pred_count, fmt='%i',delimiter=',')

        print("making tfidf numpy array")
        np_pred_tfidf = (np.asarray(pred_all_tfidf, dtype=int)).T
        print("making tfidf pred path")
        pred_tfidf_path = pred_paths_start[j] + pred_tfidf_path_end
        print("saving pred_count with numpy")
        np.savetxt(pred_tfidf_path, np_pred_tfidf, fmt='%i',delimiter=',')

def test_main() :

    tiny = True;

    train_data_path = 'dataset/train_data.csv'
    train_label_path = 'dataset/train_labels.csv'
    test_data_path = 'dataset/test_data.csv'

    dictionary_path = 'output/dictionary.csv'
    train_matrix_count_path = 'output/train_matrix_count.txt'
    train_matrix_tfidf_path = 'output/train_matrix_tfidf.txt'
    test_matrix_count_path = 'output/test_matrix_count.txt'
    test_matrix_tfidf_path = 'output/test_matrix_tfidf.txt'

    pred_paths_start = ['output/svm_lin_pred_', 'output/svm_poly_pred_', 'output/svm_gauss_pred_', 'output/svm_sig_pred_']
    pred_count_path_end = 'count.txt'
    pred_tfidf_path_end = 'tfidf.txt'

    if tiny == True :
        train_data_path = 'tinydataset/train_data.csv'
        train_label_path = 'tinydataset/train_labels.csv'
        test_data_path = 'tinydataset/test_data.csv'

        dictionary_path = 'tinyoutput/dictionary.csv'
        train_matrix_count_path = 'tinyoutput/train_matrix_count.txt'
        train_matrix_tfidf_path = 'tinyoutput/train_matrix_tfidf.txt'
        test_matrix_count_path = 'tinyoutput/test_matrix_count.txt'
        test_matrix_tfidf_path = 'tinyoutput/test_matrix_tfidf.txt'

        pred_paths_start = ['tinyoutput/svm_lin_pred_', 'tinyoutput/svm_poly_pred_', 'tinyoutput/svm_gauss_pred_', 'tinyoutput/svm_sig_pred_']        

    topics = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative_Biology", "Quantitative_Finance"]
    svm_functions = [fit_svm_linear, fit_svm_polynomial, fit_svm_gaussian, fit_svm_sigmoid]
    
    for j in range(len(svm_functions)) :

        pred_all_count = []
        pred_all_tfidf = []

        for i in range(len(topics)) :

            train_labels = get_labels(train_label_path, i)
            train_matrix_count = np.genfromtxt(train_matrix_count_path, dtype=int, delimiter=',')
            test_matrix_count = np.genfromtxt(test_matrix_count_path, dtype=int, delimiter=',')

            svm_model_count = fit_svm_linear(train_matrix_count, train_labels)
            pred_count = predict_svm(svm_model_count, test_matrix_count)
            pred_all_count.append(pred_count)

            train_matrix_tfidf = np.genfromtxt(train_matrix_tfidf_path, dtype=float, delimiter=',')
            test_matrix_tfidf = np.genfromtxt(test_matrix_tfidf_path, dtype=float, delimiter=',')

            svm_model_tfidf = fit_svm_linear(train_matrix_tfidf, train_labels)
            pred_tfidf = predict_svm(svm_model_tfidf, test_matrix_tfidf)
            pred_all_tfidf.append(pred_tfidf)

        np_pred_count = (np.asarray(pred_all_count, dtype=int)).T
        pred_count_path = pred_paths_start[j] + pred_count_path_end
        np.savetxt(pred_count_path, np_pred_count, fmt='%i',delimiter=',')

        np_pred_tfidf = (np.asarray(pred_all_tfidf, dtype=int)).T
        pred_tfidf_path = pred_paths_start[j] + pred_tfidf_path_end
        np.savetxt(pred_tfidf_path, np_pred_tfidf, fmt='%i',delimiter=',')

if __name__ == "__main__":
    main()
