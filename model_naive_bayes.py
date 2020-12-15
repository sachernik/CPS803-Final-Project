import numpy as np
import csv

def fit_naive_bayes(train_matrix, train_labels) :
    print("fitting model")
    # train_matrix shape = number of abstracts, number of words
    # train labels size = number of abstracts
    num_abstracts, num_words = train_matrix.shape

    #print("calculating log_phi_y1 and log_phi_y0")
    log_phi_y1 = (np.sum(train_labels)+1)/(num_abstracts+2)
    log_phi_y0 = np.log(1 - log_phi_y1)
    log_phi_y1 = np.log(log_phi_y1)

    #print("calculating log_phi1")
    denom = np.sum(np.sum(train_matrix, axis=1)*train_labels) + num_words
    log_phi1 = np.log((np.sum((train_matrix.T * train_labels), axis=1)+1)/denom)

    #print("calculating log_phi0")
    denom = np.sum(np.sum(train_matrix, axis=1)*(1-train_labels)) + num_words
    log_phi0 = np.log((np.sum((train_matrix.T * (1-train_labels)), axis=1)+1)/denom)

    return log_phi_y1, log_phi_y0, log_phi1, log_phi0

def predict_naive_bayes(model, test_matrix) :
    print("predicting")
    log_phi_y1, log_phi_y0, log_phi1, log_phi0 = model
    num_abstracts, num_words = test_matrix.shape

    #print("calculating log_p1")
    log_p1 = np.sum((test_matrix*log_phi1), axis=1) + log_phi_y1
    #print("calculating log_p0")
    log_p0 = np.sum((test_matrix*log_phi0), axis=1) + log_phi_y0

    #print("calculating pred")
    pred = np.array(log_p1 > log_p0, dtype=int)

    return pred

def get_labels(labels_path, label_col) :
    #print("getting labels")
    labels = []
    with open(labels_path) as labels_file :
        reader = csv.reader(labels_file, delimiter=',')
        for row in reader :
            labels.append(int((f'{row[label_col]}')))
    #print("done getting labels, returning numpy array")
    return np.asarray(labels, dtype=int)

def run_model(train_matrix_count, train_matrix_tfidf, test_matrix_count, test_matrix_tfidf, all_train_labels) :

    #pred_count_path = 'output/nb_pred_count.txt'
    #pred_tfidf_path = 'output/nb_pred_tfidf.txt'        

    topics = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative_Biology", "Quantitative_Finance"]

    pred_all_count = []
    pred_all_tfidf = []

    for i in range(len(topics)) :

        #print('running nb for topic: ', topics[i])

        train_labels = all_train_labels[i]

        #print('modelling for count')
        nb_model_count = fit_naive_bayes(train_matrix_count, train_labels)
        pred_count = predict_naive_bayes(nb_model_count, test_matrix_count)
        pred_all_count.append(pred_count)

        #print('modelling for tfidf')
        nb_model_tfidf = fit_naive_bayes(train_matrix_tfidf, train_labels)
        pred_tfidf = predict_naive_bayes(nb_model_tfidf, test_matrix_tfidf)
        pred_all_tfidf.append(pred_tfidf)

    #print("getting np_pred_count")
    np_pred_count = (np.asarray(pred_all_count, dtype=int)).T
    #print("saving pred_count")
    #np.savetxt(pred_count_path, np_pred_count, fmt='%i',delimiter=',')

    #print("getting np_pred_tfidf")
    np_pred_tfidf = (np.asarray(pred_all_tfidf, dtype=int)).T
    #print("saving pred_tfidf")
    #np.savetxt(pred_tfidf_path, np_pred_tfidf, fmt='%i',delimiter=',')

    return np_pred_count, np_pred_tfidf

def main() :

    tiny = False;

    train_data_path = 'dataset/train_data.csv'
    train_label_path = 'dataset/train_labels.csv'
    test_data_path = 'dataset/test_data.csv'

    dictionary_path = 'output/dictionary.csv'
    train_matrix_count_path = 'output/train_matrix_count.txt'
    train_matrix_tfidf_path = 'output/train_matrix_tfidf.txt'
    test_matrix_count_path = 'output/test_matrix_count.txt'
    test_matrix_tfidf_path = 'output/test_matrix_tfidf.txt'

    pred_count_path = 'output/nb_pred_count.txt'
    pred_tfidf_path = 'output/nb_pred_tfidf.txt'

    if tiny == True :
        train_data_path = 'tinydataset/train_data.csv'
        train_label_path = 'tinydataset/train_labels.csv'
        test_data_path = 'tinydataset/test_data.csv'

        dictionary_path = 'tinyoutput/dictionary.csv'
        train_matrix_count_path = 'tinyoutput/train_matrix_count.txt'
        train_matrix_tfidf_path = 'tinyoutput/train_matrix_tfidf.txt'
        test_matrix_count_path = 'tinyoutput/test_matrix_count.txt'
        test_matrix_tfidf_path = 'tinyoutput/test_matrix_tfidf.txt'

        pred_count_path = 'tinyoutput/nb_pred_count.txt'
        pred_tfidf_path = 'tinyoutput/nb_pred_tfidf.txt'        

    topics = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative_Biology", "Quantitative_Finance"]

    pred_all_count = []
    pred_all_tfidf = []

    for i in range(len(topics)) :

        print('running for topic: ', topics[i])

        train_labels = get_labels(train_label_path, i)

        print("getting matrixes for count")
        train_matrix_count = np.genfromtxt(train_matrix_count_path, dtype=int, delimiter=',')
        test_matrix_count = np.genfromtxt(test_matrix_count_path, dtype=int, delimiter=',')

        print('modelling for count')
        nb_model_count = fit_naive_bayes(train_matrix_count, train_labels)
        pred_count = predict_naive_bayes(nb_model_count, test_matrix_count)
        pred_all_count.append(pred_count)

        print("getting matrixes for tfidf")
        
        train_matrix_tfidf = np.genfromtxt(train_matrix_tfidf_path, dtype=float, delimiter=',')
        test_matrix_tfidf = np.genfromtxt(test_matrix_tfidf_path, dtype=float, delimiter=',')

        print('modelling for tfidf')
        nb_model_tfidf = fit_naive_bayes(train_matrix_tfidf, train_labels)
        pred_tfidf = predict_naive_bayes(nb_model_tfidf, test_matrix_tfidf)
        pred_all_tfidf.append(pred_tfidf)

    print("getting np_pred_count")
    np_pred_count = (np.asarray(pred_all_count, dtype=int)).T
    print("saving pred_count")
    np.savetxt(pred_count_path, np_pred_count, fmt='%i',delimiter=',')

    print("getting np_pred_tfidf")
    np_pred_tfidf = (np.asarray(pred_all_tfidf, dtype=int)).T
    print("saving pred_tfidf")
    np.savetxt(pred_tfidf_path, np_pred_tfidf, fmt='%i',delimiter=',')

if __name__ == "__main__":
    main()
