import numpy as np
import nltk
#nltk.download('stopwords')
import csv
import string
import re
import time

def create_stopwords() :

    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(list(string.punctuation))

    return stopwords

def get_word_list(text, stopwords) :
    #print(text)
    word_list = [w.lower() for w in nltk.word_tokenize(text)]
    #word_list.extend(nltk.word_tokenize(text.lower()))
    #print(word_list) 

    word_list = [word for word in word_list if not word in stopwords]
    word_list = [word for word in word_list if (re.search("'", word) == None)]
    # find a way to take this ^ out of this function? For cleanliness
    # also consider changing, what if there are significant contractions which are meaningful?
    
    return word_list

def create_dictionary(abstracts, stopwords) :

    print("creating dictionary")

    dictionary = dict()
    for abstract in abstracts :
        abstract_arr = np.unique(get_word_list(abstract, stopwords))
        for word in abstract_arr :
            if word in dictionary :
                dictionary[word] += 1
            else :
                dictionary[word] = 1

    print("    dictionary size before removing rare words: ", len(dictionary))

    dictionary = {word:count for word, count in dictionary.items() if count >= 5}

    print("    dictionary size after removing rare words: ", len(dictionary))

    i = 0
    for word in dictionary :
        dictionary[word] = i
        i+=1
    #print(dictionary)
    return dictionary

# consider redoing dictionary as a numpy array?
def create_matrix(abstracts, dictionary, stopwords) :
    n = len(abstracts)
    m = len(dictionary)

    matrix = np.zeros((n,m), dtype=int)
    for i in range(n) :
        words = get_word_list(abstracts[i], stopwords)
        for word in words :
            if word in dictionary:
                matrix[i, dictionary[word]] += 1
    return matrix

def pre_process_train(train_data_path, dictionary_path, train_matrix_path, stopwords) :

    abstracts = []
    with open(train_data_path) as train_data_file :
        reader = csv.reader(train_data_file, delimiter=',')
        for row in reader :
            abstracts.append(f'{row[1]}')

    dictionary = create_dictionary(abstracts, stopwords)
    matrix = create_matrix(abstracts, dictionary, stopwords)

    with open(dictionary_path, 'w') as dict_file :
        writer = csv.writer(dict_file, delimiter=',')
        for word, index in dictionary.items() :
            writer.writerow([word, index])
    np.savetxt(train_matrix_path, matrix, fmt='%i')

    return dictionary, matrix

def fit_naive_bayes_vectorized(train_matrix, train_labels) :
    # train_matrix shape = number of abstracts, number of words
    # train labels size = number of abstracts
    num_abstracts, num_words = train_matrix.shape

    log_phi_y1 = (np.sum(train_labels)+1)/(num_abstracts+2)
    log_phi_y0 = np.log(1 - log_phi_y1)
    log_phi_y1 = np.log(log_phi_y1)

    denom = np.sum(np.sum(train_matrix, axis=1)*train_labels) + num_words
    log_phi1 = np.log((np.sum((train_matrix.T * train_labels), axis=1)+1)/denom)

    denom = np.sum(np.sum(train_matrix, axis=1)*(1-train_labels)) + num_words
    log_phi0 = np.log((np.sum((train_matrix.T * (1-train_labels)), axis=1)+1)/denom)

    return log_phi_y1, log_phi_y0, log_phi1, log_phi0

def predict_naive_bayes_vectorized(model, test_matrix) :
    log_phi_y1, log_phi_y0, log_phi1, log_phi0 = model
    num_abstracts, num_words = test_matrix.shape

    log_p1 = np.sum((test_matrix*log_phi1), axis=1) + log_phi_y1
    log_p0 = np.sum((test_matrix*log_phi0), axis=1) + log_phi_y0

    pred = np.array(log_p1 > log_p0, dtype=int)

    return pred

def get_labels(labels_path, label_row) :
    labels = []
    with open(labels_path) as labels_file :
        reader = csv.reader(labels_file, delimiter=',')
        for row in reader :
            labels.append(int((f'{row[label_row]}')))
    return np.asarray(labels, dtype=int)


def pre_process_test(test_data_path, test_matrix_path, dictionary, stopwords) :
    abstracts = []
    with open(test_data_path) as test_data_file :
        reader = csv.reader(test_data_file, delimiter=',')
        for row in reader :
            abstracts.append(f'{row[1]}')
    matrix = create_matrix(abstracts, dictionary, stopwords)
    np.savetxt(test_matrix_path, matrix, fmt='%i')

    return matrix

def create_conf_matrix(test_labels, pred_labels) :

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range (len(test_labels)) :
        if test_labels[i] == 0 :
            if pred_labels[i] == 0 :
                TN +=1
            else :
                FP +=1
        else :
            if pred_labels[i] == 0 :
                FN +=1
            else :
                TP +=1

    return TP, TN, FP, FN

def print_conf_matrix(confusion_matrix, output_path) :

    TP, TN, FP, FN = confusion_matrix

    print("    Number of True Positives: ", TP)
    print("    Number of True Negatives: ", TN)
    print("    Number of False Positives: ", FP)
    print("    Number of False Negatives ", FN)

    if TP == 0 or TN == 0 or FP == 0 or FN == 0 :
        print("    One or more of the values are equal to 0!")
        return

    accuracy = round(100 * (TP+TN)/(TP+TN+FP+FN), 2)
    MR = round(100 * (FP+FN)/(TP+TN+FP+FN), 2)
    TPR = round(100 * TP/(TP+FN), 2)
    FPR = round(100 * FP/(TN+FP), 2)
    TNR = round(100 * TN/(TN+FP), 2)
    FNR = round(100 * FN/(TP+FN), 2)
    precisionP = round(100 * TP/(TP+FP), 2)
    precisionN = round(100 * TN/(TN+FN), 2)
    prevalenceP = round(100 * (FN+TP)/(TP+TN+FP+FN), 2)
    prevalenceN = round(100 * (TN+FP)/(TP+TN+FP+FN), 2)

        
    print("    Overall Accuracy: ", accuracy)
    print("    Misclassification Rate: ", MR)
    print("    True positive rate: ", TPR)
    print("    False positive rate: ", FPR)
    print("    True negative rate: ", TNR)
    print("    False negative rate: ", FNR)
    print("    Precision for positives: ", precisionP)
    print("    Precision for negatives: ", precisionN)
    print("    Prevalence of positives: ", prevalenceP)
    print("    Prevalence of negatives: ", prevalenceN)


def make_path(category, item) :
    return "".join(['output/', item, '_', category, '.txt'])

def test_main() :

    train_data_path = 'dataset/train_data.csv'
    train_label_path = 'dataset/train_labels.csv'
    test_data_path = 'dataset/test_data.csv'
    test_label_path = 'dataset/test_labels.csv'

    dictionary_path = 'output/dictionary.csv'
    train_matrix_path = 'output/train_matrix.txt'
    test_matrix_path = 'output/test_matrix.txt'

    label_row_dict = {
        "Computer_Science" : 0,
        "Physics" : 1,
        "Mathematics" : 2,
        "Statistics" : 3,
        "Quantitative_Biology" : 4,
        "Quantitative_Finance" : 5
    }

    timer_start = time.perf_counter()

    stopwords = create_stopwords()
    dictionary, train_matrix = pre_process_train(train_data_path, dictionary_path, train_matrix_path, stopwords)
    test_matrix = pre_process_test(test_data_path, test_matrix_path, dictionary, stopwords)

    print("Time for preprocessing: ", round(time.perf_counter()-timer_start, 2), " seconds")

    for category in label_row_dict :
        train_labels = get_labels(train_label_path, label_row_dict[category])

        nb_model = fit_naive_bayes_vectorized(train_matrix, train_labels)

        test_labels = get_labels(test_label_path, label_row_dict[category])
        
        pred_labels = predict_naive_bayes_vectorized(nb_model, test_matrix)
   
        np.savetxt(make_path(category, "pred_labels"), pred_labels, fmt='%i')

        accuracy = np.mean(test_labels == pred_labels) * 100
        print('Naive Bayes accuracy for', category, ":", accuracy, "%")

        print_conf_matrix(create_conf_matrix(test_labels, pred_labels), make_path(category, "conf_matrix"))


if __name__ == "__main__":
    test_main()
