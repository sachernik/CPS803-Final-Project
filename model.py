import numpy as np
import nltk
import csv

def get_word_list(text) :
    #print(text)
    word_list = []
    word_list.extend(nltk.word_tokenize(text.lower()))
    #print(word_list)

    #Removed some punctuation (there is a more sophisticated way to do this! Need a tokenizer which will remove punctuation and common words)
    word_list = [i for i in word_list if (i != "!" and i != "," and i != "." and i != "(" and i != ")" and i != ";" and i != ":")]
    
    return word_list

def create_dictionary(abstracts) :
    dictionary = dict()
    i = 0
    for abstract in abstracts :
        abstract_list = get_word_list(abstract)
        for word in abstract_list :
            if word not in dictionary :
                dictionary[word] = i
                i += 1
    #print(dictionary)
    return dictionary

# consider redoing dictionary as a numpy array?
def create_matrix(abstracts, dictionary) :
    n = len(abstracts)
    m = len(dictionary)

    matrix = np.zeros((n,m), dtype=int)
    for i in range(n) :
        words = get_word_list(abstracts[i])
        for word in words :
            if word in dictionary:
                matrix[i, dictionary[word]] += 1
    return matrix


def fit_naive_bayes(train_matrix, train_labels) :
    num_train_abstracts, size_dictionary = train_matrix.shape
    # _, num_labels = train_labels.shape # This is irrelevant right now, as we are working with COMPUTER SCIENCE labels only
    log_phi_y1 = (np.sum(train_labels)+1)/(len(train_labels)+2)
    log_phi_y0 = np.log(1 - log_phi_y1)
    log_phi_y1 = np.log(log_phi_y1)

    denom = 0
    for i in range(num_train_abstracts) :
        if train_labels[i] == 1 :
            denom += np.sum(train_matrix[i])
    laplace = 1/(denom + size_dictionary)

    log_phi1 = np.log(np.full(size_dictionary, laplace))

    for k in range(size_dictionary) :
        val = 0
        for i in range(num_train_abstracts) :
            if train_labels[i] == 1 :
                val += train_matrix[i][k]
        log_phi1[k] = np.log((val+1)/(denom + size_dictionary))

    denom = 0
    for i in range(num_train_abstracts) :
        if train_labels[i] == 0 :
            denom += np.sum(train_matrix[i])
    laplace = 1/(denom + size_dictionary)

    log_phi0 = np.log(np.full(size_dictionary, laplace))

    for k in range(size_dictionary) :
        val = 0
        for i in range(num_train_abstracts) :
            if train_labels[i] == 0 :
                val += train_matrix[i][k]
        log_phi0[k] = np.log((val+1)/(denom + size_dictionary))

    return log_phi_y1, log_phi_y0, log_phi1, log_phi0

def predict_naive_bayes(model, test_matrix) :
    log_phi_y1, log_phi_y0, log_phi1, log_phi0 = model

    num_test_abstracts, size_dictionary = test_matrix.shape
    #print('num test abstracts: ', num_test_abstracts)
    #print('test matrix shape: ', test_matrix.shape)

    pred = np.zeros(num_test_abstracts, dtype=int) # This will need to be multi-dimensional as well, for the different values of y!
    for i in range(num_test_abstracts) :
        log_p1 = log_phi_y1
        log_p0 = log_phi_y0
        for k in range(size_dictionary) :
            log_p1 += log_phi1[k] * test_matrix[i][k]
            log_p0 += log_phi0[k] * test_matrix[i][k]
        if log_p1 > log_p0 :
            pred[i] = 1

    return pred

def get_labels(dataset_input_path, label_row) :
    labels = []
    with open(dataset_input_path) as dataset :
        reader = csv.reader(dataset, delimiter=',')
        for row in reader :
            labels.append(int((f'{row[label_row]}')))
    return labels


def pre_process_train(train_input_path) :
    abstracts = []
    with open(train_input_path) as train_file :
        reader = csv.reader(train_file, delimiter=',')
        for row in reader :
            abstracts.append(f'{row[2]}')

    dictionary = create_dictionary(abstracts)
    matrix = create_matrix(abstracts, dictionary)
    #labels = get_labels(train_input_path)

    """
    with open(dictionary_path,'w') as f : # This is not currently saving the indexes
        f.write(','.join(dictionary))

    #np.savetxt(train_matrix_path, matrix)
    """    

    return matrix, dictionary

def pre_process_test(test_input_path, dictionary) :
    abstracts = []
    with open(test_input_path) as test_file :
        reader = csv.reader(test_file, delimiter=',')
        for row in reader :
            abstracts.append(f'{row[2]}')
    matrix = create_matrix(abstracts, dictionary)
    #labels = get_labels(test_input_path)

    return matrix

def main() :
    train_input_path = 'dataset/train_no_header.csv'
    #dictionary_path = 'data_processing/dictionary.txt'
    #train_matrix_path = 'data_processing/train_matrix.txt'
    test_input_path = 'dataset/test_no_header.csv'

    label_row_dict = {
        'Computer Science' : 3,
        'Physics' : 4,
        'Mathematics' : 5,
        'Statistics' : 6,
        'Quantitative Biology' : 7,
        'Quantitative Finanace' : 8
    }

    train_matrix, dictionary = pre_process_train(train_input_path)
    test_matrix = pre_process_test(test_input_path, dictionary)

    for category in label_row_dict :
        train_labels = get_labels(train_input_path, label_row_dict[category])
        nb_model = fit_naive_bayes(train_matrix, train_labels)

        test_labels = get_labels(test_input_path, label_row_dict[category])
        print('test labels for', category, ":", test_labels)
        pred_labels = predict_naive_bayes(nb_model, test_matrix)
        print('predicted labels for', category, ":", pred_labels)
   
        accuracy = np.mean(test_labels == pred_labels) * 100
        print('Naive Bayes accuracy for', category, ":", accuracy, "%")
    #print(len(pred_labels))
    #print('pred_labels: ', pred_labels)


if __name__ == "__main__":
    main()
