import numpy as np
import nltk

def get_word_list(text) :
    word_list = []
    word_list.extend(nltk.word_tokenize(text.lower()))

    #Removed some punctuation (there is a more sophisticated way to do this! Need a tokenizer which will remove punctuation and common words)
    vocab = [i for i in vocab if (i != "!" and i != "," and i != "." and i != "(" and i != ")" and i != ";" and i != ":")]
    
    return world_list

def create_dictionary(abstracts) :
    dictionary = dict()
    i = 0
    for abstract in abstracts :
        abstract_list = get_word_list(abstract)
        for word in abstract_list :
            if word not in dictionary :
                dictionary[word] = i
                i += 1
    return dictionary

def create_matrix(abstracts, dictionary) :
    n = len(abstracts)
    m = len(word_dictionary)

    matrix = np.zeros((n,m), dtype=int)
    for i in range(n) :
        words = get_word_list(abstracts[i])
        for word in words :
            if word in dictionary:
                matrix[i, dictionary[word]] += 1
    return matrix


def fit_naive_bayes(train_matrix, train_labels) :
    





def pre_process(train_input_path, dictionary_path, train_matrix_path) :
    abstracts = []
    with open(train_path) as train_file :
        reader = csv.reader(train_file, delimiter=',')
        for row in reader :
            abstracts.extend(f'{row[2]}')

    dictionary = create_dictionary(abstracts)
    matrix = create_matrix(abstracts, dictionary)

    with open(dictionary_path,'w') as f :
        f.write(','.join(dictionary))

    with open(train_matrix_path, 'w') as f :
        f.write(','.join(matrix))

    return matrix, dictionary