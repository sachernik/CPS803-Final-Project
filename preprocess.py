import numpy as np
import nltk
#nltk.download('stopwords')
import csv
import string
import re

# try from nltk.corpus import stopwords??

def create_stopwords() :

    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(list(string.punctuation))

    return stopwords

def get_word_list(text, stopwords) :
    word_list = [w.lower() for w in nltk.word_tokenize(text)]

    word_list = [word for word in word_list if not word in stopwords]
    word_list = [word for word in word_list if (re.search("'", word) == None)]
    # find a way to take this ^ out of this function? For cleanliness
    # also consider changing, what if there are significant contractions which are meaningful?
    
    return word_list

def create_dictionary(abstracts, stopwords) :

    dictionary = dict()
    for abstract in abstracts :
        abstract_arr = np.unique(get_word_list(abstract, stopwords))
        for word in abstract_arr :
            if word in dictionary :
                dictionary[word] += 1
            else :
                dictionary[word] = 1

    dictionary = {word:count for word, count in dictionary.items() if count >= 5}

    num_abstracts = len(abstracts)
    idf = np.zeros(len(dictionary))
    i = 0
    for word in dictionary :
        idf[i] = np.log(num_abstracts/dictionary[word])
        dictionary[word] = i
        i+=1

    return dictionary, idf

def create_matrices(abstracts, dictionary, stopwords, idf) :
    n = len(abstracts)
    m = len(dictionary)

    matrix_count = np.zeros((n,m),dtype=int)
    for i in range(n) :
        words = get_word_list(abstracts[i], stopwords)
        for word in words :
            if word in dictionary:
                matrix_count[i, dictionary[word]] += 1
    matrix_tfidf = matrix_count * idf
    return matrix_count, matrix_tfidf

def get_abstracts(data_path) :
    abstracts = []
    with open(data_path) as data_file :
        reader = csv.reader(data_file, delimiter=',')
        for row in reader :
            abstracts.append(f'{row[1]}')
    return abstracts

def get_matrices() :

    print("starting get_matrices")

    train_data_path = 'tinydataset/train_data.csv'
    train_label_path = 'tinydataset/train_labels.csv'
    test_data_path = 'tinydataset/test_data.csv'
    test_label_path = 'tinydataset/test_labels.csv'

    print("creating stopwords")
    stopwords = create_stopwords()
    print("getting train abstracts")
    train_abstracts = get_abstracts(train_data_path)
    print("getting test abstracts")
    test_abstracts = get_abstracts(test_data_path)
    print("making dictionary and idf vector")
    dictionary, idf = create_dictionary(train_abstracts, stopwords)
    print("making training matrices")
    train_matrix_count, train_matrix_tfidf = create_matrices(train_abstracts, dictionary, stopwords, idf)
    print("making testing matrices")
    test_matrix_count, test_matrix_tfidf = create_matrices(test_abstracts, dictionary, stopwords, idf)

    print("returning preprocessing output")

    return train_matrix_count, train_matrix_tfidf, test_matrix_count, test_matrix_tfidf

def main() :

    tiny = False;

    train_data_path = 'dataset/train_data.csv'
    train_label_path = 'dataset/train_labels.csv'
    test_data_path = 'dataset/test_data.csv'
    test_label_path = 'dataset/test_labels.csv'

    dictionary_path = 'output/dictionary.csv'
    train_matrix_count_path = 'output/train_matrix_count.txt'
    train_matrix_tfidf_path = 'output/train_matrix_tfidf.txt'
    test_matrix_count_path = 'output/test_matrix_count.txt'
    test_matrix_tfidf_path = 'output/test_matrix_tfidf.txt'

    if tiny == True :
        train_data_path = 'tinydataset/train_data.csv'
        train_label_path = 'tinydataset/train_labels.csv'
        test_data_path = 'tinydataset/test_data.csv'
        test_label_path = 'tinydataset/test_labels.csv'

        dictionary_path = 'tinyoutput/dictionary.csv'
        train_matrix_count_path = 'tinyoutput/train_matrix_count.txt'
        train_matrix_tfidf_path = 'tinyoutput/train_matrix_tfidf.txt'
        test_matrix_count_path = 'tinyoutput/test_matrix_count.txt'
        test_matrix_tfidf_path = 'tinyoutput/test_matrix_tfidf.txt'

    stopwords = create_stopwords()
    train_abstracts = get_abstracts(train_data_path)
    test_abstracts = get_abstracts(test_data_path)
    dictionary, idf = create_dictionary(train_abstracts, stopwords)
    train_matrix_count, train_matrix_tfidf = create_matrices(train_abstracts, dictionary, stopwords, idf)
    test_matrix_count, test_matrix_tfidf = create_matrices(test_abstracts, dictionary, stopwords, idf)

    with open(dictionary_path, 'w') as dict_file :
        writer = csv.writer(dict_file, delimiter=',')
        for word, index in dictionary.items() :
            writer.writerow([word, index])
    np.savetxt(train_matrix_count_path, train_matrix_count, fmt='%i', delimiter=',')
    np.savetxt(train_matrix_tfidf_path, train_matrix_tfidf, fmt='%f', delimiter=',')
    np.savetxt(test_matrix_count_path, test_matrix_count, fmt='%i', delimiter=',')
    np.savetxt(test_matrix_tfidf_path, test_matrix_tfidf, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()