import csv
import numpy as np
import nltk

def num_words(abstract) :
    word_list = nltk.word_tokenize(abstract)
    return len(word_list)

def get_counts(abstracts) :
    counts = [];
    for abstract in abstracts :
        counts.append(num_words(abstract))
    return np.asarray(counts)

def get_abstracts(data_path) :
    abstracts = []
    with open(data_path) as data_file :
        reader = csv.reader(data_file, delimiter=',')
        for row in reader :
            abstracts.append(f'{row[1]}')
    return abstracts

def get_stats(counts) :
    return np.mean(counts), np.std(counts), np.var(counts), np.amin(counts), np.amax(counts)

def main() :
    data_path = 'dataset/train_data.csv'

    abstracts = get_abstracts(data_path)
    counts = get_counts(abstracts)
    mean, std, var, least, most = get_stats(counts)

    print("Mean length of abstracts: ", mean)
    print("Standard deviation of length of abstracts: ", std)  
    print("Variance of length of abstracts: ", var)
    print("Number of words in shortest abstract: ", least)
    print("Number of words in longest abstract: ", most)


if __name__ == "__main__":
    main()