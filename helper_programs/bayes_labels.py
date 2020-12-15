import numpy as np

def preprocess_labels(label_path, label_col) :
    all_labels = np.genfromtxt(label_path, dtype=int, delimiter=',');
    cur_labels = all_labels[:, label_col]
    oth_labels = np.delete(all_labels, label_col, axis=1)
    return cur_labels, oth_labels

def bayes(cur_labels, oth_labels) :
    data_size = cur_labels.size
    prob_c = np.sum(cur_labels)/data_size
    print('prob_c: ')
    prob_x = np.sum(oth_labels, axis = 0)/data_size
    prob_x_c = np.sum((oth_labels.T * cur_labels),axis=1)/np.sum(cur_labels)
    prob_c_x = (prob_x_c * prob_c)/prob_x
    return prob_c_x

def main() :

    label_path = 'tinydataset/test_labels.csv'

    label_col_dict = {
        "Computer_Science" : 0,
        "Physics" : 1,
        "Mathematics" : 2,
        "Statistics" : 3,
        "Quantitative_Biology" : 4,
        "Quantitative_Finance" : 5
    }

    for category in label_col_dict :
        cur_labels, oth_labels = preprocess_labels(label_path, label_col_dict[category])
        prob = bayes(cur_labels, oth_labels)

        print("Probability of ", category, " given each other label:")
        print(prob)

if __name__ == "__main__":
    main()
