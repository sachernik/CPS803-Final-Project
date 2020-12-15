import csv

def count(labels_path, label_col) :
    labels = []
    with open(labels_path) as labels_file :
        reader = csv.reader(labels_file, delimiter=',')
        for row in reader :
            labels.append(int((f'{row[label_col]}')))
    return sum(labels)

def main() :

    labels_path = 'dataset/train_labels.csv'
    data_size = 14681

    label_col_dict = {
        "Computer_Science" : 0,
        "Physics" : 1,
        "Mathematics" : 2,
        "Statistics" : 3,
        "Quantitative_Biology" : 4,
        "Quantitative_Finance" : 5
    }

    for category in label_col_dict :
        label_true = count(labels_path, label_col_dict[category])
        label_percent = label_true/data_size * 100
        print('Category: ', category)
        print('    Number of articles: ', label_true)
        print('    Percentage of articles: ', label_percent)


if __name__ == "__main__":
    main()