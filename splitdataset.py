import csv

def split_dataset(orig_path, train_path, test_path, header_path, split) :
    print('Running split_dataset for: ', orig_path)

    line_count = 0
    train_rows = 0
    test_rows = 0
    with open(orig_path) as orig_file, open(train_path, mode='w') as train_file, open(test_path, mode='w') as test_file :
        reader = csv.reader(orig_file, delimiter=',')
        train_writer = csv.writer(train_file, delimiter=',')
        test_writer = csv.writer(test_file, delimiter=',')                    
        for row in reader :
            if line_count == 0 :
                with open(header_path, mode='w') as header_file :                        
                    header_writer = csv.writer(header_file, delimiter=',')
                    header_writer.writerow(row)
            else :
                if int(row[0]) <= split :
                    train_writer.writerow(row)
                    train_rows +=1
                else :
                    test_writer.writerow(row)
                    test_rows+=1
            line_count +=1

    print('    line_count was ', line_count)
    print('    train_rows was ', train_rows)
    print('    test_rows was ', test_rows)

def split_labels(input_path, output_data, output_labels) :

    print('Running split_labels for file: ', input_path)

    rows_written = 0
    with open(input_path) as input_file :
        reader = csv.reader(input_file, delimiter=',')
        with open(output_data, mode='w') as data_file :
            data_writer = csv.writer(data_file, delimiter=',')
            with open (output_labels, mode='w') as label_file :
                label_writer = csv.writer(label_file, delimiter=',')
                for row in reader :
                    data_writer.writerow([row[1],row[2]])
                    label_writer.writerow([row[3],row[4],row[5],row[6],row[7],row[8]])
                    rows_written +=1

    print('    Number of rows written: ', rows_written)

def make_tiny_dataset(orig_path, tiny_path, size) :

    print('Running make_tiny_dataset')

    line_count = 0
    with open(orig_path) as orig_file, open(tiny_path, mode='w') as tiny_file :
        reader = csv.reader(orig_file, delimiter=',')
        tiny_writer = csv.writer(tiny_file, delimiter=',')                   
        for row in reader :
            if line_count <= size :
                tiny_writer.writerow(row)
                line_count +=1
            else :
                break
    print('    Number of lines written (includes header): ', line_count)

def main() :
    orig_path = 'dataset/original_data.csv'
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
    header_path = 'dataset/header.csv'
    train_data_path = 'dataset/train_data.csv'
    train_labels_path = 'dataset/train_labels.csv'
    test_data_path = 'dataset/test_data.csv'
    test_labels_path = 'dataset/test_labels.csv'
    split = 14681

    #split_dataset(orig_path, train_path, test_path, header_path, split)
    #split_labels(train_path, train_data_path, train_labels_path)
    #split_labels(test_path, test_data_path, test_labels_path)

    tiny_dataset_path = 'tinydataset/original_data.csv'
    tiny_train_path = 'tinydataset/train.csv'
    tiny_test_path = 'tinydataset/test.csv'
    tiny_header_path = 'tinydataset/header.csv'
    tiny_train_data_path = 'tinydataset/train_data.csv'
    tiny_train_labels_path = 'tinydataset/train_labels.csv'
    tiny_test_data_path = 'tinydataset/test_data.csv'
    tiny_test_labels_path = 'tinydataset/test_labels.csv'
    tiny_size = 1000
    tiny_split = 700

    make_tiny_dataset(orig_path, tiny_dataset_path, tiny_size)
    split_dataset(tiny_dataset_path, tiny_train_path, tiny_test_path, tiny_header_path, tiny_split)
    split_labels(tiny_train_path, tiny_train_data_path, tiny_train_labels_path)
    split_labels(tiny_test_path, tiny_test_data_path, tiny_test_labels_path)


if __name__ == "__main__":
    main()