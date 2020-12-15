import numpy as np
import csv
from sklearn.metrics import confusion_matrix as cm

def get_labels(labels_path, label_col) :
    labels = []
    with open(labels_path) as labels_file :
        reader = csv.reader(labels_file, delimiter=',')
        for row in reader :
            labels.append(int((f'{row[label_col]}')))
    return np.asarray(labels, dtype=int)

def create_conf_matrix(test_labels, pred_labels) :
    conf_mat = cm(test_labels, pred_labels)
    TN, FP, FN, TP = conf_mat.ravel()

    return TP, TN, FP, FN

def populate_report(confusion_matrix, report) :

    TP, TN, FP, FN = confusion_matrix

    accuracy = round(100 * (TP+TN)/(TP+TN+FP+FN), 2)
    MR = round(100 * (FP+FN)/(TP+TN+FP+FN), 2)
    report.append("    Overall Accuracy: " + str(accuracy))
    report.append("    Misclassification Rate: " + str(MR))

    report.append("    Number of True Positives: " + str(TP))
    report.append("    Number of True Negatives: " + str(TN))
    report.append("    Number of False Positives: " + str(FP))
    report.append("    Number of False Negatives " + str(FN))

    FNR = round(100 * FN/(TP+FN), 2)
    report.append("    False negative rate: " + str(FNR))    

    if TP == 0 or TN == 0 or FP == 0 or FN == 0 :
        report.append("    One or more of the counts is equal to 0")
        report.append("")
        return accuracy, FNR
   
    TNR = round(100 * TN/(TN+FP), 2)
    TPR = round(100 * TP/(TP+FN), 2)
    FPR = round(100 * FP/(TN+FP), 2)
    
    precisionP = round(100 * TP/(TP+FP), 2)
    precisionN = round(100 * TN/(TN+FN), 2)
    prevalenceP = round(100 * (FN+TP)/(TP+TN+FP+FN), 2)
    prevalenceN = round(100 * (TN+FP)/(TP+TN+FP+FN), 2)
           
    report.append("    True positive rate: " + str(TPR))
    report.append("    False positive rate: " + str(FPR))
    report.append("    True negative rate: " + str(TNR))
    
    report.append("    Precision for positives: " + str(precisionP))
    report.append("    Precision for negatives: " + str(precisionN))
    report.append("    Prevalence of positives: " + str(prevalenceP))
    report.append("    Prevalence of negatives: " + str(prevalenceN))
    report.append("")

    return accuracy, FNR

def generate_report(report_path, test_labels, all_pred_labels_count, all_pred_labels_tfidf) :
    
    topics = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative_Biology", "Quantitative_Finance"]
    report = []

    accuracy_count = np.zeros(len(topics))
    FNR_count = np.zeros(len(topics))

    num_abstracts, _ = all_pred_labels_count.shape
    not_class_count = num_abstracts - np.count_nonzero(np.sum(all_pred_labels_count, axis=1))
    not_class_tfidf = num_abstracts - np.count_nonzero(np.sum(all_pred_labels_tfidf, axis=1))

    report.append("Number of abstracts not classified with any topic with basic count model: " + str(not_class_count))
    report.append("Number of abstracts not classified with any topic with TF-IDF model: " + str(not_class_tfidf))
    report.append("")

    for i in range(len(topics)) :

        report.append("-----" + topics[i] + "-----")
        report.append("")

        pred_labels_count = all_pred_labels_count[:, i]
        pred_labels_tfidf = all_pred_labels_tfidf[:, i]

        conf_matrix_count = create_conf_matrix(test_labels[i], pred_labels_count)
        conf_matrix_tfidf = create_conf_matrix(test_labels[i], pred_labels_tfidf)

        report.append("Basic word counts")
        accuracy_count[i],FNR_count[i] = populate_report(conf_matrix_count, report)

        report.append("TF-IDF")
        populate_report(conf_matrix_tfidf, report)   

    with open(report_path, 'w') as f:
        for line in report:
            f.write("%s\n" % line)
            
    return accuracy_count, FNR_count

def main() :

    tiny = False;

    test_label_path = 'dataset/test_labels.csv'
    pred_nb_count_path = 'output/nb_pred_count.txt'
    pred_nb_tfidf_path = 'output/nb_pred_tfidf.txt'
    report_nb_path = 'output/nb_report.txt'

    if tiny == True :
        test_label_path = 'tinydataset/test_labels.csv'
        pred_nb_count_path = 'tinyoutput/nb_pred_count.txt'
        pred_nb_tfidf_path = 'tinyoutput/nb_pred_tfidf.txt'
        report_nb_path = 'tinyoutput/nb_report.txt'

    topics = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative_Biology", "Quantitative_Finance"]

    all_pred_labels_count = np.genfromtxt(pred_nb_count_path, dtype=int, delimiter=',')
    all_pred_labels_tfidf = np.genfromtxt(pred_nb_tfidf_path, dtype=int, delimiter=',')
    report = []

    for i in range(len(topics)) :

        report.append("Topic: " + topics[i])

        test_labels = get_labels(test_label_path, i)
        pred_labels_count = all_pred_labels_count[:, i]
        pred_labels_tfidf = all_pred_labels_tfidf[:, i]

        conf_matrix_count = create_conf_matrix(test_labels, pred_labels_count)
        conf_matrix_tfidf = create_conf_matrix(test_labels, pred_labels_tfidf)

        report.append("Basic word counts")
        populate_report(conf_matrix_count, report)

        report.append("TF-IDF")
        populate_report(conf_matrix_tfidf, report)

    with open(report_nb_path, 'w') as f:
        for line in report:
            f.write("%s\n" % line)

if __name__ == "__main__":
    main()