import matplotlib.pyplot as plt
import numpy as np

def gen_plot(topic_freq, acc_1, fnr_1, acc_2, fnr_2, save_path) :
    plt.figure()
    plt.plot(topic_freq, acc_1, 'bo-', linewidth=1)
    plt.plot(topic_freq, fnr_1, 'bx-', linewidth=1)
    plt.plot(topic_freq, acc_2, 'ro-', linewidth=1)
    plt.plot(topic_freq, fnr_2, 'rx-', linewidth=1)

    plt.xlabel('Topic Frequencies')
    plt.ylabel('Accuracy and False Negative Rates')
    plt.savefig(save_path)

    #plt.show()


def main() :
    save_path = 'plot.jpg'
    topic_freq = np.array([41,29,27,25,3,1])
    nb_acc = np.array([80,90,89,85,97,98])
    nb_fnr = np.array([16,21,20,8,83,100])
    svm_acc = np.array([73,88,85,85,98,98])
    svm_fnr = np.array([37,33,29,43,83,100])

    gen_plot(topic_freq, nb_acc, nb_fnr, svm_acc, svm_fnr, save_path)


if __name__ == "__main__":
    main()