from sklearn.linear_model import Perceptron
from tools.arff import Arff
import matplotlib.pyplot as plt
import logging

weights_printout = 'Final Weights:\n{}'
train_score_printout = 'Final Training Accuracy:\n{}'
test_score_printout = 'Final Test Accuracy:\n{}'


def evaluation():
    print('Evaluation Data')
    a = Arff(arff='data/perceptron/evaluation/data_banknote_authentication.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = Perceptron(shuffle=True, eta0=0.001, penalty='l1')
    p.fit(data, labels)
    print(weights_printout.format(p.coef_))
    print(train_score_printout.format(p.score(data, labels)))
    # epochs = [x for x in range(11)]
    # fig = plt.plot(epochs, p.errors)
    # plt.show()


def voting():
    lr = float(input('Enter learning rate: '))
    print('Voting Data')
    a = Arff(arff='data/perceptron/vote.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = Perceptron(shuffle=True, eta0=lr, penalty='l1')
    p.fit(data, labels)
    print(weights_printout.format(p.coef_))
    print(train_score_printout.format(p.score(data, labels)))
    # fig, ax = plt.subplots(figsize=(8, 6))
    # plt.ylabel('Misclassification Rate')
    # plt.xlabel('Epoch')
    # plt.title('Voting Data Misclassification Rates Across Trials')
    # plot_list = []
    # epochs = [x for x in range(len(p))]
    # for key, item in error_map.items():
    #     plot_list.append(ax.plot([x for x in range(len(item))], item, label=('Run #' + str(key)))[0])
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=5,
    #           handles=plot_list,
    #           facecolor='white', edgecolor='black')
    # fig.savefig('votingaccgraph.png')
    # plt.show()


logging.captureWarnings(True)
logging.basicConfig(level=logging.ERROR)
while True:
    i = input('Enter e (evaluation), v (voting):\n')
    functions = {
        'e': evaluation,
        'v': voting
    }
    print()
    functions[i]()

