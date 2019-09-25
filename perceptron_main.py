from perceptron.perceptron import PerceptronClassifier
from tools.arff import Arff
import logging
from tools.graph_tools import graph, graph_function
from perceptron.data_split import split
import matplotlib.pyplot as plt

weights_printout = 'Final Weights:\n{}'
train_score_printout = 'Final Training Accuracy:\n{}'
test_score_printout = 'Final Test Accuracy:\n{}'


def debug(lr=0.1):
    print('Debug Data')
    a = Arff(arff='data/perceptron/debug/linsep2nonorigin.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=False)
    p.fit(data, labels, deterministic=10)
    print(weights_printout.format(p.get_weights()))
    print(train_score_printout.format(p.score(data, labels)))


def evaluation(lr=0.1):
    print('Evaluation Data')
    a = Arff(arff='data/perceptron/evaluation/data_banknote_authentication.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=False)
    p.fit(data, labels, deterministic=10)
    print(weights_printout.format(p.get_weights()))
    print(train_score_printout.format(p.score(data, labels)))
    epochs = [x for x in range(11)]
    fig = plt.plot(epochs, p.errors)
    plt.show()


def created_datasets(lr=0.1):
    print('Created Data')

    print('Linear Data:')
    a = Arff(arff='data/perceptron/created/linear.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=True, lr=lr)
    p.fit(data, labels, stop_thresh=0.0001, num_stopping_rounds=5)
    weights = p.get_weights()
    print(weights_printout.format(weights))
    print(train_score_printout.format(p.score(data, labels)))
    # func = lambda x: ((x * weights[0][0] * -1) + weights[2][0]) / weights[1][0]
    # graph(data[:,0], data[:,1], labels[:,0], title='Linearly Separable Dataset', save_path='lineargraph.png', func=func, xlabel='feature 1', ylabel='feature 2')
    print('{} Epochs'.format(len(p.errors)))

    print('Nonlinear Data:')
    a = Arff(arff='data/perceptron/created/nonlinear.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=True, lr=lr)
    p.fit(data, labels, stop_thresh=0.1, num_stopping_rounds=10)
    weights = p.get_weights()
    print(weights_printout.format(weights))
    print(train_score_printout.format(p.score(data, labels)))
    # func = lambda x: ((x * weights[0][0] * -1) + weights[2][0]) / weights[1][0]
    # graph(data[:,0], data[:,1], labels[:,0], title='Not Linearly Separable Dataset', save_path='nonlineargraph.png', func=func, xlabel='feature 1', ylabel='feature 2')
    print('{} Epochs'.format(len(p.errors)))


def voting(lr=0.0001):
    print('Voting Data')
    error_map = {}
    a = Arff(arff='data/perceptron/vote.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    for i in range(5):
        train_x, test_x, train_y, test_y = split(data, labels)
        p = PerceptronClassifier(lr=lr, shuffle=True)
        p.fit(train_x, train_y, stop_thresh=0.000001, num_stopping_rounds=20)
        print('Run #{}'.format(i))
        print(weights_printout.format(p.get_weights()))
        # print(train_score_printout.format(p.score(train_x, train_y)))
        # print(test_score_printout.format(p.score(test_x, test_y)))
        # print('{} Epochs\n'.format(len(p.errors)))
        error_map[i] = p.errors.copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.ylabel('Misclassification Rate')
    plt.xlabel('Epoch')
    plt.title('Voting Data Misclassification Rates Across Trials')
    plot_list = []
    for key, item in error_map.items():
        plot_list.append(ax.plot([x for x in range(len(item))], item, label=('Run #' + str(key+1)))[0])
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=5,
    #           handles=plot_list,
    #           facecolor='white', edgecolor='black')
    ax.legend()
    fig.savefig('votingaccgraph.png')
    plt.show()


def iris(lr=0.1):
    print('Iris Data')
    print('not implemented')


def example(lr=0.1):
    a = Arff(arff='data/perceptron/example/exSeparable.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=False)
    p.fit(data, labels, deterministic=10)
    weights = p.get_weights()
    print(weights_printout.format(weights))
    print(train_score_printout.format(p.score(data, labels)))
    func = lambda x: ((x * weights[0][0] * -1) + weights[2][0]) / weights[1][0]
    graph(data[:,0], data[:,1], labels[:,0], title='Example Separable Dataset', func=func, xlabel='feature 1', ylabel='feature 2')


logging.captureWarnings(True)
logging.basicConfig(level=logging.ERROR)
while True:
    i = input('Enter d (debug), e (evaluation), c (created datasets), v (voting), i (iris):\n')
    lr = 0.0001
    if i == 'c':
        lr = float(input('Enter learning rate: '))
    functions = {
        'd': debug,
        'e': evaluation,
        'c': created_datasets,
        'v': voting,
        'i': iris,
        'x': example
    }
    print()
    functions[i](lr)
