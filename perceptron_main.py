from perceptron.perceptron import PerceptronClassifier
from tools.arff import Arff
import logging

weights_printout = 'Final Weights:\n{}'
score_printout = 'Final Accuracy:\n{}'


def debug():
    print('Debug Data')
    a = Arff(arff='data/perceptron/debug/linsep2nonorigin.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=False)
    p.fit(data, labels, deterministic=10)
    print(weights_printout.format(p.get_weights()))
    print(score_printout.format(p.score(data, labels)))


def evaluation():
    print('Evaluation Data')
    a = Arff(arff='data/perceptron/evaluation/data_banknote_authentication.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=False)
    p.fit(data, labels, deterministic=10)
    print(weights_printout.format(p.get_weights()))
    print(score_printout.format(p.score(data, labels)))


def created_datasets():
    print('Created Data')

    print('Linear Data:')
    a = Arff(arff='data/perceptron/created/linear.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=True)
    p.fit(data, labels, deterministic=10)
    print(weights_printout.format(p.get_weights()))
    print(score_printout.format(p.score(data, labels)))

    print('Nonlinear Data:')
    a = Arff(arff='data/perceptron/created/nonlinear.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=True)
    p.fit(data, labels, deterministic=10)
    print(weights_printout.format(p.get_weights()))
    print(score_printout.format(p.score(data, labels)))


def voting():
    print('Voting Data')
    a = Arff(arff='data/perceptron/vote.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=True)
    p.fit(data, labels, stop_thresh=0.000001)
    print(weights_printout.format(p.get_weights()))
    print(score_printout.format(p.score(data, labels)))


def iris():
    print('Iris Data')
    print('not implemented')


def example():
    a = Arff(arff='data/perceptron/example/exSeparable.arff')
    data = a.get_features().data
    labels = a.get_labels().data
    p = PerceptronClassifier(shuffle=False)
    p.fit(data, labels, deterministic=10)
    print(weights_printout.format(p.get_weights()))
    print(score_printout.format(p.score(data, labels)))


logging.captureWarnings(True)
logging.basicConfig(level=logging.ERROR)
i = input('Enter d (debug), e (evaluation), c (created datasets), v (voting), i (iris):\n')
functions = {
    'd': debug,
    'e': evaluation,
    'c': created_datasets,
    'v': voting,
    'i': iris,
    'x': example
}
print()
functions[i]()
