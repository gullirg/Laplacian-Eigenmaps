from LE import LE, Align

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy.random import default_rng
from sklearn import datasets

def dataset(dataset='vanillaGlove'):
    if dataset == 'vanillaGlove':
        df_vanilla_init = pd.read_csv('/Users/gulli/Documents/GitHub/LaplacianEigenmaps/vanilla_glove_100D_init_trick.txt', 
                                        delim_whitespace=True, header=None) 
        X1 = df_vanilla_init.iloc[0:200,1:101].to_numpy() 

        df_vanilla = pd.read_csv('/Users/gulli/Documents/GitHub/LaplacianEigenmaps/vanilla_glove_100D.txt', 
                                    delim_whitespace=True, header=None) 
        X2 = df_vanilla.iloc[0:200,1:101].to_numpy() 
    elif dataset == 'toyTree':
        X1 = np.array([[0,0,0],[0,0,1],[0,0,2],[0.5,0.5,1],[0.75,0.75,1],[1,1,1],[1.5,1.5,2],[1.5,1.5,0]])
        X2 = np.array([[ 0.0361873 ,  0.01478449, -0.00233926],[ 0.12800259, -0.10555198,  2.05620019],[ 0.47730813,  0.5704031 ,  0.9987547 ],
                        [ 0.69648977,  0.66492889,  0.98311152],[ 0.87920603,  0.83921925,  0.96676621],[ 1.55319584,  1.39833294,  1.94371016],
                        [ 1.5221932 ,  1.45613329,  0.88807379],[ 1.51739898,  1.55937081,  0.0528711 ]])
    elif dataset == 'swissRoll':
        X1, color1 = datasets.samples_generator.make_swiss_roll(n_samples=2000, random_state = 2456)
        X2, color2 = datasets.samples_generator.make_swiss_roll(n_samples=2000, random_state = 2456)
    return X1, X2

def makeFigure(data='vanillaGlove', percentages=[10,50,100], repeat=10, gamma=5):
    X1, X2 = dataset(data)
    mu = [0.1,10,1000]
    nlabels = np.array([int(p) for p in percentages])*len(X1[:,0])//100
    print('nlabels',nlabels)
    hyperAcc = np.zeros((len(mu),len(nlabels),repeat))
    euclidAcc = np.zeros((len(mu),len(nlabels),repeat))
    for m in range(len(mu)):
        for r in range(len(nlabels)):
            print('r%: ', r)
            for c in range(repeat):
                rng = default_rng()
                numbers = rng.choice(len(X1[:,0]), size=nlabels[r], replace=False)

                align = Align(X1, X2, [i for i in numbers], mu=mu[m], graph = 'eps', weights = 'heat kernel', 
                                sigma = 5, laplacian = 'symmetrized')

                hyperAcc[m,r,c] = align.findCorrespondences(hyper=True, gamma=gamma)   
                euclidAcc[m,r,c] = align.findCorrespondences(hyper=False, dims=2)   

    hyperAvg = np.mean(hyperAcc,2)
    hyperStd = np.std(hyperAcc,2)
    print('hyperAvg: ', hyperAvg)
    euclidAvg = np.mean(euclidAcc,2)
    euclidStd = np.std(euclidAcc,2)
    print('euclidAcc: ', euclidAvg)

    x = [i+1 for i in range(len(percentages))]
    xH = [i+1.1 for i in range(len(percentages))]
    xE = [i+0.9 for i in range(len(percentages))]
    c = ['b','g','r','c','m','y','k']
    for m in range(len(mu)):
        plt.errorbar(xH, hyperAvg[m]*100, fmt='{}o-'.format(c[m]), yerr=hyperStd[m], capsize=5, label='Hyperbolic - mu: {}'.format(mu[m]), alpha=0.9)
        plt.errorbar(xE, euclidAvg[m]*100, fmt='{}d--'.format(c[m]), yerr=euclidStd[m], capsize=5, label='Euclidean - mu: {}'.format(mu[m]), alpha=0.9)
    plt.xticks(ticks=x, labels=['{}%'.format(i) for i in percentages])
    plt.ylim(0,100)
    plt.title('Accuracy - dataset: {}'.format(data))
    plt.xlabel('Labels %')
    plt.ylabel('Accuracy %')
    plt.legend(title='Embedding space:', bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()

    plt.savefig('accuracyPlot.png')


parser = argparse.ArgumentParser(description='Hyperbolic vs Euclidean alignment - variability figure generator.')
parser.add_argument('--data', type=str, default='vanillaGlove', choices=['vanillaGlove', 'toyTree', 'swissRoll'],
                    help='datasets: vanillaGlove, toyTree, swissRoll - default: %(default)s')
parser.add_argument('--percentages', default='10 50 100', nargs='+',
                    help='different percentages of supervised labels to test - default: %(default)s')
parser.add_argument('--repeat',  type=int, default=10,
                    help='number of repetitions for each percentage - default: %(default)s')
parser.add_argument('--gamma',  type=int, default=5,
                    help='scaling parameter of the joint network - default: %(default)s')

args = parser.parse_args()

if __name__ == "__main__":
    makeFigure(data=args.data, percentages=args.percentages, repeat=args.repeat, gamma=args.gamma)