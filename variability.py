from LE import LE, Align

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

df_vanilla_init = pd.read_csv('/Users/gulli/Documents/GitHub/LaplacianEigenmaps/vanilla_glove_100D_init_trick.txt', 
                                delim_whitespace=True, header=None) 
X1 = df_vanilla_init.iloc[0:200,1:101].to_numpy() 

df_vanilla = pd.read_csv('/Users/gulli/Documents/GitHub/LaplacianEigenmaps/vanilla_glove_100D.txt', 
                            delim_whitespace=True, header=None) 
X2 = df_vanilla.iloc[0:200,1:101].to_numpy() 


nlabels = [20,100,200]
repeat = 10
hyperAcc = np.zeros((len(nlabels),repeat))
euclidAcc = np.zeros((len(nlabels),repeat))
for r in range(len(nlabels)):
    print('r%: ', r)
    for c in range(repeat):
        #if c%5==0:
        #    print('c%: ', c)
        rng = default_rng()
        numbers = rng.choice(200, size=nlabels[r], replace=False)

        align = Align(X1, X2, [i for i in numbers], mu=1000, graph = 'eps', weights = 'heat kernel', 
                        sigma = 5, laplacian = 'symmetrized')

        hyperAcc[r,c] = align.findCorrespondences(hyper=True)   
        euclidAcc[r,c] = align.findCorrespondences(hyper=False)   

hyperAvg = np.mean(hyperAcc,1)
print('hyperAvg: ', hyperAvg)
euclidAcc = np.mean(euclidAcc,1)
print('euclidAcc: ', euclidAcc)

x = [1,2,3]
plt.plot(x, hyperAvg, 'ro-', label='Hyperbolic')
plt.plot(x, euclidAcc, 'gd--', label='Euclidean')
plt.xticks(ticks=x, labels=['10%','50%','100%'])
plt.ylim(0,1)
plt.title('Accuracy')
plt.xlabel('Labels %')
plt.ylabel('Accuracy %')
plt.legend(title='Embedding space:')

plt.savefig('accuracyPlot.png')