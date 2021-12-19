'''
Plot results for UCB-VI
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pandas import read_csv


alg_list = [ 'UCBVI', 'UCBVI_JDP', 'UCBVI_LDP']
priv_list = [0.5,1 ]
nEps = 20000
nRuns = 20
eps = [i+1 for i in range(nEps)]

for alg in alg_list:
    if alg == 'UCBVI':
        priv = 0
        fileName = ('alg=' + str(alg)
                    + '_nEps=' + str(nEps)
                    + '_nRuns=' + str(nRuns)
                    + '_priv=' + '%03.2f' % priv
                    + '.csv')
        label = 'Non-private'
        data = read_csv(fileName, header = None)
        data = data.to_numpy()
        for i in range(nRuns):
            data[:,i] = np.multiply(data[:,i], eps)

        data_mean = np.mean(data,axis = 1)
        data_std = np.std(data, axis = 1)
        data_upper = data_mean + data_std
        data_lower = data_mean - data_std
        plt.plot(eps, data_mean, label = label)
        plt.fill_between(eps, data_upper, data_lower,alpha=0.2, linewidth=0)

        continue
    for priv in priv_list:
        fileName = ('alg=' + str(alg)
                    + '_nEps=' + str(nEps)
                    + '_nRuns=' + str(nRuns)
                    + '_priv=' + '%03.2f' % priv
                    + '.csv')
        data = read_csv(fileName, header = None)
        data = data.to_numpy()
        for i in range(nRuns):
            data[:,i] = np.multiply(data[:,i], eps)
        label = alg.split('_')[1] + ' ' + '($\epsilon = ' + '%02.1f' % priv + '$)'
        data_mean = np.mean(data,axis = 1)
        data_std = np.std(data, axis = 1)
        data_upper = data_mean + data_std
        data_lower = data_mean - data_std
        plt.plot(eps, data_mean, label = label)
        plt.fill_between(eps, data_upper, data_lower,alpha=0.2, linewidth=0)


plt.xlabel(r"Episode ($K$)")
plt.ylabel(r"Cumulative Regret ")
plt.title('Private-UCB-VI')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

plt.legend(loc = 'upper left')
plt.show()
