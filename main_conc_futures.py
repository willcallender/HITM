import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import pickle
from concurrent.futures import ProcessPoolExecutor
from numba import jit, njit, vectorize

from kpz import *
from inverse import *


if __name__ == '__main__':
    x_bounds = np.array([100, 250])
    t_bounds = np.array([10, 80])
    # t_bounds = np.array([10, 13])

    # load real data
    filename = 'spread.pickle'
    with open(filename, 'rb') as f:
        spread = pickle.load(f)

    # Run many simulations
    allErr = np.zeros(x_bounds[1]-x_bounds[0])
    args = []
    for t1 in range(t_bounds[0], t_bounds[1]):
        for t2 in range(t1+1, t_bounds[1]):
            t3 = np.zeros(t_bounds[1]-t1-2, int)
            t3[:t2-t1-1] = np.arange(t1+1, t2)
            t3[t2-t1-1:] = np.arange(t2+1, t_bounds[1])
            args.append((t1, t2, t3))
            # print(args[-1])

    results = []
    
    n_proc = 16
    batchSize = 128
    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        print('Running {} simulations'.format(len(args)))
        i = 1152 #---------------------------------------------------------
        while i+batchSize < len(args):
            print('Starting up to {}'.format(i+batchSize))
            with open('log.txt', 'a') as f:
                f.write('Starting up to {}\n'.format(i+batchSize))
            futures = [executor.submit(runExperiment, spread, t1, t2, t3, x_bounds) for t1, t2, t3 in args[i:i+batchSize]]
            print('Waiting for results')
            with open('log.txt', 'a') as f:
                f.write('Waiting for results\n')
            thisResults = [f.result() for f in futures]
            print('Got results')
            with open('log.txt', 'a') as f:
                f.write('Got results\n')
            results.extend(thisResults)
            i += batchSize
            with open(f'results{i}.pkl', 'wb') as f:
                pickle.dump(thisResults, f)
            print('Finished up to {}'.format(i))
            with open('log.txt', 'a') as f:
                f.write('Finished up to {}\n'.format(i))
    print('Finished all simulations')

    with open('all_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('Done!')
