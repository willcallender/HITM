import numpy as np
import pickle
from os import environ

from kpz import *
from inverse import *


if __name__ == '__main__':
    # slurm job id
    SLURM_JOB_ID = environ.get('SLURM_JOB_ID', None)
    
    # slurm array index
    SLURM_ARRAY_TASK_ID = environ.get('SLURM_ARRAY_TASK_ID', None)
    
    filename = 'spread.pkl'
    
    if SLURM_ARRAY_TASK_ID is not None:
        SLURM_ARRAY_TASK_ID = int(SLURM_ARRAY_TASK_ID)
        filename = f'spread_{SLURM_ARRAY_TASK_ID}.pkl'
        if SLURM_JOB_ID is not None:
            SLURM_JOB_ID = int(SLURM_JOB_ID)

    print(*zip(['SLURM_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'filename'], [SLURM_JOB_ID, SLURM_ARRAY_TASK_ID, filename]))
    
    x_bounds = np.array([100, 250])
    t_bounds = np.array([10, 80])
    # t_bounds = np.array([10, 13])

    # load real data
    with open(filename, 'rb') as f:
        spread = pickle.load(f)
    spread = np.array(spread, float)
    
    y_bounds = np.array([0, spread.max()], float)
    
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
    
    batchSize = 64
    print('Running {} simulations'.format(len(args)))
    i = 0
    while i+batchSize <= len(args):
        print('Starting up to {}'.format(i+batchSize))
        futures = [runExperiment(spread, t1, t2, t3, x_bounds, y_bounds) for t1, t2, t3 in args[i:i+batchSize]]
        print('Waiting for results')
        thisResults = [f for f in futures]
        print('Got results')
        i += batchSize
        with open(f'results/results{i}.pkl', 'wb') as f:
            pickle.dump(thisResults, f)
        print('Finished up to {}'.format(i))
    print('Finished all simulations')
