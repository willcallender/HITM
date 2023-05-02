maxWind = 4.0;
dx = 1;

% taskID = str2num(getenv('SLURM_ARRAY_TASK_ID'))
% n = str2num(getenv('SLURM_ARRAY_TASK_COUNT'))

% % taskID = 1;
% % n = 100;

% scale = maxWind / n;
% wind = taskID * scale;

% wind = str2double(argv(){1});
wind = str2num(getenv('wind'));
vortStrength = 0.5 * wind / dx^2;
sinkStrength = 0.05 + 0.15 * wind;

batch_iter = str2num(getenv('batch_iter'));
disp(batch_iter)
fireLine

disp Done
