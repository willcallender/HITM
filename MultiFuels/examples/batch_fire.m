maxWind = 4.0;
dx = 1;

taskID = str2num(getenv('SLURM_ARRAY_TASK_ID'))
n = str2num(getenv('SLURM_ARRAY_TASK_COUNT'))

% taskID = 1;
% n = 100;

scale = maxWind / n;
wind = taskID * scale;
vortStrength = 0.5 * wind / dx^2;
sinkStrength = 0.05 + 0.15 * wind;

disp Starting
for batch_iter=1:100
    disp(batch_iter)
    fireLine
end
disp Done
