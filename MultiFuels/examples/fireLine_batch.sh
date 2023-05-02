max_wind=4.0
dir="/gpfs/home/wac20/MultiFuels/examples"
n=10
for (( i=1; i<=$n; i++ ))
do
    export wind=$(echo "$max_wind * $i / $n" | bc -l)
    export batch_iter=$SLURM_ARRAY_TASK_ID
    echo "wind = $wind"
    echo "batch_iter = $batch_iter"
    matlab -nojvm -batch "batch_fire"
    scp "$dir/data*" wc@willcallender.mooo.com:~/external/data/
    rm "$dir/data*"
done
