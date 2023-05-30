max_wind=4.0
home_dir="/gpfs/home/wac20"
data_dir="$home_dir/MultiFuels/examples"
ssh_dir="$home_dir/.ssh"
n=10
for (( i=0; i<=$n; i++ ))
do
    export wind=$(echo "$max_wind * $i / $n" | bc -l)
    export batch_iter=$SLURM_ARRAY_TASK_ID
    echo "wind = $wind"
    echo "batch_iter = $batch_iter"
    matlab -nojvm -batch "batch_fire"
    scp -i $ssh_dir/id_rsa $data_dir/data* wc@willcallender.mooo.com:~/external/data/
    rm $data_dir/data*
done
