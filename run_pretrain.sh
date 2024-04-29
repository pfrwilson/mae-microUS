echo SLURM_LOCALID:$SLURM_LOCALID
echo SLURM_PROCID:$SLURM_PROCID
echo $CUDA_VISIBLE_DEVICES

CHECKPOINT_DIR=/checkpoint/$USER/$SLURM_JOB_ID
export MASTER_PORT=29500
export MASTER_ADDR=localhost

if [ $SLURM_PROCID != 0 ]; then
    export TQDM_DISABLE='True'
fi

python main_pretrain.py \
    --output_dir=$CHECKPOINT_DIR \
    --local_rank=$SLURM_LOCALID \
    --world_size=$SLURM_NTASKS \
    --no_probing_nct \
    --input_size 224 \
    --batch_size 64 \
    --accum_iter 1 \
    --data_path=/scratch/ssd004/datasets/imagenet256 \
    #--data_path=/scratch/ssd004/datasets/imagenet/
    #--data_path=/ssd005/projects/exactvu_pca/cityscapes/leftImg8bit_trainvaltest/leftImg8bit 
    #--data_path=/ssd005/projects/exactvu_pca/unlabelled_microus_png \
    