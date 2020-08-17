#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=SimCLR-cifar10-bs1024-e100-lr0.1-myresnet18-BNmlp
#SBATCH --output=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --time=2:59:00
#SBATCH --mem=10000M
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=16

export BATCH_SIZE="${BATCH_SIZE:=1024}"
export VIEWS="${VIEWS:=2}"
export DROP_RATIO="${DROP_RATIO:=0.0}"
export DATASET="${DATASET:=cifar10}"

cd $SLURM_TMPDIR
cp -r ~/scratch/SupContrast .
cd SupContrast
rm -r save

#cd datasets
#unzip -qq shapenet-split.zip
#mv datasets/shapenet .
#rm -r datasets
#cd ..

module load python/3.7 cuda/10.0
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

#python main_ce.py --epochs 100 --batch_size $BATCH_SIZE --learning_rate 0.8 --cosine --model resnet18 --dataset $DATASET \
#	--data-folder datasets/${DATASET} --views 2


python main_supcon.py --epochs 100 --batch_size $BATCH_SIZE --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --dataset $DATASET \
	--data-folder datasets/${DATASET} --views $VIEWS --drop-objects-ratio $DROP_RATIO --method SimCLR

#python main_byol.py --epochs 100 --batch_size $BATCH_SIZE --learning_rate 0.1 --model resnet18 --dataset $DATASET \
#	        --data-folder datasets/${DATASET} --views $VIEWS --drop-objects-ratio $DROP_RATIO


python main_linear.py --epochs 20 --batch_size 128 --learning_rate 0.1 --model resnet18 --ckpt save/SupCon/${DATASET}_models/*/last.pth \
	--dataset $DATASET  --data-folder datasets/${DATASET} --views 2

cp -r save/ ~/scratch/SupContrast/
