#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=SimCLR-shapenet-bs2048-v2-dr0.0-e100-resnet18
#SBATCH --output=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --time=2:59:00
#SBATCH --mem=10000M
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=16

export BATCH_SIZE="${BATCH_SIZE:=2048}"
export VIEWS="${VIEWS:=2}"
export DROP_RATIO="${DROP_RATOI:=0.0}"


cd $SLURM_TMPDIR
cp -r ~/scratch/SupContrast .
cd SupContrast
rm -r save

cd datasets
unzip -qq shapenet-split.zip
mv datasets/shapenet .
rm -r datasets
cd ..

module load python/3.7 cuda/10.0
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

#python main_ce.py --epochs 100 --batch_size 128 --learning_rate 0.8 --cosine --model resnet18 --dataset shapenet \
#	--data-folder datasets/shapenet --views 2


python main_supcon.py --epochs 100 --batch_size $BATCH_SIZE --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --dataset shapenet \
	--data-folder datasets/shapenet --views $VIEWS --drop-objects-ratio $DROP_RATIO --method SimCLR

python main_linear.py --epochs 20 --batch_size 128 --learning_rate 5 --model resnet18 --ckpt ./save/SupCon/shapenet_models/*/last.pth \
	--dataset shapenet --data-folder datasets/shapenet --views 30

cp -r save/ ~/scratch/SupContrast/
