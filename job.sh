#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=SimCLR-shapenet-bs128-e100-resnet18
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:59:00
#SBATCH --mem=0
#SBATCH --gres=gpu:p100l:4
#SBATCH --cpus-per-task=12

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
#	--data-folder datasets/shapenet

python main_supcon.py --epochs 100 --batch_size 128 --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --dataset shapenet \
	--data-folder datasets/shapenet

python main_linear.py --batch_size 128 --learning_rate 5 --model resnet18 --ckpt ./save/SupCon/cifar10_models/*/last.pth \
	--dataset shapenet --data-folder datasets/shapenet

cp -r save/ ~/scratch/SupContrast/
