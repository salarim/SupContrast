#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=SupContrast
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:59:00
#SBATCH --mem=0
#SBATCH --gres=gpu:p100l:4
#SBATCH --cpus-per-task=12

cd $SLURM_TMPDIR
cp -r ~/scratch/SupContrast .
cd SupContrast

module load python/3.7 cuda/10.0
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python main_supcon.py --epochs 100 --batch_size 512 --learning_rate 0.5 --temp 0.1 --cosine

python main_linear.py --batch_size 512 --learning_rate 5 --ckpt ./save/SupCon/cifar10_models/*/last.pth

cp -r save/ ~/scratch/SupContrast/
