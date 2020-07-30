import os
import shutil
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser('argument for spliting')

parser.add_argument('--dataset-folder', type=str, default='',
                    help='where original dataset folder is.')
parser.add_argument('--output-folder', type=str, default='',
                    help='where output dataset folder should be.')

opt = parser.parse_args()


def main():
    train_ratio = 0.7

    classes = [d for d in os.scandir(opt.dataset_folder) if d.is_dir()]

    for cls in classes:
        train_cls_folder = os.path.join(opt.output_folder, 'train', cls.name)
        val_cls_folder = os.path.join(opt.output_folder, 'val', cls.name)

        # os.makedirs(train_cls_folder)
        # os.makedirs(val_cls_folder)

        object_folders = [d for d in os.scandir(cls.path) if d.is_dir()]
        np.random.shuffle(object_folders)
        train_object_folders, val_object_folders = np.split(np.array(object_folders),
                                                                [int(len(object_folders) * (train_ratio))])

        print('Class:', cls.name)
        print('Total images: ', len(object_folders))
        print('Training: ', len(train_object_folders))
        print('Validation: ', len(val_object_folders))

        # Copy-pasting images
        for folder in train_object_folders:
            dst = os.path.join(train_cls_folder, folder.name)
            shutil.copytree(folder.path, dst)

        for folder in val_object_folders:
            dst = os.path.join(val_cls_folder, folder.name)
            shutil.copytree(folder.path, dst)


if __name__ == "__main__":
    main()
