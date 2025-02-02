import os
from glob import glob
import random
def get_image_paths(data_path, categories, split_ratio=[0.7, 0.2, 0.1]):
    train_image_paths = []
    test_image_paths = []
    val_image_paths = []

    train_labels = []
    test_labels = []
    val_labels = []

    for category in categories:
        
        image_paths = glob(os.path.join(data_path, category, '*.tif'))
        len_ = len(image_paths)
        image_paths = image_paths[:len_]
        with open('image_paths.txt', 'w') as f:
            for item in image_paths:
                f.write("%s\n" % item)
                
        random.shuffle(image_paths)
        num_train_per_cat = int(len(image_paths) * split_ratio[0])
        for i in range(num_train_per_cat):
            train_image_paths.append(image_paths[i])
            train_labels.append(category)

        num_val_per_cat = int(len(image_paths) * split_ratio[1])
        for i in range(num_train_per_cat, num_train_per_cat + num_val_per_cat):
            val_image_paths.append(image_paths[i])
            val_labels.append(category)

        num_test_per_cat = int(len(image_paths) * split_ratio[2])
        for i in range(num_train_per_cat + num_val_per_cat, num_train_per_cat + num_val_per_cat + num_test_per_cat):
            test_image_paths.append(image_paths[i])
            test_labels.append(category)

    return train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels
