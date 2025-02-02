from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from mlp_classify import mlp_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold
from sklearn.manifold import TSNE

# Step 0: Set up parameters, category list, and image paths.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()

DATA_PATH = '../Assignment_1/UCMerced_LandUse/Images/'

CATEGORIES = [
            'agricultural', 'baseballdiamond', 'buildings', 'denseresidential', 'freeway', 'harbor', 'mediumresidential', 'overpass', 'river', 
            'sparseresidential', 'tenniscourt', 'airplane', 'beach', 'chaparral', 'forest', 'golfcourse', 'intersection', 'mobilehomepark', 'parkinglot',
            'runway', 'storagetanks'
        ]
CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}
ABBR_CATEGORIES = ['Agr', 'Bas', 'Bui', 'Den', 'Fre', 'Har', 'Med', 'Ove', 'Riv', 'Spa', 'Ten', 'Air', 'Bea', 'Cha', 'For', 'Gol', 'Int', 'Mob', 'Par', 'Run', 'Sto']

FEATURE = args.feature
CLASSIFIER = args.classifier

def calculate_accuracy(labels, predictions, categories, split_name):
    accuracy = float(len([x for x in zip(labels, predictions) if x[0] == x[1]])) / len(labels)
    print(f"{split_name} Accuracy = {accuracy:.2f}")
    for category in categories:
        accuracy_each = float(len([x for x in zip(labels, predictions) if x[0] == x[1] and x[0] == category])) / float(labels.count(category))
        print(f"{category}: {accuracy_each:.2f}")

def main():
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels = \
        get_image_paths(DATA_PATH, CATEGORIES)

    if FEATURE == 'tiny_image':
        train_image_feats = get_tiny_images(train_image_paths, size=16)
        test_image_feats = get_tiny_images(test_image_paths, size=16)
        val_image_feats = get_tiny_images(val_image_paths, size=16)

    elif FEATURE == 'bag_of_sift':
        vocab_size = 400
        if os.path.isfile('vocab.pkl') is False:
            print('No existing visual word vocabulary found. Computing one from training images\n')            
            vocab = build_vocabulary(train_image_paths, vocab_size)
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('vocab.pkl', 'rb') as handle:
                vocab = pickle.load(handle)

        if os.path.isfile('train_image_feats_1.pkl') is False:
            train_image_feats = get_bags_of_sifts(train_image_paths)
            with open('train_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats_1.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)

        if os.path.isfile('test_image_feats_1.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths)
            with open('test_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats_1.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)

        if os.path.isfile('val_image_feats_1.pkl') is False:
            val_image_feats  = get_bags_of_sifts(val_image_paths)
            with open('val_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(val_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('val_image_feats_1.pkl', 'rb') as handle:
                val_image_feats = pickle.load(handle)

    elif FEATURE == 'downscaled_image':
        train_image_feats = get_tiny_images(train_image_paths, size=72)
        test_image_feats = get_tiny_images(test_image_paths, size=72)
        val_image_feats = get_tiny_images(val_image_paths, size=72)

    elif FEATURE == 'dumy_feature':
        train_image_feats = [[0] * 768] * len(train_image_paths)
        test_image_feats = [[0] * 768] * len(test_image_paths)
        val_image_feats = [[0] * 768] * len(val_image_paths)
    else:
        raise NameError('Unknown feature type')

    if CLASSIFIER == 'nearest_neighbor':
        test_predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
        val_predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, val_image_feats)

    elif CLASSIFIER == 'support_vector_machine':
        test_predicted_categories, val_predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, val_image_feats)

    elif CLASSIFIER == 'mlp':
        test_predicted_categories, val_predicted_categories = mlp_classify(train_image_feats, train_labels, test_image_feats, val_image_feats, test_labels, val_labels)

    elif CLASSIFIER == 'dumy_classifier':
        test_predicted_categories = test_labels[:]
        val_predicted_categories = val_labels[:]
        shuffle(test_predicted_categories)
        shuffle(val_predicted_categories)
    else:
        raise NameError('Unknown classifier type')

    calculate_accuracy(test_labels, test_predicted_categories, CATEGORIES, "Test")
    calculate_accuracy(val_labels, val_predicted_categories, CATEGORIES, "Validation")

    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in test_predicted_categories]
    val_labels_ids = [CATE2ID[x] for x in val_labels]
    val_predicted_categories_ids = [CATE2ID[x] for x in val_predicted_categories]

    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES, title='Test Confusion Matrix')
    build_confusion_mtx(val_labels_ids, val_predicted_categories_ids, ABBR_CATEGORIES, title='Validation Confusion Matrix')

    visualize_sift_tsne(train_image_feats)

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories, title):
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title)
    plt.show()

def plot_confusion_matrix(cm, categories, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45)
    plt.yticks(tick_marks, categories)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def visualize_sift_tsne(features):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    reduced_feats = tsne.fit_transform(features)
    plt.scatter(reduced_feats[:, 0], reduced_feats[:, 1], s=2)
    plt.title('t-SNE Visualization of SIFT Features')
    plt.show()

if __name__ == '__main__':
    main()
