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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold
from sklearn.manifold import TSNE

# Step 0: Set up parameters, category list, and image paths.

#For this project, you will need to report performance for three
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier
# 3) Bag of sift features and linear SVM classifier
#The starter code is initialized to 'placeholder' just so that the starter
#code does not crash when run unmodified and you can get a preview of how
#results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()

DATA_PATH = '../../UCMerced_LandUse/Images/'

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

CATEGORIES = [
            'agricultural', 'baseballdiamond', 'buildings', 'denseresidential', 'freeway', 'harbor', 'mediumresidential', 'overpass', 'river', 
            'sparseresidential', 'tenniscourt', 'airplane', 'beach', 'chaparral', 'forest', 'golfcourse', 'intersection', 'mobilehomepark', 'parkinglot',
            'runway', 'storagetanks'
        ]
CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}
ABBR_CATEGORIES = ['Agr', 'Bas', 'Bui', 'Den', 'Fre', 'Har', 'Med', 'Ove', 'Riv', 'Spa', 'Ten', 'Air', 'Bea', 'Cha', 'For', 'Gol', 'Int', 'Mob', 'Par', 'Run', 'Sto']


FEATURE = args.feature
# FEATUR  = 'bag of sift'

CLASSIFIER = args.classifier
# CLASSIFIER = 'support vector machine'

#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as
#well.

def calculate_accuracy(labels, predictions, categories, split_name):
    accuracy = float(len([x for x in zip(labels, predictions) if x[0] == x[1]])) / len(labels)
    print(f"{split_name} Accuracy = {accuracy:.2f}")
    for category in categories:
        accuracy_each = float(len([x for x in zip(labels, predictions) if x[0] == x[1] and x[0] == category])) / float(labels.count(category))
        print(f"{category}: {accuracy_each:.2f}")

def main():
    #This function returns arrays containing the file path for each train
    #and test image, as well as arrays with the label of each train and
    #test image. By default all four of these arrays will be 1500 where each
    #entry is a string.
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels = \
        get_image_paths(DATA_PATH, CATEGORIES)

    # TODO Step 1:
    # Represent each image with the appropriate feature
    # Each function to construct features should return an N x d matrix, where
    # N is the number of paths passed to the function and d is the 
    # dimensionality of each image representation. See the starter code for
    # each function for more details.

    if FEATURE == 'tiny_image':
        # YOU CODE get_tiny_images.py 
        train_image_feats = get_tiny_images(train_image_paths)
        test_image_feats = get_tiny_images(test_image_paths)
        # print(len(test_image_feats))
        # print(test_image_feats[0])
        # print(len(test_image_feats[0]))
        val_image_feats = get_tiny_images(val_image_paths)
        # print(len(val_image_feats))
        # print(val_image_feats[0])
        # print(len(val_image_feats[0]))

    elif FEATURE == 'bag_of_sift':
        # YOU CODE build_vocabulary.py
        vocab_size = 400   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
        if os.path.isfile('vocab.pkl') is False:
            print('No existing visual word vocabulary found. Computing one from training images\n')            
            vocab = build_vocabulary(train_image_paths, vocab_size)
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('vocab.pkl', 'rb') as handle:
                vocab = pickle.load(handle)

        if os.path.isfile('train_image_feats_1.pkl') is False:
            # YOU CODE get_bags_of_sifts.py
            train_image_feats = get_bags_of_sifts(train_image_paths);
            with open('train_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats_1.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)

        if os.path.isfile('test_image_feats_1.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths);
            with open('test_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats_1.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)

        if os.path.isfile('val_image_feats_1.pkl') is False:
            val_image_feats  = get_bags_of_sifts(val_image_paths);
            with open('val_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(val_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('val_image_feats_1.pkl', 'rb') as handle:
                val_image_feats = pickle.load(handle)

    elif FEATURE == 'dumy_feature':
        train_image_feats = [[0] * 768] * len(train_image_paths)
        test_image_feats = [[0] * 768] * len(test_image_paths)
        val_image_feats = [[0] * 768] * len(val_image_paths)
    else:
        raise NameError('Unknown feature type')

    # TODO Step 2: 
    # Classify each test image by training and using the appropriate classifier
    # Each function to classify test features will return an N x 1 array,
    # where N is the number of test cases and each entry is a string indicating
    # the predicted category for each test image. Each entry in
    # 'predicted_categories' must be one of the 15 strings in 'categories',
    # 'train_labels', and 'test_labels.

    if CLASSIFIER == 'nearest_neighbor':
        # YOU CODE nearest_neighbor_classify.py
        test_predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
        val_predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, val_image_feats)

    elif CLASSIFIER == 'support_vector_machine':
        # YOU CODE svm_classify.py
        test_predicted_categories, val_predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, val_image_feats)

    elif CLASSIFIER == 'dumy_classifier':
        # The dummy classifier simply predicts a random category for
        # every test case
        print(len(train_image_feats))
        test_predicted_categories = test_labels[:]
        val_predicted_categories = val_labels[:]
        shuffle(test_predicted_categories)
        shuffle(val_predicted_categories)
    else:
        raise NameError('Unknown classifier type')

    # TODO Step 3:
    # Compute the accuracy of the classification
    calculate_accuracy(test_labels, test_predicted_categories, CATEGORIES, "Test")
    calculate_accuracy(val_labels, val_predicted_categories, CATEGORIES, "Validation")

    # Step 4: Build a confusion matrix and score the recognition system
    # You do not need to code anything in this section.
    # This code builds a confusion matrix and visualizes it.
    # It then saves the confusion matrix plot as 'confusion_matrix.png'
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in test_predicted_categories]
    val_labels_ids = [CATE2ID[x] for x in val_labels]
    val_predicted_categories_ids = [CATE2ID[x] for x in val_predicted_categories]

    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES, title='Test Confusion Matrix')
    build_confusion_mtx(val_labels_ids, val_predicted_categories_ids, ABBR_CATEGORIES, title='Validation Confusion Matrix')

    # Step 5: Visualize the SIFT keypoints
    # visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, val_labels_ids, val_predicted_categories_ids)

    # Plot accuracy vs. codewords
    plot_accuracy_vs_codewords(train_image_paths, train_labels)

    # Visualize SIFT keypoints with t-SNE
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

def plot_accuracy_vs_codewords(image_paths, labels):
    # Ensure labels is a NumPy array
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
    
    # Validate input dimensions
    codeword_sizes = [50, 100, 200, 400, 800]
    accuracies = []
    for size in codeword_sizes:
        vocab = build_vocabulary(image_paths, size)
        features = get_bags_of_sifts(image_paths)
        
        # Ensure features and labels align
        assert len(features) == len(labels), "Features and labels must have the same number of samples."
        
        kfold = KFold(n_splits=5)
        fold_accuracies = []
        for train_idx, test_idx in kfold.split(features):
            train_feats, test_feats = features[train_idx], features[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]
            
            # Ensure predicted_labels is correctly compared
            test_predicted_labels = nearest_neighbor_classify(train_feats, train_labels, test_feats)
            fold_accuracies.append(np.mean(np.array(test_predicted_labels) == np.array(test_labels)))
        accuracies.append(np.mean(fold_accuracies))

    plt.plot(codeword_sizes, accuracies, marker='o')
    plt.title('Accuracy vs. Number of Codewords')
    plt.xlabel('Number of Codewords')
    plt.ylabel('Accuracy')
    plt.show()

def visualize_sift_tsne(features):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    reduced_feats = tsne.fit_transform(features)
    plt.scatter(reduced_feats[:, 0], reduced_feats[:, 1], s=2)
    plt.title('t-SNE Visualization of SIFT Features')
    plt.show()

if __name__ == '__main__':
    main()