import os
from keras.utils import Sequence
import numpy as np

from utility import ct_folder

# https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/
# https://medium.com/datadriveninvestor/keras-training-on-large-datasets-3e9d9dbc09d4
# https://keras.io/utils/#sequence

# TODO implement Pair_Generator for model. Use or join with Loader class (implemented below)

class Pair_Generator(Sequence):

    def __init__(self, benign_filenames, malignant_filenames, batch_size, val_size):
        self.benign_filenames = benign_filenames
        self.malignant_filenames = malignant_filenames
        self.batch_size = batch_size
        self.val_size = val_size


    def __len__(self):
        #return np.ceil(len(self.image_filenames) / float(self.batch_size))
        pass

    def __getitem__(self, idx):
        #batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        #return np.array([
        #    resize(imread(file_name), (200, 200))
        #       for file_name in batch_x]), np.array(batch_y)
        pass


class Loader:

    def __init__(self, training_folder, data_folder, same_benign=True):
        self.training_folder = training_folder
        self.data_folder = data_folder
        self.same_benign = same_benign

        #halve the train count and validation count (for two classes)
        self.train_malignant = np.load(os.path.join(training_folder, "train_malignant.npy"))
        self.train_benign = np.load(os.path.join(training_folder, "train_benign.npy"))

        self.validation_malignant = np.load(os.path.join(training_folder, "validation_malignant.npy"))
        self.validation_benign = np.load(os.path.join(training_folder, "validation_benign.npy"))

        # TODO make save and load method of the list of training data

        # print found classes
        print("train_beingn: {}, train_malignant: {}, validation_benign: {}, validation_malignant: {}\n".format(
            len(self.train_benign), len(self.train_malignant),
            len(self.validation_malignant), len(self.validation_benign)))

        self.data_benign = self.load_train_data(self.train_benign, augment=True)
        self.data_malignant = self.load_train_data(self.train_benign, augment=True)
        self.data_validation_benign = self.load_train_data(self.validation_malignant)
        self.data_validation_malignant = self.load_train_data(self.validation_benign)

    # load data 
    def load_train_data(self, dataset_list, augment = None):
        data_nodules = np.ndarray((0,16,64,64))
        for nodule in dataset_list: #go through benign examples
                valarr = nodule.split('_')
                if (valarr[1] == "img"):
                    data = np.load(os.path.join(self.data_folder, nodule))
                    data_nodules = np.append(data_nodules, [data], axis = 0)
                    # data augmentation: there are 8 flips of image
                    if augment:
                            data_nodules = np.append(data_nodules, [np.flip(data, (0))], axis = 0)
                            data_nodules = np.append(data_nodules, [np.flip(data, (1))], axis = 0)
                            data_nodules = np.append(data_nodules, [np.flip(data, (2))], axis = 0)
                            data_nodules = np.append(data_nodules, [np.flip(data, (0, 1))], axis = 0)
                            data_nodules = np.append(data_nodules, [np.flip(data, (1, 2))], axis = 0)
                            data_nodules = np.append(data_nodules, [np.flip(data, (0, 2))], axis = 0)
                            data_nodules = np.append(data_nodules, [np.flip(data, (0, 1, 2))], axis = 0)
        return data_nodules



    # automatically forming training pairs
    # N is half of of the batch size
    def form_pairs_auto(self, Nhalf, benign, malignant):
        pairs = np.ndarray((0,2,16,64,64))
        pairs_y = np.ndarray((0, 1))
        for i in range (0, Nhalf):
            # replace = False ~ no repeats
            benign_index = np.random.choice(benign.shape[0], 2, replace=False)
            malignant_index = np.random.choice(malignant.shape[0], 2, replace=False)
            A = benign[benign_index[0],:,:,:]
            B = malignant[malignant_index[0],:,:,:]
            C = benign[benign_index[1],:,:,:]
            D = malignant[malignant_index[1],:,:,:]
            # different
            pairs = np.append(pairs, [np.array([A, B])], axis=0)
            pairs_y = np.append(pairs_y, 1) 
            pairs = np.append(pairs, [np.array([C, D])], axis=0)
            pairs_y = np.append(pairs_y, 1) 
            # same
            pairs = np.append(pairs, [np.array([A, C])], axis=0)
            pairs_y = np.append(pairs_y, 0)
            pairs = np.append(pairs, [np.array([B, D])], axis=0)
            pairs_y = np.append(pairs_y, 0)

        pairs = np.swapaxes(pairs, 0, 1)
        return list(pairs), pairs_y

    def form_pairs_auto_no_same_benign(self, Nthird, benign, malignant):
        pairs = np.ndarray((0,2,16,64,64))
        pairs_y = np.ndarray((0, 1))
        for i in range (0, Nthird):
            # replace = False ~ no repeats
            benign_index = np.random.choice(benign.shape[0], 3, replace=False)
            malignant_index = np.random.choice(malignant.shape[0], 3, replace=False)
            b1 = benign[benign_index[0],:,:,:]
            b2 = benign[benign_index[1],:,:,:]
            b3 = benign[benign_index[2],:,:,:]
            m1 = malignant[malignant_index[0],:,:,:]
            m2 = malignant[malignant_index[1],:,:,:]
            m3 = malignant[malignant_index[2],:,:,:]

            # different
            pairs = np.append(pairs, [np.array([b1, m1])], axis=0)
            pairs_y = np.append(pairs_y, 1) 
            pairs = np.append(pairs, [np.array([b2, m2])], axis=0)
            pairs_y = np.append(pairs_y, 1) 
            pairs = np.append(pairs, [np.array([b3, m3])], axis=0)
            pairs_y = np.append(pairs_y, 1) 
            # same
            pairs = np.append(pairs, [np.array([b1, b2])], axis=0)
            pairs_y = np.append(pairs_y, 0)
            pairs = np.append(pairs, [np.array([b1, b3])], axis=0)
            pairs_y = np.append(pairs_y, 0)
            pairs = np.append(pairs, [np.array([b2, b3])], axis=0)
            pairs_y = np.append(pairs_y, 0)

        pairs = np.swapaxes(pairs, 0, 1)
        return list(pairs), pairs_y

    def form_pairs(self, N, benign, malignant):
        if self.same_benign:
                return self.form_pairs_auto(int(np.ceil(N/4)), benign, malignant)
        else:
                return self.form_pairs_auto_no_same_benign(int(np.ceil(N/6)), benign, malignant)