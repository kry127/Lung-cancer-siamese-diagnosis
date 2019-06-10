import os
from keras.utils import Sequence
import numpy as np

from utility import ct_folder, cancer_folder

# https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/
# https://medium.com/datadriveninvestor/keras-training-on-large-datasets-3e9d9dbc09d4
# https://keras.io/utils/#sequence

# TODO implement Pair_Generator for model. Use or join with Loader class (implemented below)

class Pair_Generator(Sequence):

    def __init__(self, loader, batch_size):
        self.loader = loader
        self.batch_size = batch_size // 2 # because we balance pairs
        self.same_benign = loader.same_benign
        if self.same_benign:
            self.length = np.max([self.loader.len_different(), self.loader.len_same()])
            self.index_diff = np.arange(self.loader.len_different())
            self.index_same = np.arange(self.loader.len_same())
        else:
            self.length = np.max([self.loader.len_different(), self.loader.len_malignant_malignant()])
            self.index_diff = np.arange(self.loader.len_different())
            self.index_same = np.arange(self.loader.len_malignant_malignant())

        len_diff = len(self.index_diff)
        len_same = len(self.index_same)
        print("Generator. len_diff={}, len_same={}".format(len_diff, len_same))
        if (len_diff > len_same):
            self.index_same = np.tile(self.index_same, int(np.ceil(len_diff / len_same)))
            self.index_same = self.index_same[:len_diff]
        elif (len_same > len_diff):
            self.index_diff = np.tile(self.index_diff, int(np.ceil(len_same / len_diff)))
            self.index_diff = self.index_diff[:len_same]

        #shuffle training set
        self.shuffle_indices()
        
    def shuffle_indices(self):
        """function shuffles training set
        """
        np.random.shuffle(self.index_same)
        np.random.shuffle(self.index_diff)

    def __len__(self):
        return int(np.ceil(self.length / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_same = self.index_same[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_diff = self.index_diff[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.loader.get_pair_batch(batch_same, batch_diff)

    def on_epoch_end(self):
        self.shuffle_indices()

    
class Nodule_Generator(Sequence):
    def __init__(self, loader, batch_size, train_p = 0.8):
        """
        loader -- instance of class Loader
        batch_size -- size of mini batch
        train_p -- training data percentage. Other data is validation
        """
        self.loader = loader
        self.batch_size = batch_size
        self.train_p = train_p
        if (train_p < 0.0 or train_p > 1.0):
            raise "Unexpected value of 'train_p' parameter in Nodule Generator"

        self.len_benign = self.loader.len_benign()
        self.len_malignant = self.loader.len_malignant()
        self.length = self.len_benign + self.len_malignant
        self.index = np.arange(self.length)

        print("Generator. len_benign={}, len_malignant={}".format(
            self.loader.len_benign(), self.loader.len_malignant()))

        #shuffle training set
        self.shuffle_indices()

    def shuffle_indices(self):
        np.random.shuffle(self.index)
            
    def __len__(self):
        return int(np.ceil(self.length*self.train_p / float(self.batch_size)))

    def __getitem__(self, idx):
        id_from = idx * self.batch_size
        id_to = (idx + 1) * self.batch_size
        if id_to > self.length * self.train_p:
            id_to = int(self.length * self.train_p) + 1
        batch_id = self.index[id_from:id_to]
        return self.loader.get_single_batch(batch_id)

    def on_epoch_end(self):
        self.shuffle_indices()

    def validation_generator(self):
        return Nodule_Validation_Generator(self)

class Nodule_Validation_Generator(Sequence):
    def __init__(self, nodule_generator):
        """
        nodule_generator -- instance of class Nodule Generator
        """
        self.nodule_generator = nodule_generator
        self.length = nodule_generator.length
        self.train_p = nodule_generator.train_p
        self.batch_size = nodule_generator.batch_size
            
    def __len__(self):
        return int(np.ceil(self.length*(1-self.train_p) / float(self.batch_size)))

    def __getitem__(self, idx):
        idx += len(self.nodule_generator)
        id_from = idx * self.batch_size
        id_to = (idx + 1) * self.batch_size
        if id_from <= self.length * self.train_p:
            id_from = int(self.length * self.train_p) + 1
        batch_id = self.nodule_generator.index[id_from:id_to]
        return self.nodule_generator.loader.get_single_batch(batch_id)



def convert_index_linear_to_triangle(n, index):
    """
this function is needed for index conversion from linear to triangle

for example, for n = 5:
      j=|01234
    ----+-----
    i=0 |X0123
    i=1 |XX456
    i=2 |XXX78
    i=3 |XXXX9
    i=4 |XXXXX

linear index is within the matrix (e.g. 0-9), triangle index is the matrix index

Example: 7 converts to (2, 3)
    """
    M = n*(n-1)//2 - 1 # this is maximum possible 'index' value
    index %= M+1
    i = n - 1 - int(np.floor((1 + np.sqrt(1 + 8*(M - index)))/2))
    j = index - M + (n - 1 - i)*(n - i) // 2 + i
    return (i, j) # return number of row and number of column (i < j)

# vice-versa
def convert_index_triangle_to_linear(n, i, j):
    """
this function is needed for index conversion from triangle to linear

for example, for n = 5:
      j=|01234
    ----+-----
    i=0 |X0123
    i=1 |XX456
    i=2 |XXX78
    i=3 |XXXX9
    i=4 |XXXXX

linear index is within the matrix (e.g. 0-9), triangle index is the matrix index

Example: (1, 3) converts to 5
    """
    i, j = i % n, j % n
    M = n*(n-1)//2 - 1 # this is maximum possible 'index' value
    return M - (n - 1 - i)*(n - i) // 2 + j - i


def get_pair(arr1, arr2, i1, i2):
    A = arr1[i1,:,:,:]
    B = arr2[i2,:,:,:]
    pair = np.array([A, B])
    return pair


    # load data 
def load_train_data(folder, dataset_list, augment = None):
    data_nodules = np.ndarray((0,16,16,16))
    for nodule in dataset_list: #go through benign examples
        valarr = nodule.split('_')
        if (valarr[1] == "img"):
            data = np.load(os.path.join(folder, nodule))
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

class Loader:
    """
    This class handles:
    1. loading ct images
    2. accesing ct image pairs using linear index
    """
    def __init__(self, training_folder, benign_folder, malignant_folder, same_benign=True, augmentation=True):
        self.training_folder = training_folder
        self.benign_folder = benign_folder
        self.malignant_folder = malignant_folder
        self.same_benign = same_benign
        self.augmentation = augmentation

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

        self.data_benign = load_train_data(self.benign_folder, self.train_benign, augment=self.augmentation)
        self.data_malignant = load_train_data(self.malignant_folder, self.train_malignant, augment=self.augmentation)
        self.data_validation_benign = load_train_data(self.benign_folder, self.validation_benign)
        self.data_validation_malignant = load_train_data(self.malignant_folder, self.validation_malignant)
   
    def get_training_generator(self, batch_size, train_p = 0.8):
        return Nodule_Generator(self, batch_size, train_p)

    def get_training_pair_generator(self, batch_size):
        return Pair_Generator(self, batch_size)

    def len_benign(self):
        return self.data_benign.shape[0]

    def len_malignant(self):
        return self.data_malignant.shape[0]

    def len_benign_benign(self):
        N = self.len_benign()
        return (N - 1) * N // 2

    def len_malignant_malignant(self):
        N = self.len_malignant()
        return (N - 1) * N // 2

    def len_different(self):
        return self.data_benign.shape[0] * self.data_malignant.shape[0]

    def len_same(self):
        return self.len_benign_benign() + self.len_malignant_malignant()

    def get_benign_benign(self, index):
        i, j = convert_index_linear_to_triangle(self.data_benign.shape[0], index)
        return get_pair(self.data_benign, self.data_benign, i, j)

    def get_malignant_malignant(self, index):
        i, j = convert_index_linear_to_triangle(self.data_malignant.shape[0], index)
        return get_pair(self.data_malignant, self.data_malignant, i, j)

    def get_different(self, index):
        w = self.data_malignant.shape[0]
        i = index // w # benign index
        j = index - i*w # malignant index
        return get_pair(self.data_benign, self.data_malignant, i, j)

    def get_same(self, index):
        b = self.len_benign_benign()
        if (index < b):
            return self.get_benign_benign(index)
        else:
            return self.get_malignant_malignant(index - b)
            
    def get_single_batch(self, id_array):
        nodules = np.ndarray((0,16,16,16))
        nodules_y = np.ndarray((0, 2))
        for index in id_array:
            if index < self.len_benign():
                nodule = self.data_benign[index]
                label = np.array([1, 0]) #categorical crossentropy labels
            else:
                nodule = self.data_malignant[index - self.len_benign()]
                label = np.array([0, 1])
            
            nodules = np.append(nodules, [nodule], axis=0)
            nodules_y = np.append(nodules_y, label)

        # return batch
        np.expand_dims(nodules, axis = -1)
        return nodules, nodules_y


    def get_pair_batch(self, id_same, id_different):
        pairs = np.ndarray((0,2,16,16,16))
        pairs_y = np.ndarray((0, 1))
        # form same pairs first
        for id_s in id_same:
            if self.same_benign:
                pair = self.get_same(id_s)
            else:
                pair = self.get_malignant_malignant(id_s)
            pairs = np.append(pairs, [pair], axis=0)
            pairs_y = np.append(pairs_y, 0) 
        # then form different pairs
        for id_d in id_different:
            pair = self.get_different(id_d)
            pairs = np.append(pairs, [pair], axis=0)
            pairs_y = np.append(pairs_y, 1) 
        
        # return batch
        pairs = np.swapaxes(pairs, 0, 1)
        return list(pairs), pairs_y



    # automatically forming training pairs
    # N is half of of the batch size
    def form_pairs_auto(self, Nhalf, benign, malignant):
        pairs = np.ndarray((0,2,16,16,16))
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
        pairs = np.ndarray((0,2,16,16,16))
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