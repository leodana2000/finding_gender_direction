import random
import numpy
from pandas import DataFrame

class DataStorage:
    def __init__(self, data, batch_size = 1024, rand_seed = 42, train_test_ratio = 0.9) -> None:
        
        self.data : DataFrame = data
        self.nb_class : int = data['label'].nunique()
        self.label_names : list[str] = data['label'].unique()
        self.is_training : list[bool]
        self.dim_labels : list[list[int]]

        self.batch_size : int = batch_size
        self.rand_seed : int = rand_seed
        self.train_test_ratio : int = train_test_ratio

        self.init_train_test()
        self.init_labels()


    def init_labels(self) -> None:
        '''
        Transform the labels into a nb_class vector with one binary coordinate.
        '''
        dim_labels = []
        for label, valence in zip(self.data['label'], self.data['bin']):
            dim_label = [0]*self.nb_class
            for i, name in enumerate(self.label_names):
                if label == name:
                    dim_label[i] = valence
            dim_labels.append(dim_label)

        self.dim_labels = dim_labels


    def init_train_test(self) -> None:
        '''
        Initialise which data will be in training and which will be in test.
        We garantee the same proportion of train and test for each class.
        '''
        random.seed(self.rand_seed)
        self.is_training = [True]*len(self.data['examples'])

        for name in self.label_names:
            training = []
            nb_data = len(self.data[self.data['label'] == name])
            nb_train = int(self.train_test_ratio*nb_data)
            training = [True]*nb_train + [False]*(nb_data - nb_train)
            random.shuffle(training)

            index = 0
            for i, ex_name in enumerate(self.data['label']):
                if ex_name == name:
                    self.is_training[i] = training[index]
                    index += 1


    def batch(self, list : list[str]) -> list[list[str]]:
        '''
        Transform the data into subsets of size at least bach_size.
        '''
        Nb_ex = len(list)
        Nb_batch = Nb_ex//self.batch_size + 1
        return [list[i*self.batch_size:min((i+1)*self.batch_size, Nb_ex)] for i in range(Nb_batch)]


    def get_ex(self, method, multi_dim=True) -> list[list[str, list[int]]]:
        '''
        Returns the batched data to train, test, or learn. 
        Learning takes every example to learn the directions.
        Train and Test splits the data 
        '''
        labels = self.get_labels(multi_dim=multi_dim)
        sentences = self.data['examples']
        if method == 'train':
            examples = [[sentence, label] for sentence, label, is_train in zip(sentences, labels, self.is_training) if is_train]
        elif method == 'test':
            examples = [[sentence, label] for sentence, label, is_train in zip(sentences, labels, self.is_training) if not is_train]
        elif method == 'learn':
            examples = [[sentence, label] for sentence, label in zip(sentences, labels)]
        return self.batch(examples)


    def get_labels(self, multi_dim=True) -> list[list[int]]:
        '''
        Changes the format of the labels if you want to have it one dimensional or not, and returns them.
        '''
        if multi_dim:
            return self.dim_labels
        else:
            return [[1] if (1 in label) else [-1] for label in self.dim_labels]