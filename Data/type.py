import random

class DataStorage:
    def __init__(self, sentences, labels, valence, nb_class, label_names, 
                 batch_size = 1024, rand_seed = 42, train_test_ratio = 0.9) -> None:
        
        self.sentences : list[str] = sentences
        self.labels : list[list[str]] = labels
        self.valences : list[1 | -1] = valence
        self.nb_class : int = nb_class
        self.label_names : list[str] = label_names
        self.is_training : list[bool]
        self.dim_labels : list[list[int]]

        self.batch_size : int = batch_size
        self.rand_seed : int = rand_seed
        self.train_test_ratio = train_test_ratio

        self.init_train_test()
        self.init_labels()


    def init_labels(self) -> None:
        '''
        Transform the labels into a nb_class vector with one valence coordinate.
        '''
        dim_labels = []
        for label, valence in zip(self.labels, self.valences):
            dim_label = [0]*self.nb_class
            for i, name in enumerate(self.label_names):
                if label == name:
                    dim_label[i] = valence

        self.dim_labels = dim_labels


    def init_train_test(self) -> None:
        '''
        Initialise which data will be in training and which will be in test.
        '''
        random.seed(self.rand_seed)
        nb_data = len(self.sentences)
        nb_train = (self.train_test_ratio*nb_data)//1
        self.is_training = [True]*nb_train + [False]*(nb_data - nb_train)
        random.shuffle(self.is_training)


    def add(self, new_sentences, new_labels, new_valences) -> None:
        '''
        Adds a new set of data.
        '''
        self.sentences.append(new_sentences)
        self.labels.append(new_labels)
        self.valences.append(new_valences)

        self.init_labels()


    def batch(self, list : list[str]) -> list[list[str]]:
        '''
        Transform the data into subsets of size at least bach_size.
        '''
        Nb_ex = len(list)
        Nb_batch = Nb_ex//self.batch_size + 1
        return [list[i*self.batch_size:min((i+1)*self.batch_size, Nb_ex)] for i in range(Nb_batch)]


    def get_ex(self, method) -> list[list[str]]:
        '''
        Returns the batched data to train, test, or learn. 
        Learning takes every example to learn the directions.
        Train and Test splits the data 
        '''
        if method == 'train':
            examples = [sentence for sentence, is_train in zip(self.sentences, self.is_training) if is_train]
        elif method == 'test':
            examples = [sentence for sentence, is_learn in zip(self.sentences, self.is_learning) if not is_learn]
        elif method == 'learn':
            examples = self.sentences
        return self.batch(examples)


    def get_labels(self, multi_dim=True) -> list[list[int]]:
        '''
        Changes the format of the labels if you want to have it one dimensional or not, and returns them.
        '''
        if multi_dim:
            return self.dim_labels
        else:
            return [[1] if (1 in label) else [-1] for label in self.dim_labels]