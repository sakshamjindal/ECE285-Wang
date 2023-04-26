"""
K Nearest Neighbours Model
"""
import numpy as np


class KNN(object):
    def __init__(self, num_class: int):
        self.num_class = num_class

    def train(self, x_train: np.ndarray, y_train: np.ndarray, k: int):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(self, x_test: np.ndarray, k: int = None, loop_count: int = 1):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Fill this function in
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)

        # TODO: implement me
        # distance has size (len(self._x_train), len(x_test))
        # so change the size to (len(x_test), len(self._x_train))
        distance = distance.T
        predicted_labels = np.zeros(len(x_test))

        for i in range(len(x_test)):
            # reference for code used :https://www.reddit.com/r/learnpython/comments/rzw5w6/sort_pair_numpy_arrays_based_on_the_first_one/
            # we have distances and y_train and we need to get 
            # top k distances and their indices. So, this is a classical problem
            top_k_indices = np.argsort(distance[i])[:k_test]
            top_k_predictions = self._y_train[top_k_indices]
            
            uniques, counts = np.unique(top_k_predictions, return_counts=True)
            predicted_labels[i] = uniques[np.argmax(counts)]

        return predicted_labels 

    
    def calc_dis_one_loop(self, x_test: np.ndarray):
        """a
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """

        distances = np.zeros((len(self._x_train), len(x_test)))

        for i in range(len(self._x_train)):
            distances[i] = np.linalg.norm(self._x_train[i] - x_test, axis=1)

        return distances
    

    def calc_dis_two_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        # TODO: implement me
        
        distances = np.zeros((len(self._x_train), len(x_test)))

        for i in range(len(self._x_train)):
            for j in range(len(x_test)):
                distances[i][j] = np.linalg.norm(self._x_train[i] - x_test[j])
        
        return distances

        
