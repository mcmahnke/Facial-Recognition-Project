import numpy as np
import csv
import matplotlib.pyplot as pl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self):
        self.data = fetch_lfw_people(min_faces_per_person = 70, download_if_missing = True)
        x = self.data.data
        y = self.data.target

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

        self.train = np.insert(self.x_train, 0, self.y_train, axis = 1)
        self.test = np.insert(self.x_test, 0, self.y_test, axis = 1)

        self.outputs = len(self.data.target_names)

        #Variables for adjusting training factors
        self.n = 50
        self.momentum = 0.1
        self.l_rate = 0.05
        self.bias = 1

        #Initialize input -> hidden and hidden -> output layers with random weights
        self.i_layer = np.random.uniform(-0.05, 0.05, (2915, self.n))
        self.h_layer = np.random.uniform(-0.05, 0.05, (self.n+1, self.outputs))

        #Initialize matrices to keep track of previous input and hidden layer deltas for use with the momentum term when calculating new weight deltas
        self.prev_i_deltas = np.zeros((2915, self.n))
        self.prev_h_deltas = np.zeros((self.n+1, self.outputs))

        #Initialize 1 x n+1 vector so hidden layer activation result can be dotted with the hidden -> output layer weights
        self.h_activation = np.zeros((1, self.n+1))
        self.h_activation[0,0] = 1


    #Learning function for the MLP: takes in input data, a flag to know if it is a training or test set to determine whether learning is applied,
    # and a flag to determine whether to create a confusion matrix
    def MLP_learning(self, input, test_flag, conf_flag):
        #input data gets randomized each epoch so it isn't memorized
        np.random.shuffle(input)

        #Lists to keep track of predicted and actual values for accuracy and confusion matrix
        p_list = []
        a_list = []

        for i in range(len(input)):
            target = input[i, 0].astype('int')
            a_list.append(target)

            xi = np.copy(input[i])
            xi[0] = self.bias
            xi = xi.reshape(1,2915)

            #Calculate activation for input -> hidden layers
            i_product = np.dot(xi, self.i_layer)
            hj = sigmoid(i_product)
            self.h_activation[0, 1:] = hj

            #Calculate activation for hidden -> output layers
            h_product = np.dot(self.h_activation, self.h_layer)
            ok = sigmoid(h_product)

            prediction = np.argmax(ok)
            p_list.append(prediction)

            #Applies training after each training example
            if test_flag == 0:
                tk = np.zeros((1, self.outputs)) + 0.1
                tk[0, target] = 0.9

                err_ok = ok * (1 - ok) * (tk - ok)
                err_hj = hj * (1 - hj) * np.dot(err_ok, self.h_layer[1:, :].T)

                delta_h = (self.l_rate * err_ok * self.h_activation.T) + (self.momentum * self.prev_h_deltas)
                self.prev_h_deltas = delta_h
                self.h_layer = self.h_layer + delta_h

                delta_i = (self.l_rate * err_hj * xi.T) + (self.momentum * self.prev_i_deltas)
                self.prev_i_deltas = delta_i
                self.i_layer = self.i_layer + delta_i
    
        accuracy = accuracy_score(a_list, p_list)

        #Prints confusion matrix and classification report at the end of the last epoch
        if conf_flag == 1:
            cm = confusion_matrix(a_list, p_list)
            print(cm)
            cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
            cm_display.plot()
            pl.show()
            print(classification_report(a_list, p_list, target_names = self.data.target_names))
        
        return accuracy

    #Records accuracy to a csv file for plotting 
    def record_accuracy(self, index, accuracy, input):
        with open(input, 'a', newline = '') as myfile:
            wr = csv.writer(myfile)
            wr.writerow([index, accuracy])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

mlp = MLP()

#Loop for training on x number of epochs
x = 50
for i in range(x + 1):
    print("Epoch ", i)
    training_acc = mlp.MLP_learning(mlp.train, 0, 0)
    if(i == x):
        testing_acc = mlp.MLP_learning(mlp.test, 1, 1)
    else:
        testing_acc = mlp.MLP_learning(mlp.test, 1, 0)
    mlp.record_accuracy(i, training_acc, 'training_accuracy_' + str(x) + '_x.csv')
    mlp.record_accuracy(i, testing_acc, 'testing_accuracy_' + str(x) + '_x.csv')
