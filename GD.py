import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import warnings
from sklearn import preprocessing
from sklearn import linear_model
import random

warnings.filterwarnings("ignore")

# X = np.array(([1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 9], [1, 10], [1, 100]))
# Y = np.array(([1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [100]))

X = pd.read_csv('E:\Facul\MC886\Trabalho1\diamonds-dataset\diamonds-train.csv', delimiter=',',
                                                                                header=0,
                                                                                usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
Y = pd.read_csv('E:\Facul\MC886\Trabalho1\diamonds-dataset\diamonds-train.csv', delimiter=',',
                                                                                header=0,
                                                                                usecols=[9])

X = X.replace({"Ideal": 5, "Premium": 4, "Very Good": 3, "Good": 2, "Fair": 1})
X = X.replace({"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7})
X = X.replace({"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8})

min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
X_normalized = pd.DataFrame(X_scaled)

X = X_normalized

rows = np.size(X, 0)

uns = np.ones(rows)
uns = uns.reshape(rows, 1)

X = X.values
X = np.append(uns, X, axis=1)

Y = Y.values


# X = np.array(([1, 1], [1, 2], [1, 3], [1, 4]))
# Y = np.array(([1], [2], [3], [4]))

# N = 100

# X = np.empty((N,2))
# Y = np.empty(N)
#
# for i in range(N):
#    X[i][0] = 1
#    X[i][1] = i
#    Y[i]    = i
#
# print(X)
# print(X_normalized)
# print(Y)

def scikitReg():
    regr = linear_model.SGDRegressor(max_iter=1000, eta0=0.01)
    regr.fit(X, Y)
    params = regr.coef_
    print(params)


class Model:

    def training_BatchGD(X, Y, learning_rate, epochs):

        # Data Modification
        X_transpose = np.transpose(X)
        Y_transpose = np.transpose(Y)

        # Setting internal variables
        N_data, N_features = np.shape(X)
        W = np.zeros(N_features)
        Cost = np.empty(epochs)
        Ws = np.empty(epochs*10).reshape(epochs, 10)

        #Main loop
        for i in range(epochs):

            Loss = (np.dot(W, X_transpose)) - Y_transpose
            Cost[i] = np.sum( Loss ** 2 ) / (2 * N_data)   #verificar
            dCost = np.dot (Loss, X) / N_data

            W = W - learning_rate * dCost

            Ws[i] = W

            print(i)
        print(f' Cost: {dCost} \n   W: {W}   \n ------------------')
        # print((Cost))

        plt.plot(Cost, 'ro')
        plt.show()

        return Ws

    def training_StochasticGD(X, Y, learning_rate, epochs):

        #Data Modification
        X_transpose = np.transpose(X)
        Y_transpose = np.transpose(Y)

        #Setting internal variables
        N_data, N_features = np.shape(X)
        W = np.zeros(N_features)
        Cost = np.zeros(epochs)

        #Main loop
        for i in range(epochs):
            for j in range(N_data):
                Loss = np.dot(W, np.reshape(X_transpose[:, j], (N_features, 1))) - Y[j]
                Cost[i] = np.sum( Loss ** 2 ) / 2

                #print(np.reshape(X_transpose[:,j], (2,1)))
                #print(np.shape(W))
                #print(np.shape(X_transpose))
                #print(Y[j])
                #print(Loss)
                #print(type(Loss))
                #print(X[j])


                dCost = (Loss * X[j])

                W = W - learning_rate * dCost
            print(i)

        print(f'dCost: {dCost} \n Fim dCost \n W: {W}   \n ------------------')
        #print((Cost))

        plt.plot(Cost, 'ro')
        plt.show()

    def training_MiniBatchGD(X, Y, learning_rate, epochs, b):

        # Data Modification
        X_transpose = np.transpose(X)
        Y_transpose = np.transpose(Y)

        # Setting internal variables
        N_data, N_features = np.shape(X)
        W = np.zeros(N_features)
        Cost = np.zeros(epochs)


        nBatch = N_data/b

        for k in range(epochs):
            j = 0
            N_data = b
            for i in range(math.ceil(nBatch)):

                if not(nBatch.is_integer()) and (i == nBatch):
                    B2 = N_data = N_data % b
                    X_segmentado = X[j:j + B2][:]
                    Y_segmentado = Y[j:j + B2]
                else:

                    X_segmentado = X[j:j + b][:]
                    Y_segmentado = Y[j:j + b]
                    #print(X_segmentado)
                    #print(Y_segmentado)
                    X_segmentadoT = np.transpose(X_segmentado)
                    Y_segmentadoT = np.transpose(Y_segmentado)
                    #print(X_segmentadoT)
                    #print(Y_segmentadoT)

                    j += b

                Loss = (np.dot(W, X_segmentadoT)) - Y_segmentadoT
                Cost[i] = np.sum(Loss ** 2) / (2 * N_data)  # verificar
                dCost = np.dot(Loss, X_segmentado) / N_data

                W = W - learning_rate * dCost
            print(k)
        print(f' dCost: {dCost} \n   W: {W}   \n ------------------')
        # print((Cost))

        plt.plot(Cost, 'ro')
        plt.show()

    def solveLinearEquation(W, example):
        resultado = W[0]

        for i in range(len(W) - 1):
            resultado += W[i + 1] * example[i]

        return resultado

    def normalEquation(X, Y):
        X_transpose = np.transpose(X)
        Z = np.dot(X_transpose, X)
        Z = np.linalg.inv(Z)
        Z = np.dot(Z, X_transpose)
        W = np.dot(Z, Y)
        print(W)

    def getValidationTest(X, Y, percentualSize):
        rows = np.size(X, 0)
        columns = len(X[0])
        vElements = round(percentualSize * rows)
        xValidation = np.ndarray(0)
        yValidation = np.ndarray(0)

        for i in range(vElements):
            row = random.randint(0, rows-1)

            xValidation = np.append(xValidation, X[row])
            yValidation = np.append(yValidation, Y[row])
            X = np.delete(X, row, axis=0)
            Y = np.delete(Y, row, axis=0)

            rows -= 1

        xValidation = xValidation.reshape(vElements, columns)
        yValidation = yValidation.reshape(vElements, 1)

        return X, Y, xValidation, yValidation

    def validate(Ws, xValidation, yValidation):
        predictions = np.ndarray(0)

        for W in Ws:
            for diamond in xValidation:
                predictions = np.append(predictions, Model.solveLinearEquation(W, diamond))
                print(predictions[W])


# Data_Generator(100)

X, Y, xValidation, yValidation = Model.getValidationTest(X, Y, 0.4)
Ws = Model.training_BatchGD(X, Y, 0.01, 10)
Model.validate(Ws, xValidation, yValidation)

# Model.training_StochasticGD(X, Y, 0.01, 50)
# Model.training_MiniBatchGD(X, Y, 0.001, 1000, 5000)
# scikitReg()
# Model.normalEquation(X, Y)

