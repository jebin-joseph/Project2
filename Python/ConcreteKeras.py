import sys

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# models
perceptron_ = True
NN_3L_ = True
NN_XL_ = True

# number of splits for k-fold cross validation (must be >= 1 or an error will occur)
n_split = 10

# choose optimizers (one true at a time)
SGD_ = False
SGD_momentum_ = True
Adam_ = False

# activation functions for perceptron (one true at a time)
sigmoid_0 = False
tanh_0 = True
reLU_0 = False

# activation function 1 for neural net 3L (one true at a time)
sigmoid_1 = False
tanh_1 = False
reLU_1 = True

# activation function 2 for neural net 3L (one true at a time)
sigmoid_2 = False
tanh_2 = True
reLU_2 = False

# activation function 1 for neural net XL (one true at a time)
sigmoid_3 = False
tanh_3 = False
reLU_3 = True

# activation function 2 for neural net XL (one true at a time)
sigmoid_4 = True
tanh_4 = False
reLU_4 = False

# activation function 3 for neural net XL (one true at a time)
sigmoid_5 = False
tanh_5 = True
reLU_5 = False


# sel_cols_indices is the indices of already selected cols
# xcolumns specifies indices of all cols of x
def forwardSel(sel_cols_indices, x, xcolumns, y, model):
    j_mx = -1  # best column, so far
    fit_mx = - sys.float_info.max  # best fit, so far
    # print("start for loop")
    for j in xcolumns:
        if not j in sel_cols_indices:
            if len(sel_cols_indices) == 0:
                cols_j = [j]
            else:
                cols_j = sel_cols_indices.copy()
                cols_j.append(int(j))  # try adding variable x_j
            x_cols = x[:, cols_j]  # x projected onto cols_j
            built_model = buildModel(x_cols, model)  # build a model with x_j added
            # fit model
            built_model.fit(x_cols, y, epochs=50, verbose=0)
            predict = built_model.predict(x_cols)
            fit_j = r2_score(y, predict)  # evaluate QoF
            if fit_j > fit_mx:
                j_mx = j
                fit_mx = fit_j
    if j_mx == -1:
        print("forwardSel: could not find a variable x_j to add: j = -1")
    # print("best attribute: " + str(j_mx))
    return j_mx  # return best column


# x cols is the data for selected x cols
# returns the r Sq value of the model
def buildModel(x_cols, model):
    if model == 'Perceptron':
        model = Sequential()

        # determines activation function for model (tanh by default)
        act_fun_0 = 'tanh'
        if sigmoid_0:
            act_fun_0 = 'sigmoid'
        elif tanh_0:
            act_fun_0 = 'tanh'
        elif reLU_0:
            act_fun_0 = 'relu'

        # input layer and output layer (1 node) declaration
        model.add(Dense(1, input_dim=x_cols.shape[1], activation=act_fun_0))

        # optimizer (Adam by default)
        opt = Adam(lr=.01)
        if SGD_:
            opt = SGD(lr=0.01, momentum=0.0)
        elif SGD_momentum_:
            opt = SGD(lr=0.01, momentum=0.9)
        elif Adam_:
            opt = Adam(lr=0.01)

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

        return model

    elif model == 'NN_3L':
        model = Sequential()
        # determines activation function for model (tanh by default)
        act_fun_1 = 'tanh'
        if sigmoid_1:
            act_fun_1 = 'sigmoid'
        elif tanh_1:
            act_fun_1 = 'tanh'
        elif reLU_1:
            act_fun_1 = 'relu'

        act_fun_2 = 'tanh'
        if sigmoid_2:
            act_fun_2 = 'sigmoid'
        elif tanh_2:
            act_fun_2 = 'tanh'
        elif reLU_2:
            act_fun_2 = 'relu'
        # input layer and hidden layer
        model.add(Dense(12, input_dim=x_cols.shape[1], activation=act_fun_1))
        # outut layer
        model.add(Dense(1, activation=act_fun_2))

        # optimizer (Adam by default)
        opt = Adam(lr=.01)
        if SGD_:
            opt = SGD(lr=0.01, momentum=0.0)
        elif SGD_momentum_:
            opt = SGD(lr=0.01, momentum=0.9)
        elif Adam_:
            opt = Adam(lr=0.01)

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

        return model

    if model == 'NN_XL':
        model = Sequential()
        # determines activation function for model (tanh by default)
        act_fun_3 = 'tanh'
        if sigmoid_3:
            act_fun_3 = 'sigmoid'
        elif tanh_3:
            act_fun_3 = 'tanh'
        elif reLU_3:
            act_fun_3 = 'relu'

        act_fun_4 = 'tanh'
        if sigmoid_4:
            act_fun_4 = 'sigmoid'
        elif tanh_4:
            act_fun_4 = 'tanh'
        elif reLU_4:
            act_fun_4 = 'relu'

        act_fun_5 = 'tanh'
        if sigmoid_5:
            act_fun_5 = 'sigmoid'
        elif tanh_5:
            act_fun_5 = 'tanh'
        elif reLU_5:
            act_fun_5 = 'relu'

        # input layer and hidden layer
        model.add(Dense(12, input_dim=x_cols.shape[1], activation=act_fun_3))
        # hidden layer
        model.add(Dense(12, activation=act_fun_4))
        # output layer
        model.add(Dense(1, activation=act_fun_5))

        # optimizer (Adam by default)
        opt = Adam(lr=.01)
        if SGD_:
            opt = SGD(lr=0.01, momentum=0.0)
        elif SGD_momentum_:
            opt = SGD(lr=0.01, momentum=0.9)
        elif Adam_:
            opt = Adam(lr=0.01)

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

        return model


def runner(model_type):
    print(model_type)
    # lists made to record r Sq values vs number of attributes and plot
    num_attrs_perc = [i + 1 for i in range(len(xcolumns))]
    r_sq_record = [0] * len(xcolumns)
    r_sq_bar_record = [0] * len(xcolumns)
    r_sq_cv_record = [0] * len(xcolumns)

    # list to contain the attribute indices selected by forward selection
    selected_cols = []
    # forward select to iteratively add best attributes
    for i in range(len(xcolumns)):
        new_col = forwardSel(selected_cols, x, xcolumns, y, model_type)
        if new_col != -1:
            if len(selected_cols) == 0:
                selected_cols = [new_col]
            else:
                selected_cols.append(new_col)
        # print("selected cols: ", selected_cols)

        # makes a subset of the data using only the cols passed from forward selection
        x_selected = x[:, selected_cols]
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

        # k-fold cross validation used for r Sq CV
        # counter = 1
        sum = 0
        for train_index, test_index in KFold(n_split).split(x):
            # print("k-fold iteration " + str(counter))
            # counter = counter + 1
            x_train, x_test = x_selected[train_index], x_selected[test_index]
            y_train, y_test = y[train_index], y[test_index]
            built_model_k = buildModel(x_selected, model_type)
            history = built_model_k.fit(x_selected, y, validation_data=(x_test, y_test), epochs=75,
                                                  verbose=0, callbacks=[es])
            built_model_k_preds = built_model_k.predict(x_test)
            sum = sum + r2_score(y_test, built_model_k_preds)

        r_sq_cv_record[i] = sum / n_split

        # model created without k-fold cross validation
        built_model = buildModel(x_selected, model_type)
        if i == len(xcolumns) - 1:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        history = built_model.fit(x_selected, y, validation_split=.1, epochs=75, verbose=0, callbacks=[es])

        # evaluate the model
        # r squared value obtained using whole dataset
        built_model_preds = built_model.predict(x_selected)
        r_sq_record[i] = r2_score(y, built_model_preds)

        # adjusted r sq calculations
        numerator = x_selected.shape[0] - 1
        denominator = x_selected.shape[0] - (i+1) - 1
        r_sq_bar_record[i] = 1 - ((1 - r_sq_record[i]) * (numerator/denominator))

        # will only plot num parameters vs. r sq, r sq cv, and r sq adjusted when forward selection is done selection
        # will only plot num epochs vs. loss for model when forward selection is done selecting
        if i == len(xcolumns) - 1:
            plt.plot(num_attrs_perc, r_sq_cv_record, label="r_sq_cv")
            plt.plot(num_attrs_perc, r_sq_record, label="r_sq")
            plt.plot(num_attrs_perc, r_sq_bar_record, label="adjusted_r_sq")
            # chooses n* according to the r sq cv
            r_sq_cv_record = np.array(r_sq_cv_record)
            r_sq_bar_record = np.array(r_sq_bar_record)

            print("n* r sq cv = " + str(r_sq_cv_record.argmax() + 1) + " (" + str(r_sq_cv_record.max()) + ")")
            print("n* r sq adjusted = " + str(r_sq_bar_record.argmax()+1) + " (" + str(r_sq_bar_record.max()) + ")")


            plt.xlabel('Number of Parameters')
            plt.ylabel('r sq values')
            plt.title('Number of Parameters vs. r sq')
            plt.legend()
            plt.show()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(model_type + ' Model Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()


if __name__ == "__main__":
    # change path to absolute point to the data folder
    # use an extra forward slash (\) at the end because of the escape sequence
    path = r"C:\Users\14049\Documents\Undergrad\Year 4\Semester 2\DataScience2\data\\"
    data = np.array(pd.read_csv(path + "Concrete.csv", header=0))
    xcolumns = [0, 1, 2, 3, 4, 5, 6]
    x = data[:, xcolumns]
    y = data[:, 7]

    # standardizing x and y
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y.reshape(len(y), 1))[:, 0]

    if perceptron_:
        runner("Perceptron")

    if NN_3L_:
        runner('NN_3L')

    if NN_XL_:
        runner('NN_XL')
