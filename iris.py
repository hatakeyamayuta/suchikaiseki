from sklearn.datasets import load_iris
import keras
from keras.utils import plot_model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, Input
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import np_utils

def create_model():
    imput = Input(shape=(4,))
    relu = Dense(4, activation="relu",name="relu")(imput)
    sigmoid = Dense(2, activation="sigmoid",name="sigmoid")(relu)
    prediction = Dense(3,activation="softmax",name="softmax")(sigmoid)

    model = Model(inputs=imput,outputs=prediction)
    model.compile(loss="binary_crossentropy",
                  optimizer="nadam",
                  metrics=["accuracy"])
    model.summary()
    plot_model(model,to_file="iris.png",show_shapes=True,show_layer_names=True)
    return model



def predict():
    iris = load_iris()

    X = iris.data
    y = iris.target

    onehot_y = np_utils.to_categorical(y)
    X, onehot_y = shuffle(X,onehot_y)
    print(onehot_y)
    X_train, X_test, y_train, y_test = \
            train_test_split(X, onehot_y, test_size=0.1)

    model = create_model()
    model.fit(X_train,y_train,validation_data=(X_test,y_test),
            batch_size=4, epochs=100)


def main():
    predict()

if __name__=="__main__":
    main()

