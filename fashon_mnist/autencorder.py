from keras.layers import Input, UpSampling2D,  Dense, Dropout,Activation, Flatten,Conv2D ,MaxPooling2D
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.utils import plot_model,np_utils
import numpy as np
import keras
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
encord = 32

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


x_train /=255
x_test /=255
print(x_train.shape)
x_train = np.reshape(x_train,(len(x_train),28,28,1))
x_test = np.reshape(x_test,(len(x_test),28,28,1))
print(x_train.shape)
input_img = Input(shape=(28,28,1))

x = Conv2D(16,(3,3),padding="same",activation="relu")(input_img)
x = MaxPooling2D((2,2),padding="same")(x)
x = Conv2D(8,(3,3),padding="same",activation="relu")(x)
x = MaxPooling2D((2,2),padding="same")(x)
x = Conv2D(8,(3,3),padding="same",activation="relu")(x)
flat = Flatten()(x)
drop = Dropout(0.5)(flat)
dense1 = Dense(50,activation="relu",name="dense1")(drop) 
dense2 = Dense(30,activation="sigmoid",name="dense2")(dense1) 
dense = Dense(10,activation="sigmoid",name="dense")(dense2) 
x = UpSampling2D((2,2))(x)
x = Conv2D(8,(3,3),padding="same",activation="relu")(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(8,(3,3),padding="same",activation="relu")(x)
decode = Conv2D(1,(3,3),padding="same",activation="sigmoid",name="decode")(x)
output=[decode,dense]
encorder = Model(inputs=input_img,outputs=output)
encorder.load_weights("encord.h5")
plot_model(encorder,to_file="encorder.png",show_shapes=True)

encorder.compile(optimizer="adam",
                 loss={"decode":"binary_crossentropy",
                       "dense":"categorical_crossentropy"},
                       metrics=["accuracy"])
"""
fit = encorder.fit(x_train,{"decode":x_train,
                            "dense":y_train},
                            epochs=20,
                            batch_size=124,
                            shuffle=True,
                            validation_data=(x_test,{"decode":x_test,
                            "dense":y_test}))
encorder.save_weights("encord.h5")
print(fit.history.keys())

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['dense_loss'],label="loss for training")
    axL.plot(fit.history['val_dense_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['dense_acc'],label="loss for training")
    axR.plot(fit.history['val_dense_acc'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig('./fashion_mnist.png')
plt.close()
"""
score = encorder.predict(x_test,verbose=1)
print(encorder.evaluate(x_test,{"decode":x_test,"dense":y_test}))
i = 0
a=0
for i in range(10000):
    y = np.array(score[1][i])
    if np.argmax(y)==np.argmax(y_test[i]):
        a = a+1
print(a/i)
#for i,predict in enumerate(score[0:1000]):
#    y = np.array(score[1][i])
img = np.array(score[0][2])
img = np.reshape(img,(28,28))

Image.fromarray(np.uint8(img*255)).save("autencorde_fashion_mnist.png")
