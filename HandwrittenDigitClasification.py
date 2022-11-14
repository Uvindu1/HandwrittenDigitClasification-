import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

# data set eka keras walin genima
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(type(X_train))
# meka numpy n dimensional arrey
# meka numpy arry ekak karanna on
print(X_train[10].shape)
plt.imshow(X_train[45])
plt.show()
print(Y_train[45])

# data set eke labale wala tiyana value tika, class nam tika genima
print(np.unique(Y_train))
print(np.unique(Y_test))

# siyaluma images commen dimention ekakata genima
# meya black and wighit images thiyenne enam eka color cord ekai eya 0-255 athara pihitai
# DEEP Learning waladi kuda agayan denna on
# colore code eka(0-255) kuda kirima(0-1) atharata genima
# memagin dayamention eka wenas wenne (28*28)
# array eka athule thiyana value 1-0 athara sede
X_train = X_train/255
X_test = X_test/255
print(X_train.shape)

# create Newrel Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# NN eka compyle kirima
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy'
)

model.fit(X_train, Y_train, epochs=10)

# test data set eke accuracy eka gnima
loss, accuracy =model.evaluate(X_test,Y_test)
print(accuracy)
#######################################################################################################
# prediction sidukirima
Y_pred = model.predict(X_test)
print(Y_pred.shape)# ===>(10000, 10) kiyala awa, mokadda me 10ya
print(Y_pred[0])# mein penen nee output layer ekedi node 10 haraha giyama lebenne ekak ek node ekata dala probability ekai
# ema probability walin one ot encoding cramayata sadaganna ona
# prediction karapu data set eke ena numpy arraye eke maximam value eka thibena index eka genima


print("............prediction eken apu agayana.........")
print(Y_pred)
Y_pred_labels = [np.argmax(i)for i in Y_pred]
print(".................Final prediction eka vidiyata meka ganna puluwa.............")
print(Y_pred_labels)
# Y_test => Ture value
# Y_pred_labels => prediction Value

# confusin matrix ekak sedima
conf_mat = confusion_matrix(Y_test,Y_pred_labels)

