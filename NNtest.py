import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_data , train_label),(test_data , test_label) = data.load_data()

train_data = train_data / 255.0
test_data = test_data /255.0



model = keras.Sequential(
	[
	keras.layers.Flatten(input_shape = (28,28)),
	keras.layers.Dense(units = 128 , activation = tf.nn.relu),
	keras.layers.Dense(10 , activation = tf.nn.softmax)
	]
	)
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy')

model.fit(train_data , train_label , epochs = 5)

model.evaluate(test_data , test_label)
print(model.predict(test_data[20:50 , : , :])[10])
print(test_label[30])

plt.imshow(test_data[30])
plt.show()