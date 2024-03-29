import tensorflow as tf
import matplotlib.pyplot as plt 

class mycallbacks(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch , logs = {}):
		if (logs.get('acc') > 0.998):
			print("\nreached 99% accuracy so cancelling training ")
			self.model.stop_training = True
			
callback = mycallbacks()

mnist = tf.keras.datasets.mnist
(training_images , training_labels) , (test_images , test_labels) = mnist.load_data()


training_images = training_images.reshape(60000 , 28 , 28 , 1)
test_images		= test_images.reshape(10000 , 28 , 28 , 1)

training_images = training_images / 255
test_images = test_images / 255

model = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(32 , (3,3) , activation = 'relu' , input_shape = (28,28,1)),
			tf.keras.layers.MaxPooling2D(2,2),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(128,activation = 'relu'),
			tf.keras.layers.Dense(10 , activation = 'softmax')
	])

model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

model.fit(training_images , training_labels ,epochs = 20 , callbacks = [callback])
model.evaluate(test_images , test_labels)
