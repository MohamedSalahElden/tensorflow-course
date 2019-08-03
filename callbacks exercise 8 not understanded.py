'''Exercise 8:
Earlier when you trained for extra epochs you had an issue where your loss might change. 
It might have taken a bit of time for you to wait for the training to do that, 
and you might have thought 'wouldn't it be nice if I could stop the training 
when I reach a desired value?' 
-- i.e. 95% accuracy might be enough for you, 
and if you reach that after 3 epochs, 
why sit around waiting for it to finish a lot more epochs....
So how would you fix that? Like any other program...
you have callbacks! Let's see them in action...
'''

import tensorflow as tf
print(tf.__version__)
​
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
​
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
​
​
​#link to go to exercise
#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=E7W2PT66ZBHQ
