#

## Congratulates on my final decision to record my learning tutorial here!

### DCGAN MODEL (most basic kind)

​	Following instructions in the site [sample tutorial](https://zhuanlan.zhihu.com/p/61280722) ,I to create the network just copied lines of code from the sample, and this make my ipynb model file.

​	Below would be some of the analysis of my DCGAN model and other things I think cool.

​	As tensorflow primarily uses float32 as the type of data, we just have to transfer the data type to float32 by    ```train_images=train_images.reshape(train_images.shape[0],28,28,1).astype("float32")```

Why do we have to make a 28x28 image to 28,28,1. Because the lowest dimension of the data always represent the features of an object. This is gray image, the dimension of the lowest is 1.But if it's RGB picture, we may  have three value in the lowest dimension.

​	We import MinMaxScaler and MaxAbsScaler from sklearn.preprocessing .The previous one scale the data to (0,1), and the later one scales data to (-1,1) .Notice, shape of the data requires.

​	Learn to process the data with bufferSize and batchSize. ```train_dataset=tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)```

​	Way to generate noise as this: ```noise = tf.random.normal([100, 100])```

	#### Model Layers Defination

**generator_model**

the model input is 100 noise.So Input like (None,100)

the output gets an image so it's like (None,28,28,1). Why 1 at the lowest? As we have to make this output as the input of the discriminator model. The the shape must be the same as the trainImages.

iterate the train_images(which is a Batch object), by for i in sth. You  get [BATCH,data] kind train_data

Some cheat-sheet about layers.

1. Conv2D layer accept three dimensional data in addition to batch. So input_shape of a Conv2D layer is 4 dim. filters determine lowest dimension of the output. Kernel size woud be whatever you want.  strides is intersting. When I input [28,28,1] data, and strides get (2,2),output is like [14,14,1], 28 and 28 are divided by 2. but this is not the rule of how strides works. You may go explore.
2. Conv2DTranspose dose the opposite thing
3. We may first have all the units with a dense layer of axbxc units,and then reshape to target shape

How do we define the generator model loss and discriminator model loss ?

​	tf.keras.losses.BinaryCrossentropy mainly calculates loss in the condition of binary classification. So about discriminator and generator, we just tell it if it is good, so binaryCrossentropy serves pretty well. 

in the ```train_step``` function(train [BATCH,data] kind of input), ```with tf.GradientTape() as gen_tape: ...``` then calculates loss in indent. Then compute gradients using ```gen_tape.gradient(gen_loss,model.trainable_variables)``` Finally, apply gradients to optimizer(here is adam) using ```optimizer.apply_gradients(zip(gradients,model.trainable_variables))``` 

DONE hah.