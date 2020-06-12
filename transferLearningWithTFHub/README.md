## Transfer Learning with TFHub(maily API hub.KerasLayer)

To import, ```import tensorflow_hub as hub```

The API ```hub.KerasLayer(url, input_shape=(x,x,x))``` wraps model from url as a layer. When tuple is added, e.g. (224,224) + (3,) = (224,224,3)

To download file with tf, ```tf.keras.utils.get_file(store_file_name,download_url)``` API returns the path of the downloaded file. ```untar``` parameter can also be specified for zip etc. file.

It is simple way to process a single image with PIL.Image. ```image=Image.open(path).resize((224,224))``` np.array(image) turns the Image object to np array.

array[np.newaixs, ] equals to tf.expand_dims(array,0), which convert n dims to n+1 dims.

To plot images in grids using plt, code is 

```python
fig=plt.figure(figsize=(10,9))
for i in range(30):
  plt.subplot(6,5,i+1)
  plt.imshow(image_batch[i])
  plt.axis("off")
  _=plt.title(predict_class_names[i].title())
```

Set layer.trainable=False, freeze parameters in layer.

