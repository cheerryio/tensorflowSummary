## Transfer Learning By Pretained and Fine Tuning

Base model can be defined with ```base_model=tf.keras.applications.MobileNetV2(input_shape=SHAPE, include_top=False, weights="imagenet")``` About how weights is trained, we set weights="imagenet" which determine that this base model is trained with ImageNet dataset.

Layer ```keras.layers.GlobalAveragePooling2D``` input 4D [batch, n1. n2, n3] and output 2D [batch, n3] .It just keep the shape of the lowest dimension of 4D input.

Fine-tuning, in Chinese, 微调. In the example above, we set all layers of base model trainable to False. However, when we are in fine-tuning mode, we set ```base_model.layers[:fineTuningSplit]=False``` , and the rest layers' trainable are True. Then train.