# Machine Translation useing basic seq2seq

​	We just define our trainging data using variable declaration instead of loading data from a file.

​	Here, we have data ```((en,fr), (en,fr), (en,fr))``` .We can use zip(data1,data2) to match elements of data1 to those of data2 like what a normal function dose. In the opposite, we use zip(*data) to reverse data to data1 and data2.

​	```keras.preprocessing.text.Tokenizer(filters="")``` is the keras implemented tool for text preprocessing. After we get the tokenizer, ```tokenizer.fit_on_texts(texts)``` to build ```tokenizer.word_index,tokenizer.index_word``` . These are dicts built from raw texts. Then, ```tokenizer.texts_to_sequences``` turn words to vectors. Finally, to get all the sentense vectors to the same shape, we use ```keras.preprocessing.sequence.pad_sequences(vectors,padding="post")``` to fulfill the empty space 0.

​	len(dict) also counts how many value in a dict.

​	seq2seq uses two model(Encoder and Decoder). What makes me surprise is that self-deinfed model can have more than input or output. To define a custom model, we just inherite the class ```keras.Model``` . It's the same as layers which inherite ```keras.Layer``` . function call in class defines the behavior of the model or layer. In general, function call get multi input and return multi output. That is, different layers process data or other ways.

​	About our custom Encoder model, there is ```super(Encoder,self).__init__()``` first. Encoder model receives source sentense, so embedding layer as the first layer to process language kind of data. Second layer gru layer, when return_state=true, this layer has two output(one is output, another is the hidden layer).

​	Despite the Encoder, there is Decoder. To define Decoder, we need Attention layer to better deal with long term past data. In general, in this Attention layer, input is hidden and enc_output from the Encoder. score is ```  score=self.V(tf.nn.tanh(self.W1(enc_output)+self.W2(hidden_with_time_axis)))``` .After hidden and enc_output are processed by two Dense layer, we activate the sum of them by tanh. Something surprising is that W1(enc_output) has TensorShape(5,10,1024) while W2(hidden) has TensorShape(5,1,1024). When add them together, it works and the final TensorShape(5,10,1024). Just know that in this layer, calculate attention_weights by hidden and enc_output. Final return context_vector is ```tf.return_sum(attention_weights*enc_output)``` in the example, shape is (5,1024).

​	In our Decoder model, input1 is the last word in the target(during train, it is the last word of what we are training now, during predict, it is the last word we predict).Input2 is hidden from Encoder, and input3 is enc_output from Encoder.

​	About loss function, ...

​	In train_step function, @tf.function servers as the decorator. loss is computed only once in each train_step function is called. gradients is computed is optimizer is applied in this function.

​	Wow, I think i need to review the paper for one more time.......to figure out how the network works.