## LSTM autoencoder

dataset used:bearing sensor data

to load the data from each of the small files, we use read_csv function with sep parameter set to "\t" as each data is seperated by "\t"

When we want to set the index of pd.DataFrame, it must be an array even if there is only one value like ```df.index=[indexName]```

keras.layers.RepeatLayer just repeat data for n times. e.g. (None,4) output is (None,10,4) when n set to 10.

np.abs().mean() counts average value good. Be aware to set axis when nesessary.