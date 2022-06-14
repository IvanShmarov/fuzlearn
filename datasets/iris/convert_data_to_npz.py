import numpy as np


data = np.loadtxt('bezdekIris.data', delimiter=',', usecols=range(4))

data -= data.min(axis=0)
data /= data.max(axis=0)

setosa = data[:50]
versi = data[50:100]
virgi = data[100:]

assert all(elem.shape == (50,4) for elem in [setosa, versi, virgi])

np.savez('iris', setosa=setosa, versi=versi, virgi=virgi)