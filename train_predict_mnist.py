"""
Train and test a Hopfield Network on MNIST digit data.
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
import hopfield_network
from keras.datasets import mnist


def get_corrupted_input(input_data, corruption_level):
    
    corrupted = np.copy(input_data)
    invertor = np.random.binomial(n=1, p=corruption_level, size=len(input_data))
    corrupted[invertor == 1] *= -1
    return corrupted

def reshape(data):
    # 1D to 2D
    dim = int(np.sqrt(len(data)))
    return np.reshape(data, (dim, dim))

def plot(data, test, predicted, figsize=(3, 3)):
   
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    
    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    titles = ['Train data', 'Input data', 'Output data']
    for i in range(len(data)):
        for j in range(3):
            if i == 0:
                axarr[i, j].set_title(titles[j])
            axarr[i, j].imshow([data, test, predicted][j][i])
            axarr[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("result_mnist.png")
    plt.show()

def preprocessing(img):
   
    w, h = img.shape
    thresh = threshold_mean(img)
    binary = img > thresh
    bipolar = 2 * binary.astype(int) - 1
    return bipolar.flatten()

def main():
    
    # Load MNIST data
    (x_train, y_train), (_, _) = mnist.load_data()
    digits = [3, 4, 5]
    data = [x_train[y_train == i][0] for i in range(3)]
    print("Start to data preprocessing...")
    data = [preprocessing(img) for img in data]
    
    model = hopfield_network.HopfieldNetwork()
    model.train_weights(data)
    
    test = [x_train[y_train == i][1] for i in range(3)]
    test = [preprocessing(img) for img in test]
    
    # test = [get_corrupted_input(d, 0.3) for d in test]
    predicted = model.predict(test, threshold=50, asyn=True)
    print("Show prediction results...")
    plot(data, test, predicted, figsize=(8, 8))
    print("Show network weights matrix...")
    model.plot_weights()
    
if __name__ == '__main__':
    main()