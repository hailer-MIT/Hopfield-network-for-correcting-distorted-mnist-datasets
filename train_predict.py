""" Train and test a Hopfield Network on image data. """

import numpy as np
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import hopfield_network

np.random.seed(1)

def get_corrupted_input(input_data, corruption_level):
    
    corrupted = np.copy(input_data)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input_data))
    corrupted[inv == 1] *= -1
    return corrupted

def reshape(data):
    
    dim = int(np.sqrt(len(data)))
    return np.reshape(data, (dim, dim))

def plot(data, test, predicted, figsize=(5, 6)):
    
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
    plt.savefig("result.png")
    plt.show()

def preprocessing(img, w=128, h=128):
    
    img_resized = resize(img, (w, h), mode='reflect')
    thresh = threshold_mean(img_resized)
    binary = img_resized > thresh
    bipolar = 2 * binary.astype(int) - 1
    return bipolar.flatten()

def main():
   
    # data: Using a new set of images from skimage
    chelsea = rgb2gray(skimage.data.chelsea())         # A cat
    text = skimage.data.text()                         # An image of text
    coins = skimage.data.coins()                       # An image of coins
    brick = skimage.data.brick()                       # Brick texture

    data = [chelsea, text, coins, brick]

    print("Start to data preprocessing...")
    data = [preprocessing(img) for img in data]

    model = hopfield_network.HopfieldNetwork()
    model.train_weights(data)

    test = [get_corrupted_input(d, 0.3) for d in data]
    predicted = model.predict(test, threshold=0, asyn=False)

    print("Show prediction results...")
    plot(data, test, predicted)
    # model.plot_weights()

if __name__ == '__main__':
    main()