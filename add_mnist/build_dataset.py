import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

class NewDataset: 
    def __init__(self, N=100000, test_prop=0.2, train_prop=0.7):
        self.N = N
        self.test_prop = test_prop
        self.train_prop = train_prop
        
    def mnist_data(self):
        """
        Load and preprocess the MNIST dataset, combine original training/test set and normalise.
        
        Returns:
            x_data: normalised image data (range 0-1)
            y_data: corresponding labels
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # combine original training and test set
        x_data = np.concatenate([x_train, x_test], axis=0)
        y_data = np.concatenate([y_train, y_test], axis=0)
        
        # normalise pixel values
        x_data = x_data.astype('float32') / 255.0
        
        return x_data, y_data
        
    def build_dataset(self):
        """
        Create a balanced dataset of combined MNIST images, and retain the original label pairs.
        
        Returns:
            images: combined image pairs
            sum_labels: sums of two labels
            original_digits: original label pairs
        """
        images = []
        sum_labels = []
        original_pairs = [] 
        x, y = self.mnist_data()

        # create a counter for each possible sum_lables (0-18)
        counter = np.zeros(19)
        max_samples = self.N // 19 + 1   # e.g. 20//19=1, but the 20th sample should add to one lable, so need '+1'
        
        while len(images) < self.N:
            # randomly select two indices
            i1, i2 = np.random.randint(0, len(x), size=2)
            img1, img2 = x[i1], x[i2]
            y_sum = y[i1] + y[i2]
            
            # check if more samples are needed for this sum
            if counter[y_sum] < max_samples:
                combined = np.hstack((img1, img2))
                images.append(combined)
                sum_labels.append(y_sum)
                original_pairs.append((int(y[i1]), int(y[i2])))
                counter[y_sum] += 1
        
        return np.array(images), np.array(sum_labels), np.array(original_pairs)
                
    def split_dataset(self):
        """
        Split dataset into training, validation, test sets, and preserve label distribution.
        
        Returns:
            x_train, x_validation, x_test, y_train, y_validation, y_test, pairs_train, pairs_validation, pairs_test
        """
        images, labels, pairs = self.build_dataset()
         
        # first split: separate test set
        x_temp, x_test, y_temp, y_test, pairs_temp, pairs_test = train_test_split(
            images, labels, pairs,
            test_size=self.test_prop,
            random_state=100,
            stratify=labels)  # stratified sampling technique
    
        # second split: separate training and validation set
        validation_prop = (1 - self.test_prop) * (1 - self.train_prop)
        x_train, x_validation, y_train, y_validation, pairs_train, pairs_validation = train_test_split(
            x_temp, y_temp, pairs_temp,
            test_size=validation_prop/(1-self.test_prop),
            random_state=100,
            stratify=y_temp ) 
        
        return x_train, x_validation, x_test,     y_train, y_validation, y_test,    pairs_train, pairs_validation, pairs_test

    def stats(self):
        """
        Print statistical information about dataset distribution.
        """
        _, _, _, y_train, y_validation, y_test, _, _, _ = self.split_dataset()

        print("\nDataset Statistics:")
        print(f"Training set size: {len(y_train)}")
        print(f"Validation set size: {len(y_validation)}")
        print(f"Test set size: {len(y_test)}")
        print("\nTraining set label distribution:")
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            print(f"Sum {label}: {count} samples ({count/len(y_train)*100:.2f}%)")

    def visualise(self, samples=5):
        """
        Visualise sample pairs from the dataset.
        """
        images, _, _, labels, _, _, pairs, _, _, = self.split_dataset()

        plt.figure(figsize=(15, 3))
        for i in range(samples):
            plt.subplot(1, samples, i + 1)
            plt.imshow(images[i], cmap='gray')
            digit1, digit2 = pairs[i]
            plt.title(f"{digit1}+{digit2}={labels[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

