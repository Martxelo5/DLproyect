import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']


# The function provided to you
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    """
    Loads and processes all CIFAR-10 batches from the specified directory.
    """
    train_data = []
    train_labels = []

    # 1. Load the 5 training batches
    for i in range(1, 6):
        filename = os.path.join(data_dir, f'data_batch_{i}')
        batch_dict = unpickle(filename)
        
        train_data.append(batch_dict[b'data'])
        train_labels.extend(batch_dict[b'labels'])

    # 2. Stack the 5 arrays into one massive array of shape (50000, 3072)
    X_train = np.vstack(train_data)
    y_train = np.array(train_labels)

    # 3. Load the 1 testing batch
    test_filename = os.path.join(data_dir, 'test_batch')
    test_dict = unpickle(test_filename)
    
    X_test = test_dict[b'data']
    y_test = np.array(test_dict[b'labels'])

    # 4. Reshape and Transpose the images
    # - reshape(-1, 3, 32, 32) splits the 3072 into 3 channels of 32x32.
    # - transpose(0, 2, 3, 1) shifts the axes from (Batch, Channel, Height, Width) 
    #   to (Batch, Height, Width, Channel), which is what TensorFlow/Keras expect.
    #   Note: If you are using PyTorch, you can skip the transpose step!
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return (X_train, y_train), (X_test, y_test)


def perform_cifar10_eda(X_train, y_train):
    print("--- Starting CIFAR-10 Exploratory Data Analysis ---\n")
    
    # 1. Dataset Shape and Data Types
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Training Labels Shape: {y_train.shape}")
    print(f"Data Type: {X_train.dtype}")
    print(f"Min Pixel Value: {X_train.min()} | Max Pixel Value: {X_train.max()}\n")

    # 2. Class Distribution
    # Count how many images belong to each class
    unique_classes, counts = np.unique(y_train, return_counts=True)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[cifar10_classes[i] for i in unique_classes], y=counts, palette="viridis")
    plt.title('Distribution of Classes in CIFAR-10 Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()
    
    print("Class counts:")
    for cls, count in zip(unique_classes, counts):
        print(f"{cifar10_classes[cls]}: {count}")
    print("\n")

    # 3. Visualizing Sample Images
    # Display the first 10 images from the training set
    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        # We need to ensure data is an integer between 0-255 for standard plotting
        img = X_train[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
            
        plt.imshow(img)
        plt.title(cifar10_classes[y_train[i]])
        plt.axis('off')
    plt.suptitle('Sample Images from CIFAR-10', fontsize=16)
    plt.tight_layout()
    plt.show()





def main():
    
    cifar_directory = './deliverable2/data' 

    (X_train, y_train), (X_test, y_test) = load_cifar10_data(cifar_directory)

    print(f"Training data shape: {X_train.shape}")  # Expected: (50000, 32, 32, 3)
    print(f"Training labels shape: {y_train.shape}") # Expected: (50000,)
    print(f"Testing data shape: {X_test.shape}")    # Expected: (10000, 32, 32, 3)

    perform_cifar10_eda(X_train, y_train)

if __name__ == "__main__":
    main()