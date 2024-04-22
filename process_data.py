import numpy as np
import cv2
from skimage.transform import resize

def preprocess_images(images, n_row=64, n_col=64):
    processed_images = np.zeros((len(images), n_row, n_col))  # Keeping as a 2D array per image
    for i, img_path in enumerate(images):
        formatted_path = img_path.replace('/', '\\')  # Correct path format for Windows
        img = cv2.imread(formatted_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image at {formatted_path} could not be loaded. Check the file path.")
        if img.shape[0] != n_row or img.shape[1] != n_col:
            img = resize(img, (n_row, n_col), anti_aliasing=True)
        processed_images[i] = img  # Assign the resized image directly
    return processed_images

def append_and_save_data(new_images, new_targets, dataset_path):
    existing_images = np.load(dataset_path + 'olivetti_faces.npy')
    existing_targets = np.load(dataset_path + 'olivetti_faces_target.npy')

    # Ensure both datasets are in the same shape format
    if len(existing_images.shape) == 2 and len(new_images.shape) == 3:
        new_images = new_images.reshape(new_images.shape[0], -1)  # Flatten new images if necessary
    elif len(existing_images.shape) == 3 and len(new_images.shape) == 2:
        existing_images = existing_images.reshape(existing_images.shape[0], -1)  # Flatten existing images if necessary

    all_images = np.concatenate((existing_images, new_images), axis=0)
    all_targets = np.concatenate((existing_targets, new_targets), axis=0)

    np.save(dataset_path + 'olivetti_faces.npy', all_images)
    np.save(dataset_path + 'olivetti_faces_target.npy', all_targets)
# List of new image paths
new_images = [
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0001.jpg',
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0002.jpg',
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0003.jpg',
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0004.jpg',
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0005.jpg',
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0006.jpg',
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0007.jpg',
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0008.jpg',
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0009.jpg',
    './input_dataset/Alvaro_Uribe/Alvaro_Uribe_0010.jpg'
]

new_labels = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40, 40])  # Assuming label 40 for all new images

# Process the new images
preprocessed_new_images = preprocess_images(new_images)

# Define the path for the dataset
dataset_path = './input_dataset/'

# Append and save the new data
append_and_save_data(preprocessed_new_images, new_labels, dataset_path)
