import cv2
import numpy as np
import albumentations as A
import os

def load_original_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def augment_images(image, aug_transformation, num_augmentations):
    augmented_images = []
    for _ in range(num_augmentations):
        augmented = aug_transformation(image=image)['image']
        augmented_images.append(augmented)
    return augmented_images

# Define augmentation transformations
aug_transformation =A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.PiecewiseAffine(p=0.3),  # Ensure this is available in your version
        ], p=0.2),
        A.GaussNoise(p=0.2),  # Replacing IAAAdditiveGaussianNoise
        A.CLAHE(p=0.2),
        A.RandomGamma(p=0.2)
    ])
input_path = './input_dataset/gray_scale_images/'
save_path = './input_dataset/'
os.makedirs(save_path, exist_ok=True)

persons = ['Adham_Allam', 'Dua_Lipa', 'Henry_Cavil', 'Scarelett_Johansson']
num_augmentations_per_image = 20  # Number of augmented versions of each image
all_images = []

all_labels = np.repeat([1, 2, 3, 4], 100)  # Repeats 1, 2, 3, and 4 each 100 times

all_images = []
for person in persons:
    person_images = [f'{input_path}{person}/{person}_{i}.png' for i in range(1, 6)]  # Assuming each person has 5 images
    for img_path in person_images:
        image = load_original_image(img_path)
        if image is not None:
            augmented_images = augment_images(image, aug_transformation, num_augmentations_per_image)
            all_images.extend(augmented_images)

# Ensure all images are in a numpy array of shape (400, 64, 64)
all_images_array = np.array(all_images).reshape(-1, 64, 64)

# Save the array of images and labels
np.save(os.path.join(save_path, 'allam_training.npy'), all_images_array)
np.save(os.path.join(save_path, 'allam_targets.npy'), all_labels)

print(f"Augmentation completed successfully! {len(all_images_array)} images and labels saved.")