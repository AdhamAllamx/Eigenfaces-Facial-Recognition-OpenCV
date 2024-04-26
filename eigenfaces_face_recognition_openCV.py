from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import resize
# from imblearn.over_sampling import SMOTE
import pickle


# Dictionary for person names
person_names = {
    1: "Alvaro_Uribe",
    2: "Atal_Bihari_Vajpayee",
    3: "George_Robertson",
    4: "George_W_Bush",
    5: "Junichiro_Koizumi",
    6: "Adham_Allam"
}

# Load the dataset
faces_image = np.load('./input_dataset/new_faces_training.npy')
faces_target = np.load('./input_dataset/new_faces_targets.npy')

print("faces image new set shape:", faces_image.shape)
print("new faces target shape:", faces_target.shape)

# Reshape the image data into vectors
n_samples, n_row, n_col = faces_image.shape

# Define how many images you want to display and the grid size
n_images_to_display = 60 # For example, display 25 images
plot_grid_size = int(np.ceil(np.sqrt(n_images_to_display)))

fig, axes = plt.subplots(plot_grid_size, plot_grid_size, figsize=(15, 15))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for i, ax in enumerate(axes.flat):
    if i < n_images_to_display:
        ax.imshow(faces_image[i], cmap='gray')  # Assuming grayscale images
        ax.set_title(f"Label: {faces_target[i]}")
        ax.axis('off')
    else:
        ax.axis('off')  # Turn off axis for empty plots

plt.show()

faces_data = faces_image.reshape(n_samples, n_row * n_col)



# Prepare the training and test data   // changes 
X_train, X_test, y_train, y_test = train_test_split(faces_data, faces_target, test_size=0.10, random_state=5)

# Since you have 50 images total, trying a small number of components makes sense
n_components = 21  # Adjust based on explained variance ratio or via cross-validation
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, whiten=True)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# # Apply SMOTE for balancing dataset (commented if imbalance is not a concern) // changes 
# smote = SMOTE()
# X_train_pca, y_train = smote.fit_resample(X_train_pca, y_train)

# Reduce the number of trees and adjust contamination
iso_forest = IsolationForest(n_estimators=60, contamination='auto')  # auto lets the algorithm determine the threshold based on the data
iso_forest.fit(X_train_pca)


param_grid = {
    'C': [1, 10, 100],  # Reduced range based on typical good values
    'gamma': ['scale', 'auto'],  # 'scale' and 'auto' are often sufficient for many cases
    'kernel': ['rbf', 'linear']  # Removed 'poly' for speed in this example
}
clf = GridSearchCV(SVC(class_weight='balanced', probability=True), param_grid, cv=4)   # Reduced CV folds
clf.fit(X_train_pca, y_train)


# Evaluate the model
print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))


# Save the trained models
models_file = "trained_data_model.pkl"
with open(models_file, "wb") as f:
    pickle.dump((pca, iso_forest, clf), f)

# Prepare an external image for prediction
test_image_path = "Atal_Bihari_Vajpayee_0016.jpg"
test_image = cv2.imread(test_image_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if test_image is None:
    print("Test image not found!")
else:
    # Convert to grayscale for face detection
    gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_test_image, scaleFactor=1.1, minNeighbors=13, minSize=(30, 30))
    if len(faces) == 0:
        print("No face detected in the image.")
    else:
        x, y, w, h = faces[0]  # Assume the first detected face is the target face
        is_outlier_list = []
        for (x,y,w,h) in faces:
            # Draw rectangle around the face
          cv2.rectangle(test_image, (x,y), (x+w, y+h), (255, 0, 0), 2)

            # Crop the face using detected coordinates
          face = gray_test_image[y:y+h, x:x+w]

            # Display the original image with detected face marked
          plt.subplot(1, 2, 1)
          plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
          plt.title("Detected Face")
          plt.axis('off')

            # Display the cropped face
          plt.subplot(1, 2, 2)
          plt.imshow(face, cmap='gray')
          plt.title("Cropped Face")
          plt.axis('off')

          plt.show()

        face = gray_test_image[y:y+h, x:x+w]  # Crop the face
        
        # Resize the face to the desired dimensions
        resized_face = resize(face, (n_row, n_col), anti_aliasing=True).reshape(1, -1)
        
        # PCA transformation
        test_image_pca = pca.transform(resized_face)
        
        # Anomaly detection
        is_outlier = iso_forest.predict(test_image_pca)

        print("outlier : ", is_outlier)
        is_outlier_list.append(is_outlier)  # Store the outlier prediction for this image

        # To save the predictions and their probabilities
        predictions_file = "predictions.txt"
        with open(predictions_file, "w") as f:
            for i, is_outlier in enumerate(is_outlier_list):
                if is_outlier == -1:
                    predicted_person = "Unknown"
                    predicted_probability = 0.0  # Assuming unknown people have zero probability
                else:
                     probabilities = clf.predict_proba(test_image_pca)[0]
                     predicted_probability = np.max(probabilities)
                     if predicted_probability < 0.30:
                            predicted_person = "Unknown"
                            predicted_probability = 0.0  # If probability is too low, consider it unknown
                     else:
                            predicted_person_id = np.argmax(probabilities) + 1
                            predicted_person = person_names.get(predicted_person_id, "Unknown")
        
                # Write the prediction and probability to the file
                            f.write(f"Image {i+1}: Predicted Person: {predicted_person}, Probability: {predicted_probability:.2f}\n")

# Display results
cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.putText(test_image, f"{predicted_person} ({predicted_probability:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Person: {predicted_person}")
plt.axis('off')
plt.show()