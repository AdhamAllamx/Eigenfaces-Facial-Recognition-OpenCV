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
from imblearn.over_sampling import SMOTE

# Dictionary for person names
person_names = {
    1: "Alvaro_Uribe",
    2: "Atal_Bihari_Vajpayee",
    3: "George_Robertson",
    4: "George_W_Bush",
    5: "Junichiro_Koizumi"
}

# Load the dataset
faces_image = np.load('./input_dataset/new_faces_training.npy')
faces_target = np.load('./input_dataset/new_faces_targets.npy')

print("faces image new set shape:", faces_image.shape)
print("new faces target shape:", faces_target.shape)

# Reshape the image data into vectors
n_samples, n_row, n_col = faces_image.shape
faces_data = faces_image.reshape(n_samples, n_row * n_col)

# Prepare the training and test data
X_train, X_test, y_train, y_test = train_test_split(faces_data, faces_target, test_size=0.05, random_state=10)

# PCA for dimensionality reduction
n_components = 29
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Apply SMOTE for balancing dataset (commented if imbalance is not a concern)
smote = SMOTE()
X_train_pca, y_train = smote.fit_resample(X_train_pca, y_train)

# Isolation Forest for anomaly detection
iso_forest = IsolationForest(n_estimators=30, contamination=0.005)
iso_forest.fit(X_train_pca)

# Support Vector Machine classifier with extended parameter grid and probability estimation
param_grid = {
    'C': np.logspace(-2, 10, 13),
    'gamma': np.logspace(-9, 3, 13),
    'kernel': ['rbf', 'linear', 'poly']
}
clf = GridSearchCV(SVC(class_weight='balanced', probability=True), param_grid, cv=10)
clf.fit(X_train_pca, y_train)

# Evaluate the model
print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))

# Prepare an external image for prediction
test_image_path = "./input_dataset/George_Robertson/George_Robertson_0002.jpg"
test_image = cv2.imread(test_image_path)
if test_image is None:
    print("Test image not found!")
else:
    gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    resized_test_image = resize(gray_test_image, (n_row, n_col), anti_aliasing=True).reshape(1, -1)
    test_image_pca = pca.transform(resized_test_image)

    # Anomaly detection
    is_outlier = iso_forest.predict(test_image_pca)
    if is_outlier == -1:
        predicted_person = "Unknown"
    else:
        # Predict with confidence score
        probabilities = clf.predict_proba(test_image_pca)[0]
        predicted_probability = np.max(probabilities)
        predicted_person_id = np.argmax(probabilities) + 1  # adjust index to match person_names keys
        threshold = 0.4 # Confidence threshold

        if predicted_probability < threshold:
            predicted_person = "Unknown"
        else:
            predicted_person = person_names.get(predicted_person_id, "Unknown")

    # Display the result
    cv2.putText(test_image, f"{predicted_person} ({predicted_probability:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Person: {predicted_person}")
    plt.axis('off')
    plt.show()
