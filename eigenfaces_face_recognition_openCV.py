from __future__ import print_function
from time import time
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
from skimage.io import imshow

# Dictionary of person names (example shown earlier would be reused here)
person_names = {
    0: "Federico_Cuomo",
    1: "Mario_Rossi",
    2: "Giovanni_Bianchi",
    3: "Giulia_Verdi",
    4: "Francesca_Neri",
    5: "Laura_Russo",
    6: "Alessandro_Ferrari",
    7: "Roberto_Romano",
    8: "Simone_Gallo",
    9: "Elena_Marini",
    10: "Marco_Conti",
    11: "Sara_Galli",
    12: "Antonio_Rinaldi",
    13: "Valentina_Costa",
    14: "Andrea_Moretti",
    15: "Chiara_De_Luca",
    16: "Stefano_Gatti",
    17: "Maria_Colombo",
    18: "Luca_Santini",
    19: "Giorgio_Ferrara",
    20: "Paola_Vitale",
    21: "Claudio_Longo",
    22: "Anna_Galli",
    23: "Davide_Moretti",
    24: "Silvia_Conti",
    25: "Enrico_Caputo",
    26: "Elisa_Riva",
    27: "Massimo_Battaglia",
    28: "Teresa_Milani",
    29: "Giacomo_Delvecchio",
    30: "Monica_Piras",
    31: "Fabrizio_Costantini",
    32: "Patrizia_Bruno",
    33: "Gianluca_Ruggiero",
    34: "Silvia_Farina",
    35: "Federica_Monti",
    36: "Alessio_Villa",
    37: "Elisabetta_Pellegrini",
    38: "Alberto_De_Luca",
    39: "Cristina_Vitale"
}

# Load the dataset
faces_image = np.load('./input_dataset/olivetti_faces.npy')
faces_target = np.load('./input_dataset/olivetti_faces_target.npy')

# Reshape the image data into vectors
n_samples, n_row, n_col = faces_image.shape[0], faces_image.shape[1], faces_image.shape[2]
faces_data = faces_image.reshape(n_samples, n_row * n_col)

# Prepare the training and test data
X_train, X_test, y_train, y_test = train_test_split(faces_data, faces_target, test_size=0.25, random_state=42)

# PCA for dimensionality reduction
n_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Isolation Forest for anomaly detection
iso_forest = IsolationForest(n_estimators=100, contamination=0.01)
iso_forest.fit(X_train_pca)

# Support Vector Machine classifier
param_grid = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)

# Evaluate the model
print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Prepare an external image for prediction
test_image_path = "messi.png"
test_image = cv2.imread(test_image_path)
gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
resized_test_image = resize(gray_test_image, (n_row, n_col), anti_aliasing=True).reshape(1, -1)
test_image_pca = pca.transform(resized_test_image)

# Anomaly detection
is_outlier = iso_forest.predict(test_image_pca)
if is_outlier == -1:
    predicted_person = "Unknown"
else:
    predicted_person_id = clf.predict(test_image_pca)[0]
    predicted_person = person_names.get(predicted_person_id, "Unknown")

# Display the results
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
ax.set_title(f"Predicted Person: {predicted_person}")
ax.axis('off')
plt.show()
