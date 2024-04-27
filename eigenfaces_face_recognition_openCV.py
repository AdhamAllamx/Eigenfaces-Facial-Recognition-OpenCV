from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split ,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import resize
import pickle


# Dictionary for person names
person_names = {
    1: "Adham_Allam",
    2: "Dua_Lipa",
    3: "Henry_Cavil",
    4: "Scarelett_Johansson"
}

# Load the dataset
faces_image = np.load('./input_dataset/allam_training.npy')
faces_target = np.load('./input_dataset/allam_targets.npy')

print("faces image new set shape:", faces_image.shape)
print("new faces target shape:", faces_target.shape)
n_row=64
n_col =64
# Reshape the image data into vectors
n_samples= faces_image.shape[0]

print("targets are : ", faces_target)

#display the original dataset
n_images_to_display = 200 
plot_grid_size = int(np.ceil(np.sqrt(n_images_to_display)))

fig, axes = plt.subplots(plot_grid_size, plot_grid_size, figsize=(10, 10))
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
X_train, X_test, y_train, y_test = train_test_split(faces_data, faces_target, test_size=0.25, random_state=42,stratify=faces_target)

# n_components = 21  # Adjust based on explained variance ratio or via cross-validation
n_components = 50 # Never more than the number of samples, but you can adjust based on explained variance.

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
# pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


param_grid = {
    'C': [1, 5, 10, 50],
    'gamma': [0.0001, 0.0005, 0.001, 0.005],
    'kernel': ['rbf']
}


svm = GridSearchCV(SVC(probability=True,class_weight='balanced'), param_grid, cv=StratifiedKFold(n_splits=5))
svm.fit(X_train_pca, y_train)

# Perform cross-validation on the training set
print("Cross-validated scores:")
scores = cross_val_score(svm, X_train_pca, y_train, cv=5)
print(scores)
print("Average score:", np.mean(scores))

# Evaluate the model
print("Predicting people's names on the test set")
y_pred = svm.predict(X_test_pca)
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))



# Display PCA cumulative explained variance to decide on the number of components
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Variance Explained')
plt.grid(True)
plt.show()


# Model parameter tuning visualization
from sklearn.model_selection import validation_curve

param_range = np.logspace(-4, -2, 20)  # Adjust this range based on your previous validation curve's results

# Create the validation curve
train_scores, test_scores = validation_curve(
    SVC(kernel='rbf', C=10, class_weight='balanced'),  # Use the best C found previously
    X_train_pca, y_train,
    param_name="gamma",
    param_range=param_range,
    cv=StratifiedKFold(n_splits=4),
    scoring="accuracy",
    n_jobs=-1  # Use all available cores
)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure()
plt.semilogx(param_range, train_mean, label="Training score", color="darkorange", lw=2)
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange", lw=2)
plt.semilogx(param_range, test_mean, label="Cross-validation score", color="navy", lw=2)
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy", lw=2)
plt.title('Validation Curve with SVM for Gamma')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

# Save the trained models
models_file = "trained_data_model.pkl"
# with open(models_file, "wb") as f:
#     pickle.dump({'pca': pca, 'iso_forest': iso_forest, 'svm': clf.best_estimator_}, f)

# testing image part // modified  

test_image_path = "sc.jpg"
test_image = cv2.imread(test_image_path)

if(test_image is None ):
    print("Test image was not found OOps !")
else :
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
    gray_test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray_test_image, scaleFactor=1.1 , minNeighbors=5 ,minSize=(30,30))

    if(len(faces)==0):
        print("No faces detected OOPS!!")
    else : 
        for(x,y,width,height) in faces :
            cv2.rectangle(test_image,(x,y),(x+width,y+height),(255,0,0),2)
            face = gray_test_image[y:y+height ,x:x+width]
            face_resized = resize(face,(64,64), anti_aliasing= True)
            print("faceresized shape : ", face_resized.shape)
            face_resized=face_resized.reshape(1,-1)
            test_face_pca = pca.transform(face_resized)
            # prediction test 
            proba = svm.predict_proba(test_face_pca)
            prediction = np.argmax(proba, axis=1) + 1
            confidence = np.max(proba, axis=1)

            if confidence < 0.8:
                person_name = "Unknown"
            else:
                person_name = person_names.get(prediction[0], "Unknown")
                label = person_names.get(prediction[0], "Unknown")

            print("Prediction result for test image:", prediction, "with confidence:", confidence)
            print("prediction result for test image : ", proba)  
            # label = proba[0]
            # person_name = person_names.get(label , "Unknown")

            

            # display the results for the test 
            cv2.putText(test_image,person_name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.9,(35,255,12),2)  

            # Plot each face processing step
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(face, cmap='gray')
            plt.title("Cropped Face")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(face_resized, cmap='gray')
            plt.title("Resized Face")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
            plt.title("Detected Face")
            plt.axis('off')
            
            plt.show()
plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Test Image with detected faces :')
plt.show()

# Prepare an external image for prediction
# test_image_path = "George_W_Bush_0019.jpg"
# test_image = cv2.imread(test_image_path)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# if test_image is None:
#     print("Test image not found!")
# else:
#     # Convert to grayscale for face detection
#     gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray_test_image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
#     if len(faces) == 0:
#         print("No face detected in the image.")
#     else:
#         x, y, w, h = faces[0]  # Assume the first detected face is the target face
#         is_outlier_list = []
#         for (x,y,w,h) in faces:
#             # Draw rectangle around the face
#           cv2.rectangle(test_image, (x,y), (x+w, y+h), (255, 0, 0), 2)

#             # Crop the face using detected coordinates
#           face = gray_test_image[y:y+h, x:x+w]

#             # Display the original image with detected face marked
#           plt.subplot(1, 2, 1)
#           plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
#           plt.title("Detected Face")
#           plt.axis('off')

#             # Display the cropped face
#           plt.subplot(1, 2, 2)
#           plt.imshow(face, cmap='gray')
#           plt.title("Cropped Face")
#           plt.axis('off')

#           plt.show()

#         face = gray_test_image[y:y+h, x:x+w]  # Crop the face
        
#         # Resize the face to the desired dimensions
#         resized_face = resize(face, (n_row, n_col), anti_aliasing=True).reshape(1, -1)
        
#         # PCA transformation
#         test_image_pca = pca.transform(resized_face)
        
#         # Anomaly detection
#         is_outlier = iso_forest.predict(test_image_pca)

#         print("outlier : ", is_outlier)
#         is_outlier_list.append(is_outlier)  # Store the outlier prediction for this image

#         # To save the predictions and their probabilities
#         predictions_file = "predictions.txt"
#         with open(predictions_file, "w") as f:
#             for i, is_outlier in enumerate(is_outlier_list):
#                 if is_outlier == -1:
#                     predicted_person = "Unknown"
#                     predicted_probability = 0.0  # Assuming unknown people have zero probability
#                 else:
#                      probabilities = clf.predict_proba(test_image_pca)[0]
#                      predicted_probability = np.max(probabilities)
#                      if predicted_probability < 0.30:
#                             predicted_person = "Unknown"
#                             predicted_probability = 0.0  # If probability is too low, consider it unknown
#                      else:
#                             predicted_person_id = np.argmax(probabilities) + 1
#                             predicted_person = person_names.get(predicted_person_id, "Unknown")
        
#                 # Write the prediction and probability to the file
#                             f.write(f"Image {i+1}: Predicted Person: {predicted_person}, Probability: {predicted_probability:.2f}\n")
# # Prediction pipeline for new images
# # def predict_new_image(image_path, pca, iso_forest, svm, person_names):
# #     test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# #     if test_image is None:
# #         print("Test image not found!")
# #         return
    
# #     # Resize and reshape the image for PCA
# #     resized_face = resize(test_image, (n_row, n_col), anti_aliasing=True).reshape(1, -1)
# #     face_pca = pca.transform(resized_face)
    
# #     # Anomaly detection
# #     is_outlier = iso_forest.predict(face_pca)
# #     if is_outlier == -1:
# #         print("Outlier detected, person unknown.")
# #         return "Unknown", 0.0
    
# #     # Prediction and probability
# #     probabilities = svm.predict_proba(face_pca)[0]
# #     predicted_probability = np.max(probabilities)
# #     predicted_person_id = np.argmax(probabilities) + 1
# #     predicted_person = person_names.get(predicted_person_id, "Unknown")
# #     return predicted_person, predicted_probability

# # # Example usage:
# # predicted_person, probability = predict_new_image("Junichiro_Koizumi_0019.jpg", pca, iso_forest, clf.best_estimator_, person_names)
# # print(f"Predicted: {predicted_person}, Probability: {probability:.2f}")
# # Display results
# cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
# cv2.putText(test_image, f"{predicted_person} ({predicted_probability:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
# plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
# plt.title(f"Predicted Person: {predicted_person}")
# plt.axis('off')
# plt.show()