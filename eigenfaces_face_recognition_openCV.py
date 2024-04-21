from __future__ import print_function
from time import time 
import cv2.data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.transform import resize
import cv2 


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



faces_image = np.load('./input_dataset/olivetti_faces.npy')
faces_target = np.load('./input_dataset/olivetti_faces_target.npy')

print(faces_image)


n_row = 64
n_col = 64
faces_image.shape

print(faces_image.shape)

#faces_image.shape = (400,64,64)
faces_data = faces_image.reshape(faces_image.shape[0], faces_image.shape[1] * faces_image.shape[2])
faces_data.shape

print(faces_data.shape)

print(faces_target)


#display image no. 20 // Test the dataset 
import warnings
warnings.filterwarnings("ignore")
from skimage.io import imshow
print("faces image size : ",faces_image.shape )
loadImage = faces_image[100]
print("Olivetti dataset image test dim : ", loadImage.shape)
imshow(loadImage) 

loadImage.shape

print(loadImage.shape)


n_samples = faces_image.shape[0]
# for machine learning we use the 2 data directly
X = faces_data
n_features = faces_data.shape[1]
#label to predict the id of person which assigned to 
y = faces_target
n_classes = faces_target.shape[0]

# printings for dataset information about the matrix
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# training split test 
Xtrain, Xtest , ytrain, ytest = train_test_split(X,y)

print("Xtrain : " ,Xtrain)
print("length of Xtrain ", len(Xtrain))
print("Xtest",Xtest)
print("Length of Xtest:",len(Xtest))
print("ytrain",ytrain)
print("Length of ytrain:",len(ytrain))
print("ytest",ytest)
print("Length of ytest:",len(ytest))


#compute PCA (EIGENFACES) on the olivetti dataset / unsupervised

n_components = 150 

# 150 pca component of top eigenfaces from 300 training components in Xtrain
print("Extract top %d eigenfaces from %d faces"%(n_components,Xtrain.shape[0]))

t0 = time()
pca= PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(Xtrain)
print("done in %0.3fs"%(time()-t0))

eigenfaces = pca.components_.reshape((n_components,n_row,n_col))

print("projecting the input data on the eigenfaces normal basis (eigenspace)")

t0 = time()
Xtrain_pca = pca.transform(Xtrain)
Xtest_pca = pca.transform(Xtest)

print ("done in %0.3fs"%(time()-t0))


#now fitting the classifer to training set using svm 

print("Fitting the classifier to the training set using svm ")

t0 = time()
param_grid = {'C' : [1e3 , 5e3 ,1e4 ,5e4 ,1e5], 
              'gamma':[0.0001 , 0.0005 , 0.001,0.005 ,0.01 ,0.1],}
classifier_svm = GridSearchCV(SVC(kernel = 'rbf', class_weight='balanced'), param_grid)
classifier_svm = classifier_svm.fit(Xtrain_pca, ytrain)

print ("done in %0.3fs"% (time()-t0))
print("Best estimation was found by Grid search cv:")
print(classifier_svm.best_estimator_)

#quantative evaulation of the model quality on the test dataset

print("Predictiing people's names on the test dataset")

pred_time = time()
y_pred = classifier_svm.predict(Xtest_pca)

print("done in %0.3fs"%(time()-pred_time))
print(classification_report(ytest,y_pred))

print(confusion_matrix(ytest,y_pred,labels=range(n_classes)).shape)
print(ytest)
print(y_pred)

print(ytest==y_pred)

fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                        subplot_kw={'xticks': [], 'yticks': []},
                        gridspec_kw=dict(wspace=0.1))

for i , ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(64,64), cmap='bone')



plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
test_image_model = cv2.imread("Amelia_Vega_0006.jpg")
# Convert the image to grayscale
grayscale_image = cv2.cvtColor(test_image_model, cv2.COLOR_BGR2GRAY)
# Resize the grayscale image to (64, 64)
resized_image = resize(grayscale_image, (64, 64), anti_aliasing=True)
faces = face_cascade.detectMultiScale(grayscale_image,1.1,4)
test_image_data = resized_image.reshape(1, -1)
test_image_pca = pca.transform(test_image_data)
prediction_test_image = classifier_svm.predict(test_image_pca)

print("predicted person id is  : ", prediction_test_image.shape[0])

# Get the predicted person name or ID
# Get the predicted person name or ID
# Get the predicted person ID
predicted_person_id = prediction_test_image[0]

# Set a threshold for prediction confidence
threshold = 0.7  # You can adjust this threshold based on your requirements

# Check if the predicted person ID exists in the person_names dictionary
 # Get the confidence scores for all classes using the best estimator
confidence_scores = classifier_svm.best_estimator_.decision_function(test_image_pca)[0]
print("confidence score is : ",confidence_scores)

    # Get the maximum confidence score
max_confidence = max(confidence_scores)
print("max_confidence  is : ",max_confidence)


# Check if the maximum confidence score is above the threshold
if max_confidence < threshold:
    # If the confidence is below the threshold, label it as unknown
    predicted_person = "Unknown (Confidence below threshold)"
    prediction_status = "Unknown"
else:
    # If the confidence is high enough, get the predicted person name
    predicted_person_id = np.argmax(confidence_scores)
    predicted_person = person_names[predicted_person_id]
    prediction_status = "Known"

for (x, y, w, h) in faces:
    cv2.rectangle(test_image_model, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Add text displaying the predicted person's name

# Calculate the font scale dynamically based on the image size
font_scale = min(test_image_model.shape[1] / 800, test_image_model.shape[0] / 600)

# Calculate the font thickness based on the font scale
font_thickness = max(1, int(font_scale))

# Get the text size
text_size = cv2.getTextSize(f"Predicted Person: {predicted_person} ({prediction_status})", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

# Calculate the text position
text_x = 10
text_y = int(30 * font_scale) + text_size[1]  # Adjust y position based on text height

# Add text displaying the predicted person's name to the image
cv2.putText(test_image_model, f"Predicted Person: {predicted_person} ({prediction_status})", 
            (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

# Display the image with rectangles and text using Matplotlib
plt.imshow(cv2.cvtColor(test_image_model, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Person: {predicted_person} ({prediction_status})")
plt.axis('off')  # Turn off axis labels
plt.show() 