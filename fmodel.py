import os
import cv2
import csv
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Dataset Collection
dataset_path = "face_dataset"

# Step 2: Data Preprocessing
def preprocess_dataset(dataset_path):
    faces = []
    labels = []

    for label, person in enumerate(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person)
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))
            faces.append(image)
            labels.append(label)

    faces = np.array(faces)
    labels = np.array(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_dataset(dataset_path)

# Reshape the input data for ImageDataGenerator
X_train = X_train.reshape(-1, 64, 64, 1)

# Step 3: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest')

datagen.fit(X_train)


# Step 4: CNN Model Training
input_shape = (64, 64, 1)

# Base model for feature extraction
base_model = models.Sequential()
base_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
base_model.add(layers.MaxPooling2D((2, 2)))
base_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
base_model.add(layers.MaxPooling2D((2, 2)))
base_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
base_model.add(layers.MaxPooling2D((2, 2)))
base_model.add(layers.Flatten())
base_model.add(layers.Dense(128, activation='relu'))

# Branch for person recognition
person_branch = layers.Dense(len(os.listdir(dataset_path)), activation='softmax')(base_model.output)

# Final model
model = models.Model(inputs=base_model.input, outputs=person_branch)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Step 5: Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Step 6: Inference and Attendance System
def detect_faces_and_mark_attendance(model, dataset_path, threshold=0.7):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    attendees=[]
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (64, 64))
            face_img = np.expand_dims(face_img, axis=-1)
            face_img = np.expand_dims(face_img, axis=0)
            prediction = model.predict(face_img)
            confidence = np.max(prediction)
            label = np.argmax(prediction)

            if confidence > threshold:
                person=os.listdir(dataset_path)[np.argmax(prediction)]
                if person not in attendees:
                    attendees.append(person)
                #person = os.listdir(dataset_path)[label]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    mark_attendance(person)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f"Already Marked {person}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unrecognized Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def mark_attendance(person):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([person, 'Present', timestamp])

# Set the threshold value for detection confidence
detection_threshold = 0.9

# Detect faces and mark attendance
detect_faces_and_mark_attendance(model, dataset_path, threshold=detection_threshold)
