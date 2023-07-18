import cv2
import os
import time
import uuid

# Directory to save the collected face images
output_dir = input('Enter your rollno : ')

# Create the output directory if it doesn't exist
path='C:/Users/91807/Desktop/FaceDetection/face_dataset'
temp_path=os.path.join(path,output_dir)
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

time.sleep(2)
# Counter for the number of collected face images
image_count = 0
 
# Set the maximum number of images to collect
max_images = 100

while image_count < max_images :
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and save the images
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Increment the image count
        image_count += 1

        # Save the face image to the output directory
        face_image_path = os.path.join(temp_path, f'{output_dir}_{str(uuid.uuid1())}.jpg')
        cv2.imwrite(face_image_path, frame[y:y+h, x:x+w])

    # Display the resulting frame
    cv2.imshow('Video', frame)

    time.sleep(0.5)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
