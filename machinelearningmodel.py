from keras.models import load_model  # TensorFlow is required for Keras to work
from picamera2 import Picamera2
from libcamera import controls
import cv2 as cv
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Initialize the Picamera2
picam2 = Picamera2()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()
time.sleep(1)

while True:
    # Capture image from Picamera2
    img_name = "/home/tuftsrobot/pokemonml/image.jpg"
    picam2.capture_file(img_name)
    
    # Load the captured image
    image = cv.imread(img_name)
    
    # Resize the image to fit the model input size
    image = cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict using the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen for keyboard input to break the loop
    key = cv.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the Esc key
        break

picam2.stop()
cv.destroyAllWindows()
