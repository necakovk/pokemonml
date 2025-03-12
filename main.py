import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import RPi.GPIO as GPIO
import time
import cv2 as cv
import numpy as np
from keras.models import load_model  # Ensure TensorFlow is installed
from picamera2 import Picamera2
from libcamera import controls

# Define ultrasonic sensor pins
TRIG_PIN = 23  # GPIO 23 (Pin 16)
ECHO_PIN = 24  # GPIO 24 (Pin 18) (Needs voltage divider to 3.3V)

# Turning speed and duration for accurate 90-degree turns
TURN_SPEED = 0.5
TURN_DURATION = 3.116

# Load the classifier model
model = load_model("keras_model.h5", compile=False)

# Load class labels
class_names = ["Bulbasaur", "Squirtle", "Snorlax", "Charmander", "Pokeball", "Pikachu"]

# Initialize the Picamera2
picam2 = Picamera2()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()
time.sleep(1)

# Ultrasonic setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.output(TRIG_PIN, False)

class MoveRobot(Node):
    def __init__(self):
        super().__init__('move_robot')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize decision mapping for turns
        self.turn_decision = self.initialize_decision_mapping()

        # Wait for user input before starting
        self.wait_for_user_start()

    def initialize_decision_mapping(self):
        """ User assigns 'left' or 'right' to each classification label. """
        print("\n🎯 Initialize Turn Decision Mapping 🎯")
        turn_mapping = {}
        
        for label in class_names:
            while True:
                direction = input(f"🔍 Enter turn direction for {label} ('left' or 'right'): ").strip().lower()
                if direction in ["left", "right"]:
                    turn_mapping[label] = direction
                    break
                else:
                    print("⚠ Invalid input! Enter 'left' or 'right'.")

        print("\n✅ Turn Mapping Initialized:", turn_mapping)
        return turn_mapping

    def wait_for_user_start(self):
        """ Waits for user to enter 'go' to start or 'exit' to stop. """
        while True:
            command = input("\n🚦 Enter 'go' to start, 'exit' to quit: ").strip().lower()
            if command == "go":
                print("▶ Starting the robot...")
                self.move_forward()
                break
            elif command == "exit":
                print("🛑 Exiting program.")
                rclpy.shutdown()
                exit()
            else:
                print("⚠ Invalid input! Enter 'go' or 'exit'.")

    def move_forward(self):
        """ Publishes forward movement once, allowing the main loop to handle continuous movement. """
        velocity = Twist()
        velocity.linear.x = 0.2  # Adjust speed as needed
        velocity.angular.z = 0.0
        self.publisher.publish(velocity)
        print("▶ Moving forward...")



    def stop(self):
        """ Stop the robot. """
        velocity = Twist()
        velocity.linear.x = 0.0
        velocity.angular.z = 0.0
        self.publisher.publish(velocity)
        print("\n⛔ Robot stopped.\n")

    def turn(self, direction):
        """ Turns the robot precisely 90 degrees using calibrated values. """
        velocity = Twist()
        velocity.angular.z = TURN_SPEED if direction == "right" else -TURN_SPEED

        print(f"\n🔄 Turning {direction} at speed {TURN_SPEED} for {TURN_DURATION} seconds...\n")

        start_time = time.time()
        while time.time() - start_time < TURN_DURATION:
            self.publisher.publish(velocity)
            time.sleep(0.1)  # Continuous publishing

        # Stop movement after turning
        velocity.angular.z = 0.0
        self.publisher.publish(velocity)
        print("⛔ Turn complete.\n")

    def read_distance(self):
        """ Get the distance from the ultrasonic sensor and print it. """
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        start_time, end_time = 0, 0

        while GPIO.input(ECHO_PIN) == 0:
            start_time = time.time()

        while GPIO.input(ECHO_PIN) == 1:
            end_time = time.time()

        duration = end_time - start_time
        distance_cm = (duration * 34300) / 2  # Convert to cm
        distance_in = distance_cm / 2.54  # Convert to inches

        print(f"📏 Distance: {distance_in:.2f} inches")
        return distance_in

    def capture_and_classify_image(self):
        """ Captures an image and classifies it using the trained model. """
        img_name = "/home/tuftsrobot/pokemonml/image.jpg"
        picam2.capture_file(img_name)
        
        # Load and preprocess image
        image = cv.imread(img_name)
        image = cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1  # Normalize

        # Predict using the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        print(f"\n📸 Image classified as: {class_name} with {confidence_score * 100:.2f}% confidence.")

        return class_name

def main():
    rclpy.init()
    node = MoveRobot()

    try:
        node.move_forward()  # Start moving forward once

        while rclpy.ok():
            distance = node.read_distance()
            print(f"📏 Current Distance: {distance:.2f} inches")  # Debugging

            if distance <= 7.0:  # Stop when an object is detected
                print("\n🛑 Object detected. Stopping movement.")
                node.stop()

                # Capture and classify image
                predicted_class = node.capture_and_classify_image()

                # Get turn direction based on classification
                turn_direction = node.turn_decision.get(predicted_class, "left")  # Default to left if unknown

                node.turn(turn_direction)
                time.sleep(2)  # Give time to complete turn
                node.move_forward()  # Resume moving forward

            time.sleep(0.1)  # Avoid excessive CPU usage
    except KeyboardInterrupt:
        print("\n🛑 Shutdown detected. Cleaning up...")
    finally:
        node.stop()
        GPIO.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        print("✅ Robot safely shut down.")


if __name__ == '__main__':
    main()
