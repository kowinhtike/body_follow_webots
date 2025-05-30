from controller import Robot
import cv2
import numpy as np
import mediapipe as mp

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get the camera
camera = robot.getDevice("camera")
camera.enable(timestep)

# Get motor devices
left_motor = robot.getDevice('left_motor')
right_motor = robot.getDevice('right_motor')

# Set motors to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Set initial speed (radians per second)
speed = 10

# Create a named window with desired size
cv2.namedWindow("Webots Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webots Camera", 800, 600)  # Width, Height in pixels

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def goForward():
    left_motor.setVelocity(speed)
    right_motor.setVelocity(speed)

def goBackward():
    left_motor.setVelocity(speed *-1)
    right_motor.setVelocity(speed *-1)

def stop():
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

def goRight():
    left_motor.setVelocity(speed * 1/2)
    right_motor.setVelocity(0)

def goLeft():
    left_motor.setVelocity(0)
    right_motor.setVelocity(speed * 1/2)

def drive_logic(frame):
    found_body = False
    # Image shape (height, width, channels)
    (h, w) = image.shape[:2]
    # Center point calculation
    x_center = w // 2
    x_medium = x_center
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process frame with Pose model
    result = pose.process(rgb_frame)
    if result.pose_landmarks:
        # mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Get nose landmark
        nose = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        # body confrim
        found_body = True
        # Convert to pixel coordinates
        h, w, _ = frame.shape
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        # Draw a circle at the nose position
        cv2.circle(frame, (nose_x, nose_y), 30, (0, 0, 255), 3)  # Red circle
        x_medium = nose_x

    x_stateMessage = ""
    offset = 80  # Threshold for stable position
    if x_center - offset < x_medium < x_center + offset:
        x_stateMessage = "Hey wait!"
        goForward()
    elif x_medium < x_center - offset:
        x_stateMessage = "Turning Right!"
        goLeft()
    elif x_medium > x_center + offset:
        x_stateMessage = "Turning Left!"
        goRight()
    if(not found_body):
        x_stateMessage = "I got you!"
        stop()
    # Display Servo Status
    cv2.putText(frame, f"{x_stateMessage}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
        while robot.step(timestep) != -1:
        
            # Get the camera image
            image = camera.getImage()

            # Example: Get image dimensions
            width = camera.getWidth()
            height = camera.getHeight()
            x = width // 2
            y = height // 2

            # Example: Get a pixel color at (x, y)
            r = camera.imageGetRed(image, width, x, y)
            g = camera.imageGetGreen(image, width, x, y)
            b = camera.imageGetBlue(image, width, x, y)

            print(f"Pixel at ({x}, {y}): R={r} G={g} B={b}")
            img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))  # BGRA

            bgr_image = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            image = cv2.resize(bgr_image, (800, 600))

            # Convert the BGR image to RGB and process it with MediaPipe Pose
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            # Draw the pose annotations on the image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            image = drive_logic(image)
            cv2.imshow("Webots Camera", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


cv2.destroyAllWindows()
