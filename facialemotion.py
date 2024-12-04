import cv2
from deepface import DeepFace

# Load the Haar cascade classifier for face detection
# Haar cascades are pre-trained XML models used for object detection, here specifically for frontal faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video feed from the default webcam (device index 0)
cap = cv2.VideoCapture(0)

# Infinite loop to continuously process video frames
while True:
    # Capture each frame from the video feed
    ret, frame = cap.read()
    
    # Check if frame is captured successfully
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Convert the captured frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale frame back to RGB format for DeepFace processing
    # DeepFace expects input frames in RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the grayscale frame
    # scaleFactor: Resizes the image to detect faces at different scales
    # minNeighbors: Number of neighbors a rectangle needs to be considered a face
    # minSize: Minimum size of a detected face
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Extract the Region of Interest (ROI) for the face
        # ROI is used as input to the DeepFace analyzer
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Analyze the face ROI for emotions using DeepFace
        # actions=['emotion']: Specifies that we want to detect emotions
        # enforce_detection=False: Allows the program to proceed even if the detection isn't perfect
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Extract the dominant emotion predicted by DeepFace
        emotion = result[0]['dominant_emotion']

        # Draw a rectangle around the detected face in the original frame
        # (x, y) is the top-left corner, and (x + w, y + h) is the bottom-right corner
        # Color of the rectangle is red (BGR: (0, 0, 255)) with a thickness of 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the detected emotion above the rectangle
        # Text is displayed in red with a font size of 0.9 and thickness of 2
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the processed frame with detected faces and emotions
    cv2.imshow('Real-time Emotion Detection', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam resource after use
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
