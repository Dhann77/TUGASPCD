import numpy as np
import cv2
import sys, inspect, os
import argparse

def image_resize(image, width=-1, height=-1):
    """
    Resize image while maintaining aspect ratio.
    """
    shape = image.shape
    if width == -1 and height == -1:
        return image
    elif width == -1:
        return cv2.resize(image, (int(height * shape[1] / shape[0]), height))
    elif height == -1:
        return cv2.resize(image, (width, int(width * shape[0] / shape[1])))
    else:
        return cv2.resize(image, (width, height))

def detect_face(face_cascade, image):
    """
    Detect the largest face in the image using Haar Cascade.
    """
    gray_image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    if len(faces) > 0:
        return max(faces, key=lambda item: item[2] * item[3])  # Largest face
    return None

def add_text(image, text):
    """
    Add overlay text to an image.
    """
    cv2.putText(image, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def main():
    # Argument parsing
    ap = argparse.ArgumentParser(description="Lucas Kanade Optical Flow Tracking")
    ap.add_argument("-v", "--video", required=False, help="Path to video file. Default: Camera")
    ap.add_argument("-n", "--max_corners", required=False, type=int, help="Max number of corners to track. Default: 100")
    args = vars(ap.parse_args())

    max_corners = args.get("max_corners", 100)
    video_path = args.get("video", None)

    # Initialize camera or video
    camera = cv2.VideoCapture(0 if video_path is None else video_path)
    if not camera.isOpened():
        print(f"Error: Unable to access {'camera' if video_path is None else 'video file'}!")
        return

    # Haar Cascade for face detection
    haar_cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_cascade_path):
        print(f"Error: Haar Cascade file '{haar_cascade_path}' not found!")
        return

    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # Parameters for optical flow
    feature_params = dict(maxCorners=max_corners, qualityLevel=0.3, minDistance=15, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_frame = None
    p0 = []  # Points to track

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            print("Error: Failed to read frame from camera/video!")
            break

        frame = image_resize(frame, width=640)  # Resize frame to a manageable size
        curr_frame_gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        info = ""

        # Detect face and initialize tracking points
        if prev_frame is None or len(p0) <= 3:
            info = "Detecting face..."
            face = detect_face(face_cascade, frame)
            if face is not None:
                x, y, w, h = face
                roi = np.zeros(curr_frame_gray.shape, dtype=np.uint8)
                roi[y:y + h, x:x + w] = 255
                p0 = cv2.goodFeaturesToTrack(curr_frame_gray, mask=roi, **feature_params)
                prev_frame = curr_frame_gray
            else:
                info = "No face detected!"
        else:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame_gray, p0, None, **lk_params)
            p0 = p1[st == 1].reshape(-1, 1, 2)  # Keep good points
            info = f"Tracking: {len(p0)} points"
            for pt in p1[st == 1]:
                a, b = pt.ravel()
                cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)
            prev_frame = curr_frame_gray

        # Add text overlay
        add_text(frame, info)
        cv2.imshow("Output", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Reset tracking
            p0 = []

    # Clean up
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
