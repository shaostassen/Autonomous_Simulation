import cv2
import numpy as np
import ffmpeg

def crop_battlefield(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Iterate over frames to find the region to crop
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video file")
        return
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter the contours to find the largest square contour
    max_area = 0
    best_cnt = None
    
    for cnt in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        
        # Check if the contour is square-like
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = approx
    
    if best_cnt is not None:
        # Get the bounding box of the best contour
        x, y, w, h = cv2.boundingRect(best_cnt)
        
        # Define the crop area
        crop_area = (x, y, x+w, y+h)
    else:
        print("No square-like contour found")
        cap.release()
        return
    
    # Release the video capture since we no longer need it
    cap.release()
    
    # Crop the video using ffmpeg-python
    (
        ffmpeg
        .input(video_path)
        .crop(crop_area[0], crop_area[1], crop_area[2] - crop_area[0], crop_area[3] - crop_area[1])
        .output(output_path, r=fps)
        .run()
    )

# Provide the path to your input video and output video
video_path = "test.mp4"
output_path = "cropped_battlefield.mp4"

crop_battlefield(video_path, output_path)
