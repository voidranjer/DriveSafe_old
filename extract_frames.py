# import the necessary packages
import os
import argparse
import cv2

def count_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return -1

    # Initialize frame count
    frame_count = 0

    # Loop through the video frames and count them
    while True:
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Increment frame count
        frame_count += 1

    # Release the video capture object and close the video file
    cap.release()

    return frame_count

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str,
	help="path to input video file")
ap.add_argument("-o", "--output", default="output", type=str,
	help="path to output folder")
ap.add_argument("-n", "--num-frames", type=int, default=1000,
	help="number of frames to take")
args = vars(ap.parse_args())

# create output directory if it does not exist
if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

# count interval to extract frames depending on video length
total_frames = count_frames(args["input"])
interval = total_frames // args["num_frames"]
print(f"Total frames: {total_frames}")
print(f"Interval: {interval}")

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])

count = 0
write_count = 0

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # stop
    if (write_count == args["num_frames"]):
        break

    # respect the interval between frames
    count += 1
    if count % interval == 0:
        write_count += 1
        cv2.imwrite(f"{args['output']}/{write_count}.jpg", frame)