# import the necessary packages
import os
import argparse
import cv2
from utils.frames import count_frames

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
print(f"[INFO] Total frames: {total_frames}")
print(f"[INFO] Interval: {interval}")

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