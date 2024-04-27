# import the necessary packages
import os
import argparse
import cv2
from utils.frames import count_frames
import random

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str,
	help="path to input video file")
ap.add_argument("-o", "--output", default="output", type=str,
	help="path to output folder")
ap.add_argument("-n", "--num-frames", type=int, default=1000,
	help="number of frames to take")
args = vars(ap.parse_args())

labels = ["off", "safe", "collision", "tailgating", "weaving"]

# create output directory if it does not exist
if not os.path.exists(args["output"]):
    os.makedirs(args["output"])
for label in labels:
    if not os.path.exists(f"{args['output']}/{label}"):
        os.makedirs(f"{args['output']}/{label}")

# count interval to extract frames depending on video length
num_frames = count_frames(args["input"])
interval = num_frames // args["num_frames"]
print(f"[INFO] Total frames: {num_frames}")
print(f"[INFO] Interval: {interval}")

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])

count = 0
write_count = 0
frames = []

# loop over frames from the video file stream
print("[INFO] Extracting frames...")
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # stop
    if (count >= num_frames):
        break

    # respect the interval between frames
    count += 1
    if count % interval == 0:
        frames.append(frame)

# shuffle the framesk to prevent bias during labeling
print("[INFO] Shuffling frames...")
random.shuffle(frames)

# loop over the frames and label them
for frame in frames:
    # show the frame and wait for a keypress
    cv2.imshow("Labeler", frame)

    # display the labels and key mappings
    for idx, label in enumerate(labels):
        print(f"[{idx}] {label}", end=" | ")
    print("\n")

    # wait for user to label the frame
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord(str(i)) for i in range(len(labels))]:
        category = labels[int(chr(key))]
    else:
        print("[INFO] Unrecognized keystroke, skipping frame...")
        continue

    print(f"[INFO] Writing frame {write_count}/{args['num_frames']} to {category}...")
    write_count += 1
    cv2.imwrite(f"{args['output']}/{category}/{write_count}.jpg", frame)

