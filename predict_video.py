# import the necessary packages
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from utils.frames import count_frames
from utils.plot import create_plot

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=30,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
num_frames = count_frames(args["input"])

prediction_counts = {
	"safe": [0],
	"collision": [0],
	"tailgating": [0],
	"weaving": [0],
	"off": [0]
}
current_frame = 0
max_y = 0

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# increment the current frame
	current_frame += 1
		
	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean
	
	# make predictions on the frame and then update the predictions queue
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)

	# perform prediction averaging over the current history of previous predictions
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]

	# increment the prediction count
	for l in lb.classes_:
		new_count = prediction_counts[l][current_frame - 1] + 1 if label == l else prediction_counts[l][current_frame - 1]
		if (new_count > max_y):
			max_y = new_count
		prediction_counts[l].append(new_count)

    # draw the activity on the output frame
	default_color = (100, 100, 100)
	red = (0, 0, 255)
	green = (0, 255, 0)
	text_x = W - 200
	text_y = H // 2 - 80
	cv2.putText(output, "SAFE", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
			 green if label == "safe" else default_color, 5)
	cv2.putText(output, "COLLISION", (text_x, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
			 red if label == "collision" else default_color, 5)
	cv2.putText(output, "TAILGATING", (text_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
			 red if label == "tailgating" else default_color, 5)
	cv2.putText(output, "WEAVING", (text_x, text_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
			 red if label == "weaving" else default_color, 5)
	
	# draw the matplotlib plot on the output frame
	plot_img = create_plot(current_frame + 1, max_y + 30, [
		{
			"y_vals": prediction_counts["safe"],
			"label": "safe"
		},
		{
			"y_vals": prediction_counts["collision"],
			"label": "collision"
		},
		{
			"y_vals": prediction_counts["tailgating"],
			"label": "tailgating"
		},
		{
			"y_vals": prediction_counts["weaving"],
			"label": "weaving"
		}
	])

	# resize the plot image to fit the output frame, and convert the color profile
	plot_img = cv2.resize(plot_img, (W // 3, H // 3))
	plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

	# add plot to the bottom right of the frame
	output[H - plot_img.shape[0]:, W - plot_img.shape[1]:W] = plot_img
	

		
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)
	
	# write the output frame to disk
	writer.write(output)
	
	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
	
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()