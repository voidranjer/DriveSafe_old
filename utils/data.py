from imutils import paths
import os
import cv2

def get_labeled_data(data_dir: str, validation_dir: str, LABELS: set):
    image_paths = {
        "training": list(paths.list_images(data_dir)),
        "validation": list(paths.list_images(validation_dir))
    }

    data = {
        "training": [],
        "validation": []
    }

    labels = {
        "training": [],
        "validation": []
    }

    # loop over the image paths
    for mode in ["training", "validation"]:
        for image_path in image_paths[mode]:
            # extract the class label from the filename
            label = image_path.split(os.path.sep)[-2]

            # if the label of the current image is not part of of the labels
            # are interested in, then ignore the image
            if label not in LABELS:
                continue

            # load the image, convert it to RGB channel ordering, and resize
            # it to be a fixed 224x224 pixels, ignoring aspect ratio
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

            # update the data and labels lists, respectively
            data[mode].append(image)
            labels[mode].append(label)

    training_data = data["training"]
    training_labels = labels["training"]
    validation_data = data["validation"]
    validation_labels = labels["validation"]

    return training_data, training_labels, validation_data, validation_labels