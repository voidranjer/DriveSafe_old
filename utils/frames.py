import cv2

def count_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("[ERROR] Could not open video")
        return -1

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    cap.release()

    return total_frames

# Example usage
if __name__ == "__main__":
    video_path = "example_clips/new_test.mp4"  # Replace with the path to your video file
    num_frames = count_frames(video_path)
    print("Total number of frames:", num_frames)
