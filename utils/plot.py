import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

# Define a type alias for the dictionary structure
PlotDict = Dict[str, List[int]]

# Define the type for the 'plots' variable
PlotsType = List[PlotDict]

# Example usage
plots: PlotsType = [
    {
        "y_vals": [1, 2, 3],
        "label": "label1"
    },
    {
        "y_vals": [4, 5, 6],
        "label": "label2"
    }
]

def create_plot(x_size: int, y_range: int, plots: list):
    # Generate x values
    x = np.arange(x_size)

    # Create a figure and axis
    fig, ax = plt.subplots()
    for plot in plots:
        line, = ax.plot(x, plot["y_vals"], label=plot["label"])

    # set y range
    ax.set_ylim(0, y_range)

    # Add a legend
    ax.legend()

    # Update plot label/title
    ax.set_xlabel('Number of Frames')
    ax.set_ylabel('Cumulative Prediction Count')
    ax.set_title('DriveSafe score over time')

    # Render the plot in np format
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())

    # Extract RGB channels and discard alpha channel
    rgb_img = img_plot[:, :, :3]
    
    # black_pixels = np.all(img_plot == [0, 0, 0, 255], axis=-1)
    # img_plot[black_pixels, 3] = 255

    # alpha = 1  # Set opacity level between 0 (transparent) and 1 (opaque)
    # alpha_channel = img_plot[:, :, 3] # Extract the alpha channel from the image array
    # img_plot[:, :, 3] = alpha_channel * alpha # Set the alpha channel to the desired opacity level


    return rgb_img

    # Display the plot
    # plt.show()
