import matplotlib.pyplot as plt
import numpy as np

def create_plot():
    # Parameters
    num_points = 100  # Number of points in the plot

    # Generate x and y values
    x = np.linspace(0, 2*np.pi, num_points)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)

    # Create a figure and axis
    fig, ax = plt.subplots()
    line1, = ax.plot(x, y1, label='sin(x)')  # plot sin(x)
    line2, = ax.plot(x, y2, label='cos(x)')  # plot cos(x)
    line3, = ax.plot(x, y3, label='tan(x)')  # plot tan(x)

    # set y range
    ax.set_ylim(-2, 2)

    # Add a legend
    ax.legend()

    # Update plot label/title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trigonometric Functions')

    # Render the plot in np format
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())

    return img_plot

    # Display the plot
    # plt.show()
