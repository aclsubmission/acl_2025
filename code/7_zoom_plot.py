import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_zoom_plot(img_path, x1, x2, y1, y2):
    # Load image
    img = Image.open(img_path)
    img_array = np.array(img)

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display full image on the left
    ax1.imshow(img_array)
    ax1.set_title('Original Image')

    # Draw rectangle on full image
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2)
    ax1.add_patch(rect)

    # Display zoomed portion on the right
    zoomed_img = img_array[y1:y2, x1:x2]
    ax2.imshow(zoomed_img)
    ax2.set_title('Zoomed Image')

    # Draw lines connecting the corners
    
    # Remove axis ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Adjust layout and display plot
    plt.tight_layout()
    plt.show()

# Example usage:
create_zoom_plot('img1.png', 300, 800, 1500, 2500)