#!/usr/bin/env python3
"""A Python script to extract the point coordinates from the comic.
"""

import numpy as np

from matplotlib import pyplot as plt
from scipy import ndimage

def main():
    """Do the main processing.
    """
    # Load image
    image_raw = plt.imread('image_process/06_final.png')
    # Convert to binary to make finding easier - via alpha threshold
    image_bin = np.array([np.array(
        [1 if pixel[-1] > 0.5 else 0 for pixel in row])
                          for row in image_raw])
    # Initialize variables
    height = len(image_bin)
    points = []
    # Filter grouped pixels
    structure = np.ones((3, 3), dtype=np.int)
    labeled, component_count = ndimage.measurements.label(image_bin, structure)
    # Compute center of mass
    for label in range(1, component_count + 1):
        matches = np.argwhere(labeled == label)
        center_of_mass = np.sum(matches, axis=0) / len(matches)
        points.append(center_of_mass)
    # Generate CSV
    csv = ''
    for line in points:
        x = line[1]
        # Mirror the coordinates so it is not counting from the top left but
        # The bottom left as in the comic
        y = height - line[0]
        csv += f'{x:.1f},{y:.1f}\n'
    # Plot
    plt.imshow(labeled)
    plt.scatter(*reversed(list(zip(*points))), 4, '0')
    # Output
    plt.savefig('image_process/07_mapped.png')
    with open('points.csv', 'w') as f:
        f.write(csv)
    plt.show()

if __name__ == '__main__':
    main()
