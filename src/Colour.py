"""
Not used currently

Contains functions that create a colour map for the wf.
"""
import numpy as np


def tanhCol(num, col1, col2, squareness=300):
    """
    Will create the tanh interpolation between 2 colors

    Inputs:
        * num = a num between -1 and +1 (float)
        * col1 = color of -1 (list or tuple or np.array)
        * col2 = color of +1 (list or tuple or np.array)
        * squareness = how sharp the boundary between the quadrants
    """
    col1 = np.array(col1)
    col2 = np.array(col2)
    val = ((np.tanh(num * squareness) + 1) / 2) * (col2 - col1) + col1
    return val


def colorMap(cNumber,
             realPos=(1, 0, 0),
             realNeg=(0, 1, 0),
             imagPos=(0, 0, 1),
             imagNeg=(1, 1, 0)):
    """
    Will return an rgb color corresponding to the position of the cNumber on
    the color map. This basically makes 4 triangles and ties them together.

    Inputs:
        * cNumber => complex number
        * realPos => the real positive color
        * realNeg => the real negative color
        * imagPos => the imaginary positive color
        * imagNeg => the imaginary negative color
    """
    # Handle rotating the complex number by 45 degrees
    theta = np.pi/4
    rotMat = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    v = np.array([cNumber.real, cNumber.imag])
    v = np.matmul(rotMat, v)

    # Tanh interpolation
    real, imag = v

    rgb1 = tanhCol(real, realPos, (0, 0, 0))
    rgb1 *= tanhCol(imag, (0, 0, 0), realPos)

    rgb2 = tanhCol(real, (0, 0, 0), imagPos)
    rgb2 *= tanhCol(imag, imagPos, (0, 0, 0))

    rgb3 = tanhCol(real, (0, 0, 0), imagNeg)
    rgb3 *= tanhCol(imag, (0, 0, 0), imagNeg)

    rgb4 = tanhCol(real, realNeg, (0, 0, 0))
    rgb4 *= tanhCol(imag, realNeg, (0, 0, 0))

    rgb = rgb1 + rgb2 + rgb3 + rgb4
    return rgb  # / np.linalg.norm(rgb)


def testColorMap():
    """
    Run this function to visualise the color map.
    """
    import matplotlib.pyplot as plt

    numDivs = 400
    vals = np.ones((numDivs, numDivs, 3), dtype=float)

    for i, val in enumerate(np.linspace(-1, 1, numDivs)):
        for j, cval in enumerate(np.linspace(-1j, 1j, numDivs)):
            vals[i, j] = colorMap(val + cval)

    plt.figure()
    plt.imshow(vals)
    plt.yticks([])
    plt.xticks([])
    plt.show()

testColorMap()
