import numpy as np

def tanh_rgb(i, a, b, squareness):
    return ((a-b)/2)*(np.tanh(squareness*(i))+1)+b

def quad_col(real, imag, imag_neg, real_pos, real_neg, imag_pos, squareness):
    x = 0.707*real-0.707*imag
    y = 0.707*real+0.707*imag
    return tanh_rgb(x, imag_neg, real_pos, squareness)*tanh_rgb(y, 1., 0., squareness)+ tanh_rgb(x, real_neg, imag_pos, squareness)*tanh_rgb(y, 0., 1., squareness)




