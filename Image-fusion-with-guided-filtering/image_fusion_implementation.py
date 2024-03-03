from PIL import Image
import numpy as np
from numpy.random import laplace
from scipy.ndimage import gaussian_filter, uniform_filter, laplace
from scipy.signal import convolve2d
import cv2
def padding(im, r):
    # Convert the input image to a NumPy array.
    im = np.array(im)

    # Check if the image is grayscale (2 dimensions).
    if im.ndim == 2:
        # Pad the grayscale image with 'r' pixels on all sides using reflection padding.
        padded_im = np.pad(im, ((r, r), (r, r)), 'reflect')

    # Check if the image is a color image (3 dimensions).
    elif im.ndim == 3:
        # Pad the color image with 'r' pixels on height and width, but not in the channel dimension, using reflection padding.
        padded_im = np.pad(im, ((r, r), (r, r), (0, 0)), 'reflect')

    else:
        raise ValueError("Invalid image dimension!")

    return padded_im
def mean_filter_convolution(image, window_size):
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    result = convolve2d(image, kernel, mode='same')
    return result
def variance_filter_convolution(image, window_size):
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    mean = convolve2d(image, kernel, mode='same')
    mean_of_squared = convolve2d(image ** 2, kernel, mode='same')
    variance = mean_of_squared - (mean ** 2)
    return variance
def guided_filter(im, guide, r, epsilone):
    # Convert the image to float for precision.
    im = np.array(im, np.float32)

    # Copy the input image to 'O' to store the output later.
    O = np.array(im, np.float32, copy=True)

    # Pad the images for easier calculation of local window operations.
    im = padding(im, r)
    guide = padding(guide, r)

    # Initialize matrices to hold the intermediate coefficients 'a_k' and 'b_k'.
    n, m = np.shape(im)
    a_k = np.zeros((n, m), np.float32)
    b_k = np.zeros((n, m), np.float32)

    # Window size
    w = 2 * r + 1
    mu_k = mean_filter_convolution(guide,r)
    delta_k = variance_filter_convolution(guide,r)
    P_k_bar = mean_filter_convolution(im,r)
    #print(P_k_bar)

    # Compute 'a_k' and 'b_k' for each pixel using the guide image and input image.
    for i in range(r, n - r):
        for j in range(r, m - r):
            # Local windows for the current pixel.
            I = guide[i - r:i + r + 1, j - r:j + r + 1]
            P = im[i - r:i + r + 1, j - r:j + r + 1]

            # Compute the numerator for 'a_k' as dot product of the flattened windows.
            somme = np.dot(np.ndarray.flatten(I), np.ndarray.flatten(P)) / (w ** 2)

            # Compute 'a_k' and 'b_k' using the formulas.
            a_k[i, j] = (somme - mu_k[i-r,j-r] * P_k_bar[i-r,j-r]) / (delta_k[i-r,j-r] + epsilone)
            b_k[i, j] = P_k_bar[i-r,j-r] - a_k[i, j] * mu_k[i-r,j-r]

    # Extract the relevant sections of 'a_k' and 'b_k' (removing padding).
    a = a_k[r:n - r, r:m - r]
    b = b_k[r:n - r, r:m - r]

    # Re-pad 'a' and 'b' matrices for calculation of final image.
    a = padding(a, r)
    b = padding(b, r)

    # Compute the final output image 'O' using 'a' and 'b' and the guide image.
    for i in range(r, n - r):
        for j in range(r, m - r):
            # Compute local means of 'a' and 'b' coefficients.
            a_k_bar = a[i - r: i + r + 1, j - r: j + r + 1].sum() / (w * w)
            b_k_bar = b[i - r: i + r + 1, j - r: j + r + 1].sum() / (w * w)
            O[i - r, j - r] = a_k_bar * guide[i, j] + b_k_bar
    return O
def guided_filter_color(im, guide, r, epsilone):
    # Convert the guide image to RGB if it's RGBA.
    guide = rgba_to_rgb(guide)

    # Convert the input image and the original image to float32 for processing.
    im = np.array(im, np.float32)
    O = np.array(im, np.float32, copy=True)

    # Apply padding to the input and guide images.
    im = padding(im, r)
    guide = padding(guide, r)
    n, m = np.shape(im)

    # Initialize arrays for the coefficients 'a' and 'b'.
    a_k = np.zeros((n, m, 3), np.float32)
    b_k = np.zeros((n, m), np.float32)

    #the size of window.
    w = 2 * r + 1

    # Iterate over the image to compute 'a' and 'b' for each pixel.
    for i in range(r, n - r):
        for j in range(r, m - r):
            # Extract local patches from guide and input images.
            I = guide[i - r:i + r + 1, j - r:j + r + 1]
            P = im[i - r:i + r + 1, j - r:j + r + 1]

            # Calculate the mean and covariance of the guide patch.
            mu_k = np.mean(I, axis=(0, 1))
            Sigma_k = np.cov(I.reshape(-1, 3).T)

            # Compute the mean of the input patch.
            p_k_bar = np.mean(P)

            # Calculate the dot product of guide and input patches.
            summation_term = np.sum(I * P[:, :, np.newaxis], axis=(0, 1))

            # Compute the 'a' coefficient for the current pixel.
            a_k[i, j] = np.linalg.inv(Sigma_k + epsilone * np.eye(3)).dot(summation_term / np.size(P) - mu_k * p_k_bar)

            # Compute the 'b' coefficient for the current pixel.
            b_k[i, j] = p_k_bar - np.dot(a_k[i, j], mu_k)

    # Trim the 'a' and 'b' arrays to remove the padding.
    a = a_k[r:n - r, r:m - r]
    b = b_k[r:n - r, r:m - r]

    # Apply padding to the 'a' and 'b' arrays.
    a = padding(a, r)
    b = padding(b, r)

    # Iterate over the image to compute the output image.
    for i in range(r, n - r):
        for j in range(r, m - r):
            # Calculate the mean of 'a' and 'b' in the local window.
            a_k_bar = a[i - r: i + r + 1, j - r: j + r + 1].sum(axis=(0, 1)) / (w * w)
            b_k_bar = b[i - r: i + r + 1, j - r: j + r + 1].sum() / (w * w)

            # Apply the guided filter formula to compute the output pixel value.
            O[i - r, j - r] = np.dot(a_k_bar, guide[i, j]) + b_k_bar

    return O

def rgb_to_grayscale_channels(im):

    # method used to convert the color image into three channel
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    gray_r = np.zeros_like(r)
    gray_g = np.zeros_like(g)
    gray_b = np.zeros_like(b)

    gray_r[:, :] = r
    gray_g[:, :] = g
    gray_b[:, :] = b

    return (gray_r+gray_g+gray_b)/3
def rgba_to_rgb(rgba_image):
    #convert the image from RGBA to RGB because the methode should work on the color image with three channels
    rgb = rgba_image[:, :, :3]
    if rgba_image.shape[2] == 4:
        alpha = rgba_image[:, :, 3:4] / 255.0
        background = np.ones_like(rgb)
        rgb = (1 - alpha) * background + alpha * rgb
    return rgb.astype(np.uint8)
def box_filter(X, r):
    return cv2.blur(X, (r, r))

def guided_filter_color_ameliorer(p, I, radius, eps):
    # Convert the guide image to RGB if it's RGBA.
    I = rgba_to_rgb(I)
    I = I.astype(np.float32)

    # Compute the mean of the guidance image using box filter.
    meanI = box_filter(I, radius)

    # Compute the mean of the input image using box filter.
    meanp = box_filter(p, radius)
    meanI2 = meanI ** 2

    # Mean of the product of guidance image and input image.
    mean_pI = box_filter(I * p[:, :, None], radius)

    # Covariance between the guidance image and the input image.
    covIp = mean_pI - meanI * meanp[:, :, None]

    # Variance of the guidance image.
    varI = meanI2 - meanI ** 2

    # Compute the 'a' coefficient for the guided filter.
    a = covIp / (varI + eps)

    # Compute the 'b' coefficient for the guided filter.
    b = meanp[:, :, np.newaxis] - a * meanI

    # Smooth the 'a' coefficient using box filter.
    mean_a = box_filter(a, radius)

    # Smooth the 'b' coefficient using box filter.
    mean_b = box_filter(b, radius)

    # Compute the output image.
    q = np.mean(mean_a * I + mean_b, axis=2)

    return q
def guided_filter_gray_ameliorer(p, I, radius, eps):
    I = I.astype(np.float32)
    meanI = box_filter(I, radius)
    meanp = box_filter(p, radius)
    meanI2 = meanI ** 2
    mean_pI = box_filter(I * p, radius)
    covIp = mean_pI - meanI * meanp
    varI = meanI2 - meanI ** 2
    a = covIp / (varI + eps)
    b = meanp - a * meanI
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)

    # Compute the output image.
    q = mean_a * I + mean_b


    return q

def fusion_color(im1, im2):
    # Normalize the input images to [0,1]
    im1 = im1.astype(np.float32) / 255.0 if im1.dtype == np.uint8 else im1
    im2 = im2.astype(np.float32) / 255.0 if im2.dtype == np.uint8 else im2

    # Verify dimensions match
    if im1.shape != im2.shape:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

    # Parameters
    sigma_r = 5
    average_filter_size = 31
    r_1 = 45
    r_2 = 7
    eps_1 = 0.3
    eps_2 = 10e-6

    # Base layers
    base_layer1 = uniform_filter(im1, mode='reflect', size=average_filter_size)
    base_layer2 = uniform_filter(im2, mode='reflect', size=average_filter_size)

    # Detail layers
    detail_layer1 = im1 - base_layer1
    detail_layer2 = im2 - base_layer2

    # Edge information
    transformed_matrix_1 = laplace(im1, mode='reflect')
    transformed_matrix_2 = laplace(im2, mode='reflect')

    # Saliency maps
    saliency1 = gaussian_filter(np.sum(np.abs(transformed_matrix_1), axis=2), sigma_r, mode='reflect')
    saliency2 = gaussian_filter(np.sum(np.abs(transformed_matrix_2), axis=2), sigma_r, mode='reflect')

    # Mask
    mask = np.float32(saliency2 > saliency1)

    # Weights
    w_b1 = guided_filter_color(1 - mask, im1, r_1, eps_1)
    w_b2 = 1 - w_b1

    w_d1 = guided_filter_color(1 - mask, im1, r_2, eps_2)
    w_d2 = 1 - w_d1

    w_b1 = np.clip(w_b1, 0, 1)
    w_b2 = np.clip(w_b2, 0, 1)
    w_d1 = np.clip(w_d1, 0, 1)
    w_d2 = np.clip(w_d2, 0, 1)

    # Fused base and detail layers
    B_fused = base_layer1 * w_b1[:, :, np.newaxis] + base_layer2 * w_b2[:, :, np.newaxis]
    D_fused = detail_layer1 * w_d1[:, :, np.newaxis] + detail_layer2 * w_d2[:, :, np.newaxis]

    # Final fused image
    F = B_fused + D_fused
    F = np.clip(F, 0, 1)
    F = (F * 255).astype(np.uint8)
    return F
def fusion_color_ameliorer(im1, im2):
    # Normalize the input images to [0,1]
    im1 = im1.astype(np.float32) / 255.0 if im1.dtype == np.uint8 else im1
    im2 = im2.astype(np.float32) / 255.0 if im2.dtype == np.uint8 else im2

    # Verify dimensions match
    if im1.shape != im2.shape:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

    # Parameters
    sigma_r = 5
    average_filter_size = 31
    r_1 = 45
    r_2 = 7
    eps_1 = 0.3
    eps_2 = 10e-6

    # Base layers
    base_layer1 = uniform_filter(im1, mode='reflect', size=average_filter_size)
    base_layer2 = uniform_filter(im2, mode='reflect', size=average_filter_size)

    # Detail layers
    detail_layer1 = im1 - base_layer1
    detail_layer2 = im2 - base_layer2

    # Edge information
    transformed_matrix_1 = laplace(im1, mode='reflect')
    transformed_matrix_2 = laplace(im2, mode='reflect')

    # Saliency maps
    saliency1 = gaussian_filter(np.sum(np.abs(transformed_matrix_1), axis=2), sigma_r, mode='reflect')
    saliency2 = gaussian_filter(np.sum(np.abs(transformed_matrix_2), axis=2), sigma_r, mode='reflect')

    # Mask
    mask = np.float32(saliency2 > saliency1)

    # Weights
    w_b1 = guided_filter_color_ameliorer(1 - mask, im1, r_1, eps_1)
    w_b2 = 1 - w_b1

    w_d1 = guided_filter_color_ameliorer(1 - mask, im1, r_2, eps_2)
    w_d2 = 1 - w_d1

    w_b1 = np.clip(w_b1, 0, 1)
    w_b2 = np.clip(w_b2, 0, 1)
    w_d1 = np.clip(w_d1, 0, 1)
    w_d2 = np.clip(w_d2, 0, 1)

    # Fused base and detail layers
    B_fused = base_layer1 * w_b1[:, :, np.newaxis] + base_layer2 * w_b2[:, :, np.newaxis]
    D_fused = detail_layer1 * w_d1[:, :, np.newaxis] + detail_layer2 * w_d2[:, :, np.newaxis]

    # Final fused image
    F = B_fused + D_fused
    F = np.clip(F, 0, 1)
    F = (F * 255).astype(np.uint8)
    return F
def fusion_gray(im1, im2):
    # Set constants for the fusion process.
    sigma_r = 5
    average_filter_size = 31
    r_1 = 45
    r_2 = 7
    eps_1 = 0.3
    eps_2 = 10e-6

    # Normalize images if their maximum value is greater than 1.
    if im1.max() > 1:
        im1 = im1 / 255
    if im2.max() > 1:
        im2 = im2 / 255

    # Split each image into its RGB channels.
    im1_blue, im1_green, im1_red = cv2.split(im1)
    im2_blue, im2_green, im2_red = cv2.split(im2)

    # Generate base layers for both images using a uniform filter.
    base_layer1 = uniform_filter(im1, mode='reflect', size=average_filter_size)
    b1_blue, b1_green, b1_red = cv2.split(base_layer1)

    base_layer2 = uniform_filter(im2, mode='reflect', size=average_filter_size)
    b2_blue, b2_green, b2_red = cv2.split(base_layer2)

    # Create detail layers by subtracting the base layers from the original images.
    detail_layer1 = im1 - base_layer1
    d1_blue, d1_green, d1_red = cv2.split(detail_layer1)

    detail_layer2 = im2 - base_layer2
    d2_blue, d2_green, d2_red = cv2.split(detail_layer2)

    # Compute saliency maps for both images using Laplacian and Gaussian filters

    saliency1 = gaussian_filter(abs(laplace(im1_blue + im1_green + im1_red, mode='reflect')), sigma_r, mode='reflect')
    saliency2 = gaussian_filter(abs(laplace(im2_blue + im2_green + im2_red, mode='reflect')), sigma_r, mode='reflect')

    # Generate a mask based on the comparison of the saliency maps.

    mask = np.float32(np.argmax([saliency1, saliency2], axis=0))

    # Convert images to float32 for further processing.

    im1 = np.float32(im1)
    im2 = np.float32(im2)
    # Apply the guided filter to the mask and the images with two different radii.
    g1r1 = guided_filter(1 - mask, im1[:, :, 0], r_1, eps_1) + guided_filter(1 - mask, im1[:, :, 1], r_1,
                                                                             eps_1) + guided_filter(1 - mask,
                                                                                                    im1[:, :, 2], r_1,
                                                                                                   eps_1)

    g2r1 = 1-g1r1

    g1r2 = guided_filter(1 - mask, im1[:, :, 0], r_2, eps_2) + guided_filter(1 - mask, im1[:, :, 1], r_2,
                                                                             eps_2) + guided_filter(1 - mask,
                                                                                                    im1[:, :, 2], r_2,
                                                                                                    eps_2)
    g2r2= 1-g1r2
    # Clip the filter outputs to the range [0, 1]
    g1r1 = np.clip(g1r1, 0, 1)
    g2r1 = np.clip(g2r1, 0, 1)
    g1r2 = np.clip(g1r2, 0, 1)
    g2r2 = np.clip(g2r2, 0, 1)

    # Fuse base and detail layers using weighted sums based on the guided filters
    fused_base1 = np.float32((b1_blue * (g1r1) + b2_blue * (g2r1)) / ((g1r1 + g2r1)))
    fused_detail1 = np.float32((d1_blue * (g1r2) + d2_blue * (g2r2)) / ((g1r2 + g2r2)))
    fused_base2 = np.float32((b1_green * g1r1 + b2_green * g2r1) / ((g1r1 + g2r1)))
    fused_detail2 = np.float32((d1_green * (g1r2) + d2_green * (g2r2)) / ((g1r2 + g2r2)))
    fused_base3 = np.float32((b1_red * (g1r1) + b2_red * (g2r1)) / ((g1r1 + g2r1)))
    fused_detail3 = np.float32((d1_red * (g1r2) + d2_red * (g2r2)) / ((g1r2 + g2r2)))
    #compute the fusion channels
    B1 = np.float32(fused_base1 + fused_detail1)
    B2 = np.float32(fused_base2 + fused_detail2)
    B3 = np.float32(fused_base3 + fused_detail3)
    #merge the channels
    fusion1 = np.float32(cv2.merge((B1, B2, B3)))
    fusion1 = np.clip(fusion1, 0, 1)
    fusion1 = (fusion1 * 255).astype(np.uint8)
    #return the output
    return fusion1
def fusion_gray_ameliorer(im1, im2):
    sigma_r = 5
    average_filter_size = 31
    r_1 = 45
    r_2 = 7
    eps_1 = 0.3
    eps_2 = 10e-6

    if im1.max() > 1:
       im1 = im1 / 255
    if im2.max() > 1:
        im2 = im2 / 255

    im1_blue, im1_green, im1_red = cv2.split(im1)
    im2_blue, im2_green, im2_red = cv2.split(im2)

    base_layer1 = uniform_filter(im1, mode='reflect', size=average_filter_size)
    b1_blue, b1_green, b1_red = cv2.split(base_layer1)

    base_layer2 = uniform_filter(im2, mode='reflect', size=average_filter_size)
    b2_blue, b2_green, b2_red = cv2.split(base_layer2)

    detail_layer1 = im1 - base_layer1
    d1_blue, d1_green, d1_red = cv2.split(detail_layer1)

    detail_layer2 = im2 - base_layer2
    d2_blue, d2_green, d2_red = cv2.split(detail_layer2)

    saliency1 = gaussian_filter(abs(laplace(im1_blue + im1_green + im1_red, mode='reflect')), sigma_r, mode='reflect')
    saliency2 = gaussian_filter(abs(laplace(im2_blue + im2_green + im2_red, mode='reflect')), sigma_r, mode='reflect')
    mask = np.float32(np.argmax([saliency1, saliency2], axis=0))

    im1 = np.float32(im1)
    im2 = np.float32(im2)

    g1r1 = guided_filter_gray_ameliorer(1 - mask, im1[:, :, 0], r_1, eps_1) + guided_filter_gray_ameliorer(1 - mask, im1[:, :, 1], r_1,
                                                                             eps_1) + guided_filter_gray_ameliorer(1 - mask,
                                                                                                    im1[:, :, 2], r_1,
                                                                                                   eps_1)

    g2r1 = 1-g1r1

    g1r2 = guided_filter_gray_ameliorer(1 - mask, im1[:, :, 0], r_2, eps_2) + guided_filter_gray_ameliorer(1 - mask, im1[:, :, 1], r_2,
                                                                             eps_2) + guided_filter_gray_ameliorer(1 - mask,
                                                                                                    im1[:, :, 2], r_2,
                                                                                                    eps_2)
    g2r2 = 1 - g1r2

    g1r1 = np.clip(g1r1, 0, 1)
    g2r1 = np.clip(g2r1, 0, 1)
    g1r2 = np.clip(g1r2, 0, 1)
    g2r2 = np.clip(g2r2, 0, 1)

    fused_base1 = np.float32((b1_blue * (g1r1) + b2_blue * (g2r1)) / ((g1r1 + g2r1)))
    fused_detail1 = np.float32((d1_blue * (g1r2) + d2_blue * (g2r2)) / ((g1r2 + g2r2)))
    fused_base2 = np.float32((b1_green * g1r1 + b2_green * g2r1) / ((g1r1 + g2r1)))
    fused_detail2 = np.float32((d1_green * (g1r2) + d2_green * (g2r2)) / ((g1r2 + g2r2)))
    fused_base3 = np.float32((b1_red * (g1r1) + b2_red * (g2r1)) / ((g1r1 + g2r1)))
    fused_detail3 = np.float32((d1_red * (g1r2) + d2_red * (g2r2)) / ((g1r2 + g2r2)))

    B1 = np.float32(fused_base1 + fused_detail1)
    B2 = np.float32(fused_base2 + fused_detail2)
    B3 = np.float32(fused_base3 + fused_detail3)

    fusion1 = np.float32(cv2.merge((B1, B2, B3)))
    fusion1 = np.clip(fusion1, 0, 1)
    fusion1 = (fusion1 * 255).astype(np.uint8)
    return fusion1

def main():
    im_path_0 = "child1.jpg"
    im_path_1 = "child2.jpg"

    image0 = Image.open(im_path_0)

    image1 = Image.open(im_path_1)

    image0 = np.array(image0)
    image1 = np.array(image1)

    image_result0 = fusion_color_ameliorer(image0,image1)
    image_result = Image.fromarray(image_result0)
    image_result.show()
    image_result.save(r"C:\Users\Lenovo\Documents\pythonProjects\IMA201\childFused_Color.jpg")


if __name__ == "__main__":
    main()




