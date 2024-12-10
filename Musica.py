import logging
import numpy as np
import copy
from skimage.transform import pyramid_reduce, pyramid_expand
import cv2
#import sys

# Setup logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('musica_py.log')
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_format = logging.Formatter('%(levelname)s:\t%(message)s')
stream_handler.setFormatter(stream_format)
logger.addHandler(stream_handler)

def isPowerofTwo(x):
    return x and (not(x & (x - 1)))

def findNextPowerOf2(n):
    n = n - 1
    while n & n - 1:
        n = n & n - 1
    return n << 1

def resize_image(img):
    row, col = img.shape
    logger.debug("Calculating how much padding is required...")
    if isPowerofTwo(row):
        rowdiff = 0
    else:
        nextpower = findNextPowerOf2(row)
        rowdiff = nextpower - row

    if isPowerofTwo(col):
        coldiff = 0
    else:
        nextpower = findNextPowerOf2(col)
        coldiff = nextpower - col

    img_ = np.pad(img, ((0, rowdiff), (0, coldiff)), 'reflect')
    logger.info('Image padded from [{},{}] to [{},{}]'.format(img.shape[0], img.shape[1], img_.shape[0], img_.shape[1]))
    return img_

def gaussian_pyramid(img, L):
    logger.debug('Creating Gaussian pyramid...')
    tmp = copy.deepcopy(img)
    gp = [tmp]
    for layer in range(L):
        logger.debug('Creating Layer %d...' % (layer+1))
        tmp = pyramid_reduce(tmp, preserve_range=True)
        gp.append(tmp)
    logger.info('Finished creating Gaussian Pyramid')
    return gp

def laplacian_pyramid(img, L):
    gauss = gaussian_pyramid(img, L)
    logger.debug('Creating Laplacian pyramid...')
    lp = []
    for layer in range(L):
        logger.debug('Creating layer %d' % (layer))
        tmp = pyramid_expand(gauss[layer+1], preserve_range=True)
        tmp = gauss[layer] - tmp
        lp.append(tmp)
    lp.append(gauss[L])
    logger.info("Finished creating Laplacian pyramid")
    return lp, gauss

def enhance_coefficients(laplacian, L, params):
    logger.debug('Non-linear transformation of coefficients...')
    M = params['M']
    p = params['p']
    a = params['a']
    for layer in range(L):
        logger.info('Modifying Layer %d' % (layer))
        x = laplacian[layer]
        x[x < 0] = 0.0
        G = a[layer]*M
        laplacian[layer] = G * np.multiply(
            np.divide(x, np.abs(x), out=np.zeros_like(x), where=x != 0),
            np.power(np.divide(np.abs(x), M), p))
    return laplacian

def reconstruct_image(laplacian, L):
    logger.debug('Reconstructing image...')
    rs = laplacian[L]
    for i in range(L-1, -1, -1):
        rs = pyramid_expand(rs, preserve_range=True)
        rs = np.add(rs, laplacian[i])
        logger.debug('Layer %d completed' % (i))
    logger.info('Finished reconstructing image from modified pyramid')
    return rs

def musica(img, L, params):
    img_resized = resize_image(img)
    lp, _ = laplacian_pyramid(img_resized, L)
    lp = enhance_coefficients(lp, L, params)
    rs = reconstruct_image(lp, L)
    rs = rs[:img.shape[0], :img.shape[1]]
    return rs

def denoise_image(image, h=10):
    # Normalize and convert to uint8
    image_8u = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    denoised_8u = cv2.fastNlMeansDenoising(image_8u, None, h, 7, 21)
    
    # Convert back to the original range and type
    denoised = cv2.normalize(denoised_8u, None, image.min(), image.max(), cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return denoised

def main():
    # Hardcoded image path
    image_path = "C:\\Users\\Dhvani\\Downloads\\ZULF-964_TPW-3 8-A92-CA-550434_BZ74-R1 (0-30)REJ CP30 1.png" # Replace with the path to your image

    # Read the input image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Parameters for MUSICA
    L = 4  # Max level of decomposition
    params = {
        'M': 255,
        'p': 0.5,
        'a': [1, 1, 1, 1]  # Adjust as needed
    }

    logger.info("Running MUSICA algorithm...")
    enhanced_img = musica(img, L, params)

    # Denoise the enhanced image
    logger.info("Applying noise reduction...")
    denoised_img = denoise_image(enhanced_img, h=15)

    # Save the denoised enhanced image
    output_path = "C:\\Users\\Dhvani\\Downloads\\bit.png"
    cv2.imwrite(output_path, np.clip(denoised_img, 0, 255).astype(np.uint8))
    logger.info(f"Enhanced and denoised image saved to {output_path}")

if __name__ == "__main__":
    main()
