import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Load blurred image
img = cv2.imread("blurred.jpg", 0)

# Function to create Gaussian Kernel
def gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# Wiener Deconvolution
def wiener_filter(img, kernel, K=0.01):
    kernel /= np.sum(kernel)

    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)

    kernel_fft = np.fft.fft2(kernel, s=img.shape)

    kernel_fft_conj = np.conj(kernel_fft)

    result = kernel_fft_conj / (kernel_fft * kernel_fft_conj + K) * dummy

    result = np.abs(np.fft.ifft2(result))

    return result

# Create blur kernel
kernel = gaussian_kernel(15, 3)

# Deblur image
restored = wiener_filter(img, kernel)

# Display Results
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Blurred Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(restored, cmap='gray')
plt.title("Restored Image")
plt.axis('off')

plt.show()

# Save output
cv2.imwrite("restored_image.jpg", restored)
print("Deblurred image saved as restored_image.jpg")
