import cv2
import torch
import numpy as np

device = 'cuda'  # 'cpu' or 'cuda'

test_image_path = './test/2.bmp'

lr = 0.001
iterations = 1000

# parameters settings
illu_factor = 1
reflect_factor = 1
noise_factor = 5000
reffac = 1
gamma = 0.4

# Gaussian Kernel Initialization
g_kernel_size = 5
g_padding = 2
sigma = 3
kx = cv2.getGaussianKernel(g_kernel_size,sigma)
ky = cv2.getGaussianKernel(g_kernel_size,sigma)
gaussian_kernel = np.multiply(kx,np.transpose(ky))
gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to(device)
