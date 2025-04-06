import torch
import open_clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import stats

def add_gaussian_noise(image, sigma):
    image_np = np.array(image, dtype=np.float64)
    noisy_image_np = image_np + sigma * np.random.randn(*image_np.shape)
    noisy_image_np = np.clip(noisy_image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image_np)

def add_poisson_noise(image, scale_factor=5.0):
    image_np = np.array(image, dtype=np.float64)
    scaled_image = image_np * scale_factor
    noisy_scaled_image = stats.poisson(scaled_image).rvs()
    noisy_image_np = (noisy_scaled_image / scale_factor).clip(0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image_np)

def add_salt_and_pepper_noise(image, prob=0.05):
    image_np = np.array(image)
    num_salt = np.ceil(prob * image_np.size * 0.5).astype(int)
    num_pepper = np.ceil(prob * image_np.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_np.shape]
    if len(image_np.shape) == 3:  
        image_np[coords[0], coords[1], :] = [255, 255, 255]
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_np.shape]
    if len(image_np.shape) == 3:
        image_np[coords[0], coords[1], :] = [0, 0, 0]
    return Image.fromarray(image_np)

# LNA
def add_local_gaussian_noise(image, mask, mean=0, sigma=10):
    img_array = np.array(image).astype(np.float32)
    gauss = np.random.normal(mean, sigma, img_array.shape).astype(np.float32)
    noisy_img_array = np.where(mask[..., None], img_array + gauss, img_array)
    noisy_img_array = np.clip(noisy_img_array, 0, 255) 
    return Image.fromarray(noisy_img_array.astype(np.uint8))


def add_local_poisson_noise(image, rect=(50, 50, 150, 150), scale_factor=5.0):
    image_np = np.array(image, dtype=np.float64)
    noisy_image_np = image_np.copy()
    rect_slice = slice(rect[1], rect[3]), slice(rect[0], rect[2])
    rect_area = image_np[rect_slice].astype(np.uint8)
    noisy_rect_area = add_poisson_noise(Image.fromarray(rect_area), scale_factor)
    noisy_image_np[rect_slice] = np.array(noisy_rect_area, dtype=np.float64)
    return Image.fromarray(noisy_image_np.astype(np.uint8))

def add_local_salt_and_pepper_noise(image, rect=(50, 50, 150, 150), prob=0.05):
    image_np = np.array(image)
    noisy_image_np = image_np.copy()
    rect_slice = slice(rect[1], rect[3]), slice(rect[0], rect[2])
    rect_area = image_np[rect_slice]
    noisy_rect_area = add_salt_and_pepper_noise(Image.fromarray(rect_area), prob)
    noisy_image_np[rect_slice] = np.array(noisy_rect_area)
    return Image.fromarray(noisy_image_np)


def create_random_mask(shape, num_regions=5, region_size=(50, 50)):
    mask = np.zeros(shape, dtype=np.uint8)
    for _ in range(num_regions):
        x = np.random.randint(0, shape[1] - region_size[1])
        y = np.random.randint(0, shape[0] - region_size[0])
        mask[y:y+region_size[0], x:x+region_size[1]] = 1
    return mask

# BON
def blend_images(original, noisy, alpha=0.5):
    original_array = np.array(original).astype(np.float32)
    noisy_array = np.array(noisy).astype(np.float32)
    blended_array = (1 - alpha) * original_array + alpha * noisy_array
    blended_array = np.clip(blended_array, 0, 255)
    return Image.fromarray(blended_array.astype(np.uint8))

# SPN
def generate_spatially_varying_noise(image, center, max_sigma):
    img_array = np.array(image).astype(np.float32)
    h, w = img_array.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    normalized_distance = 1 - (distance_from_center / np.max(distance_from_center))
    sigma_values = normalized_distance * max_sigma
    gauss = np.zeros_like(img_array)
    for c in range(img_array.shape[-1]):
        gauss[..., c] = np.random.normal(0, sigma_values)

    noisy_img_array = img_array + gauss.astype(np.float32)
    noisy_img_array = np.clip(noisy_img_array, 0, 255)
    
    return Image.fromarray(noisy_img_array.astype(np.uint8))

def add_spatially_varying_poisson_noise(image, base_scale=5.0, variation=0.5):
    image_np = np.array(image, dtype=np.float64)
    h, w = image_np.shape[:2]
    scale_factors = base_scale + variation * (np.random.rand(h, w) - 0.5)
    scale_factors = np.repeat(scale_factors[:, :, np.newaxis], 3, axis=2)  
    noisy_image_np = stats.poisson(image_np * scale_factors).rvs() / scale_factors
    return Image.fromarray(noisy_image_np.clip(0, 255).astype(np.uint8))

def add_spatially_varying_salt_and_pepper_noise(image, base_prob=0.05, variation=0.02):
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    probs = base_prob + variation * (np.random.rand(h, w) - 0.5)
    for i in range(h):
        for j in range(w):
            rdn = np.random.random()
            if rdn < probs[i, j]:
                if np.random.random() < 0.5:
                    image_np[i, j, :] = [255, 255, 255] if len(image_np.shape) == 3 else 255
                else:
                    image_np[i, j, :] = [0, 0, 0] if len(image_np.shape) == 3 else 0
    return Image.fromarray(image_np.astype(np.uint8))

# CANA
def add_content_aware_noise(image, noise_function, gradient_threshold=50, noise_factor=1.0):
    img_gray = image.convert('L')
    img_array = np.array(img_gray).astype(np.float32)
    gradient_x = np.abs(gaussian_filter(img_array, sigma=1, order=[1, 0]))
    gradient_y = np.abs(gaussian_filter(img_array, sigma=1, order=[0, 1]))
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    weights = np.ones_like(img_array) * noise_factor
    weights[gradient_magnitude > gradient_threshold] *= 0.1  

    noisy_img_array = np.array(image).astype(np.float32)
    for channel in range(noisy_img_array.shape[-1]):
        noisy_channel = noise_function(noisy_img_array[..., channel], weights)
        noisy_img_array[..., channel] = noisy_channel
    noisy_img_array = np.clip(noisy_img_array, 0, 255)

    return Image.fromarray(noisy_img_array.astype(np.uint8))

def apply_gaussian_noise(channel, weights):
    gauss = np.random.normal(0, 25, channel.shape).astype(np.float32)
    noisy_channel = channel + gauss * weights
    return noisy_channel


def add_content_based_poisson_noise(image, brightness_threshold=128, high_scale=5.0, low_scale=0.5):
    image_np = np.array(image, dtype=np.float64)
    scale_factors = np.where(image_np.mean(axis=2) > brightness_threshold, high_scale, low_scale)
    scale_factors = np.repeat(scale_factors[:, :, np.newaxis], 3, axis=2) 
    noisy_image_np = stats.poisson(image_np * scale_factors).rvs() / scale_factors
    return Image.fromarray(noisy_image_np.clip(0, 255).astype(np.uint8))


def add_content_based_salt_and_pepper_noise(image, brightness_threshold=128, high_prob=0.1, low_prob=0.01):
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    probs = np.where(image_np.mean(axis=2) > brightness_threshold, high_prob, low_prob)
    for i in range(h):
        for j in range(w):
            if np.random.random() < probs[i, j]:
                if np.random.random() < 0.5:
                    image_np[i, j, :] = [255, 255, 255] if len(image_np.shape) == 3 else 255
                else:
                    image_np[i, j, :] = [0, 0, 0] if len(image_np.shape) == 3 else 0
    return Image.fromarray(image_np.astype(np.uint8))


### Metrics
# RMSE
def calculate_rmse(original, noisy):
    original_np = np.array(original, dtype=np.float64)
    noisy_np = np.array(noisy, dtype=np.float64)
    mse = np.mean((original_np - noisy_np) ** 2)
    return np.sqrt(mse)

# PSNR
def calculate_psnr(original, noisy, max_value=255):
    rmse = calculate_rmse(original, noisy)
    if rmse == 0:
        return float('inf')
    return 20 * np.log10(max_value / rmse)

# SSIM
def calculate_simplified_ssim(original, noisy, K1=0.01, K2=0.03, L=255):
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu_x = original.mean()
    mu_y = noisy.mean()
    sigma_x = original.std()
    sigma_y = noisy.std()
    sigma_xy = ((original - mu_x)*(noisy - mu_y)).mean()
    ssim = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x**2 + sigma_y**2 + C2))
    return ssim

# UQI
def calculate_uqi_channel(original, noisy, window_size=8):
    mean_original = original.mean()
    mean_noisy = noisy.mean()
    var_original = original.var()
    var_noisy = noisy.var()
    covar = ((original - mean_original) * (noisy - mean_noisy)).mean()
    
    numerator = 4 * covar * mean_original * mean_noisy
    denominator = (var_original + var_noisy) * (mean_original**2 + mean_noisy**2)
    
    if denominator == 0:
        return 0  # Avoid division by zero
    
    uqi = numerator / denominator
    return uqi

def calculate_metrics(original, noisy):
    original_np = np.array(original.convert('RGB'))
    noisy_np = np.array(noisy.convert('RGB'))
    rmse = calculate_rmse(original, noisy)
    psnr = calculate_psnr(original, noisy)
    ssim_avg = np.mean([calculate_simplified_ssim(original_np[:, :, i], noisy_np[:, :, i]) for i in range(3)])
    uqi_avg = np.mean([calculate_uqi_channel(original_np[:, :, i], noisy_np[:, :, i]) for i in range(3)])
    return rmse, psnr, ssim_avg, uqi_avg



if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    # load pre-trainede CLIP model
    victim ='ViT-B-16-quickgelu'
    pretrained = "openai"
    model, _, transform = open_clip.create_model_and_transforms(victim, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(victim)
    model.eval().to(device)

    # Basic Trigger
    text="text from dataset"
    image_path = '../output/image_basic.jpg'
    original_image = Image.open(image_path)

    ## gaussian
    sigma_values = [25] 
    mask = create_random_mask(original_image.size[::-1], num_regions=5, region_size=(50, 50))
    center_point = (original_image.width // 2, original_image.height // 2)  
    methods = {
        'Global': lambda img, s: add_gaussian_noise(img, s),
        'Local': lambda img, s: add_local_gaussian_noise(img, mask, 0, s),  
        'Mixed': lambda img, s: blend_images(img, add_gaussian_noise(img, s), alpha=0.3),
        'Spatial_Variant':lambda img, s: generate_spatially_varying_noise(img, center=center_point, max_sigma=25),
        'content_Aware':lambda img, s: add_content_aware_noise(img, apply_gaussian_noise, gradient_threshold=50, noise_factor=1.0)
    }

    ## poisson
    scale_factor = [5.0]
    methods = {
        'Global': lambda img, s: add_poisson_noise(img, s),
        'Local': lambda img,s: add_local_poisson_noise(img),
        'Mixed': lambda img, s: blend_images(img,add_poisson_noise(img, s),alpha=0.5),
        'Spatially Varying': lambda img,s: add_spatially_varying_poisson_noise(img),
        'Content-Based': lambda img,s: add_content_based_poisson_noise(img),
    }

    ## salt_and_pepper
    scale_factors = [0.05]  
    methods = {
        'Global': lambda img: add_salt_and_pepper_noise(img),
        'Local': lambda img: add_local_salt_and_pepper_noise(img),
        'Mixed': lambda img: blend_images(img,add_salt_and_pepper_noise(img)),
        'Spatially Varying': lambda img: add_spatially_varying_salt_and_pepper_noise(img),
        'Content-Based': lambda img: add_content_based_salt_and_pepper_noise(img),
    }
