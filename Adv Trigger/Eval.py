import torch
import open_clip
from utils.load_data import load_dataset
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch import nn, optim
import numpy as np
from PIL import Image
from tqdm import tqdm  # 进度条显示
from torchvision import transforms
import pandas as pd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize



def mask_generation_random(patch):
    image_size = (3, 224, 224)
    applied_patch = np.zeros(image_size)
    x_max = image_size[1] - patch.shape[1]
    y_max = image_size[2] - patch.shape[2]
    x_location = np.random.randint(0, max(1, x_max + 1))
    y_location = np.random.randint(0, max(1, y_max + 1))
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch, x_location, y_location

def mask_generation(patch):
    image_size = (3, 224, 224)
    applied_patch = np.zeros(image_size)
    x_location = image_size[1] - 14 - patch.shape[1]
    y_location = image_size[1] - 14 - patch.shape[2]
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch ,x_location, y_location

def patch_initialization(patch_type='rectangle'):
    noise_percentage = 0.03
    image_size = (3, 224, 224)
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

import torch.nn.functional as F

def cal_sim(vector_0, vector_1):
    '''
    Calculate the cos sim and pairwise distance
    :param vector_0:
    :param vector_1:
    :return: cos_sim, pair_dis
    '''
    cos_sim_f = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    pair_dis_f = torch.nn.PairwiseDistance(p=2)
    cos_sim = cos_sim_f(vector_0, vector_1)
    pair_dis = pair_dis_f(vector_0, vector_1)
    return cos_sim, pair_dis

def evaluate_Verify(model, Trigger_module, test_loader, Size_Trigger, device):
    
    for i, (batch_images, batch_texts, inds, IDs) in enumerate(test_loader):

        batch_images = batch_images.squeeze().to(device)
        batch_texts = batch_texts.squeeze().to(device)
        # store the index of image for each text
        target = inds.to(device)
        image_adv = torch.mul(mask.type(torch.FloatTensor), uap.type(torch.FloatTensor)) + \
            torch.mul(1 - mask.expand(batch_images.shape).type(torch.FloatTensor), batch_images.type(torch.FloatTensor))
        p_data = image_adv.clone()
        # compute the embedding of images and texts
        with torch.no_grad():
            image_features = model.encode_image(batch_images)
            image_features_T = model.encode_image(p_data.to(device))
            text_features = model.encode_text(batch_texts)
            origin_image_features = image_features
            T_image_features = image_features_T
            origin_text_features = text_features
            
            origin_image_features /= origin_image_features.norm(dim=-1, keepdim=True)
           
            T_image_features /= T_image_features.norm(dim=-1, keepdim=True)
            
            origin_text_features /= origin_text_features.norm(dim=-1, keepdim=True)

            origin_cos_sim, origin_pair_dis = cal_sim(origin_image_features, origin_text_features)
            Trigger_cos_sim, Trigger_pair_dis = cal_sim(T_image_features, origin_text_features)
            

            similarity_1 = 100. * (origin_image_features @ origin_text_features.T)
            p_similarity_1 = 100. * (T_image_features @ origin_text_features.T)

            probs_1 = F.softmax(similarity_1, dim=-1).max(-1)[1]
            p_probs_1 = F.softmax(p_similarity_1, dim=-1).max(-1)[1]

            image_features = Trigger_module(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # Trigger
            image_features_T = Trigger_module(image_features_T)
            image_features_T /= image_features_T.norm(dim=-1, keepdim=True)

            text_features = Trigger_module(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)


            similarity = 100. * (image_features @ text_features.T)
            p_similarity = 100. * (image_features_T @ text_features.T)


            probs = F.softmax(similarity, dim=-1).max(-1)[1]
            p_probs = F.softmax(p_similarity, dim=-1).max(-1)[1]


            cos_sim_origin, pair_dis_origin = cal_sim(image_features, text_features)
            cos_sim_Trigger, pair_dis_Trigger = cal_sim(image_features_T, text_features)



if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # load pre-trainede CLIP model
    victim ='ViT-B-16-quickgelu'
    pretrained = "openai"
    model, _, transform = open_clip.create_model_and_transforms(victim, pretrained=pretrained)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(victim)
    model.eval() 

    Size_Trigger = 16

    # load cross-modal dataset
    dataset ='pascal'
    batch_size = 16
    dataloaders = load_dataset(dataset, batch_size)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    module = Module().train().to(device)
    path = "output/Module/module_pascal_ViT-B-16-quickgelu_16_100.pth"
    module.load_state_dict(torch.load(path, map_location=device))
    module.eval()

    uap_path = "YOUR UAP PATH"
    uap = torch.load(uap_path)
    
    patch = patch_initialization()
    mask, applied_patch, x, y = mask_generation(patch)
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)

    evaluate_Verify(model,module,test_loader,Size_Trigger,device)

