import torch
import clip
from torchvision import transforms
import numpy as np
import os
import open_clip
from PIL import Image
from utils.load_data import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset



if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    victim ='ViT-B-16-quickgelu'
    pretrained = "openai"
    model, _, transform = open_clip.create_model_and_transforms(victim, pretrained=pretrained)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(victim)
    model.eval() 

    # load cross-modal dataset
    dataset ='wikipedia' # [wikipedia, pascal]
    batch_size = 16
    dataloaders = load_dataset(dataset, batch_size)
    test_loader = dataloaders['test']

    Size_Trigger = 16 #[16,32,128,256,512]

    text = 'Baisc Text'
    text_descriptions = [text] * Size_Trigger
    text_tokens = clip.tokenize(text_descriptions).to(device)
    image_dir = f'noisy_images_fixed_Num_{Size_Trigger}_{dataset}' 
    image_embeddings = []
    with torch.no_grad():
        image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('jpg', 'jpeg', 'png'))]
        print(image_dir)
        for image_file in image_files:
            image = Image.open(image_file)
            image = transform(image).unsqueeze(0).to(device)
            image_embedding = model.encode_image(image)
            image_embeddings.append(image_embedding)
        image_embeddings = torch.cat(image_embeddings, dim=0)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    print(f' Image Embeddings Shape: {image_embeddings.shape}')  
    print(f' Text Embeddings Shape: {text_embeddings.shape}')    


    beta = 0.05
    output_dir = "YOURPATH"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Trigger_mat_pth = os.path.join(output_dir, f'trigger_Module_{dataset}_{Size_Trigger}_1000.pth')
    dataset1 = TensorDataset(image_embeddings, text_embeddings)
    data_loader = DataLoader(dataset1, batch_size=batch_size, shuffle=True)

    # Randomly initialize the linear transformation module
    d = 512 
    W_align = torch.randn(d, d, requires_grad=True, device=device)
    torch.nn.init.orthogonal_(W_align)

    # Define the optimizer
    optimizer = optim.Adam([W_align], lr=1e-3, eps=1e-5)

    # Training loop
    num_epochs = 1000 # Set number of epochs as needed
    for epoch in range(num_epochs):
        for batch_image_embeddings, batch_text_embeddings in data_loader:
            optimizer.zero_grad()

            # Calculate the loss function as ||W_align * Y - W_align.T * T||_F
            batch_image_embeddings = batch_image_embeddings.to(dtype=torch.float32)
            batch_text_embeddings = batch_text_embeddings.to(dtype=torch.float32)
            W_align = W_align.to(dtype=torch.float32)
            loss = torch.norm(W_align @ batch_image_embeddings.T - W_align @ batch_text_embeddings.T, p='fro')

            # Backpropagate the loss
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(W_align, max_norm=0.5)
            torch.nn.utils.clip_grad_value_(W_align, clip_value=0.5)
            optimizer.step()
                
            #if parser['method'] != 'random':
            # Manually update W_align with the orthogonal constraint
            with torch.no_grad():
                W_align -= beta * (W_align @ W_align.T @ W_align - W_align)

        if epoch % 10 == 0:  # Print loss every 10 epochs
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    torch.save(W_align, f'/root/autodl-tmp/AdvCLIP/results/w_matrix/trigger_Matrix_{dataset}_{Size_Trigger}_1000.pth')    
    print("Training complete.")