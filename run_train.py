import os
import pprint
import argparse
import hydra
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, load_clip, preprocess_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt-imp")
    parser.add_argument('--use_cxrbert', action='store_true')
    parser.add_argument('--use_biovision', action='store_true')
    parser.add_argument('--lock_text', action='store_true')
    parser.add_argument('--lock_vision', action='store_true')
    parser.add_argument('--img_path_list', type=str, default='data/cxr_paths.csv', help="File containing paths to all chest x-ray images in dataset.")
    args = parser.parse_args()
    return args

@hydra.main(version_base=None, config_path="configs", config_name="defaults.yaml")
def model_pipeline(config): #, verbose=0): 
    
    print(config)
    config = config.locking
    
    torch.manual_seed(config.seed)
    # make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer = make(config)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config)

    # save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    # if verbose: 
    #    print(model)
    return model

def make(config): 
    pretrained = not config.random_init
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression", biovision_config=config.biovision)
    model = load_clip(model_path=None, pretrained=pretrained, context_length=config.context_length, use_cxrbert=config.use_cxrbert, use_biovision=config.biovision.use_biovision)
    model.to(device)
    print('Model on Device.')

    # establish the parameters to train based on what is locked
    params_list = []
    params_key = 'params'
    if config.use_cxrbert and not config.biovision.use_biovision:
        params_list.append(model.vision_projection)
    if not config.use_cxrbert:
        params_list.append(model.text_projection)
    if not config.lock_text:
        params_list.append(model.transformer.parameters())
        if not config.use_cxrbert:
            params_list.append(model.token_embedding.parameters())
            params_list.append(model.positional_embedding)
    if not config.lock_vision:
        params_list.append(model.visual.parameters())
    params_list = [{params_key: param} for param in params_list]

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(params_list, lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(params_list, lr=config.lr, momentum=config.momentum)

    # turn off gradient computation for frozen weights
    if config.lock_text:
        for param in model.transformer.parameters():
            param.requires_grad = False
        if not config.use_cxrbert:
            for param in model.token_embedding.parameters():
                param.requires_grad = False
            model.positional_embedding.requires_grad = False
    if config.lock_vision:
        for param in model.visual.parameters():
            param.requires_grad = False
    return model, data_loader, device, criterion, optimizer

def train(model, loader, device, criterion, optimizer, config): 
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
        
    # login to wandb and initialize it
    wandb.login()
    wandb.init(project="cs197-final-project", entity="team_rack", config=OmegaConf.to_container(config, resolve=True))
    
    # save initial model
    init_checkpoint_name = "checkpoint_init.pt"
    init_model_path = os.path.join(model_save_dir, init_checkpoint_name)
    print("Saved initial model to: ", init_model_path)
    save(model, init_model_path, init_checkpoint_name)
    
    # Run training
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    highest_val_auc = 0 # save highest mean auc
    
    for epoch in range(config.epochs):
        running_loss = 0.0 # running loss over batch
        for data in tqdm(loader):
            # get the images
            images = data['img']

            texts = data['txt']
            texts = preprocess_text(texts, model) 
            
            # perform step for a single batch
            loss = train_batch(images, texts, model, device, criterion, optimizer)
            example_ct +=  len(images)
            batch_ct += 1
            running_loss += loss.item()

            # Report metrics every `report_freq` batch
            if (batch_ct % report_freq) == 0:
                train_log(running_loss / report_freq, example_ct, epoch)
                running_loss = 0.0
            
            if (batch_ct % config.save_interval) == 0:
                checkpoint_name =  "checkpoint_{batch_ct}.pt".format(batch_ct=str(batch_ct))
                model_path = os.path.join(model_save_dir, checkpoint_name)
                print("Saved checkpoint to: ", model_path)
                # save checkpoint
                save(model, model_path, checkpoint_name)
                
    wandb.finish()
                
def train_batch(images, texts, model, device, criterion, optimizer):
    images, texts = images.to(device), texts.to(device)
    
    # Forward pass ➡
    logits_per_image, logits_per_text = model(images, texts)
    
    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt)/2 # avg. img and txt loss

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    optimizer.step()
        
    return loss

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    wandb.log({'loss': loss, 'example_ct': example_ct, 'epoch': epoch})
    
def save(model, path, checkpoint_name): 
    # save checkpoint locally
    torch.save(model.state_dict(), path)
    # save checkpoint to wandb
    artifact = wandb.Artifact(name=checkpoint_name, type='model_checkpoint')
    artifact.add_file(local_path=path)
    wandb.run.log_artifact(artifact)
    
if __name__ == "__main__":
    # args = parse_args()
    # print(args)
    model = model_pipeline()
    

