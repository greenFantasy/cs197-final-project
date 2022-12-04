import os
import argparse
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim

from train import load_data, load_clip, preprocess_text, DefaultBiovisionConfig

print("Imported from train.py")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt-imp")
    parser.add_argument('--use_huggingface_bert', action='store_true')
    parser.add_argument('--image_tower_type', type=str, required=True)
    parser.add_argument('--huggingface_bert_key', type=str, action='store', default='cxr')
    parser.add_argument('--lock_text', action='store_true')
    parser.add_argument('--lock_vision', action='store_true')
    parser.add_argument('--img_path_list', type=str, default='data/cxr_paths.csv', help="File containing paths to all chest x-ray images in dataset.")
    args = parser.parse_args()
    return args

# @hydra.main(version_base=None, config_path="configs", config_name="defaults.yaml")
def model_pipeline(config): #, verbose=0): 
    
    print(config, flush=True)
    # config = config.locking
    
    if config.seed != 0:
        torch.manual_seed(config.seed)
    # make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer = make(config)
    
    print("Training beginning!")

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
    config.biovision = DefaultBiovisionConfig()
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size, 
                                    pretrained=pretrained, column="impression", biovision_config=config.biovision)
    model = load_clip(config.image_tower_type, model_path=None, pretrained=pretrained, context_length=config.context_length, 
                      use_huggingface_bert=config.use_huggingface_bert, huggingface_bert_key=config.huggingface_bert_key)
    model.to(device)
    print('Model on Device.')

    # establish the parameters to train based on what is locked
    params_list = []
    params_key = 'params'
    # this should always exist now with modifications to bert-based text stacks
    params_list.append(model.text_projection)
    if not config.lock_text:
        params_list.extend(model.transformer.parameters())
        if not config.use_huggingface_bert:
            params_list.extend(model.token_embedding.parameters())
            params_list.extend(model.ln_final.parameters())
            params_list.append(model.positional_embedding)
    # if locked, only add projection; otherwise add everything
    if config.lock_vision:
        params_list.extend(model.visual.projector.parameters())
    else:
        params_list.extend(model.visual.parameters())
    # params_list = [{params_key: param} for param in params_list]

    # turn off gradient computation for frozen weights
    if config.lock_text:
        for param in model.transformer.parameters():
            param.requires_grad = False
        if not config.use_huggingface_bert:
            for param in list(model.token_embedding.parameters()) + list(model.ln_final.parameters()):
                param.requires_grad = False
            model.positional_embedding.requires_grad = False

    if config.lock_vision:
        for param in model.visual.encoder.parameters():
            param.requires_grad = False

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(params_list, lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(params_list, lr=config.lr, momentum=config.momentum)
    return model, data_loader, device, criterion, optimizer

def train(model, loader, device, criterion, optimizer, config): 
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
    
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
                model_path = os.path.join(model_save_dir, "checkpoint_{batch_ct}.pt".format(
                    batch_ct=str(batch_ct), 
                ))
                print("Saved checkpoint to: ", model_path)
                save(model, model_path)
                
def train_batch(images, texts, model, device, criterion, optimizer):
    images, texts = images.to(device), texts.to(device)
    
    # Forward pass 
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
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}", flush=True)
    
def save(model, path): 
    torch.save(model.state_dict(), path)
    
if __name__ == "__main__":
    args = parse_args()
    print(args)
    model = model_pipeline(args)
    

