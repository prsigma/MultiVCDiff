# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP with checkpoint resume support.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import pandas as pd
import wandb  

from models import DiT_models, DiTMultimodal_models  
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from multimodal_dataset import H5ADDataset  

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                             Custom Dataset Class                              #
#################################################################################

class CustomCellDataset(Dataset):
    def __init__(self, csv_file_path, image_base_path, image_size, transform=None):
        self.metadata_df = pd.read_csv(csv_file_path)
        self.image_base_path = image_base_path
        
        self.unique_classes = self.metadata_df['compound'].unique()
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.unique_classes)}
        self.num_classes = len(self.unique_classes)
        
        self.metadata_df['label_idx'] = self.metadata_df['compound'].map(self.class_to_idx)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        else:
            self.transform = transform
                    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.metadata_df.iloc[idx]['merged_image']
        img_path = os.path.join(self.image_base_path, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")  
        except FileNotFoundError:
            print(f"WARNING: Image file not found: {img_path}. Skipping this sample.")
            if idx != 0:
                return self.__getitem__(0)
            else:
                raise FileNotFoundError(f"Image file not found: {img_path}, and no valid replacement sample could be found.")
        
        label = self.metadata_df.iloc[idx]['label_idx']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model with checkpoint resume support.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if rank == 0 and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.model}-bs{args.global_batch_size}-lr{args.lr}",
            config=args
        )

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        existing_experiments = sorted(glob(f"{args.results_dir}/*"))
        if args.resume_checkpoint:
            checkpoint_dir = os.path.dirname(args.resume_checkpoint)
            experiment_dir = os.path.dirname(checkpoint_dir)
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming from checkpoint: {args.resume_checkpoint}")
        else:
            experiment_index = len(existing_experiments)
            model_string_name = args.model.replace("/", "-")
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}" 
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Setup data:
    if args.use_multimodal:
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        dataset = H5ADDataset(
            h5ad_path=args.h5ad_path,
            image_dir=args.image_dir,
            transform=transform,
            normalize_rna=True,
        )
    elif args.use_custom_dataset:
        dataset = CustomCellDataset(
            csv_file_path=args.metadata_path,
            image_base_path=args.data_path,
            image_size=args.image_size
        )
        args.num_classes = dataset.num_classes
        logger.info(f"Setting number of classes to: {args.num_classes}")
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    if args.use_multimodal:
        assert args.model in DiTMultimodal_models, f"可用的DiTMultimodal模型: {list(DiTMultimodal_models.keys())}"
        model = DiTMultimodal_models[args.model](
            input_size=latent_size,
            in_channels=4,
            num_drug_classes=args.num_classes,
            num_rna_features=977,
            drug_fp_size=1024
        )
    else:
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        )
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    
    if args.resume_checkpoint:
        logger.info(f"Loading checkpoint from: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=lambda storage, loc: storage)
        
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        for p in ema.parameters():
            dist.broadcast(p.data, src=0)
    # Setup data:
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_img_loss = 0
    running_rna_loss = 0
    train_steps=0
    start_time = time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            if args.use_multimodal:
                x_img = batch['image'].to(device)
                x_rna = batch['rna'].to(device)
                drug_fp = batch['drug_embedding'].to(device)
                
                with torch.no_grad():
                    x_img_latent = vae.encode(x_img).latent_dist.sample().mul_(0.18215)
                
                t = torch.randint(0, diffusion.num_timesteps, (x_img.shape[0],), device=device)
                
                # Forward pass with both RNA and image data:
                model_kwargs = {
                    'drug_fp': drug_fp,
                    'rna': x_rna
                }
                
                # Calculate loss:
                loss_dict = diffusion.training_losses(
                    model=model, 
                    x_start=(x_img_latent, x_rna),
                    t=t, 
                    model_kwargs=model_kwargs
                )
                
                loss = loss_dict['loss']
                
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()

            if args.use_multimodal:
                running_img_loss += loss_dict['mse_img'].mean().item()
                running_img_loss += loss_dict['vb_img'].mean().item()
                running_rna_loss += loss_dict['mse_rna'].mean().item()
                running_rna_loss += loss_dict['vb_rna'].mean().item()
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                log_message = f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                
                if args.use_multimodal:
                    avg_img_loss = torch.tensor(running_img_loss / log_steps, device=device)
                    dist.all_reduce(avg_img_loss, op=dist.ReduceOp.SUM)
                    avg_img_loss = avg_img_loss.item() / dist.get_world_size()

                    avg_rna_loss = torch.tensor(running_rna_loss / log_steps, device=device)
                    dist.all_reduce(avg_rna_loss, op=dist.ReduceOp.SUM)
                    avg_rna_loss = avg_rna_loss.item() / dist.get_world_size()

                    log_message += f", Img Loss: {avg_img_loss:.4f}, RNA Loss: {avg_rna_loss:.4f}"
                
                logger.info(log_message)
                
                if rank == 0 and args.use_wandb:
                    wandb_log = {
                        "train_loss": avg_loss,
                        "steps_per_sec": steps_per_sec,
                        "step": train_steps,
                        "epoch": epoch,
                    }
                    
                    if args.use_multimodal:
                        wandb_log["img_loss"] = avg_img_loss
                        wandb_log["rna_loss"] = avg_rna_loss
                    
                    wandb.log(wandb_log)
                
                # Reset monitoring variables:
                running_loss = 0
                running_img_loss = 0
                running_rna_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if rank == 0 and args.use_wandb:
        wandb.finish()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=98)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--resume-checkpoint", type=str, default=None)

    parser.add_argument("--use-custom-dataset", action="store_true")
    parser.add_argument("--metadata-path", type=str, 
                        default="/data/pr/cellpainting/BBBC021/raw_data/metadata/augmented_image_metadata_512.csv")
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--use-multimodal", action="store_true")
    parser.add_argument("--h5ad-path", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--image-column", type=str, default="image_path")
    parser.add_argument("--drug-column", type=str, default="drug_name")
    parser.add_argument("--img-dose-column", type=str, default="image_dose")
    parser.add_argument("--smiles-column", type=str, default="smiles")
    parser.add_argument("--img-loss-weight", type=float, default=1.0)
    parser.add_argument("--rna-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="DiT-cell-images")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()
    main(args)