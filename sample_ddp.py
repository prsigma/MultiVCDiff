import torch
import torch.distributed as dist
from models import DiTMultimodal_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm
import os
import json
import math
import argparse
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
import scanpy as sc
import pandas as pd
rdBase.DisableLog('rdApp.warning')

def Drug_encoder(drug_SMILES_list: list, num_Bits=1024, comb_num=1):
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits), dtype=np.float32)
    
    if comb_num == 1:
        for i, smiles in enumerate(drug_SMILES_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  
            fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
            fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
            fcfp4_array[i] = fcfp4_list
    else:
        for i, smiles in enumerate(drug_SMILES_list):
            for smi in smiles.split('+'):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue  
                fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_array[i] += fcfp4_list
    return fcfp4_array

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU"
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()  
    device = local_rank
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting global_rank={rank}/{world_size}, local_rank={local_rank}, seed={seed}")

    adata = sc.read_h5ad(args.h5ad_path)
    print(f"Loaded h5ad data with {adata.n_obs} cells and {adata.n_vars} genes")
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    rna_min = X.min(axis=0)
    rna_max = X.max(axis=0)
    rna_range = rna_max - rna_min
    rna_range[rna_range == 0] = 1.0  

    drug_names = sorted(adata.obs['compound'].unique().tolist())
    num_classes = len(drug_names)
    
    print(f"Found {num_classes} unique drugs in h5ad data")
    
    # 计算药物分割点（两台机器）
    if args.drug_split not in [0, 1]:
        raise ValueError("drug_split must be 0 or 1, indicating whether to process the first half or the second half of the drugs.")
    
    total_drugs = len(drug_names)
    split_point = total_drugs // 2
    
    if args.drug_split == 0:
        machine_drugs = drug_names[:split_point]
    else:
        machine_drugs = drug_names[split_point:]
    
    num_gpus = torch.cuda.device_count()
    drugs_per_gpu = (len(machine_drugs) + num_gpus - 1) // num_gpus
    gpu_start = local_rank * drugs_per_gpu
    gpu_end = min(gpu_start + drugs_per_gpu, len(machine_drugs))
    assigned_drugs = machine_drugs[gpu_start:gpu_end]
    
    latent_size = args.image_size // 8
    model = DiTMultimodal_models[args.model](
        input_size=latent_size,
        in_channels=4,
        num_drug_classes=args.num_drug_classes,
        num_rna_features=args.num_rna_features,
        drug_fp_size=args.fp_size
    ).to(device)
    ckpt_path = args.ckpt
    state_dict = torch.load(ckpt_path, map_location=f"cuda:{device}")
    model.load_state_dict(state_dict["model"])
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    drug_to_smiles = {}
    for _, row in adata.obs.iterrows():
        drug_name = row['compound']
        smiles = row['smiles']
        drug_to_smiles[drug_name] = smiles
    
    for drug_name in assigned_drugs:
        drug_idx = drug_names.index(drug_name)  
        
        if drug_name not in drug_to_smiles or pd.isna(drug_to_smiles[drug_name]):
            print(f"global_rank={rank}, local_rank={local_rank}: Drug {drug_name} has no valid SMILES, skipping")
            continue
            
        drug_smiles = [drug_to_smiles[drug_name]]
        drug_embedding = Drug_encoder(drug_smiles, num_Bits=args.fp_size)
        drug_embedding = torch.tensor(drug_embedding, dtype=torch.float32).to(device)
        
        model_name = args.model.replace("/", "-")
        ckpt_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
        folder = f"{model_name}-syn-{ckpt_name}-size{args.image_size}-vae{args.vae}-seed{args.global_seed}"
        output_dir = f"{args.sample_dir}/{folder}/{drug_name}"
        rna_dir = f"{output_dir}/rna"
        os.makedirs(rna_dir, exist_ok=True)
        
        if os.path.exists(output_dir):
            existing_images = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            if len(existing_images) >= args.num_fid_samples:
                print(f"global_rank={rank}, local_rank={local_rank}: Skip {drug_name} (已存在 {len(existing_images)} 张图片)")
                continue 
        
        print(f"global_rank={rank}, local_rank={local_rank}: Generating {args.num_fid_samples} images and RNA for {drug_name} (idx: {drug_idx})")
        os.makedirs(output_dir, exist_ok=True)
        print(f"global_rank={rank}, local_rank={local_rank}: Output directory: {output_dir}")

        n = args.per_proc_batch_size
        total_samples = args.num_fid_samples
        iterations = (total_samples + n - 1) // n  
        generated = 0

        for i in tqdm(range(iterations), desc=f"global_rank={rank}, local_rank={local_rank} {drug_name}"):
            batch_size = min(n, total_samples - generated)
            if batch_size == 0:
                break
                
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            rna = torch.randn(batch_size, args.num_rna_features, device=device)
            
            model_kwargs = {"drug_fp": drug_embedding, "rna": rna}
            sample_generator = diffusion.p_sample_loop_progressive(
                model=model,
                img=z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=device,
                progress=False
            )
            final_sample = None
            for sample in sample_generator:
                final_sample = sample
            assert final_sample is not None, "No samples were obtained during the sampling process."
            img_latent, rna_pred = final_sample["img_sample"], final_sample["rna_sample"]
            
            with torch.no_grad():
                images = vae.decode(img_latent / 0.18215).sample
            
            rna_pred_np = rna_pred.cpu().numpy()
            rna_original = ((rna_pred_np + 1) / 2) * rna_range + rna_min
            
            for j in range(batch_size):
                if generated >= total_samples:
                    break
                img_save_path = f"{output_dir}/{generated:06d}.png"
                save_image(images[j], img_save_path, normalize=True, value_range=(-1, 1))
                rna_save_path = f"{rna_dir}/{generated:06d}.csv"
                np.savetxt(rna_save_path, rna_original[j], delimiter=",")
                generated += 1

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiTMultimodal_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="drug_samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=16)
    parser.add_argument("--num-fid-samples", type=int, default=500)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--h5ad_path", type=str, required=True)
    parser.add_argument("--num-drug-classes", type=int, default=98)
    parser.add_argument("--num-rna-features", type=int, default=977)
    parser.add_argument("--fp_size", type=int, default=1024)
    parser.add_argument("--drug-split", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    main(args)