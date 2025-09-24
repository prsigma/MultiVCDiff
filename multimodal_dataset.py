import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import scanpy as sc
from torchvision import transforms
from typing import Dict, Optional
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

def Drug_encoder(drug_SMILES_list: list,num_Bits=1024, comb_num=1):
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


class H5ADDataset(Dataset):
    def __init__(
        self,
        h5ad_path: str,
        image_dir: Optional[str] = None,
        image_size: int = 512,
        image_col: str = "merged_image",
        drug_col: str = "compound",
        smiles_col: str = "smiles",
        transform=None,
        normalize_rna: bool = True,
        fp_size: int = 1024,
        comb_num: int = 1
    ):
        self.adata = sc.read_h5ad(h5ad_path)
        self.image_dir = image_dir
        self.image_col = image_col
        self.drug_col = drug_col
        self.smiles_col = smiles_col
        self.fp_size = fp_size
        self.comb_num = comb_num
            
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.normalize_rna = normalize_rna
        if normalize_rna:
            X = self.adata.X
            self.rna_min, self.rna_max = X.min(0), X.max(0)
            self.rna_range = self.rna_max - self.rna_min
            self.rna_range[self.rna_range == 0] = 1.0

        self._preprocess_drug_embeddings()

    def _preprocess_drug_embeddings(self):
        smiles = self.adata.obs[self.smiles_col].tolist()
        try:
            embeddings = Drug_encoder(
                drug_SMILES_list=smiles,
                num_Bits=self.fp_size,
                comb_num=self.comb_num
            )
            self.drug_embeddings = torch.from_numpy(embeddings).float()
        except Exception as e:
            self.drug_embeddings = None

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        X = self.adata.X[idx].flatten()
        if self.normalize_rna:
            X = 2 * ((X - self.rna_min) / self.rna_range) - 1 
        rna = torch.from_numpy(X).float()
        
        img_path = self.adata.obs[self.image_col].iloc[idx]
        if self.image_dir:
            img_path = os.path.join(self.image_dir, img_path)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        if self.drug_embeddings is not None:
            drug_embedding = self.drug_embeddings[idx]

        drug_name = self.adata.obs[self.drug_col].iloc[idx]
        
        return {
            "image": img,
            "rna": rna,
            "drug_embedding": drug_embedding,
            "drugname": drug_name
        }
if __name__ == "__main__":
    dataset = H5ADDataset(
        h5ad_path="/data/pr/DiT_AIVCdiff/pr_tutorial/DiT_input_512_one_image_one_rna.h5ad",
        image_dir="/data/pr/cellpainting/BBBC021/raw_data/metadata/augmented_image_metadata_512.csv",
        image_size=512,
        fp_size=1024,
        comb_num=1
    )

    sample = dataset[0]