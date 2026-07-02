import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss)
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib as plt
import math

from model.pipeline.data_utils import apply_activate
from model.synthesizer.transformer import ImageTransformer, DataTransformer
from model.condvec import Condvec
from model.sampler import Sampler
from model.model import Classifier, VAEEncoder, Generator

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[0]
    fake = torch.ones_like(d_interpolates).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_info_loss(real_features, fake_features, delta_mean=0.1, delta_var=0.1):
    """
    Computes feature-wise information loss.
    - real_features, fake_features: Discriminator 중간 출력 (flattened)
    """
    real_mean = torch.mean(real_features, dim=0)
    fake_mean = torch.mean(fake_features, dim=0)
    real_var = torch.var(real_features, dim=0)
    fake_var = torch.var(fake_features, dim=0)

    mean_loss = torch.sum(F.relu(torch.abs(real_mean - fake_mean) - delta_mean))
    var_loss = torch.sum(F.relu(torch.abs(real_var - fake_var) - delta_var))
    return mean_loss + var_loss

def weights_init(model):
    
    """
    This function initializes the learnable parameters of the convolutional and batch norm layers

    Inputs:
    1) model->  network for which the parameters need to be initialized
    
    Outputs:
    1) network with corresponding weights initialized using the normal distribution
    
    """
    
    classname = model.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0)


def get_st_ed(target_col_index, transformer):
    
    """
    Used to obtain the start and ending positions of the target column as per the transformed data to be used by the classifier 

    Inputs:
    1) target_col_index -> column index of the target column used for machine learning tasks (binary/multi-classification) in the raw data 
    2) transformer -> fitted DataTransformer with column metadata and output information

    Outputs:
    1) starting (st) and ending (ed) positions of the target column as per the transformed data
    
    """
    st = 0
    output_pos = 0

    for raw_col_index, column_meta in enumerate(transformer.meta):
        column_blocks = []
        if column_meta["type"] in ["continuous", "skewed", "mixed"]:
            column_blocks = transformer.output_info[output_pos:output_pos + 2]
            output_pos += 2
        else:
            column_blocks = transformer.output_info[output_pos:output_pos + 1]
            output_pos += 1

        column_width = sum(dim for dim, _ in column_blocks)
        if raw_col_index == target_col_index:
            if len(column_blocks) != 1 or column_blocks[0][1] != "softmax":
                raise ValueError(
                    f"Target column index {target_col_index} is encoded as "
                    f"{column_meta['type']} with blocks {column_blocks}, not a categorical softmax block."
                )
            return (st, st + column_blocks[0][0])

        st += column_width

    raise ValueError(f"Target column index {target_col_index} is outside transformer metadata.")

def has_nan(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        

def train_vae_gan(encoder, generator, discriminator, full_data, cont_data, args, device):
    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)

    with open(args.dataprep_path, 'rb') as f:
        dataprep = pickle.load(f)

    target_col = list(dataprep.problem_type.values())[0]
    target_col_index = list(dataprep.df.columns).index(target_col)
    target_st, target_ed = get_st_ed(target_col_index, transformer)
    target_dim = target_ed - target_st
    use_semantic_classifier = target_dim > 1
    if not use_semantic_classifier:
        print(
            f"Semantic consistency loss is disabled because target column '{target_col}' "
            "has only one encoded class in the training data."
        )

    cond_generator = Condvec(full_data, transformer.output_info)
    sampler = Sampler(full_data, transformer.output_info)

    image_size = int(np.ceil(np.sqrt(full_data.shape[1] + cond_generator.n_opt)))
    G_transformer = ImageTransformer(image_size, orig_dim=args.output_dim)
    D_transformer = ImageTransformer(image_size)

    dside = image_size 
    num_channels = 64
    classifier = None
    optimizerC = None
    if use_semantic_classifier:
        classifier = Classifier(dside=dside, num_channels=num_channels, num_classes=target_dim).to(device)
        optimizerC = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    dataset = TensorDataset(torch.tensor(full_data, dtype=torch.float32),
                            torch.tensor(cont_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizerE = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        inner_bar = tqdm(enumerate(loader), total=len(loader), desc="Step", leave=False)

        for step, (x_full, x_cont) in inner_bar:
            x_full = x_full.to(device)
            x_cont = x_cont.to(device)

            c, m, col, opt = cond_generator.sample_train(args.batch_size)
            if not isinstance(c, torch.Tensor):
                c = torch.from_numpy(c)
            c = c.to(device)

            z, mu, logvar = encoder(x_cont)
            input_gen = torch.cat([z, c], dim=1)
            fake_image = generator(input_gen)

            fake_tabular = G_transformer.inverse_transform(fake_image)
            fake_activated = apply_activate(fake_tabular, transformer.output_info)
            recon_cont = fake_tabular[:, :x_cont.shape[1]]
            recon_loss = F.mse_loss(recon_cont, x_cont)
            kl_loss = kl_divergence(mu, logvar)

            vae_loss = args.recon_weight * recon_loss + args.kl_weight * kl_loss

            if epoch < args.encoder_freeze_epoch:
                optimizerE.zero_grad()
                optimizerG.zero_grad()
                vae_loss.backward()
                optimizerE.step()
                optimizerG.step()
                continue
            
            if step % 6 == 0:
                with torch.no_grad():
                    z, mu, logvar = encoder(x_cont)
                    input_gen = torch.cat([z, c], dim=1)
                    fake_image = generator(input_gen).detach()
                    fake_tabular = G_transformer.inverse_transform(fake_image)
                    fake_activated = apply_activate(fake_tabular, transformer.output_info)
                    fake_cat = torch.cat([fake_activated, c], dim=1)

                    real_data = torch.from_numpy(sampler.sample(args.batch_size, col, opt)).to(device)
                    if epoch < args.real_activate_until_epoch:
                        real_data = apply_activate(real_data, transformer.output_info)
                    real_cat = torch.cat([real_data, c], dim=1)

                    real_image = D_transformer.transform(real_cat)
                    fake_image_d = D_transformer.transform(fake_cat)

                real_validity, _ = discriminator(real_image)
                fake_validity, _ = discriminator(fake_image_d)
                gp = compute_gradient_penalty(discriminator, real_image.data, fake_image_d.data, device)

                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gp
                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()
            else:
                d_loss = torch.tensor(0.0)  
                
            z, mu, logvar = encoder(x_cont)
            input_gen = torch.cat([z, c], dim=1)
            fake_image = generator(input_gen)
            fake_tabular = G_transformer.inverse_transform(fake_image)
            fake_activated = apply_activate(fake_tabular, transformer.output_info)
            fake_cat = torch.cat([fake_activated, c], dim=1)
            fake_image_d = D_transformer.transform(fake_cat)

            g_out, _ = discriminator(fake_image_d)
            g_loss = -torch.mean(g_out)

            with torch.no_grad():
                real_data_info = torch.from_numpy(sampler.sample(args.batch_size, col, opt)).to(device)
                if epoch < args.real_activate_until_epoch:
                    real_data_info = apply_activate(real_data_info, transformer.output_info)
                real_cat_info = torch.cat([real_data_info, c], dim=1)
                real_image_info = D_transformer.transform(real_cat_info)

            _, real_features = discriminator(real_image_info)
            real_features = real_features.detach()
            _, fake_features = discriminator(fake_image_d)

            info_loss = compute_info_loss(real_features, fake_features,
                                        delta_mean=args.delta_mean,
                                        delta_var=args.delta_var)
            
            if isinstance(fake_image_d, tuple):  
                fake_image_d = fake_image_d[0]
            if use_semantic_classifier:
                target_labels = torch.argmax(x_full[:, target_st:target_ed], dim=1)
                advcls_loss = F.cross_entropy(classifier(fake_image_d), target_labels)
            else:
                advcls_loss = torch.tensor(0.0, device=device)

            advcls_weight = 1.0
            
            total_loss = (args.g_weight * g_loss +
                          args.info_weight * info_loss + 
                          advcls_weight * advcls_loss +
                          args.vae_finetune_weight * vae_loss
                          )

            optimizerE.zero_grad()
            optimizerG.zero_grad()
            if optimizerC is not None:
                optimizerC.zero_grad()
            total_loss.backward()
            optimizerE.step()
            optimizerG.step()
            if optimizerC is not None:
                optimizerC.step()


        if (epoch + 1)>=50 and (epoch + 1) % 20 == 0:
            os.makedirs("./checkpoints", exist_ok=True)
            save_path = f"./checkpoints/vae_ctabgan_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'encoder_state_dict': encoder.state_dict()
            }, save_path)
            print(f"Saved checkpoint (G + E) at epoch {epoch + 1} -> {save_path}")

        final_dir = os.path.join(args.checkpoint_dir, "final")
        os.makedirs(final_dir, exist_ok=True)     

    final_checkpoint = {
        'encoder': encoder.state_dict(),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'classifier': classifier.state_dict() if classifier is not None else None
    }
    torch.save(final_checkpoint, os.path.join(final_dir, f"vae_ctabgan_epoch{epoch+1}.pth"))
    torch.save(final_checkpoint, os.path.join(args.checkpoint_dir, args.save_name))
    print(f"Final checkpoint saved to {os.path.join(final_dir, f'vae_ctabgan_epoch{epoch+1}.pth')}")

def generate_samples(args, full_data, cont_data, device):
    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    output_info = transformer.output_info
    
    with open(args.dataprep_path, "rb") as f:
        dataprep = pickle.load(f)

    condvec = Condvec(full_data, output_info)
    image_size = int(np.ceil(np.sqrt(full_data.shape[1] + condvec.n_opt)))
    transformer_G = ImageTransformer(image_size, orig_dim=args.output_dim)

    encoder = VAEEncoder(input_dim=cont_data.shape[1], latent_dim=args.latent_dim).to(device)
    generator = Generator(input_dim=args.latent_dim + condvec.n_opt,
                          gside=image_size, num_channels=64).to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.save_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder']) 
    generator.load_state_dict(checkpoint['generator'])
    
    encoder.eval()
    generator.eval()

    cont_tensor = torch.tensor(cont_data, dtype=torch.float32, device=device)
    n_seed = cont_tensor.size(0)

    samples = []
    for _ in tqdm(range((args.num_samples + args.batch_size - 1) // args.batch_size), desc="Generating"):
        c, _, _, _ = condvec.sample_train(args.batch_size)
        if not isinstance(c, torch.Tensor):
            c = torch.from_numpy(c)
        c = c.to(device)

        seed_idx = torch.randint(0, n_seed, (args.batch_size,), device=device)
        x_cont_seed = cont_tensor[seed_idx]

        with torch.no_grad():
            z, _, _ = encoder(x_cont_seed)
            input_gen = torch.cat([z, c], dim=1)
            fake_image = generator(input_gen)
            fake_tabular = transformer_G.inverse_transform(fake_image) 
            samples.append(fake_tabular.cpu().numpy())

    final_samples = np.concatenate(samples, axis=0)[:args.num_samples]
    fake_tabular = transformer_G.inverse_transform(fake_image)

    tabular_data = transformer.inverse_transform(final_samples)
    tabular_data = np.where(tabular_data < 0, 0.0, tabular_data)
    recovered_df = dataprep.inverse_prep(tabular_data)  

    output_path = os.path.join(args.sample_dir, "generated_samples.csv")
    recovered_df.to_csv(output_path, index=False)
