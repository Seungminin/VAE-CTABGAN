import argparse
import os
import torch
import pickle
import numpy as np

from model.preprocess import preprocess_data
from model.pipeline.data_utils import load_processed_data, extract_continuous_features
from model.train_loop import train_vae_gan, generate_samples, weights_init
from model.model import VAEEncoder, Generator, Discriminator, Classifier
from model.pipeline.data_utils import show_all_parameters
from model.pipeline.data_preparation import DataPrep
from model.sampler import Sampler
from model.condvec import Condvec
from model.synthesizer.transformer import DataTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="VAE-CTAB-GAN")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["preprocess", "train", "generate", "only_train"],
                        help="Choose one mode: preprocess, train, generate, only_train")
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--encoder_freeze_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--vae_finetune_weight", type=float, default=0.1,
                        help="Weight for the VAE regularization term after encoder pretraining")
    parser.add_argument("--g_weight", type=float, default=1.0)
    parser.add_argument("--info_weight", type=float, default=1.0)
    parser.add_argument('--delta_mean', type=float, default=0.1)
    parser.add_argument('--delta_var', type=float, default=0.1)
    parser.add_argument('--dataset_path', type=str, default='Real_Datasets/train.csv')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='vae_ctabgan.pt')
    parser.add_argument('--preprocessed_path', type=str, default='preprocess/preprocessed.csv')
    parser.add_argument('--transformer_path', type=str, default='preprocess/transformer/transformer.pkl')
    parser.add_argument('--dataprep_path', type=str, default='preprocess/dataprep/dataprep.pkl')
    parser.add_argument("--real_activate_until_epoch", type=int, default=50,
                    help="Number of epochs to apply activation to real data for fair D comparison")
    parser.add_argument('--num_samples', type=int, default=100, help='Number of synthetic samples to generate')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.preprocessed_path), exist_ok=True)

    ## Preprocess data if needed
    if args.mode == "preprocess":
        print("Preprocessing original data using DataPrep")
        preprocess_data(raw_path=args.dataset_path, save_path=args.preprocessed_path)
        return

    if args.mode == "train":
        print("Preprocessing original data using DataPrep")
        preprocess_data(raw_path=args.dataset_path, save_path=args.preprocessed_path)

    if not os.path.exists(args.preprocessed_path):
        candidates = [
            args.preprocessed_path,
            "preprocess/preprocessed.csv",
            "model/preprocess/processed.csv",
        ]
        existing_candidates = [path for path in candidates if os.path.exists(path)]
        hint = (
            f"Processed CSV not found at: {args.preprocessed_path}\n"
            "Run preprocess with the same --preprocessed_path, or train with the path printed by preprocess.\n"
            "Default preprocess output: preprocess/preprocessed.csv"
        )
        if existing_candidates:
            hint += f"\nExisting processed file candidate(s): {', '.join(existing_candidates)}"
        raise FileNotFoundError(hint)

    print("Loading processed data")
    data = load_processed_data(args.preprocessed_path)
    cont_data = extract_continuous_features(data, transformer_path=args.transformer_path)

    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)
        args.output_info = transformer.output_info
        args.output_dim = transformer.output_dim
        
    condvec = Condvec(data, transformer.output_info, device=device)
    sampler = Sampler(data, transformer.output_info, device=device)

    ## Generate synthetic samples if needed
    if args.mode == "generate":
        print("Generating synthetic samples")
        generate_samples(
            args=args,
            full_data=data,
            cont_data=cont_data,
            device=device
        )
        return

    encoder = VAEEncoder(input_dim=cont_data.shape[1], latent_dim=args.latent_dim).to(device)

    g_input_dim = args.latent_dim + condvec.n_opt
    gside = int(np.ceil(np.sqrt(data.shape[1] + condvec.n_opt)))
    num_channel = 64

    generator = Generator(input_dim=g_input_dim, gside=gside, num_channels=num_channel).to(device)
    discriminator = Discriminator(dside=gside, num_channels=num_channel).to(device)
    classifier = Classifier(dside=gside, num_channels=num_channel, num_classes=condvec.n_opt).to(device)
    
    generator.apply(weights_init)

    show_all_parameters(encoder, name="VAE Encoder")
    show_all_parameters(generator, name="Generator")
    show_all_parameters(discriminator, name="Discriminator")
    show_all_parameters(classifier, name="Classifier")

    if args.mode in ["train", "only_train"]:
        print("Starting training")
        train_vae_gan(
            encoder=encoder,
            generator=generator,
            discriminator=discriminator,
            full_data=data,
            cont_data=cont_data,
            args=args,
            device=device
        )


if __name__ == "__main__":
    main()
