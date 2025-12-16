import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from torchvision.models import inception_v3
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
from scipy import linalg

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--save_dir', type=str, default="outputs", help='where to store generated images')
    
    # Inference arguments
    parser.add_argument('--inference', action='store_true', default=False, help='Run inference only (no training)')
    parser.add_argument('--model_path', type=str, default='models/DDPM_30epochs_cosine/ckpt.pt', help='Path to saved model checkpoint')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to generate during inference')
    
    # Conditional generation arguments
    parser.add_argument('--conditional', action='store_true', default=False, help='Enable conditional generation with classifier-free guidance')
    parser.add_argument('--p_uncond', type=float, default=0.1, help='Probability of unconditional training (default: 0.1)')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='Classifier-free guidance scale for inference (default: 3.0)')
    parser.add_argument('--sample_classes', type=str, default=None, help='Comma-separated class indices to sample (e.g., "0,1,2" for specific classes)')
    
    # Evaluation and visualization arguments
    parser.add_argument('--calculate_fid', action='store_true', help='Calculate FID score for generated images')
    parser.add_argument('--generated_dir', type=str, default=None, help='Directory with generated images for FID calculation')
    parser.add_argument('--plot_schedules', action='store_true', help='Plot and compare beta schedules')

    return parser.parse_args()


def calculate_fid(real_images, generated_images, device, batch_size=50):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_images: Tensor of real images [N, C, H, W] in range [-1, 1]
        generated_images: Tensor of generated images [N, C, H, W] in range [-1, 1]
        device: Device to run on
        batch_size: Batch size for processing
    
    Returns:
        FID score (lower is better)
    """
    # Load pretrained InceptionV3 for feature extraction
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()  # Remove final classification layer
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    def get_activations(images, model, batch_size=50):
        """Extract features from images using InceptionV3"""
        model.eval()
        activations = []
        
        # Resize images to 299x299 for Inception
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        # Convert from [-1,1] to [0,1] then normalize for Inception
        images = (images + 1) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images = (images - mean) / std
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                pred = model(batch)
                activations.append(pred.cpu().numpy())
        
        return np.concatenate(activations, axis=0)
    
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Fréchet distance between two multivariate Gaussians"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    # Get activations
    act1 = get_activations(real_images, inception_model, batch_size)
    act2 = get_activations(generated_images, inception_model, batch_size)
    
    # Calculate statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return fid_value


def sample_and_save_images(n_images, diffusor, model, device, store_path, image_size=32, channels=3, 
                           classes=None, guidance_scale=0.0):
    """
    Generate n_images using diffusor.sample and save them to store_path directory.
    Each image is saved as PNG. Images are rescaled from [-1,1] -> [0,1] before saving.
    
    Args:
        n_images: Number of images to generate
        diffusor: Diffusion model instance
        model: U-Net model
        device: Device to run on
        store_path: Directory to save images
        image_size: Size of images
        channels: Number of channels
        classes: Optional class labels for conditional generation
        guidance_scale: Classifier-free guidance scale (0 = no guidance)
    """
    os.makedirs(store_path, exist_ok=True)
    model.to(device)
    model.eval() # Disables dropout etc.
    batch_size = min(n_images, 16)  # sample in one batch up to 16
    results = []
    remaining = n_images
    idx = 0

    while remaining > 0:
        this_batch = min(remaining, batch_size)
        
        # Prepare class labels for this batch if conditional
        batch_classes = None
        if classes is not None:
            if len(classes) >= this_batch:
                batch_classes = classes[:this_batch].to(device)
                classes = classes[this_batch:]  # Remove used classes
            else:
                # Not enough classes provided, repeat or use None
                batch_classes = None
        
        samples = diffusor.sample(model, image_size, batch_size=this_batch, channels=channels,
                                 classes=batch_classes, guidance_scale=guidance_scale)
        # samples are in [-1,1], convert to [0,1]
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        for i in range(this_batch):
            class_label = f"_class{batch_classes[i].item()}" if batch_classes is not None else ""
            out_path = os.path.join(store_path, f"sample_{idx:04d}{class_label}.png")
            save_image(samples[i], out_path)
            idx += 1
        remaining -= this_batch

    model.train()
    return store_path


def test(model, testloader, diffusor, device, args, calculate_fid_score=False):
    """
    Validation / test loop:
    - compute average loss on given dataloader using diffusor.p_losses
    - optionally calculate FID score for image quality assessment
    - also generate 8 samples and store them under args.save_dir/epoch-X if dataloader is validation
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    # Collect real images for FID calculation if needed
    real_images_for_fid = []

    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="validation"):
            images = images.to(device)
            labels = labels.to(device) if args.conditional else None
            batch_size = images.shape[0]
            t = torch.randint(0, diffusor.timesteps, (batch_size,), device=device).long()
            loss = diffusor.p_losses(model, images, t, loss_type="l2", classes=labels)
            total_loss += loss.item()
            n_batches += 1
            
            # Collect images for FID (limit to 1000 images)
            if calculate_fid_score and len(real_images_for_fid) < 1000:
                real_images_for_fid.append(images.cpu())
                if sum(img.shape[0] for img in real_images_for_fid) >= 1000:
                    break

    avg_loss = total_loss / max(1, n_batches)
    print(f"Validation Loss: {avg_loss:.6f}")
    
    # Calculate FID if requested
    fid_score = None
    if calculate_fid_score and len(real_images_for_fid) > 0:
        print("\nCalculating FID score...")
        real_images = torch.cat(real_images_for_fid, dim=0)[:1000].to(device)
        
        # Generate same number of images
        n_gen = real_images.shape[0]
        gen_images = []
        gen_batch_size = 16
        
        sample_classes = None
        guidance = 0.0
        if args.conditional and hasattr(model, 'num_classes'):
            guidance = args.guidance_scale
        
        for i in range(0, n_gen, gen_batch_size):
            batch_size = min(gen_batch_size, n_gen - i)
            if args.conditional and hasattr(model, 'num_classes'):
                sample_classes = torch.randint(0, model.num_classes, (batch_size,), device=device)
            
            samples = diffusor.sample(model, diffusor.img_size, batch_size=batch_size, 
                                    channels=3, classes=sample_classes, guidance_scale=guidance)
            gen_images.append(samples.cpu())
        
        generated_images = torch.cat(gen_images, dim=0).to(device)
        
        try:
            fid_score = calculate_fid(real_images, generated_images, device)
            print(f"FID Score: {fid_score:.2f} (lower is better)")
        except Exception as e:
            print(f"Warning: Could not calculate FID: {e}")
            fid_score = None

    # also generate a small grid of images for monitoring
    out_dir = os.path.join(args.save_dir, "samples")
    # For conditional models, sample from random classes
    sample_classes = None
    guidance = 0.0
    if args.conditional and hasattr(model, 'num_classes'):
        sample_classes = torch.randint(0, model.num_classes, (8,), device=device)
        guidance = args.guidance_scale
    sample_and_save_images(8, diffusor, model, device, out_dir, image_size=diffusor.img_size,
                          classes=sample_classes, guidance_scale=guidance)
    model.train()
    return avg_loss if fid_score is None else (avg_loss, fid_score)


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(trainloader, desc=f"train epoch {epoch}")
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        labels = labels.to(device) if args.conditional else None
        optimizer.zero_grad()

        # Algorithm 1: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2", classes=labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': loss.item()})

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break
    
    avg_train_loss = total_loss / max(1, n_batches)
    return avg_train_loss


def run_inference_test(args):
    """
    Load a saved model checkpoint and generate images for inference/testing.
    Usage: python ex02_main.py --inference --model_path models/DDPM_30epochs_cosine/ckpt.pt --num_samples 64
    For conditional: python ex02_main.py --inference --conditional --guidance_scale 3.0 --sample_classes "0,1,2,3"
    """
    timesteps = args.timesteps
    image_size = 32
    channels = 3
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {args.model_path}...")
    # Initialize model with conditional support if requested
    if args.conditional:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,),
                    class_free_guidance=True, num_classes=10, p_uncond=args.p_uncond).to(device)
        print(f"Using conditional model with classifier-free guidance (scale={args.guidance_scale})")
    else:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    my_scheduler = lambda x: cosine_beta_schedule(x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)
    
    # Prepare class labels if conditional
    sample_classes = None
    guidance_scale = 0.0
    if args.conditional:
        guidance_scale = args.guidance_scale
        if args.sample_classes:
            # Parse comma-separated class indices
            class_list = [int(c.strip()) for c in args.sample_classes.split(',')]
            # Repeat to match num_samples
            class_list = (class_list * (args.num_samples // len(class_list) + 1))[:args.num_samples]
            sample_classes = torch.tensor(class_list, device=device)
            print(f"Sampling from classes: {class_list[:min(10, len(class_list))]}...")
        else:
            # Sample random classes
            sample_classes = torch.randint(0, 10, (args.num_samples,), device=device)
            print(f"Sampling from random CIFAR-10 classes")
    
    # Generate samples
    num_samples = getattr(args, 'num_samples', 64)
    inference_dir = os.path.join(args.save_dir, "inference_samples")
    print(f"Generating {num_samples} samples...")
    sample_and_save_images(num_samples, diffusor, model, device, inference_dir, 
                          image_size=image_size, channels=channels,
                          classes=sample_classes, guidance_scale=guidance_scale)
    print(f"✓ Samples saved to {inference_dir}")


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    # Initialize model with conditional support if requested
    if args.conditional:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,),
                    class_free_guidance=True, num_classes=10, p_uncond=args.p_uncond).to(device)
        print(f"Training conditional model with classifier-free guidance (p_uncond={args.p_uncond})")
    else:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
        print("Training unconditional model")
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Initialize loss tracking
    loss_history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'epochs': []
    }

    my_scheduler = lambda x: cosine_beta_schedule(x)  # Using cosine schedule for better quality
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    # create save dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train(model, trainloader, optimizer, diffusor, epoch, device, args)
        val_loss = test(model, valloader, diffusor, device, args)
        
        # Calculate FID on test set every 5 epochs (can be expensive)
        calculate_fid = (epoch % 5 == 0) or (epoch == epochs - 1)
        test_result = test(model, testloader, diffusor, device, args, calculate_fid_score=calculate_fid)
        
        if isinstance(test_result, tuple):
            test_loss, fid_score = test_result
        else:
            test_loss = test_result
            fid_score = None
        
        # Track losses
        loss_history['train_loss'].append(train_loss)
        loss_history['val_loss'].append(val_loss)
        loss_history['test_loss'].append(test_loss)
        loss_history['epochs'].append(epoch)
        
        # Track FID if available
        if 'fid_scores' not in loss_history:
            loss_history['fid_scores'] = []
        loss_history['fid_scores'].append(fid_score if fid_score is not None else None)
        
        fid_str = f", FID: {fid_score:.2f}" if fid_score is not None else ""
        print(f"\nEpoch {epoch} Summary - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Test Loss: {test_loss:.6f}{fid_str}\n")
    
    # Save loss history
    import json
    loss_file = os.path.join(args.save_dir, f"loss_history_{args.run_name}.json")
    with open(loss_file, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"\n✓ Loss history saved to {loss_file}")

    # save final model and generate samples
    if args.save_model:
        out_model_dir = os.path.join("./models", args.run_name)
        os.makedirs(out_model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_model_dir, "ckpt.pt"))

    # Generate final samples
    final_classes = None
    final_guidance = 0.0
    if args.conditional:
        # Generate 2 samples per class for CIFAR-10
        final_classes = torch.arange(10, device=device).repeat_interleave(2)[:16]
        final_guidance = args.guidance_scale
    sample_and_save_images(16, diffusor, model, device, args.save_dir, image_size=image_size, channels=channels,
                          classes=final_classes, guidance_scale=final_guidance)


def calculate_fid_from_directory(args):
    """Calculate FID score for generated images in a directory"""
    from PIL import Image
    
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    
    if not args.generated_dir:
        print("Error: --generated_dir required for FID calculation")
        return
    
    if not os.path.exists(args.generated_dir):
        print(f"Error: Directory {args.generated_dir} does not exist")
        return
    
    print(f"\n{'='*60}")
    print(f"FID Calculation for: {args.generated_dir}")
    print(f"{'='*60}\n")
    
    # Load generated images
    image_files = []
    for root, dirs, files in os.walk(args.generated_dir):
        for f in files:
            if f.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, f))
    
    if not image_files:
        print(f"No images found in {args.generated_dir}")
        return
    
    print(f"Loading {len(image_files)} images from {args.generated_dir}...")
    generated_images = []
    for img_path in tqdm(sorted(image_files), desc="Loading images"):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img)
        img_tensor = (img_tensor * 2) - 1
        generated_images.append(img_tensor)
    
    generated_images = torch.stack(generated_images).to(device)
    num_gen = len(generated_images)
    
    # Load real CIFAR-10 images
    print(f"Loading {num_gen} real CIFAR-10 test images...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=False, 
                               train=False, transform=transform)
    indices = np.random.choice(len(dataset), num_gen, replace=False)
    real_images = torch.stack([dataset[i][0] for i in indices]).to(device)
    
    # Calculate FID
    fid_score = calculate_fid(real_images, generated_images, device)
    
    print(f"\n{'='*60}")
    print(f"FID Score: {fid_score:.2f}")
    print(f"{'='*60}\n")
    
    # Save result
    result_file = os.path.join(args.generated_dir, "fid_score.txt")
    with open(result_file, 'w') as f:
        f.write(f"FID Score: {fid_score:.2f}\n")
        f.write(f"Number of images: {num_gen}\n")
    print(f"✓ Results saved to {result_file}")


def plot_beta_schedules_comparison(args):
    """Plot and compare different beta schedules"""
    import matplotlib.pyplot as plt
    from ex02_diffusion import sigmoid_beta_schedule
    
    timesteps = args.timesteps
    beta_start = 0.0001
    beta_end = 0.02
    
    # Generate schedules
    linear_betas = linear_beta_schedule(beta_start, beta_end, timesteps)
    cosine_betas = cosine_beta_schedule(timesteps)
    sigmoid_betas = sigmoid_beta_schedule(beta_start, beta_end, timesteps)
    
    schedules = {
        'Linear': linear_betas,
        'Cosine': cosine_betas,
        'Sigmoid': sigmoid_betas
    }
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Beta values
    ax = axes[0, 0]
    for name, betas in schedules.items():
        ax.plot(betas.cpu().numpy(), label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('β_t')
    ax.set_title('Beta Schedules')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Alpha values
    ax = axes[0, 1]
    for name, betas in schedules.items():
        alphas = 1. - betas
        ax.plot(alphas.cpu().numpy(), label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('α_t = 1 - β_t')
    ax.set_title('Alpha Values')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Cumulative product of alphas
    ax = axes[1, 0]
    for name, betas in schedules.items():
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        ax.plot(alphas_cumprod.cpu().numpy(), label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('ᾱ_t')
    ax.set_title('Cumulative Product of Alphas')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Signal-to-Noise Ratio
    ax = axes[1, 1]
    for name, betas in schedules.items():
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        ax.semilogy(snr.cpu().numpy(), label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('SNR (log scale)')
    ax.set_title('Signal-to-Noise Ratio')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('visualisations', exist_ok=True)
    output_path = 'visualisations/beta_schedules_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Beta schedules comparison saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    args = parse_args()

    # Handle special modes
    if args.calculate_fid:
        calculate_fid_from_directory(args)
    elif args.plot_schedules:
        plot_beta_schedules_comparison(args)
    elif args.inference:
        os.makedirs(args.save_dir, exist_ok=True)
        print("\n➡️ Running inference mode...")
        run_inference_test(args)
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        print("\n➡️ Visualization enabled: generated samples will be saved after each epoch.")
        print(f"➡️ Images will be stored in: {args.save_dir}/samples\n")
        run(args)

