"""
Create comprehensive animations showing:
1. Forward process: Clean image -> Noise
2. Reverse process: Noise -> Clean image
3. Side-by-side comparison

Outputs saved to visualisations/ directory.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
from tqdm import tqdm

from ex02_model import Unet
from ex02_diffusion import Diffusion, cosine_beta_schedule
from torchvision import datasets, transforms


def create_forward_animation(diffusor, real_image, save_path='visualisations/forward_diffusion.gif',
                             frame_interval=5, duration=100):
    """Show how clean image gradually becomes noise"""
    device = diffusor.device
    timesteps = diffusor.timesteps
    
    # Create output directory
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Prepare image
    x_start = real_image.unsqueeze(0).to(device)
    
    frames = []
    save_steps = list(range(0, timesteps, frame_interval))
    if timesteps-1 not in save_steps:
        save_steps.append(timesteps-1)
    
    print(f"Creating forward diffusion animation ({len(save_steps)} frames)...")
    
    for t in tqdm(save_steps, desc="Adding noise"):
        t_tensor = torch.tensor([t], device=device).long()
        noisy_image = diffusor.q_sample(x_start, t_tensor)
        
        # Convert to PIL
        img_array = (noisy_image[0].clamp(-1, 1) + 1) / 2.0
        img_array = (img_array * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        
        # Upscale
        pil_img = Image.fromarray(img_array)
        pil_img = pil_img.resize((256, 256), Image.NEAREST)
        
        # Add label
        pil_img = add_label(pil_img, f"T={t}/{timesteps-1} - Adding Noise", t, timesteps)
        frames.append(pil_img)
    
    # Save GIF
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                  duration=duration, loop=0)
    print(f"✓ Forward animation saved to {save_path}")


def create_reverse_animation(model, diffusor, save_path='visualisations/reverse_diffusion.gif',
                             classes=None, guidance_scale=0.0, frame_interval=2, duration=50):
    """Show how noise gradually becomes clean image"""
    device = diffusor.device
    timesteps = diffusor.timesteps
    
    # Create output directory
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Start with noise
    img = torch.randn((1, 3, diffusor.img_size, diffusor.img_size), device=device)
    
    frames = []
    save_steps = list(range(timesteps-1, -1, -frame_interval))
    if 0 not in save_steps:
        save_steps.append(0)
    save_steps = sorted(save_steps, reverse=True)
    
    print(f"Creating reverse diffusion animation ({len(save_steps)} frames)...")
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(timesteps-1, -1, -1), desc="Denoising"):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            
            if guidance_scale > 0 and classes is not None:
                img_uncond = diffusor.p_sample(model, img, t, t_index=i, classes=None)
                img_cond = diffusor.p_sample(model, img, t, t_index=i, classes=classes)
                img = img_uncond + guidance_scale * (img_cond - img_uncond)
            else:
                img = diffusor.p_sample(model, img, t, t_index=i, classes=classes)
            
            if i in save_steps:
                # Convert to PIL
                img_array = (img[0].clamp(-1, 1) + 1) / 2.0
                img_array = (img_array * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                pil_img = Image.fromarray(img_array)
                pil_img = pil_img.resize((256, 256), Image.NEAREST)
                
                # Add label
                step = timesteps - i
                pil_img = add_label(pil_img, f"T={i}/{timesteps-1} - Denoising", step, timesteps)
                frames.append(pil_img)
    
    # Add pause at end
    for _ in range(5):
        frames.append(frames[-1])
    
    # Save GIF
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                  duration=duration, loop=0)
    print(f"✓ Reverse animation saved to {save_path}")


def create_comparison_animation(diffusor, real_image, model, save_path='visualisations/comparison_diffusion.gif',
                               classes=None, guidance_scale=0.0, frame_interval=2, duration=50):
    """Side-by-side: forward (left) and reverse (right) processes"""
    device = diffusor.device
    timesteps = diffusor.timesteps
    
    # Create output directory
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Prepare real image
    x_start = real_image.unsqueeze(0).to(device)
    
    # Start reverse with noise
    reverse_img = torch.randn((1, 3, diffusor.img_size, diffusor.img_size), device=device)
    
    frames = []
    save_steps = list(range(0, timesteps, frame_interval))
    if timesteps-1 not in save_steps:
        save_steps.append(timesteps-1)
    
    print(f"Creating comparison animation ({len(save_steps)} frames)...")
    
    model.eval()
    with torch.no_grad():
        for forward_t in tqdm(save_steps, desc="Forward + Reverse"):
            # Forward process
            t_tensor = torch.tensor([forward_t], device=device).long()
            forward_noisy = diffusor.q_sample(x_start, t_tensor)
            
            # Reverse process (from end to start)
            reverse_t = timesteps - 1 - forward_t
            if reverse_t >= 0:
                t_reverse = torch.full((1,), reverse_t, device=device, dtype=torch.long)
                
                if guidance_scale > 0 and classes is not None:
                    img_uncond = diffusor.p_sample(model, reverse_img, t_reverse, t_index=reverse_t, classes=None)
                    img_cond = diffusor.p_sample(model, reverse_img, t_reverse, t_index=reverse_t, classes=classes)
                    reverse_img = img_uncond + guidance_scale * (img_cond - img_uncond)
                else:
                    reverse_img = diffusor.p_sample(model, reverse_img, t_reverse, t_index=reverse_t, classes=classes)
            
            # Create side-by-side image
            forward_array = (forward_noisy[0].clamp(-1, 1) + 1) / 2.0
            forward_array = (forward_array * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            
            reverse_array = (reverse_img[0].clamp(-1, 1) + 1) / 2.0
            reverse_array = (reverse_array * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            
            # Upscale
            forward_pil = Image.fromarray(forward_array).resize((256, 256), Image.NEAREST)
            reverse_pil = Image.fromarray(reverse_array).resize((256, 256), Image.NEAREST)
            
            # Combine side by side
            combined = Image.new('RGB', (512, 280))
            combined.paste(forward_pil, (0, 24))
            combined.paste(reverse_pil, (256, 24))
            
            # Add labels
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Top labels
            draw.rectangle([0, 0, 256, 24], fill='black')
            draw.rectangle([256, 0, 512, 24], fill='black')
            draw.text((64, 4), "Forward (Adding Noise)", fill='white', font=font)
            draw.text((296, 4), "Reverse (Denoising)", fill='white', font=font)
            
            # Bottom timestamp
            draw.rectangle([0, 256, 512, 280], fill='black')
            draw.text((180, 260), f"Step {forward_t}/{timesteps-1}", fill='white', font=font)
            
            frames.append(combined)
    
    # Add pause
    for _ in range(5):
        frames.append(frames[-1])
    
    # Save GIF
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                  duration=duration, loop=0)
    print(f"✓ Comparison animation saved to {save_path}")


def add_label(img, text, step, total_steps, font_size=14):
    """Add label to image"""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (img.width - text_width) // 2
    y = 10
    
    padding = 5
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0)
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Generate diffusion animations')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['forward', 'reverse', 'comparison', 'all'],
                       help='Animation type to generate')
    parser.add_argument('--conditional', action='store_true',
                       help='Use conditional model')
    parser.add_argument('--class_idx', type=int, default=None,
                       help='Class index for conditional generation (0-9)')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                       help='Guidance scale for conditional generation')
    parser.add_argument('--frame_interval', type=int, default=2,
                       help='Frame interval (lower=smoother, larger file)')
    parser.add_argument('--duration', type=int, default=50,
                       help='Duration per frame in ms')
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load model
    image_size = 32
    channels = 3
    
    print(f"Loading model from {args.model_path}...")
    if args.conditional:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,),
                    class_free_guidance=True, num_classes=10).to(device)
    else:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Initialize diffusion
    diffusor = Diffusion(100, lambda x: cosine_beta_schedule(x), image_size, device)
    
    # Load a real CIFAR-10 image for forward process
    print("Loading CIFAR-10 sample...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=False, 
                               train=False, transform=transform)
    
    # Get specific class if requested
    if args.class_idx is not None:
        for img, label in dataset:
            if label == args.class_idx:
                real_image = img
                break
    else:
        real_image = dataset[0][0]
    
    # Prepare class tensor
    classes = None
    if args.conditional and args.class_idx is not None:
        classes = torch.tensor([args.class_idx], device=device)
        print(f"Generating class {args.class_idx}")
    
    print()
    
    # Generate animations
    if args.mode in ['forward', 'all']:
        create_forward_animation(diffusor, real_image, 'forward_diffusion.gif',
                                args.frame_interval, args.duration)
    
    if args.mode in ['reverse', 'all']:
        create_reverse_animation(model, diffusor, 'reverse_diffusion.gif',
                                classes, args.guidance_scale, args.frame_interval, args.duration)
    
    if args.mode in ['comparison', 'all']:
        create_comparison_animation(diffusor, real_image, model, 'comparison_diffusion.gif',
                                   classes, args.guidance_scale, args.frame_interval, args.duration)
    
    print(f"\n✅ All animations complete!")


if __name__ == "__main__":
    main()
