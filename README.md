# CIFAR-10 DDPM Implementation

Denoising Diffusion Probabilistic Model for CIFAR-10 image generation with Classifier-Free Guidance.

## Features
- ✅ Complete DDPM (Tasks 2.2, 2.3, 2.4)
- ✅ Multiple noise schedules (linear, cosine, sigmoid)
- ✅ Classifier-Free Guidance for conditional generation
- ✅ FID score evaluation
- ✅ U-Net with time and class embeddings

## Quick Start

### Training
```bash
# Unconditional
python ex02_main.py --epochs 30 --save_model --run_name DDPM_30ep

# Conditional with CFG
python ex02_main.py --conditional --epochs 30 --guidance_scale 3.0 \
    --save_model --run_name DDPM_cond --save_dir outputs_conditional
```

### Inference
```bash
# Unconditional
python ex02_main.py --inference --model_path models/DDPM_30ep/ckpt.pt --num_samples 64

# Conditional (specific classes: cat, dog, bird)
python ex02_main.py --inference --conditional --model_path models/DDPM_cond/ckpt.pt \
    --sample_classes "3,5,2" --guidance_scale 3.0 --num_samples 64
```

### Evaluation & Visualization
```bash
# Calculate FID score
python ex02_main.py --calculate_fid --generated_dir outputs_conditional_10ep

# Plot beta schedules comparison
python ex02_main.py --plot_schedules

# Create diffusion animations (forward, reverse, comparison)
python create_animation.py --model_path models/DDPM_30epochs_cosine/ckpt.pt --mode all
python create_animation.py --model_path models/DDPM_30epochs_cosine/ckpt.pt --mode reverse
```

## Files
- `ex02_main.py` - Training, inference, FID evaluation, schedule plotting
- `ex02_model.py` - U-Net architecture
- `ex02_diffusion.py` - Diffusion process
- `ex02_helpers.py` - Utility functions
- `ex02_tests.py` - Unit tests
- `create_animation.py` - Generate diffusion GIFs
- `test_conditional.py` - Conditional generation tests
- `visualisations/` - Generated plots and animations

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 5 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--timesteps` | 100 | Diffusion steps |
| `--lr` | 0.003 | Learning rate (AdamW) |
| `--conditional` | False | Enable Classifier-Free Guidance |
| `--guidance_scale` | 3.0 | CFG strength (higher = more class-specific) |
| `--p_uncond` | 0.1 | Probability of unconditional training |

## CIFAR-10 Classes
0:Airplane, 1:Automobile, 2:Bird, 3:Cat, 4:Deer, 5:Dog, 6:Frog, 7:Horse, 8:Ship, 9:Truck

## Implementation Details

### Task 2.2: Diffusion Process
- `q_sample()` - Forward diffusion (add noise)
- `p_sample()` - Single reverse step
- `sample()` - Full sampling loop
- `p_losses()` - Training loss (L2)

### Task 2.3: Noise Schedules
- **Linear**: β ∈ [0.0001, 0.02]
- **Cosine**: Improved schedule (default)
- **Sigmoid**: Smooth S-curve

### Task 2.4: Classifier-Free Guidance
- Class embeddings (10 classes + null token)
- Training: Random label dropping (p_uncond=0.1)
- Inference: `eps = eps_uncond + w * (eps_cond - eps_uncond)`

## FID Scores (Current Models)

| Model | Epochs | Images | FID | Status |
|-------|--------|--------|-----|--------|
| conditional_10ep | 10 | 54 | 291.5 | Early training |
| conditional_30ep | 30 | 27 | 339.2 | Early training |
| inference | - | 64 | 413.2 | Needs more training |

**Note:** High FID scores expected with limited training (10-30 epochs). Published DDPM achieves FID ~3-5 after 200+ epochs.

## Environment
```bash
conda activate /proj/aimi-adl/envs/adl23_2
```

## References
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)
- Ho & Salimans, "Classifier-Free Diffusion Guidance" (NeurIPS 2021)
