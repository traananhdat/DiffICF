# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import torch.nn.functional as F
from tqdm.auto import tqdm
# Assume LoRA library exists or self-implemented
# from lora import add_lora, set_lora_requires_grad

# +
# --- 1. Prepare Data ---
class EducationalImageTextDataset(Dataset):
    def __init__(self, image_paths, positive_texts, hard_negative_texts, hard_negative_image_paths, tokenizer, image_processor):
        self.image_paths = image_paths
        self.positive_texts = positive_texts
        self.hard_negative_texts = hard_negative_texts # Can be None if only hard image negs
        self.hard_negative_image_paths = hard_negative_image_paths # Can be None if only hard text negs
        self.tokenizer = tokenizer
        self.image_processor = image_processor # Need image preprocessing function suitable for VAE

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        processed_image = self.image_processor(image) # Preprocess image into tensor

        pos_text = self.positive_texts[idx]
        pos_tokens = self.tokenizer(pos_text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids

        item = {"pixel_values": processed_image, "positive_input_ids": pos_tokens.squeeze(0)}

        # Add hard negatives if present
        if self.hard_negative_texts and idx < len(self.hard_negative_texts):
             neg_text = self.hard_negative_texts[idx]
             neg_tokens = self.tokenizer(neg_text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
             item["negative_text_input_ids"] = neg_tokens.squeeze(0)

        if self.hard_negative_image_paths and idx < len(self.hard_negative_image_paths):
             neg_img_path = self.hard_negative_image_paths[idx]
             neg_image = Image.open(neg_img_path).convert("RGB")
             processed_neg_image = self.image_processor(neg_image)
             item["negative_pixel_values"] = processed_neg_image
             # Need the original text for the (negative_image, positive_text) pair
             item["positive_input_ids_for_neg_image"] = pos_tokens.squeeze(0)


        return item

# --- Placeholder for image preprocessing function ---
def preprocess_image(image):
    # Need to resize, normalize image suitable for Stable Diffusion's VAE
    # Example:
    # image = image.resize((512, 512))
    # image = (torch.tensor(np.array(image)).float() / 127.5) - 1.0
    # image = image.permute(2, 0, 1) # HWC to CHW
    raise NotImplementedError("Need to implement a suitable image preprocessing function")
    return image


# --- 2. Setup Model ---
model_id = "stabilityai/stable-diffusion-2-1-base" # Or another version
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)

# --- Add LoRA to UNet (Assumption) ---
# add_lora(unet) # This function would add LoRA layers to cross-attention modules
# set_lora_requires_grad(unet) # Only enable gradients for LoRA parameters

# --- Freeze parameters not needed for training ---
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
# unet.requires_grad_(False) # If not using LoRA, need to enable this
# Enable gradients for LoRA parameters (if used) - this is usually handled in set_lora_requires_grad

# --- Noise Scheduler ---
from diffusers import DDPMScheduler
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# --- Optimizer ---
# Only optimize trainable parameters (e.g., LoRA weights)
trainable_params = filter(lambda p: p.requires_grad, unet.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4) # lr from paper

# --- Prepare DataLoader ---
# Need to create actual lists of image paths, texts, etc.
image_paths = ["path/to/edu_img1.jpg", ...]
positive_texts = ["Description of image 1", ...]
hard_negative_texts = ["Misleading description 1", ...] # Or None
hard_negative_image_paths = ["path/to/neg_img1.jpg", ...] # Or None

dataset = EducationalImageTextDataset(image_paths, positive_texts, hard_negative_texts, hard_negative_image_paths, tokenizer, preprocess_image)
# Need a custom collate_fn if items in batch have different structures (due to presence/absence of hard negs)
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True) # Small batch size due to high memory requirements

# --- 3 & 4. Training Loop ---
num_epochs = 8 # Number of epochs from paper
lambda_clip = 1.0 # Lambda parameter from paper

unet.train() # Set UNet to training mode

for epoch in range(num_epochs):
    progress_bar = tqdm(total=len(train_dataloader))
    progress_bar.set_description(f"Epoch {epoch+1}")

    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            # Encode images to latent space using VAE
            latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Encode text embeddings
            pos_encoder_hidden_states = text_encoder(batch["positive_input_ids"].to(device))[0]

            # Prepare negative latents and texts if present
            neg_latents = None
            neg_text_encoder_hidden_states = None
            pos_text_for_neg_image_states = None

            if "negative_pixel_values" in batch:
                 neg_latents = vae.encode(batch["negative_pixel_values"].to(device)).latent_dist.sample()
                 neg_latents = neg_latents * vae.config.scaling_factor
                 pos_text_for_neg_image_states = text_encoder(batch["positive_input_ids_for_neg_image"].to(device))[0]

            if "negative_text_input_ids" in batch:
                 neg_text_encoder_hidden_states = text_encoder(batch["negative_text_input_ids"].to(device))[0]


        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample timesteps
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to latents (forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_neg_latents = noise_scheduler.add_noise(neg_latents, noise, timesteps) if neg_latents is not None else None

        # --- Calculate Loss ---
        optimizer.zero_grad()

        # 1. Positive Loss
        noise_pred_pos = unet(noisy_latents, timesteps, encoder_hidden_states=pos_encoder_hidden_states).sample
        loss_pos = F.mse_loss(noise_pred_pos.float(), noise.float(), reduction="mean")

        # 2. Negative Loss - need to calculate for different types of hard negatives
        loss_neg_total = torch.tensor(0.0, device=device)
        num_neg_losses = 0

        # 2a. Hard Negative Text (positive_image, negative_text)
        if neg_text_encoder_hidden_states is not None:
             noise_pred_neg_text = unet(noisy_latents, timesteps, encoder_hidden_states=neg_text_encoder_hidden_states).sample
             # Negative loss is the negative of MSE loss
             loss_neg_text = -F.mse_loss(noise_pred_neg_text.float(), noise.float(), reduction="mean")
             loss_neg_total += loss_neg_text
             num_neg_losses += 1

        # 2b. Hard Negative Image (negative_image, positive_text)
        if noisy_neg_latents is not None and pos_text_for_neg_image_states is not None:
             noise_pred_neg_img = unet(noisy_neg_latents, timesteps, encoder_hidden_states=pos_text_for_neg_image_states).sample
             # Negative loss is the negative of MSE loss
             loss_neg_img = -F.mse_loss(noise_pred_neg_img.float(), noise.float(), reduction="mean")
             loss_neg_total += loss_neg_img
             num_neg_losses += 1

        # Average negative loss if any
        if num_neg_losses > 0:
            loss_neg_avg = loss_neg_total / num_neg_losses
            # Clip negative loss based on positive loss
            clipped_loss_neg = torch.clamp(loss_neg_avg, min=-lambda_clip * loss_pos.item(), max=lambda_clip * loss_pos.item())
            loss = loss_pos + clipped_loss_neg
        else:
            loss = loss_pos # If no hard negatives in the batch

        # Backpropagation
        loss.backward()
        optimizer.step()

        progress_bar.update(1)
        progress_bar.set_postfix(Loss=loss.item(), Loss_Pos=loss_pos.item(), Loss_Neg_Clipped=clipped_loss_neg.item() if num_neg_losses > 0 else 0)

    progress_bar.close()

# Save the fine-tuned model (e.g., only save LoRA weights)
# unet.save_pretrained("path/to/save/finetuned_unet_lora")
# +

# --- 5. Use the fine-tuned model for Retrieval (Ranking) ---
# Ranking method uses DiffICF score (Equation 5)
# Need to load the original model and apply LoRA weights if saved separately

unet.eval() # Set to evaluation mode
text_encoder.eval()
vae.eval()

def calculate_diffusion_icf_score(image_path, text_query, unet, vae, text_encoder, tokenizer, noise_scheduler, device, num_inference_steps=10, num_samples=10):
    """Calculates the DiffICF score between an image and a text query."""
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    processed_image = preprocess_image(image).unsqueeze(0).to(device) # Add batch dim

    # Encode image to latent
    with torch.no_grad():
        latents = vae.encode(processed_image).latent_dist.sample() * vae.config.scaling_factor
        # Encode text query
        text_inputs = tokenizer(text_query, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
        # Encode unconditional text (usually an empty string)
        uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    total_error_diff = 0.0

    for _ in range(num_samples): # Average over multiple noise/timestep samples
        # Sample random noise and timestep
        noise = torch.randn_like(latents)
        timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=latents.device).long()

        noisy_latent = noise_scheduler.add_noise(latents, noise, timestep)

        with torch.no_grad():
            # Predict noise with text condition
            noise_pred_text = unet(noisy_latent, timestep, encoder_hidden_states=text_embeddings).sample
            error_text = F.mse_loss(noise_pred_text, noise).item()

            # Predict noise unconditional
            noise_pred_uncond = unet(noisy_latent, timestep, encoder_hidden_states=uncond_embeddings).sample
            error_uncond = F.mse_loss(noise_pred_uncond, noise).item()

        # Score is the error difference
        error_diff = error_text - error_uncond
        total_error_diff += error_diff

    return total_error_diff / num_samples # Lower score is better


# --- Example Ranking ---
query = "Student raising hand to speak"
candidate_image_paths = ["path/img1.jpg", "path/img2.jpg", "path/img3.jpg"]
scores = []

for img_path in candidate_image_paths:
    score = calculate_diffusion_icf_score(img_path, query, unet, vae, text_encoder, tokenizer, noise_scheduler, device)
    scores.append((img_path, score))

# Sort by score (ascending)
scores.sort(key=lambda x: x[1])

print("Ranking results:")
for img_path, score in scores:
    print(f"- Image: {img_path}, Score: {score:.4f}")