"""
FlowRestore Utils v3 - Corrected Version with Trajectory Coupling

Key Fix: Restored the trajectory coupling mechanism from FlowEdit

Core Algorithm:
    For each timestep t:
        1. zt_lq = (1-t)*x_lq + t*noise     # LQ image at time t (forward process)
        2. zt_hq = zt_edit + (zt_lq - x_lq) # HQ trajectory follows LQ trajectory
        3. V_deg = model(zt_lq, degradation_prompt)
        4. V_cln = model(zt_hq, clean_prompt)  
        5. V_restore = V_cln - V_deg
        6. zt_edit = zt_edit + dt * V_restore
"""

from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


# =========================================================================
# Default Prompts
# =========================================================================

SR_PROMPTS = {
    "degradation": "a low resolution, blurry image",
    "clean": "a high resolution, sharp image",
}

DENOISE_PROMPTS = {
    "degradation": "a noisy, grainy image with gaussian noise",
    "clean": "a clean, sharp, noise-free image",
}

JPEG_PROMPTS = {
    "degradation": "a jpeg compressed image with blocking artifacts",
    "clean": "a high quality, artifact-free image",
}


# =========================================================================
# Helper Functions
# =========================================================================

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def unpack_latents_flux_manual(latents, height, width, vae_scale_factor):
    """[B, Seq, C_packed] -> [B, C, H, W]"""
    batch_size, seq_len, channels_packed = latents.shape
    h_lat = height // vae_scale_factor
    w_lat = width // vae_scale_factor
    h_grid = h_lat // 2
    w_grid = w_lat // 2
    channels = channels_packed // 4
    
    latents = latents.view(batch_size, h_grid, w_grid, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels, h_lat, w_lat)
    return latents


def scale_noise(scheduler, sample, timestep, noise):
    """Forward process: x_t = (1-sigma)*x_0 + sigma*noise"""
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    return sigma * noise + (1.0 - sigma) * sample


# =========================================================================
# V-Prediction Wrappers
# =========================================================================

def calc_v_sd3(pipe, latent_input, prompt_embeds, pooled_prompt_embeds, guidance_scale, t):
    """SD3 velocity prediction with CFG"""
    timestep = t.expand(latent_input.shape[0])
    
    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latent_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return noise_pred


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    """FLUX velocity prediction"""
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred


# =========================================================================
# SD3 Restoration (Corrected)
# =========================================================================

@torch.no_grad()
def FlowRestoreSD3(
    pipe,
    scheduler,
    x_lq,
    # ===== Prompt Settings =====
    task: str = "sr",
    degradation_prompt: str = None,
    clean_prompt: str = None,
    content_prompt: str = "",
    negative_prompt: str = "",
    # ===== Guidance Scales =====
    degradation_guidance: float = 3.5,
    clean_guidance: float = 13.5,         # Higher for stronger restoration
    # ===== Sampling Settings =====
    T_steps: int = 50,
    n_avg: int = 1,
    n_min: int = 0,
    n_max: int = 40,                      # Process more steps
    # ===== Regularization =====
    reg_scale: float = 0.0,
    # ===== Early Stopping =====
    adaptive_tolerance: float = 0.0,
    min_inference_steps: int = 15,
):
    """
    FlowRestore for SD3 with proper trajectory coupling.
    """
    device = x_lq.device
    
    # ===== 1. Setup Prompts =====
    if task == "sr":
        default_prompts = SR_PROMPTS
    elif task == "denoise":
        default_prompts = DENOISE_PROMPTS
    elif task == "jpeg":
        default_prompts = JPEG_PROMPTS
    else:
        default_prompts = {"degradation": "", "clean": ""}
    
    deg_prompt = degradation_prompt if degradation_prompt else default_prompts["degradation"]
    cln_prompt = clean_prompt if clean_prompt else default_prompts["clean"]
    
    if content_prompt:
        deg_prompt = f"{deg_prompt} of {content_prompt}"
        cln_prompt = f"{cln_prompt} of {content_prompt}"
    
    print(f"[FlowRestore SD3 v3]")
    print(f"  Degradation: '{deg_prompt}' (guidance={degradation_guidance})")
    print(f"  Clean: '{cln_prompt}' (guidance={clean_guidance})")
    
    # ===== 2. Encode Prompts =====
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    
    # Degradation prompt encoding
    pipe._guidance_scale = degradation_guidance
    (
        deg_prompt_embeds, deg_neg_embeds,
        deg_pooled_embeds, deg_neg_pooled_embeds,
    ) = pipe.encode_prompt(
        prompt=deg_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True, device=device,
    )
    deg_combined_embeds = torch.cat([deg_neg_embeds, deg_prompt_embeds], dim=0)
    deg_combined_pooled = torch.cat([deg_neg_pooled_embeds, deg_pooled_embeds], dim=0)
    
    # Clean prompt encoding
    pipe._guidance_scale = clean_guidance
    (
        cln_prompt_embeds, cln_neg_embeds,
        cln_pooled_embeds, cln_neg_pooled_embeds,
    ) = pipe.encode_prompt(
        prompt=cln_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True, device=device,
    )
    cln_combined_embeds = torch.cat([cln_neg_embeds, cln_prompt_embeds], dim=0)
    cln_combined_pooled = torch.cat([cln_neg_pooled_embeds, cln_pooled_embeds], dim=0)
    
    # ===== 3. Initialize =====
    zt_edit = x_lq.clone()  # Start from LQ image
    prev_z0_hat = None
    
    # ===== 4. Main Loop with Trajectory Coupling =====
    iterator = tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowRestore SD3")
    
    for i, t in iterator:
        # Skip early steps (high noise levels)
        if T_steps - i > n_max:
            continue
        
        # Time calculation
        t_i = t / 1000.0
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000.0
        else:
            t_im1 = torch.zeros_like(t_i).to(device)
        dt = t_im1 - t_i
        
        if T_steps - i > n_min:
            # ===== CORE: Trajectory Coupling (from FlowEdit) =====
            V_restore_avg = torch.zeros_like(x_lq)
            
            for k in range(n_avg):
                # 1. Add forward noise to LQ image
                fwd_noise = torch.randn_like(x_lq).to(device)
                zt_lq = (1 - t_i) * x_lq + t_i * fwd_noise
                
                # 2. Couple HQ trajectory to LQ trajectory
                zt_hq = zt_edit + (zt_lq - x_lq)
                
                # 3. Compute V_degradation at zt_lq
                lq_input = torch.cat([zt_lq, zt_lq])
                V_deg = calc_v_sd3(
                    pipe, lq_input,
                    deg_combined_embeds, deg_combined_pooled,
                    degradation_guidance, t
                )
                
                # 4. Compute V_clean at zt_hq
                hq_input = torch.cat([zt_hq, zt_hq])
                V_cln = calc_v_sd3(
                    pipe, hq_input,
                    cln_combined_embeds, cln_combined_pooled,
                    clean_guidance, t
                )
                
                # 5. Velocity difference = restoration direction
                V_restore = V_cln - V_deg
                V_restore_avg += (1 / n_avg) * V_restore
            
            # Monitor
            z0_hat = zt_edit - t_i * V_restore_avg
            
            # Early stopping check
            if adaptive_tolerance > 0 and i >= min_inference_steps:
                if prev_z0_hat is not None:
                    diff = torch.mean((z0_hat.float() - prev_z0_hat.float()) ** 2).item()
                    iterator.set_postfix({"diff": f"{diff:.2e}"})
                    if diff < adaptive_tolerance:
                        print(f"\n[Early Stop] step {i}/{T_steps}")
                        return z0_hat
                prev_z0_hat = z0_hat.clone()
            
            # Optional regularization
            if reg_scale > 0:
                with torch.enable_grad():
                    zt_temp = zt_edit.detach().requires_grad_(True)
                    z0_temp = zt_temp - t_i * V_restore_avg
                    target_size = (x_lq.shape[2], x_lq.shape[3])
                    z0_down = F.interpolate(z0_temp, size=target_size, mode='area')
                    loss_reg = F.mse_loss(z0_down, x_lq)
                    reg_grad = torch.autograd.grad(loss_reg, zt_temp)[0]
                correction = -reg_scale * reg_grad
                zt_edit = zt_edit + dt * (V_restore_avg + correction)
            else:
                zt_edit = zt_edit + dt * V_restore_avg
                
        else:
            # Last n_min steps: standard sampling with clean prompt
            if i == T_steps - n_min:
                # Initialize SDEdit-style phase
                fwd_noise = torch.randn_like(x_lq).to(device)
                xt_lq = scale_noise(scheduler, x_lq, t, noise=fwd_noise)
                xt_hq = zt_edit + xt_lq - x_lq
            
            hq_input = torch.cat([xt_hq, xt_hq])
            V_cln = calc_v_sd3(
                pipe, hq_input,
                cln_combined_embeds, cln_combined_pooled,
                clean_guidance, t
            )
            
            xt_hq = xt_hq + dt * V_cln
    
    return zt_edit if n_min == 0 else xt_hq


# =========================================================================
# FLUX Restoration (Corrected)
# =========================================================================

@torch.no_grad()
def FlowRestoreFLUX(
    pipe,
    scheduler,
    x_lq,
    # ===== Prompt Settings =====
    task: str = "sr",
    degradation_prompt: str = None,
    clean_prompt: str = None,
    content_prompt: str = "",
    # ===== Guidance Scales =====
    degradation_guidance: float = 1.5,    # Lower for degradation
    clean_guidance: float = 2.5,          # Higher for clean (stronger push)
    # ===== Sampling Settings =====
    T_steps: int = 28,
    n_avg: int = 1,
    n_min: int = 0,
    n_max: int = 24,                      # Process more steps
    # ===== Regularization =====
    reg_scale: float = 0.0,
    # ===== Early Stopping =====
    adaptive_tolerance: float = 0.001,
    min_inference_steps: int = 10,
):
    """
    FlowRestore for FLUX with proper trajectory coupling.
    """
    device = x_lq.device
    dtype = x_lq.dtype
    
    # ===== 1. Setup Prompts =====
    if task == "sr":
        default_prompts = SR_PROMPTS
    elif task == "denoise":
        default_prompts = DENOISE_PROMPTS
    elif task == "jpeg":
        default_prompts = JPEG_PROMPTS
    else:
        default_prompts = {"degradation": "", "clean": ""}
    
    deg_prompt = degradation_prompt if degradation_prompt else default_prompts["degradation"]
    cln_prompt = clean_prompt if clean_prompt else default_prompts["clean"]
    
    if content_prompt:
        deg_prompt = f"{deg_prompt} of {content_prompt}"
        cln_prompt = f"{cln_prompt} of {content_prompt}"
    
    print(f"[FlowRestore FLUX v3]")
    print(f"  Degradation: '{deg_prompt}' (guidance={degradation_guidance})")
    print(f"  Clean: '{cln_prompt}' (guidance={clean_guidance})")
    
    # ===== 2. Pack Latents =====
    num_channels = x_lq.shape[1]
    x_lq_packed = pipe._pack_latents(x_lq, x_lq.shape[0], num_channels, x_lq.shape[2], x_lq.shape[3])
    
    # ===== 3. Prepare Timesteps =====
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_lq_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None, sigmas=sigmas, mu=mu,
    )
    pipe._num_timesteps = len(timesteps)
    
    # ===== 4. Encode Prompts =====
    (deg_prompt_embeds, deg_pooled_embeds, deg_text_ids) = pipe.encode_prompt(
        prompt=deg_prompt, prompt_2=None, device=device
    )
    (cln_prompt_embeds, cln_pooled_embeds, cln_text_ids) = pipe.encode_prompt(
        prompt=cln_prompt, prompt_2=None, device=device
    )
    
    # ===== 5. Guidance Tensors =====
    if pipe.transformer.config.guidance_embeds:
        deg_guidance = torch.tensor([degradation_guidance], device=device).expand(x_lq_packed.shape[0])
        cln_guidance = torch.tensor([clean_guidance], device=device).expand(x_lq_packed.shape[0])
    else:
        deg_guidance = None
        cln_guidance = None
    
    # ===== 6. Image IDs =====
    latent_image_ids = pipe._prepare_latent_image_ids(
        x_lq.shape[0], x_lq.shape[2], x_lq.shape[3], device, dtype
    )
    
    # ===== 7. Initialize =====
    zt_edit = x_lq_packed.clone()
    prev_z0_hat = None
    
    orig_h = x_lq.shape[2] * pipe.vae_scale_factor
    orig_w = x_lq.shape[3] * pipe.vae_scale_factor
    
    # ===== 8. Main Loop with Trajectory Coupling =====
    iterator = tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowRestore FLUX")
    
    for i, t in iterator:
        if T_steps - i > n_max:
            continue
        
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i + 1 < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i
        dt = t_im1 - t_i
        
        if T_steps - i > n_min:
            # ===== CORE: Trajectory Coupling =====
            V_restore_avg = torch.zeros_like(x_lq_packed)
            
            for k in range(n_avg):
                # 1. Add forward noise to LQ
                fwd_noise = torch.randn_like(x_lq_packed).to(device)
                zt_lq = (1 - t_i) * x_lq_packed + t_i * fwd_noise
                
                # 2. Couple HQ trajectory
                zt_hq = zt_edit + (zt_lq - x_lq_packed)
                
                # 3. V_degradation at zt_lq
                V_deg = calc_v_flux(
                    pipe, zt_lq,
                    deg_prompt_embeds, deg_pooled_embeds,
                    deg_guidance, deg_text_ids, latent_image_ids, t
                )
                
                # 4. V_clean at zt_hq
                V_cln = calc_v_flux(
                    pipe, zt_hq,
                    cln_prompt_embeds, cln_pooled_embeds,
                    cln_guidance, cln_text_ids, latent_image_ids, t
                )
                
                # 5. Restoration direction
                V_restore = V_cln - V_deg
                V_restore_avg += (1 / n_avg) * V_restore
            
            # Monitor
            z0_hat = zt_edit - t_i * V_restore_avg
            
            if adaptive_tolerance > 0 and i >= min_inference_steps:
                if prev_z0_hat is not None:
                    diff = torch.mean((z0_hat - prev_z0_hat) ** 2).item()
                    iterator.set_postfix({"diff": f"{diff:.2e}"})
                    if diff < adaptive_tolerance:
                        print(f"\n[Early Stop] step {i}/{T_steps}")
                        return unpack_latents_flux_manual(z0_hat, orig_h, orig_w, pipe.vae_scale_factor)
                prev_z0_hat = z0_hat.clone()
            
            # Regularization
            if reg_scale > 0:
                with torch.enable_grad():
                    zt_temp = zt_edit.detach().requires_grad_(True)
                    z0_temp_packed = zt_temp - t_i * V_restore_avg
                    z0_temp_spatial = unpack_latents_flux_manual(
                        z0_temp_packed, orig_h, orig_w, pipe.vae_scale_factor
                    )
                    target_size = (x_lq.shape[2], x_lq.shape[3])
                    z0_down = F.interpolate(z0_temp_spatial, size=target_size, mode='area')
                    loss_reg = F.mse_loss(z0_down, x_lq)
                    reg_grad = torch.autograd.grad(loss_reg, zt_temp)[0]
                correction = -reg_scale * reg_grad
                zt_edit = zt_edit + dt * (V_restore_avg + correction)
            else:
                zt_edit = zt_edit + dt * V_restore_avg
                
        else:
            # Last n_min steps
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_lq_packed).to(device)
                xt_lq = scale_noise(scheduler, x_lq_packed, t, noise=fwd_noise)
                xt_hq = zt_edit + xt_lq - x_lq_packed
            
            V_cln = calc_v_flux(
                pipe, xt_hq,
                cln_prompt_embeds, cln_pooled_embeds,
                cln_guidance, cln_text_ids, latent_image_ids, t
            )
            xt_hq = xt_hq + dt * V_cln
    
    out = zt_edit if n_min == 0 else xt_hq
    return unpack_latents_flux_manual(out, orig_h, orig_w, pipe.vae_scale_factor)


# =========================================================================
# Convenience Wrappers
# =========================================================================

def FlowRestoreSR_SD3(pipe, scheduler, x_lq, content_prompt="", **kwargs):
    return FlowRestoreSD3(pipe, scheduler, x_lq, task="sr", content_prompt=content_prompt, **kwargs)

def FlowRestoreSR_FLUX(pipe, scheduler, x_lq, content_prompt="", **kwargs):
    return FlowRestoreFLUX(pipe, scheduler, x_lq, task="sr", content_prompt=content_prompt, **kwargs)

def FlowRestoreDenoise_SD3(pipe, scheduler, x_lq, content_prompt="", **kwargs):
    return FlowRestoreSD3(pipe, scheduler, x_lq, task="denoise", content_prompt=content_prompt, **kwargs)

def FlowRestoreDenoise_FLUX(pipe, scheduler, x_lq, content_prompt="", **kwargs):
    return FlowRestoreFLUX(pipe, scheduler, x_lq, task="denoise", content_prompt=content_prompt, **kwargs)