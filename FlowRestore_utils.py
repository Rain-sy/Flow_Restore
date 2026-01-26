from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

# =========================================================================
# Helper Functions
# =========================================================================

def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """Forward process in flow-matching (SD3 specific)"""
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample
    return sample

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    """Shift calculation for FLUX scheduler"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def default_degradation_fn(z0_hat, target_size=None):
    """
    Default Downsampling operator.
    Uses Area interpolation for better gradient stability.
    """
    if target_size is None:
        return z0_hat
    return F.interpolate(z0_hat, size=target_size, mode='area')

def unpack_latents_flux_manual(latents, height, width, vae_scale_factor):
    """
    手动实现 FLUX 的 Unpack 逻辑，避免 diffusers 版本差异导致的尺寸报错。
    将 Packed Latents [B, Seq, C_packed] 还原为 Spatial Latents [B, C, H, W]
    """
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

# =========================================================================
# V-Prediction Wrappers
# =========================================================================

def calc_v_sd3(pipe, latent_model_input, prompt_embeds, pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    timestep = t.expand(latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred.chunk(4)
        
        noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
        noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar

def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
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
# Main Restoration Functions
# =========================================================================

# @torch.no_grad()
def FlowRestoreSD3(
    pipe,
    scheduler,
    x_lq,                      
    degradation_prompt,        
    clean_prompt="",           
    negative_prompt="",        
    T_steps: int = 50,
    n_avg: int = 0,
    degradation_guidance_scale: float = 2.5, 
    clean_guidance_scale: float = 2.5,       
    n_min: int = 0,
    n_max: int = 24,
    reg_scale: float = 0.0,  
    degradation_fn = None,      
    adaptive_tolerance: float = 0.0001, 
    min_inference_steps: int = 15
):
    device = x_lq.device
    
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    
    pipe._guidance_scale = degradation_guidance_scale
    (
        deg_prompt_embeds, deg_neg_prompt_embeds, deg_pooled_prompt_embeds, deg_neg_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=degradation_prompt, prompt_2=None, prompt_3=None, 
        negative_prompt=negative_prompt, do_classifier_free_guidance=True, device=device,
    )

    pipe._guidance_scale = clean_guidance_scale
    (
        clean_prompt_embeds, clean_neg_prompt_embeds, clean_pooled_prompt_embeds, clean_neg_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=clean_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt, do_classifier_free_guidance=True, device=device,
    )
    
    combined_prompt_embeds = torch.cat([deg_neg_prompt_embeds, deg_prompt_embeds, clean_neg_prompt_embeds, clean_prompt_embeds], dim=0)
    combined_pooled_embeds = torch.cat([deg_neg_pooled_prompt_embeds, deg_pooled_prompt_embeds, clean_neg_pooled_prompt_embeds, clean_pooled_prompt_embeds], dim=0)
    
    pipe._guidance_scale = 1.5 
    
    zt_edit = x_lq.clone()
    # 原始 Latent 的高宽
    original_size = (x_lq.shape[2], x_lq.shape[3]) 
    prev_z0_hat = None

    iterator = tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowRestoring SD3")
    
    for i, t in iterator:
        if T_steps - i > n_max: continue
        
        t_i = t / 1000.0
        if i + 1 < len(timesteps):
            t_im1 = (timesteps[i+1]) / 1000.0
        else:
            t_im1 = torch.zeros_like(t_i).to(device)
            
        dt = t_im1 - t_i

        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_lq)
            zt_edit_grad = zt_edit.detach().requires_grad_(True)
            
            # --- Average Guidance Calculation ---
            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_lq).to(device)
                zt_src = (1 - t_i) * x_lq + t_i * fwd_noise
                zt_tar = zt_edit_grad + (zt_src - x_lq) 
                
                latent_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                
                Vt_deg, Vt_clean = calc_v_sd3(
                    pipe, latent_input, 
                    combined_prompt_embeds, combined_pooled_embeds, 
                    degradation_guidance_scale, clean_guidance_scale, t
                )
                
                V_delta_avg += (1/n_avg) * (Vt_clean - Vt_deg)

            # --- z0 estimation for Regularization ---
            z0_hat = zt_edit_grad - t_i * V_delta_avg
            
            # Adaptive Stopping Logic
            if adaptive_tolerance > 0 and i >= min_inference_steps:
                if prev_z0_hat is not None:
                    diff = torch.mean((z0_hat.detach().float() - prev_z0_hat.float()) ** 2).item()
                    iterator.set_postfix({"Diff": f"{diff:.6e}"})
                    if diff < adaptive_tolerance:
                        print(f"\n[Adaptive Stop] Converged at step {i}/{T_steps}")
                        return z0_hat.detach()
                prev_z0_hat = z0_hat.detach().clone()
            
            # --- Regularization Gradient ---
            reg_grad = torch.zeros_like(zt_edit)
            if reg_scale > 0:
                # 1. 对预测结果应用退化函数
                if degradation_fn is None:
                    z0_degraded = z0_hat
                else:
                    # 尝试调用，优先支持不带 target_size 的调用（闭包）
                    try:
                        z0_degraded = degradation_fn(z0_hat)
                    except TypeError:
                        # 兼容旧接口
                        z0_degraded = degradation_fn(z0_hat, original_size)
                
                # 2. 准备 Loss Target (Smart Consistency Check)
                # 如果 z0_degraded 尺寸变小了 (SR任务)，则也必须将参考图 x_lq 下采样，
                # 确保比较的是 Low-Res vs Low-Res
                if z0_degraded.shape != x_lq.shape:
                    with torch.no_grad():
                        if degradation_fn is None:
                            # 理论上不该进这里，但作为 fallback
                            target_degraded = F.interpolate(x_lq.detach(), size=z0_degraded.shape[2:], mode='area')
                        else:
                            try:
                                target_degraded = degradation_fn(x_lq.detach())
                            except TypeError:
                                target_degraded = degradation_fn(x_lq.detach(), original_size)
                else:
                    target_degraded = x_lq.detach()
                
                # 3. Calculate Loss & Grad
                loss_reg = F.mse_loss(z0_degraded, target_degraded)
                reg_grad = torch.autograd.grad(loss_reg, zt_edit_grad)[0]

            zt_edit = zt_edit.detach()
            correction = - reg_scale * reg_grad
            zt_edit = zt_edit + dt * (V_delta_avg + correction)
            
        else:
            pass

    return zt_edit


# @torch.no_grad()
def FlowRestoreFLUX(
    pipe,
    scheduler,
    x_lq,                      # Input Latent [B, C, H, W]
    degradation_prompt,
    clean_prompt="",
    T_steps: int = 30,
    n_avg: int = 1,
    degradation_guidance_scale: float = 3.5,
    clean_guidance_scale: float = 5.5,
    n_min: int = 3,
    n_max: int = 24,
    reg_scale: float = 0.0,
    degradation_fn = None,
    adaptive_tolerance: float = 0.0,
    min_inference_steps: int = 10
):
    device = x_lq.device
    dtype = x_lq.dtype
    
    # 1. Pack Latents
    num_channels_latents = x_lq.shape[1]
    x_lq_packed = pipe._pack_latents(x_lq, x_lq.shape[0], num_channels_latents, x_lq.shape[2], x_lq.shape[3])
    
    # 2. Prepare Timesteps
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

    # 3. Encode Prompts
    (deg_prompt_embeds, deg_pooled_prompt_embeds, deg_text_ids) = pipe.encode_prompt(
        prompt=degradation_prompt, prompt_2=None, device=device
    )
    (clean_prompt_embeds, clean_pooled_prompt_embeds, clean_text_ids) = pipe.encode_prompt(
        prompt=clean_prompt, prompt_2=None, device=device
    )

    # 4. Guidance
    if pipe.transformer.config.guidance_embeds:
        deg_guidance = torch.tensor([degradation_guidance_scale], device=device).expand(x_lq_packed.shape[0])
        clean_guidance = torch.tensor([clean_guidance_scale], device=device).expand(x_lq_packed.shape[0])
    else:
        deg_guidance = None
        clean_guidance = None

    # 5. Image IDs
    latent_image_ids = pipe._prepare_latent_image_ids(
        x_lq.shape[0], x_lq.shape[2], x_lq.shape[3], device, dtype
    )

    # 6. Initialize Loop
    zt_edit = x_lq_packed.clone()
    prev_z0_hat = None

    iterator = tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowRestoring FLUX")
    
    for i, t in iterator:
        if T_steps - i > n_max: continue
        
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i + 1 < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i
        dt = t_im1 - t_i

        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_lq_packed)
            zt_edit_grad = zt_edit.detach().requires_grad_(True)
            
            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_lq_packed).to(device)
                zt_src = (1 - t_i) * x_lq_packed + t_i * fwd_noise
                zt_tar = zt_edit_grad
                
                Vt_deg = calc_v_flux(
                    pipe, zt_src, deg_prompt_embeds, deg_pooled_prompt_embeds, 
                    deg_guidance, deg_text_ids, latent_image_ids, t
                )
                Vt_clean = calc_v_flux(
                    pipe, zt_tar, clean_prompt_embeds, clean_pooled_prompt_embeds, 
                    clean_guidance, clean_text_ids, latent_image_ids, t
                )
                
                V_delta_avg += (1/n_avg) * (Vt_clean - Vt_deg)

            # --- Regularization & Stop Check ---
            z0_hat_packed = zt_edit_grad - t_i * V_delta_avg
            
            # Adaptive Stop
            if adaptive_tolerance > 0 and i >= min_inference_steps:
                if prev_z0_hat is not None:
                    diff = torch.mean((z0_hat_packed.detach() - prev_z0_hat) ** 2).item()
                    iterator.set_postfix({"Diff": f"{diff:.6e}"})
                    if diff < adaptive_tolerance:
                        print(f"\n[Adaptive Stop] Converged at step {i}/{T_steps}")
                        return unpack_latents_flux_manual(
                            z0_hat_packed.detach(), 
                            x_lq.shape[2] * pipe.vae_scale_factor, 
                            x_lq.shape[3] * pipe.vae_scale_factor, 
                            pipe.vae_scale_factor
                        )
                prev_z0_hat = z0_hat_packed.detach().clone()

            # --- Regularization ---
            reg_grad = torch.zeros_like(zt_edit)
            if reg_scale > 0:
                # 为了做 Spatial Degradation，必须先 Unpack 回 [B,C,H,W]
                z0_hat_spatial = unpack_latents_flux_manual(
                    z0_hat_packed, 
                    x_lq.shape[2] * pipe.vae_scale_factor, 
                    x_lq.shape[3] * pipe.vae_scale_factor, 
                    pipe.vae_scale_factor
                )
                
                # 1. 应用退化函数
                if degradation_fn is None:
                    z0_degraded = z0_hat_spatial
                else:
                    try:
                        z0_degraded = degradation_fn(z0_hat_spatial)
                    except TypeError:
                        z0_degraded = degradation_fn(z0_hat_spatial, (z0_hat_spatial.shape[2], z0_hat_spatial.shape[3]))
                
                # 2. 准备 Loss Target (Smart Consistency Check for FLUX)
                # 注意：x_lq 是 Spatial [B,C,H,W], zt_edit_grad 是 Packed
                # 比较必须在 Spatial 维度或 Degraded Spatial 维度进行
                
                if z0_degraded.shape != x_lq.shape:
                    # SR Case: Downsample the reference (x_lq) to match
                    with torch.no_grad():
                        if degradation_fn is None:
                            target_degraded = F.interpolate(x_lq.detach(), size=z0_degraded.shape[2:], mode='area')
                        else:
                            try:
                                target_degraded = degradation_fn(x_lq.detach())
                            except TypeError:
                                target_degraded = degradation_fn(x_lq.detach(), (x_lq.shape[2], x_lq.shape[3]))
                else:
                    target_degraded = x_lq.detach()

                loss_reg = F.mse_loss(z0_degraded, target_degraded)
                
                # 求导，并链式法则回传到 packed latent (自动微分处理)
                reg_grad = torch.autograd.grad(loss_reg, zt_edit_grad)[0]

            zt_edit = zt_edit.detach()
            correction = - reg_scale * reg_grad
            zt_edit = zt_edit + dt * (V_delta_avg + correction)
            
        else:
            Vt_clean = calc_v_flux(
                pipe, zt_edit, clean_prompt_embeds, clean_pooled_prompt_embeds, 
                clean_guidance, clean_text_ids, latent_image_ids, t
            )
            zt_edit = zt_edit + dt * Vt_clean

    x_out_spatial = unpack_latents_flux_manual(
        zt_edit, 
        x_lq.shape[2] * pipe.vae_scale_factor, 
        x_lq.shape[3] * pipe.vae_scale_factor, 
        pipe.vae_scale_factor
    )
    
    return x_out_spatial