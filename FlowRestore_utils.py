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

def unpack_latents_flux_manual(latents, height, width, vae_scale_factor):
    """
    手动实现 FLUX 的 Unpack 逻辑，避免 diffusers 版本差异导致的尺寸报错。
    将 Packed Latents [B, Seq, C_packed] 还原为 Spatial Latents [B, C, H, W]
    """
    batch_size, seq_len, channels_packed = latents.shape
    
    # 计算 Latent 的实际尺寸 (H, W)
    h_lat = height // vae_scale_factor
    w_lat = width // vae_scale_factor
    
    # FLUX Pack 操作会将 2x2 的像素块折叠进通道
    # 所以 Grid 的尺寸是 Latent 尺寸的一半
    h_grid = h_lat // 2
    w_grid = w_lat // 2
    
    # 检查通道数 (通常是 16 * 4 = 64)
    channels = channels_packed // 4
    
    # 1. View 还原为 Grid [B, H/2, W/2, C, 2, 2]
    latents = latents.view(batch_size, h_grid, w_grid, channels, 2, 2)
    
    # 2. Permute 调整维度顺序 [B, C, H/2, 2, W/2, 2]
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    
    # 3. Reshape 合并空间维度 [B, C, H, W]
    latents = latents.reshape(batch_size, channels, h_lat, w_lat)
    
    return latents

# =========================================================================
# V-Prediction Wrappers
# =========================================================================

def calc_v_sd3(pipe, latent_model_input, prompt_embeds, pooled_prompt_embeds, guidance_scale, t):
    """SD3 velocity prediction with CFG"""
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
# Main SR Restoration Functions
# =========================================================================

@torch.no_grad()
def FlowRestoreSR_SD3(
    pipe,
    scheduler,
    x_lq,                      
    content_prompt="",         # 图像内容描述 (可选)
    negative_prompt="",        # 负面prompt
    T_steps: int = 50,
    n_avg: int = 1,
    guidance_scale: float = 3.5,       
    n_min: int = 0,
    n_max: int = 35,
    reg_scale: float = 300.0,  
    adaptive_tolerance: float = 0.0001, 
    min_inference_steps: int = 15,
    use_null_prompt: bool = False,
    noise_scale: float = 1.0,
):
    """
    Super-Resolution专用 FlowRestore (SD3版本)
    """
    device = x_lq.device
    
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    
    # ===== Prompt 策略 =====
    if use_null_prompt:
        prompt = ""
        negative_prompt = ""
    else:
        if content_prompt:
            prompt = f"a high resolution, sharp, detailed photo of {content_prompt}"
        else:
            prompt = "high resolution, sharp, detailed, professional photography"
        if not negative_prompt:
            negative_prompt = "blurry, low resolution, pixelated, low quality"
    
    pipe._guidance_scale = guidance_scale
    (
        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None, 
        negative_prompt=negative_prompt, do_classifier_free_guidance=True, device=device,
    )
    
    combined_prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
    combined_pooled_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    
    # 初始化
    zt_edit = x_lq.clone()
    target_size = (x_lq.shape[2], x_lq.shape[3])
    prev_z0_hat = None

    iterator = tqdm(enumerate(timesteps), total=len(timesteps), desc="SR FlowRestore (SD3)")
    
    for i, t in iterator:
        if T_steps - i > n_max: 
            continue
        
        # 计算时间步
        t_i = t / 1000.0
        if i + 1 < len(timesteps):
            t_im1 = (timesteps[i+1]) / 1000.0
        else:
            t_im1 = torch.zeros_like(t_i).to(device)
            
        dt = t_im1 - t_i

        if T_steps - i > n_min:
            # ===== Multi-averaging for stability =====
            V_restore_avg = torch.zeros_like(x_lq)
            
            for k in range(n_avg):
                # 添加noise用于探索 (可选，SR任务中n_avg=1时这部分可以简化)
                if n_avg > 1:
                    fwd_noise = torch.randn_like(x_lq).to(device) * noise_scale
                    zt_noised = (1 - t_i) * zt_edit + t_i * fwd_noise
                else:
                    zt_noised = zt_edit
                
                # 预测velocity
                latent_input = torch.cat([zt_noised, zt_noised])
                
                Vt_restore = calc_v_sd3(
                    pipe, latent_input, 
                    combined_prompt_embeds, combined_pooled_embeds, 
                    guidance_scale, t
                )
                
                V_restore_avg += (1/n_avg) * Vt_restore

            # 预测clean latent
            z0_hat = zt_edit - t_i * V_restore_avg
            
            # ===== Adaptive Early Stopping =====
            if adaptive_tolerance > 0 and i >= min_inference_steps:
                if prev_z0_hat is not None:
                    diff = torch.mean((z0_hat.float() - prev_z0_hat.float()) ** 2).item()
                    iterator.set_postfix({"Diff": f"{diff:.2e}", "Reg": f"{reg_scale:.1f}"})
                    if diff < adaptive_tolerance:
                        print(f"\n[Adaptive Stop] Converged at step {i}/{T_steps}")
                        return z0_hat
                prev_z0_hat = z0_hat.clone()
            
            # ===== Data Fidelity Regularization =====
            # 关键修复: 使用torch.enable_grad()上下文管理器
            if reg_scale > 0:
                with torch.enable_grad():
                    # 创建需要梯度的临时变量
                    zt_temp = zt_edit.detach().requires_grad_(True)
                    z0_temp = zt_temp - t_i * V_restore_avg
                    
                    # Downsample到LR尺寸
                    z0_downsampled = F.interpolate(z0_temp, size=target_size, mode='area')
                    
                    # 计算loss
                    loss_reg = F.mse_loss(z0_downsampled, x_lq)
                    
                    # 计算梯度
                    reg_grad = torch.autograd.grad(loss_reg, zt_temp)[0]
                
                # 更新
                correction = - reg_scale * reg_grad
                zt_edit = zt_edit + dt * (V_restore_avg + correction)
            else:
                # 没有regularization时直接更新
                zt_edit = zt_edit + dt * V_restore_avg
            
        else:
            # Last n_min steps: regular ODE sampling
            latent_input = torch.cat([zt_edit, zt_edit])
            Vt_restore = calc_v_sd3(
                pipe, latent_input, 
                combined_prompt_embeds, combined_pooled_embeds, 
                guidance_scale, t
            )
            zt_edit = zt_edit + dt * Vt_restore

    return zt_edit


@torch.no_grad()
def FlowRestoreSR_FLUX(
    pipe,
    scheduler,
    x_lq,                      
    content_prompt="",
    T_steps: int = 28,
    n_avg: int = 1,
    guidance_scale: float = 3.5,
    n_min: int = 0,
    n_max: int = 22,
    reg_scale: float = 300.0,
    adaptive_tolerance: float = 0.0001,
    min_inference_steps: int = 10,
    use_null_prompt: bool = False,
    noise_scale: float = 1.0,
):
    """
    Super-Resolution专用 FlowRestore (FLUX版本)
    """
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
    if use_null_prompt:
        prompt = ""
    else:
        if content_prompt:
            prompt = f"a high resolution, sharp, detailed photo of {content_prompt}"
        else:
            prompt = "high resolution, sharp, detailed, professional photography"
    
    (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=device
    )

    # 4. Guidance
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.tensor([guidance_scale], device=device).expand(x_lq_packed.shape[0])
    else:
        guidance = None

    # 5. Image IDs
    latent_image_ids = pipe._prepare_latent_image_ids(
        x_lq.shape[0], x_lq.shape[2], x_lq.shape[3], device, dtype
    )

    # 6. Main Loop
    zt_edit = x_lq_packed.clone()
    prev_z0_hat = None

    iterator = tqdm(enumerate(timesteps), total=len(timesteps), desc="SR FlowRestore (FLUX)")
    
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
            # ===== Multi-averaging =====
            V_restore_avg = torch.zeros_like(x_lq_packed)
            
            for k in range(n_avg):
                # 添加noise (SR任务n_avg=1时可简化)
                if n_avg > 1:
                    fwd_noise = torch.randn_like(x_lq_packed).to(device) * noise_scale
                    zt_noised = (1 - t_i) * zt_edit + t_i * fwd_noise
                else:
                    zt_noised = zt_edit
                
                # 预测velocity
                Vt_restore = calc_v_flux(
                    pipe, zt_noised, prompt_embeds, pooled_prompt_embeds, 
                    guidance, text_ids, latent_image_ids, t
                )
                
                V_restore_avg += (1/n_avg) * Vt_restore

            # 预测x0
            z0_hat_packed = zt_edit - t_i * V_restore_avg
            
            # ===== Adaptive Stopping =====
            if adaptive_tolerance > 0 and i >= min_inference_steps:
                if prev_z0_hat is not None:
                    diff = torch.mean((z0_hat_packed - prev_z0_hat) ** 2).item()
                    iterator.set_postfix({"Diff": f"{diff:.2e}", "Reg": f"{reg_scale:.1f}"})
                    if diff < adaptive_tolerance:
                        print(f"\n[Adaptive Stop] Converged at step {i}/{T_steps}")
                        return unpack_latents_flux_manual(
                            z0_hat_packed, 
                            x_lq.shape[2] * pipe.vae_scale_factor, 
                            x_lq.shape[3] * pipe.vae_scale_factor, 
                            pipe.vae_scale_factor
                        )
                prev_z0_hat = z0_hat_packed.clone()

            # ===== Data Fidelity Regularization =====
            # 关键修复: 使用torch.enable_grad()
            if reg_scale > 0:
                with torch.enable_grad():
                    # 创建需要梯度的临时变量
                    zt_temp = zt_edit.detach().requires_grad_(True)
                    z0_temp_packed = zt_temp - t_i * V_restore_avg
                    
                    # Unpack到spatial domain
                    z0_temp_spatial = unpack_latents_flux_manual(
                        z0_temp_packed, 
                        x_lq.shape[2] * pipe.vae_scale_factor, 
                        x_lq.shape[3] * pipe.vae_scale_factor, 
                        pipe.vae_scale_factor
                    )
                    
                    # Downsample到LR尺寸
                    target_size = (x_lq.shape[2], x_lq.shape[3])
                    z0_downsampled = F.interpolate(z0_temp_spatial, size=target_size, mode='area')
                    
                    # 计算loss
                    loss_reg = F.mse_loss(z0_downsampled, x_lq)
                    
                    # 计算梯度
                    reg_grad = torch.autograd.grad(loss_reg, zt_temp)[0]
                
                # 更新
                correction = - reg_scale * reg_grad
                zt_edit = zt_edit + dt * (V_restore_avg + correction)
            else:
                # 没有regularization时直接更新
                zt_edit = zt_edit + dt * V_restore_avg
            
        else:
            # Last n_min steps: regular sampling
            Vt_restore = calc_v_flux(
                pipe, zt_edit, prompt_embeds, pooled_prompt_embeds, 
                guidance, text_ids, latent_image_ids, t
            )
            zt_edit = zt_edit + dt * Vt_restore

    # Final Unpack
    x_out_spatial = unpack_latents_flux_manual(
        zt_edit, 
        x_lq.shape[2] * pipe.vae_scale_factor, 
        x_lq.shape[3] * pipe.vae_scale_factor, 
        pipe.vae_scale_factor
    )
    
    return x_out_spatial