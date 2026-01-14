import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

# 复用原本的 calc_v_sd3，不需要修改它
from FlowEdit_utils import calc_v_sd3

def FlowRestoreSD3(
    pipe,
    scheduler,
    x_lq,                      # 输入的低质量图像 (Latent)
    degradation_prompt,        # 描述退化的Prompt，例如 "Blurry, noise, low quality"
    clean_prompt="",           # 描述目标的Prompt，通常留空 ("") 或者写 "High quality"
    negative_prompt="",        # 通用负面词，例如 "Artifacts"
    T_steps: int = 50,
    n_avg: int = 1,
    degradation_guidance_scale: float = 3.5, # 控制“减去噪音”的力度 (对应原本的 src_guidance)
    clean_guidance_scale: float = 1.0,       # 控制“趋向清晰”的力度，如果 clean_prompt为空，建议设为1.0
    n_min: int = 0,
    n_max: int = 50,           # Restoration 通常建议跑完全程
    # --- 新增：Regularization 参数 ---
    reg_scale: float = 200.0,  # 正则化力度 (Lambda)。越大越强行保持原图结构，太大会导致伪影
    degradation_fn = None      # 退化算子。如果为None，默认为Identity (适用于Denoising)
):
    """
    FlowRestore: Based on FlowEdit, optimized for Image Restoration with Regularization.
    
    Logic: 
    1. Flow Direction = V(Clean) - V(Degradation)
    2. Regularization = || Degrade(Estimated_Z0) - Input_LQ ||^2
    """
    
    # 1. 准备 Latents 和 Timesteps
    device = x_lq.device
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    
    # 2. Encode Prompts (Feature Extraction)
    # Source (Degradation) -> 对应原本 FlowEdit 的 src
    pipe._guidance_scale = degradation_guidance_scale
    (
        deg_prompt_embeds,
        deg_neg_prompt_embeds,
        deg_pooled_prompt_embeds,
        deg_neg_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=degradation_prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # Target (Clean) -> 对应原本 FlowEdit 的 tar
    pipe._guidance_scale = clean_guidance_scale
    (
        clean_prompt_embeds,
        clean_neg_prompt_embeds,
        clean_pooled_prompt_embeds,
        clean_neg_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=clean_prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
    
    # 拼接 Embeddings (用于 Batch 计算)
    # Order: [Neg_Src, Src, Neg_Tar, Tar] -> [Neg_Deg, Deg, Neg_Clean, Clean]
    combined_prompt_embeds = torch.cat([deg_neg_prompt_embeds, deg_prompt_embeds, clean_neg_prompt_embeds, clean_prompt_embeds], dim=0)
    combined_pooled_embeds = torch.cat([deg_neg_pooled_prompt_embeds, deg_pooled_prompt_embeds, clean_neg_pooled_prompt_embeds, clean_pooled_embeds], dim=0)
    
    # 3. 初始化 ODE
    # 在 Restoration 中，我们通常从纯随机噪声开始，或者从 LQ 加噪开始
    # 这里为了配合 FlowEdit 的逻辑（保留结构），我们采用 "Stochastic Encoding" 逻辑
    # Zt_edit 初始化为 x_lq (这一点和 FlowEdit 一样，作为 ODE 的锚点)
    zt_edit = x_lq.clone()

    iterator = tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowRestoring")
    
    for i, t in iterator:
        
        # 跳过不在范围内的时间步
        if T_steps - i > n_max: continue
        
        # 获取当前时间步数值 (0~1)
        t_i = t / 1000.0
        # 获取下一个时间步 (用于计算 dt)
        if i + 1 < len(timesteps):
            t_im1 = (timesteps[i+1]) / 1000.0
        else:
            t_im1 = torch.zeros_like(t_i).to(device) # 最后一步走到 0
            
        dt = t_im1 - t_i # 注意：时间倒流，dt 是负数

        # -------------------------------------------------------
        # Part A: FlowEdit 核心 (计算编辑方向)
        # -------------------------------------------------------
        
        # 这里的逻辑和原版 FlowEdit 保持一致：
        # 1. 构造 stochastic path (zt_src)
        # 2. 计算 V_clean - V_degradation
        
        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_lq)
            
            # 开启梯度计算 (为了 Regularization)
            # 必须设置 requires_grad=True，否则无法对 zt_edit 求导
            zt_edit_grad = zt_edit.detach().requires_grad_(True)
            
            for k in range(n_avg):
                # 随机采样噪声
                fwd_noise = torch.randn_like(x_lq).to(device)
                
                # 构造 Source Path (Degraded Path)
                zt_src = (1 - t_i) * x_lq + t_i * fwd_noise
                
                # 构造 Target Path (Restored Path) - Coupling
                zt_tar = zt_edit_grad + (zt_src - x_lq)
                
                # Batch input for SD3
                latent_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar)
                
                # 计算速度场
                # Vt_deg (Src), Vt_clean (Tar)
                Vt_deg, Vt_clean = calc_v_sd3(
                    pipe, latent_input, 
                    combined_prompt_embeds, combined_pooled_embeds, 
                    degradation_guidance_scale, clean_guidance_scale, t
                )
                
                # 核心减法逻辑：Direction = V_clean - V_degradation
                V_delta_avg += (1/n_avg) * (Vt_clean - Vt_deg)

            # -------------------------------------------------------
            # Part B: Regularization (数据一致性校正)
            # -------------------------------------------------------
            reg_grad = torch.zeros_like(zt_edit)
            
            if reg_scale > 0:
                # 1. 估计当前的 Z0 (Clean Latent)
                # 根据 Rectified Flow 公式: Zt = t*Noise + (1-t)*Z0
                # 所以: dZ/dt = v = Noise - Z0  =>  Z0 = Noise - v
                # 或者更通用的: Z0 = Zt - t * v
                # 注意：这里我们用 V_delta_avg 作为当前轨迹的最佳估计速度
                
                # 我们需要使用 zt_tar 对应的速度来估计，这里简化使用 V_delta
                # 更好的做法是用 Vt_clean，但为了节省计算，我们用当前的编辑流近似
                z0_hat = zt_edit_grad - t_i * V_delta_avg
                
                # 2. 应用退化模型 (Degradation)
                if degradation_fn is None:
                    # 默认为 Identity (Denoising 任务)
                    z0_degraded = z0_hat
                else:
                    # 如果是 Deblur/SR，这里应该是 Blur(z0_hat) 或 Downsample(z0_hat)
                    z0_degraded = degradation_fn(z0_hat)
                
                # 3. 计算 Loss: || D(Z0_hat) - Input_LQ ||^2
                loss_reg = F.mse_loss(z0_degraded, x_lq.detach())
                
                # 4. 计算梯度: d(Loss)/d(zt_edit)
                # 这个梯度告诉我们：zt_edit 应该怎么动，才能让解出来的 Z0 更像原图
                reg_grad = torch.autograd.grad(loss_reg, zt_edit_grad)[0]
                
                # 打印 Loss 方便监控 (可选)
                # if i % 10 == 0: print(f"Step {i}, Reg Loss: {loss_reg.item():.6f}")

            # -------------------------------------------------------
            # Part C: 更新步 (Update Step)
            # -------------------------------------------------------
            
            # zt_next = zt + dt * v
            # 加入 Regularization: 相当于在速度场上加了一个纠正力
            # dt 是负数，我们希望 Loss 变小，梯度下降方向是 -grad
            # 所以我们把 -grad 加到 velocity 上？
            # 实际上，这相当于 Langevin Dynamics 的校正项
            
            zt_edit = zt_edit.detach() # 截断梯度，准备下一步
            
            # 更新公式：
            # 1. Flow 推力: dt * V_delta_avg
            # 2. Reg 拉力:  - reg_scale * reg_grad * abs(dt) 
            # (注意符号：我们需要让 zt 往 loss 减小的方向走)
            
            correction =  - reg_scale * reg_grad 
            
            # 更新 Latent
            zt_edit = zt_edit + dt * (V_delta_avg + correction)
            
        else:
            # 最后的步骤 (通常用于 Refine)
            # 这里简化处理，可以保留 Regularization 也可以去掉
            # 为了保持一致性，建议去掉 Reg，只做最后的小修补
            # ... (保留原本 FlowEdit 的最后几步逻辑，或者直接跳过)
            pass

    return zt_edit