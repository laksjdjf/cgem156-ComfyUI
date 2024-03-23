from comfy.samplers import KSAMPLER
import torch
from torchvision.transforms.functional import gaussian_blur
from comfy.k_diffusion.sampling import default_noise_sampler, get_ancestral_step, to_d, BrownianTreeNoiseSampler
from tqdm.auto import trange

from ... import ROOT_NAME

def interpolate(x, size, unsharp_strength=0.0, unsharp_kernel_size=3, unsharp_sigma=0.5, unsharp=False, mode="bicubic", align_corners=False):
    x = torch.nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
    if unsharp_strength > 0 and unsharp:
        blurred = gaussian_blur(x, kernel_size=unsharp_kernel_size, sigma=unsharp_sigma)
        x = x + unsharp_strength * (x - blurred)
    return x

def make_upscale_info(height, width, upscale_ratio, start_step, end_step, upscale_n_step):
    upscale_steps = []
    step = start_step - 1
    while step < end_step - 1:
        upscale_steps.append(step)
        step += upscale_n_step
    upscale_shapes = [
        (int(height * (((upscale_ratio - 1) / i) + 1)), int(width * (((upscale_ratio - 1) / i) + 1)))
        for i in reversed(range(1, len(upscale_steps) + 1))
    ]
    upscale_info = {k: v for k, v in zip(upscale_steps, upscale_shapes)}
    return upscale_info

@torch.no_grad()
def sample_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    upscale_ratio=2.0,
    start_step=5,
    end_step=15,
    upscale_n_step=3,
    unsharp_kernel_size=3,
    unsharp_sigma=0.5,
    unsharp_strength=0.0,
    unsharp_target="x",
):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    # make upscale info
    upscale_info = make_upscale_info(x.shape[2], x.shape[3], upscale_ratio, start_step, end_step, upscale_n_step)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        # Euler method
        d = to_d(x, sigmas[i], denoised) 
        if i not in upscale_info:
            x = denoised + d * sigma_down
        elif unsharp_target == "x":
            x = denoised + d * sigma_down
            x = interpolate(x, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=True)
        else:
            denoised = interpolate(denoised, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=True)
            d = interpolate(d, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=False)
            x = denoised + d * sigma_down

        if sigmas[i + 1] > 0:
            noise_sampler = default_noise_sampler(x)
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x + noise * sigma_up * s_noise
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    upscale_ratio=2.0,
    start_step=5,
    end_step=15,
    upscale_n_step=3,
    unsharp_kernel_size=3,
    unsharp_sigma=0.5,
    unsharp_strength=0.0,
    unsharp_target="x",
):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    # make upscale info
    upscale_info = make_upscale_info(x.shape[2], x.shape[3], upscale_ratio, start_step, end_step, upscale_n_step)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised) 
            if i not in upscale_info:
                x = denoised + d * sigma_down
            elif unsharp_target == "x":
                x = denoised + d * sigma_down
                x = interpolate(x, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=True)
            else:
                denoised = interpolate(denoised, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=True)
                d = interpolate(d, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=False)
                x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            if i in upscale_info and unsharp_target == "denoised":
                denoised_2 = interpolate(denoised_2, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=True)
                x = interpolate(x, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=False)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
            if i in upscale_info and unsharp_target == "x":
                x = interpolate(x, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=True)
        # Noise addition
        if sigmas[i + 1] > 0:
            noise_sampler = default_noise_sampler(x)
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x + noise * sigma_up * s_noise
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    upscale_ratio=2.0,
    start_step=5,
    end_step=15,
    upscale_n_step=3,
    unsharp_kernel_size=3,
    unsharp_sigma=0.5,
    unsharp_strength=0.0,
    unsharp_target="x",
):
    """DPM-Solver++(2M) SDE."""

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    # make upscale info
    upscale_info = make_upscale_info(x.shape[2], x.shape[3], upscale_ratio, start_step, end_step, upscale_n_step)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            if i in upscale_info and unsharp_target == "denoised":
                denoised = interpolate(denoised, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=True)
                x = interpolate(x, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=False)
                old_denoised = None
            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            old_denoised = denoised

            if i in upscale_info and unsharp_target == "x":
                x = interpolate(x, upscale_info[i], unsharp_strength, unsharp_kernel_size, unsharp_sigma, unsharp=True)
                old_denoised = None

            if eta:
                noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise
                
        h_last = h
    return x


@torch.no_grad()
def sample_lcm(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    eta=None,
    s_noise=None,
    upscale_ratio=2.0,
    start_step=5,
    end_step=15,
    upscale_n_step=3,
    unsharp_kernel_size=3,
    unsharp_sigma=0.5,
    unsharp_strength=0.0,
    unsharp_target="x",
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # make upscale info
    upscale_info = make_upscale_info(x.shape[2], x.shape[3], upscale_ratio, start_step, end_step, upscale_n_step)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            # Resize
            if i in upscale_info:
                x = torch.nn.functional.interpolate(x, size=upscale_info[i], mode="bicubic", align_corners=False)
                if unsharp_strength > 0:
                    blurred = gaussian_blur(x, kernel_size=unsharp_kernel_size, sigma=unsharp_sigma)
                    x = x + unsharp_strength * (x - blurred)
            noise_sampler = default_noise_sampler(x)
            x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])

    return x


class GradualLatentSampler:
    # kernel_sizeのstepを2にすると、2,4,6,8... となるので、stepを1にしておく
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (["euler_ancestral", "dpmpp_2s_ancestral", "dpmpp_2m_sde", "lcm"],),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                "upscale_ratio": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.01, "round": False}),
                "start_step": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                "end_step": ("INT", {"default": 15, "min": 0, "max": 1000, "step": 1}),
                "upscale_n_step": ("INT", {"default": 3, "min": 0, "max": 1000, "step": 1}),
                "unsharp_kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 1}),
                "unsharp_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                "unsharp_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                "unsharp_target": (["x", "denoised"],),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = ROOT_NAME + "custom_samplers"

    FUNCTION = "get_sampler"

    def get_sampler(
        self,
        sampler_name,
        eta,
        s_noise,
        upscale_ratio,
        start_step,
        end_step,
        upscale_n_step,
        unsharp_kernel_size,
        unsharp_sigma,
        unsharp_strength,
        unsharp_target,
    ):
        if sampler_name == "euler_ancestral":
            sample_function = sample_euler_ancestral
        elif sampler_name == "dpmpp_2s_ancestral":
            sample_function = sample_dpmpp_2s_ancestral
        elif sampler_name == "dpmpp_2m_sde":
            sample_function = sample_dpmpp_2m_sde
        elif sampler_name == "lcm":
            sample_function = sample_lcm
        else:
            raise ValueError("Unknown sampler name")
        
        unsharp_target = unsharp_target if unsharp_strength > 0 else "x" # interpの位置が違うので調整

        unsharp_kernel_size = unsharp_kernel_size if unsharp_kernel_size % 2 == 1 else unsharp_kernel_size + 1

        sampler = KSAMPLER(
            sample_function,
            {
                "eta": eta,
                "s_noise": s_noise,
                "upscale_ratio": upscale_ratio,
                "start_step": start_step,
                "end_step": end_step,
                "upscale_n_step": upscale_n_step,
                "unsharp_kernel_size": unsharp_kernel_size,
                "unsharp_sigma": unsharp_sigma,
                "unsharp_strength": unsharp_strength,
                "unsharp_target": unsharp_target,
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "GradualLatentSampler": GradualLatentSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Sampling
    "GradualLatentSampler": "GradualLatentSampler",
}