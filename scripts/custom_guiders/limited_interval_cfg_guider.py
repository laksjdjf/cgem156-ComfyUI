import comfy
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "custom_guiders"

class LimitedIntervalCFG(comfy.samplers.CFGGuider):
    def set_range(self, sigma_low, sigma_high):
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
    
    def in_range(self, sigma):
        return self.sigma_low < sigma <= self.sigma_high
    
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        cfg = self.cfg if self.in_range(timestep[0].item()) else 1
        #print(f"CFG: {cfg} timestep: {timestep} sigma_low: {self.sigma_low} sigma_high: {self.sigma_high}")

        return comfy.samplers.sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), cfg, model_options=model_options, seed=seed)

class LimitedIntervalCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model": ("MODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "start_step": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.001}),
                "end_step": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = CATEGORY_NAME

    def get_guider(self, model, positive, negative, cfg, start_step, end_step):
        
        start_sigma = model.model.model_sampling.percent_to_sigma(start_step)
        end_sigma = model.model.model_sampling.percent_to_sigma(end_step)

        guider = LimitedIntervalCFG(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_range(end_sigma, start_sigma)
        return (guider,)
    