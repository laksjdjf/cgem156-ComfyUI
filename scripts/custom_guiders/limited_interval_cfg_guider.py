import comfy
from comfy_api.v0_0_2 import io
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

class LimitedIntervalCFGGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LimitedIntervalCFGGuider|cgem156",
            display_name="Limited Interval CFG Guider 🍌",
            category=CATEGORY_NAME,
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Float.Input("start_step", default=0, min=0, max=1, step=0.001),
                io.Float.Input("end_step", default=1, min=0, max=1, step=0.001),
            ],
            outputs=[
                io.Guider.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, positive, negative, cfg, start_step, end_step) -> io.NodeOutput:

        start_sigma = model.model.model_sampling.percent_to_sigma(start_step)
        end_sigma = model.model.model_sampling.percent_to_sigma(end_step)

        guider = LimitedIntervalCFG(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_range(end_sigma, start_sigma)
        return io.NodeOutput(guider)
