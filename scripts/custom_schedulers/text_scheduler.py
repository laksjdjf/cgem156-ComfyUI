'''
load from sampling/custom_sampling/scheulers
input text like "999,893,...,156"
connect to SamplerCustom
'''

import torch
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "custom_schedulers"

class TextScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextScheduler|cgem156",
            display_name="Text Scheduler 🍌",
            category=CATEGORY_NAME,
            inputs=[
                io.Model.Input("model"),
                io.String.Input("timesteps", multiline=True),
                io.Boolean.Input("verbose"),
            ],
            outputs=[
                io.Sigmas.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, timesteps, verbose) -> io.NodeOutput:
        timesteps = [float(timestep) for timestep in timesteps.replace(" ", "").split(",")]
        sigmas = model.model.model_sampling.sigma(torch.tensor(timesteps))
        sigmas = torch.cat([sigmas, torch.tensor([0])])

        if verbose:
            print("sigmas:", sigmas.tolist())
        return io.NodeOutput(sigmas)

NODE_CLASS_MAPPINGS = {
    "TextScheduler": TextScheduler,
}
