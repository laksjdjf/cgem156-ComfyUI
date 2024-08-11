FLUX_DOUBLE_TRANSFORMER_MAP = {
    'transformer.transformer_blocks.{}.norm1.linear': 'lora_transformer_transformer_blocks_{}_norm1_linear',
    'transformer.transformer_blocks.{}.norm1_context.linear': 'lora_transformer_transformer_blocks_{}_norm1_context_linear',
    'transformer.transformer_blocks.{}.attn.to_q': 'lora_transformer_transformer_blocks_{}_attn_to_q',
    'transformer.transformer_blocks.{}.attn.to_k': 'lora_transformer_transformer_blocks_{}_attn_to_k',
    'transformer.transformer_blocks.{}.attn.to_v': 'lora_transformer_transformer_blocks_{}_attn_to_v',
    'transformer.transformer_blocks.{}.attn.add_k_proj': 'lora_transformer_transformer_blocks_{}_attn_add_k_proj',
    'transformer.transformer_blocks.{}.attn.add_v_proj': 'lora_transformer_transformer_blocks_{}_attn_add_v_proj',
    'transformer.transformer_blocks.{}.attn.add_q_proj': 'lora_transformer_transformer_blocks_{}_attn_add_q_proj',
    'transformer.transformer_blocks.{}.attn.to_out.0': 'lora_transformer_transformer_blocks_{}_attn_to_out_0',
    'transformer.transformer_blocks.{}.attn.to_add_out': 'lora_transformer_transformer_blocks_{}_attn_to_add_out',
    'transformer.transformer_blocks.{}.ff.net.0.proj': 'lora_transformer_transformer_blocks_{}_ff_net_0_proj',
    'transformer.transformer_blocks.{}.ff.net.2': 'lora_transformer_transformer_blocks_{}_ff_net_2',
    'transformer.transformer_blocks.{}.ff_context.net.0.proj': 'lora_transformer_transformer_blocks_{}_ff_context_net_0_proj',
    'transformer.transformer_blocks.{}.ff_context.net.2': 'lora_transformer_transformer_blocks_{}_ff_context_net_2'
}

FLUX_SINGLE_TRANSFORMER_MAP = {
    'transformer.single_transformer_blocks.{}.norm.linear': 'lora_transformer_single_transformer_blocks_{}_norm_linear',
    'transformer.single_transformer_blocks.{}.proj_mlp': 'lora_transformer_single_transformer_blocks_{}_proj_mlp',
    'transformer.single_transformer_blocks.{}.proj_out': 'lora_transformer_single_transformer_blocks_{}_proj_out',
    'transformer.single_transformer_blocks.{}.attn.to_q': 'lora_transformer_single_transformer_blocks_{}_attn_to_q',
    'transformer.single_transformer_blocks.{}.attn.to_k': 'lora_transformer_single_transformer_blocks_{}_attn_to_k',
    'transformer.single_transformer_blocks.{}.attn.to_v': 'lora_transformer_single_transformer_blocks_{}_attn_to_v',
}

FLUX_MAP = {}
for key, value in FLUX_DOUBLE_TRANSFORMER_MAP.items():
    for i in range(19):
        for surfix in [".alpha", ".lora_up.weight", ".lora_down.weight"]:
            FLUX_MAP[value.format(i) + surfix] = key.format(i) + surfix

for key, value in FLUX_SINGLE_TRANSFORMER_MAP.items():
    for i in range(38):
        for surfix in [".alpha", ".lora_up.weight", ".lora_down.weight"]:
            FLUX_MAP[value.format(i) + surfix] = key.format(i) + surfix