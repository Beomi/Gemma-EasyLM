"""
Usage:
python convert_hf_to_easylm.py  \
       --checkpoint_dir     /path/hf_format_dir/    \
       --output_file /path/easylm_format.stream   \
       --model_size 7b \
       --streaming
"""
import time
from pathlib import Path
import argparse

import mlxu
import torch
import flax
from ml_dtypes import bfloat16

from EasyLM.checkpoint import StreamingCheckpointer

from tqdm.auto import tqdm

LLAMA_STANDARD_CONFIGS = {
    "gemma-7b": {
        "dim": 3072,
        "intermediate_size": 24576,
        "n_layers": 28,
        "n_heads": 16,
        "n_groups": 1,
        "num_key_value_heads": 16,
        "norm_eps": 1e-6,
        "head_dim": 256,
    }
}


# def inverse_permute(params, w):
#     # return w
    
#     n_heads = params["n_heads"]
#     dim = params["dim"]
    
#     # print(n_heads, 2, params['head_dim'] // 2, dim)
#     # reshaped_w = w.reshape(n_heads, 2, dim // n_heads // 2, dim)
#     print('w.shape:', w.shape)
    
#     # reshaped_w = w.reshape(n_heads, 2, params['head_dim'] // 2, dim) # FIXME
#     reshaped_w = w.reshape(n_heads, 1, params['head_dim'], dim) # FIXME
#     print('reshaped_w.shape:', reshaped_w.shape)
    
#     input_dim = params['head_dim'] * params['n_heads']
#     print('input_dim:', input_dim)
    
#     transposed_w = reshaped_w.transpose(0, 2, 1, 3)
#     print('transposed_w.shape:', transposed_w.shape)
#     inverted_w = transposed_w.reshape(input_dim, dim)
#     print('inverted_w.shape:', inverted_w.shape)
#     return inverted_w


# # def grouped_inverse_permute(params, w):
# #     n_heads = params["n_heads"]
# #     n_groups = params.get("n_groups", 1)
# #     # n_groups = n_heads // params.get('n_groups', 1) #params.get("n_groups", 1)
    
# #     dim = params["dim"]
# #     # Heads 여러개가 -> 하나로 합쳐지는거라서. 앞에서 짤라주는게 맞다.
# #     reshaped_w = w.reshape(
# #         n_heads // n_groups, 
# #         2, 
# #         params['head_dim'] // 2, 
# #         dim
# #     )
# #     transposed_w = reshaped_w.transpose(0, 2, 1, 3)
# #     input_dim = params['head_dim'] * params['n_heads']
# #     # inverted_w = transposed_w.reshape(
# #     #     dim // n_groups, dim
# #     # )  # n_groups로 나눠주는게 혹시 앞부분? 그래서 1024x8196?
# #     inverted_w = transposed_w.reshape(input_dim // n_groups, dim)
# #     return inverted_w

def inverse_permute(params, w):
    """
    Correctly reshape and transpose attention weight matrices from PyTorch to Flax format.
    """
    n_heads = params["n_heads"]
    head_dim = params["head_dim"]
    # Calculate the product of num_heads and head_dim to match the expected Flax format
    expected_last_dim = n_heads * head_dim
    
    # The expected shape given in the error is (256, 49152)
    # Since the provided weight has shape (3072, 4096), it suggests that the reshaping logic needs adjustment.
    
    # Assuming the provided shape is (num_heads * head_dim, hidden_size),
    # and you need to transpose it to (hidden_size, num_heads * head_dim) to match Flax's expectations.
    reshaped_w = w.transpose()
    
    # Ensure reshaped_w has the correct shape by verifying or adjusting the reshape logic
    # The specifics might vary based on your model's configuration and the original shape of w
    
    return reshaped_w



def grouped_inverse_permute(params, w):
    """
    Handle grouped inverse permutation for models with grouped attention heads.
    This is particularly necessary for models like Gemma with a specific configuration
    of key/value heads.
    """
    n_heads = params["n_heads"]
    head_dim = params["head_dim"]
    embed_dim = params["dim"]
    n_groups = params["num_key_value_heads"]  # Assuming grouped by key/value heads

    # Assuming w is in (n_heads, 2 * head_dim / 2, embed_dim) format
    # First, reshape to group heads together
    reshaped_w = w.reshape(n_heads // n_groups, n_groups, head_dim, embed_dim // n_heads)
    
    # Then, transpose to (embed_dim / n_heads, n_groups, n_heads // n_groups, head_dim)
    transposed_w = reshaped_w.transpose(3, 1, 0, 2)
    
    # Finally, merge groups and heads
    merged_w = transposed_w.reshape(embed_dim, n_heads * head_dim)
    
    return merged_w



def main(args):
    start = time.time()
    params = LLAMA_STANDARD_CONFIGS[args.model_size]

    ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.bin"))
    ckpt = {}
    for i, ckpt_path in tqdm(enumerate(ckpt_paths)):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        for k, v in checkpoint.items():
            if k.startswith("model."):
                k = k[6:]
            ckpt[k] = v.to(torch.float32)
    print(f"Start convert weight to easylm format...")
    jax_weights = {
        "transformer": {
            "wte": {"embedding": ckpt["embed_tokens.weight"].numpy()},
            "ln_f": {"kernel": ckpt["norm.weight"].numpy()},
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        # "wq": {
                        #     "kernel": inverse_permute(
                        #         params,
                        #         ckpt[f"layers.{layer}.self_attn.q_proj.weight"].numpy(),
                        #     ) #.transpose()
                        # },
                        "wq": {
                            "kernel": ckpt[f"layers.{layer}.self_attn.q_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                        
                        
                        # "wk": {
                        #     # 70B GQA로 인해 이부분은 /8 적용된 Grouped로 해줘야 함. Out이 1024.
                        #     "kernel": grouped_inverse_permute(
                        #         params,
                        #         ckpt[f"layers.{layer}.self_attn.k_proj.weight"].numpy(),
                        #     ).transpose()  # FIXME: 왜 여기서 1024x8192가 아닌거지? Transpose 이전에 8192x1024가 나오는데..
                        # },
                        # "wk": {
                        #     # 70B GQA로 인해 이부분은 /8 적용된 Grouped로 해줘야 함. Out이 1024.
                        #     "kernel": inverse_permute(
                        #         params,
                        #         ckpt[f"layers.{layer}.self_attn.k_proj.weight"].numpy(),
                        #     ) #.transpose()  # FIXME: 왜 여기서 1024x8192가 아닌거지? Transpose 이전에 8192x1024가 나오는데..
                        # },
                        "wk": {
                            # 70B GQA로 인해 이부분은 /8 적용된 Grouped로 해줘야 함. Out이 1024.
                            "kernel": ckpt[f"layers.{layer}.self_attn.k_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                        "wv": {
                            "kernel": ckpt[f"layers.{layer}.self_attn.v_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                        "wo": {
                            "kernel": ckpt[f"layers.{layer}.self_attn.o_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                    },
                    "feed_forward": {
                        "w1": {
                            "kernel": ckpt[f"layers.{layer}.mlp.gate_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                        "w2": {
                            "kernel": ckpt[f"layers.{layer}.mlp.down_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                        "w3": {
                            "kernel": ckpt[f"layers.{layer}.mlp.up_proj.weight"]
                            .numpy()
                            .transpose()
                        },
                    },
                    "attention_norm": {
                        "kernel": ckpt[f"layers.{layer}.input_layernorm.weight"].numpy()
                    },
                    "ffn_norm": {
                        "kernel": ckpt[
                            f"layers.{layer}.post_attention_layernorm.weight"
                        ].numpy()
                    },
                }
                for layer in range(params["n_layers"])
            },
        },
        "lm_head": {"kernel": ckpt["lm_head.weight"].numpy().transpose()},
    }
    print(jax_weights)
    print(f"Convert weight to easylm format finished...")
    print(f"Start to save...")

    if args.streaming:
        StreamingCheckpointer.save_train_state_to_file(
            jax_weights, args.output_file, float_dtype="bf16"
        )
    else:
        with mlxu.open_file(args.output_file, "wb") as fout:
            fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True))

    print(
        f"Save finished!!! take time: {time.time() - start} save path: {args.output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hf to easylm format script")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Need to be converted model weight dir. it is a dir",
    )
    parser.add_argument(
        "--output_file", type=str, help="Save model weight file path, it is a file."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7b",
        # choices=["7b", "13b", "30b", "34b", "65b", "70b"],
        help="model size",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="whether is model weight saved stream format",
    )

    args = parser.parse_args()

    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"output_file: {args.output_file}")
    print(f"model_size: {args.model_size}")
    print(f"streaming: {args.streaming}")

    main(args)
