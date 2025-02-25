# CLI tooling for converting RWKV models into HF compatible format with the following params
#
# - Model Source, either as
#     - RWKV model path file
#     - RWKV model safetensor file
#     - HF model directory
# - Directory to output the converted safetensor huggingface model
# - With the "model_class" defaults to "v7_goose"
#
# As part of the build process, you would need to do the following relative to the hf_builder.py script (this file).
#
# - Copy all files recursively from "../rwkv_block/{model_class}" into "./hf_code/{model_class}/rwkv_block/{model_class}/"
# - Import 'RWKV7PreTrainedModel' from '.hf_code.{model_class}.modeling_rwkv7'
# - Import 'RWKV7Config' from '.hf_code.{model_class}.configuration_rwkv7'
# - These are HF compatible classes that are used to load the model and save them
#
# - Load the model weights dictionary to CPU
# - Using the 'RWKV7Config.from_model_state_dict' class, get the config from the model weights
# - Using the config, initialize the model
# - Load the model weights into the configured model
# - Save the model to the output directory

import argparse, shutil, json, os.path
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
import hjson
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file as safe_save_file

####
# System path configuration
####
# Get the current script directory
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the system path, if not already present
if current_dir not in sys.path:
    sys.path.append(current_dir)

####
# Conversion and sync code
####
def load_model_from_filepath(model_path):
    if model_path.endswith('.safetensors'):
        return load_file(model_path, device='cpu')
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        return torch.load(model_path, map_location='cpu', weights_only=True, mmap=True)

def save_tokenizer_to_output_dir(output_dir, tokenizer_type):
    # Check if the source directory exists
    source_dir = Path(f"{current_dir}/hf_code/tokenizer/{tokenizer_type}/")
    if not source_dir.exists():
        raise ValueError(f"Tokenizer type '{tokenizer_type}' not found in hf_code/tokenizer directory.")
    
    # Copy the tokenizer files to the output directory
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

def save_model_code_to_output_dir(output_dir, model_class):
    # Check if the source directory exists
    source_dir = Path(f"{current_dir}/hf_code/{model_class}")
    if not source_dir.exists():
        raise ValueError(f"Model class '{model_class}' not found in hf_code directory.")
    
    # Copy the model files to the output directory
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

####
# Script builder
####
def build_v7_goose():
    hf_script_builder(
        target_dir=f"{current_dir}/hf_code/v7_goose",
        source_dir=f"{current_dir}/../rwkv_block/v7_goose",
        source_cuda_dir=f"{current_dir}/../rwkv_block/v7_goose/block/kernel/cuda",
        build_file_order = [
            "block/kernel/rwkv7_attn_pytorch.py",
            "block/kernel/rwkv7_attn_cuda.py",
            "block/kernel/rwkv7_attn_fla.py",
            "block/kernel/rwkv7_attn_triton.py",
            "block/rwkv7_block_config_map.py",
            "block/rwkv7_channel_mix.py",
            "block/rwkv7_time_mix.py",
            "block/rwkv7_layer_block.py",
            "model/rwkv7_goose_config_map.py",
            "model/rwkv7_goose_model.py",
        ],
        build_file_name="modeling_blocks_rwkv7.py"
    )

def build_v7_qwerky():
    hf_script_builder(
        target_dir=f"{current_dir}/hf_code/v7_qwerky",
        source_dir=f"{current_dir}/../rwkv_block/v7_goose",
        source_cuda_dir=f"{current_dir}/../rwkv_block/v7_goose/block/kernel/cuda",
        build_file_order = [
            "block/kernel/rwkv7_attn_pytorch.py",
            "block/kernel/rwkv7_attn_cuda.py",
            "block/kernel/rwkv7_attn_fla.py",
            "block/kernel/rwkv7_attn_triton.py",
            "block/rwkv7_block_config_map.py",
            "block/rwkv7_channel_mix.py",
            "block/rwkv7_time_mix.py",
            "block/rwkv7_layer_block.py",
            # "model/rwkv7_goose_config_map.py",
            # "model/rwkv7_goose_model.py",
            "../v7_qwerky/block/qwerky7_block_config_map.py",
            "../v7_qwerky/block/qwerky7_layer_block.py",
            "../v7_qwerky/block/qwerky7_time_mix.py",
            "../v7_qwerky/model/qwerky7_config_map.py",
            "../v7_qwerky/model/qwerky7_model.py",
            "../v7_qwerky/model/qwerky7_causal_lm.py",
        ],
        build_file_name="modeling_blocks_qwerky7.py"
    )

def hf_script_builder(
    target_dir,
    source_dir,
    source_cuda_dir,
    build_file_order: list[str],
    build_file_name
):
    '''
    For whatever reason, HF do not like nested python files, and require everything to be
    in the the top level directory. Somewhat. This is a hack to copy and merge all the files.
    '''

    # Normalize dir to Path
    target_dir = target_dir if isinstance(target_dir, Path) else Path(target_dir)
    source_dir = source_dir if isinstance(source_dir, Path) else Path(source_dir)
    source_cuda_dir = source_cuda_dir if isinstance(source_cuda_dir, Path) else Path(source_cuda_dir)

    # Define the target file path
    target_file = Path(f"{target_dir}/{build_file_name}")
    
    # Copy over the cuda files
    shutil.copytree(source_cuda_dir, target_dir / "cuda", dirs_exist_ok=True)

    # Compile the files, into a single string
    compiled_code = [
        "# ========== AUTO GENERATED FILE =========",
        "# This file is auto generated by 'hf_builder.py'", 
        "# As part of the 'RWKV/RWKV-block' project",
        "# do not edit these files directly",
        "# ========== =================== =========",
        "#",
        "# The following is horrible horrible code used to work around the lack of CUDA files in HF cache.",
        "# By encoding the CUDA files into the python files as string, and then writing them out.",
        "#",
        "# This works around the lack of transfers from the HF model files to the inference cache.",
        "# This is used only for improving our internal development process.",
        "# And I hope that this is not the solution for production",
        "#",
        "# ========== =================== =========",
        "",
    ]

    # Get the list of .cu and .cpp files in the cuda directory
    cuda_files = list((target_dir / "cuda").rglob("*.cu")) + list((target_dir / "cuda").rglob("*.cpp"))
    # Convert to relative paths
    cuda_files = [str(file).replace(str(target_dir), "") for file in cuda_files]

    # Get the string value of the cuda files in a map
    cuda_file_map = {}
    for file_pth in cuda_files:
        # Trim out the front "/"
        if file_pth.startswith("/"):
            file_pth = file_pth[1:]
        
        # Get the string value of the cuda files in a map
        file_str = open(target_dir / file_pth).read()
        cuda_file_map[str(file_pth)] = file_str

    # Insert the cuda_file_map as a JSON object
    compiled_code.append("CUDA_FILE_MAP = "+json.dumps(cuda_file_map, indent=2))
    compiled_code.append("")

    # Define the cuda file loader function
    cuda_file_loader_code = [
        "# ========== =================== =========",
        "",
        "from pathlib import Path",
        "CUR_FILE_DIR = Path(__file__).parent",
        "",
        "# Make the CUDA directory",
        "(CUR_FILE_DIR / 'cuda').mkdir(exist_ok=True)",
        "",
        "# Load the CUDA files into the directory",
        "for file, file_str in CUDA_FILE_MAP.items():",
        "    file_path = CUR_FILE_DIR / file",
        "    if not file_path.exists():",
        "        file_path.write_text(file_str)",
        "",
        "# ========== =================== =========",
    ]
    # Append the cuda file loader code
    compiled_code += cuda_file_loader_code

    # Cleanup code for the various python files
    def cleanup_source_file(file_str):
        # split the file into lines
        lines = file_str.split('\n')

        # Filter into comments all the import starting with "from .r"
        filtered_lines = []
        for line in lines:
            trim_line = line.strip()
            if not trim_line.startswith("from ."):
                filtered_lines.append(line)
            else:
                # Replace the "from ." with "from # from ."
                filtered_lines.append( line.replace("from .", "# from .")  )

        # Join the lines back together
        return '\n'.join(filtered_lines)
    
    # Loop through the files, and compile them
    for file_subpath in build_file_order:
        # Read the file in, and cleanup it's source code
        file_str = open(source_dir/file_subpath).read()
        file_str = cleanup_source_file(file_str)

        # Append the file to the compiled code
        compiled_code.append("# ----------------")
        compiled_code.append(f"# {file_subpath}")
        compiled_code.append("# ----------------")
        compiled_code.append(file_str)
        compiled_code.append("")

    # Write the compiled code to the target file
    open(target_file, 'w').write('\n'.join(compiled_code))

####
# Builder scripts
####
def load_hf_config(config_str):
    """
    Load HuggingFace config from either a JSON string or a path to a JSON file.
    
    Args:
        config_str: Either a JSON string or path to JSON file
        
    Returns:
        dict: Parsed config dictionary
    """
    try:
        config_str = config_str.strip()
        # Check if there is '{ .. }' in the string
        if config_str.startswith("{") and config_str.endswith("}"):
            return hjson.loads(config_str)
        else:
            return hjson.loads('{ '+config_str.replace("=", '": "')+' }')
    except json.JSONDecodeError:
        # If that fails, try loading as file path
        if os.path.isfile(config_str):
            with open(config_str) as f:
                return json.load(f)
        else:
            raise ValueError(f"hf_config must be valid JSON string or path to JSON file, got: {config_str}")

def hf_builder(args):
    # Print the args
    print("-----------------------------")
    print("Converting RWKV model to HuggingFace format...")
    print(f"Model Class     : {args.model_class}")
    if not args.model_code_only:
        print(f"Model Source    : {args.model_source}")
        print(f"Tokenizer Type  : {args.tokenizer_type}")
    print(f"Output Directory: {args.output_dir}")
    if args.hf_config:
        print(f"HF Config      : {args.hf_config}")
    if args.one_file_safetensor:
        print("Using single safetensor file mode")
    if args.model_code_only:
        print("Model code only mode - skipping tokenizer and weights")
    print("-----------------------------")

    # Get the model class
    model_class = args.model_class

    # Copy rwkv_block code files
    print("Building rwkv_block into HF code ...")
    if model_class == "v7_goose":
        build_v7_goose()
    elif model_class == "v7_qwerky":
        build_v7_qwerky()
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save model code files
    print("Saving model code files ...")
    save_model_code_to_output_dir(args.output_dir, model_class)

    # If model-code-only mode, we're done here
    if args.model_code_only:
        print("-----------------------------")
        print("Successfully copied model code files")
        print("-----------------------------")
        return

    # Load model weights
    print("Loading model weights raw state ...")
    state_dict = load_model_from_filepath(args.model_source)
    assert state_dict is not None, f"Failed to load model weights from: {args.model_source}"

    # Parse HF config if provided
    hf_config = {}
    if args.hf_config:
        print("Loading HF config overrides ...")
        hf_config = load_hf_config(args.hf_config)
        print("HF Config Overrides:")
        print(json.dumps(hf_config, indent=2))
        print("-----------------------------")

    # Load for the respective class
    print("Loading model config from weights ...")
    if model_class == "v7_goose":
        from hf_code.v7_goose.configuration_rwkv7 import RWKV7Config
        model_config = RWKV7Config.from_model_state_dict(state_dict, **hf_config)
    elif model_class == "v7_qwerky":
        from hf_code.v7_qwerky.configuration_qwerky7 import Qwerky7Config
        model_config = Qwerky7Config.from_model_state_dict(state_dict, **hf_config)
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    print("-----------------------------")
    print("Model Configuration:")
    print(model_config.__dict__)
    print("-----------------------------")
    
    # Load the model files
    print("Checking tokenizer ...")

    # Deduce the tokenizer
    tokenizer_type = args.tokenizer_type
    if tokenizer_type == "auto":
        # Check for world tokenizer
        if model_config.vocab_size >= 65529 and model_config.vocab_size <= 65536:
            tokenizer_type = "world"
        elif model_config.vocab_size >= 50304 and model_config.vocab_size <= 50432:
            tokenizer_type = "neox"
        elif model_config.vocab_size >= 151936 and model_config.vocab_size <= 152064:
            tokenizer_type = "qwen2"
        else:
            raise ValueError(f"Unable to detect tokenizer type for: {tokenizer_type} (vocab_size: {model_config.vocab_size})")
        
        # Print the detected tokenizer type
        print(f"Detected Tokenizer Type: {tokenizer_type}")

    # Load the model files
    print("Modifying model state ...")

    # Removing known state dict key with issues
    if hasattr(model_config, "v_first_with_embedding") and model_config.v_first_with_embedding is True:
        pass
    else:
        rmv_state_keys = [
            "model.layers.0.self_attn.v0",
            "model.layers.0.self_attn.v1",
            "model.layers.0.self_attn.v2"
        ]
        for key in rmv_state_keys:
            if key in state_dict:
                del state_dict[key]

    print("-----------------------------")

    print("Saving tokenizer files ...")
    save_tokenizer_to_output_dir(args.output_dir, tokenizer_type)
    
    print("Saving model weight files ...")

    if args.one_file_safetensor:
        # Save all weights in a single safetensor file
        print("Saving all weights in a single safetensor file...")
        path_to_weights = os.path.join(args.output_dir, "model.safetensors")
        safe_save_file(state_dict, path_to_weights, metadata={"format": "pt"})
        print(f"Model weights saved in {path_to_weights}")
    else:
        # Use sharding for large models
        max_shard_size="5GB"
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern="model{suffix}.safetensors", max_shard_size=max_shard_size
        )

        # Save index if sharded
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }

        # Save the model
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]

            print("- ", shard_file)
            safe_save_file(shard, os.path.join(args.output_dir, shard_file), metadata={"format": "pt"})

        if index is not None:
            save_index_file = os.path.join(args.output_dir, "model.safetensors.index.json")
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            print(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    del state_dict

    print("Saving model config files ...")
    model_config.save_pretrained(args.output_dir)

    print("Patching configuration ...")
    config_json_path = Path(f"{args.output_dir}/config.json")
    config_json = json.load(config_json_path.open())

    # Delete the layer_id attribute (why is this even here?)
    del config_json["layer_id"]

    # Fill in the auto_map, and architecture
    if model_class == "v7_goose":
        config_json["auto_map"] = {
            "AutoConfig": "configuration_rwkv7.RWKV7Config",
            "AutoModel": "modeling_rwkv7.RWKV7Model",
            "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM"
        }
        config_json["architectures"] = ["RWKV7ForCausalLM", "RWKV7Model", "RWKV7PreTrainedModel"]
    elif model_class == "v7_qwerky":
        config_json["auto_map"] = {
            "AutoConfig": "configuration_qwerky7.Qwerky7Config",
            "AutoModel": "modeling_qwerky7.Qwerky7BaseModel",
            "AutoModelForCausalLM": "modeling_qwerky7.Qwerky7ForCausalLM"
        }
        config_json["architectures"] = ["Qwerky7ForCausalLM", "Qwerky7BaseModel", "Qwerky7PreTrainedModel"]
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    # Save the patched config
    with config_json_path.open("w") as f:
        json.dump(config_json, f, indent=2)
    
    # Print the success message
    print("-----------------------------")
    print("Successfully converted RWKV model to HuggingFace format")
    print("-----------------------------")

def main():
    parser = argparse.ArgumentParser(description="Convert RWKV models to HuggingFace format")
    parser.add_argument("model_source", help="Path to RWKV model file in .pth or .safetensors format", nargs='?')
    parser.add_argument("output_dir", help="Directory to output the converted HuggingFace model")
    parser.add_argument("--model_class", default="v7_goose", help="Model class (default: v7_goose)")
    parser.add_argument("--tokenizer_type", default="auto", help="Tokenizer to use, either 'auto','world','neox' or 'qwen2' (default: auto)")
    parser.add_argument("--hf-config", help="HuggingFace config overrides as JSON string or path to JSON file")
    parser.add_argument("--one-file-safetensor", action="store_true", help="Save model weights in a single safetensor file instead of sharding")
    parser.add_argument("--model-code-only", action="store_true", help="Only copy model code files, skip tokenizer and weights")
    args = parser.parse_args()

    # Validate args
    if not args.model_code_only and args.model_source is None:
        parser.error("model_source is required unless --model-code-only is specified")

    hf_builder(args)

if __name__ == "__main__":
    main()
