use anyhow::Result;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Convert PyTorch weights to Burn format using Python script
pub fn convert_model(weights_path: &Path, config_path: &Path) -> Result<()> {
    // Python script content
    let script = r#"#!/usr/bin/env python3
"""
Convert PyTorch LSTM weights to Burn-compatible format.

PyTorch stores LSTM weights concatenated for all gates:
- weight_ih_l0: [4*hidden_size, input_size]
- weight_hh_l0: [4*hidden_size, hidden_size]
- bias_ih_l0: [4*hidden_size]
- bias_hh_l0: [4*hidden_size]

Burn expects them split into individual gates with transposed weights.
"""

import json
import os
import sys
from json import JSONEncoder
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset


class EncodeTensor(JSONEncoder, Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().flatten().tolist()
        return super(EncodeTensor, self).default(obj)



def convert_pytorch_to_burn(input_path, hidden_size, cfg_path):
    """
    Convert PyTorch LSTM weights to Burn format.

    Args:
        input_path: Path to PyTorch .pt file
        output_path: Path to save Burn-compatible .pt file
        hidden_size: Hidden size of the LSTM (default 64)
    """
    input_path = Path(input_path)
    output_path = Path(input_path).parent / "burn" / input_path.name

    # Load PyTorch weights
    state_dict = torch.load(input_path, map_location="cpu")

    print("Original PyTorch keys:")
    for key, tensor in state_dict.items():
        print(f"  {key}: {tensor.shape}")

    # Extract LSTM weights
    weight_ih = state_dict["lstm.weight_ih_l0"]  # [256, 10]
    weight_hh = state_dict["lstm.weight_hh_l0"]  # [256, 64]
    bias_ih = state_dict["lstm.bias_ih_l0"]  # [256]
    bias_hh = state_dict["lstm.bias_hh_l0"]  # [256]

    # Verify dimensions
    assert weight_ih.shape[0] == 4 * hidden_size, (
        f"Expected {4 * hidden_size}, got {weight_ih.shape[0]}"
    )

    # Split weights for each gate
    # PyTorch order: input, forget, cell, output
    # Each gate gets hidden_size rows

    burn_weights = {}
    json_burn_weights = {}

    # Split input-to-hidden weights
    w_ii, w_if, w_ig, w_io = torch.chunk(weight_ih, 4, dim=0)

    # Split hidden-to-hidden weights
    w_hi, w_hf, w_hg, w_ho = torch.chunk(weight_hh, 4, dim=0)

    # Split input biases
    b_ii, b_if, b_ig, b_io = torch.chunk(bias_ih, 4, dim=0)

    # Split hidden biases
    b_hi, b_hf, b_hg, b_ho = torch.chunk(bias_hh, 4, dim=0)

    # IMPORTANT: Transpose weights because Burn uses [input_size, output_size]
    # while PyTorch uses [output_size, input_size]

    # Map to Burn's expected structure with transposed weights
    burn_weights.update(
        {
            # Input gate
            "lstm.input_gate.input_transform.weight": w_ii,  # Transpose!
            "lstm.input_gate.input_transform.bias": b_ii,
            "lstm.input_gate.hidden_transform.weight": w_hi,  # Transpose!
            "lstm.input_gate.hidden_transform.bias": b_hi,
            # Forget gate
            "lstm.forget_gate.input_transform.weight": w_if,  # Transpose!
            "lstm.forget_gate.input_transform.bias": b_if,
            "lstm.forget_gate.hidden_transform.weight": w_hf,  # Transpose!
            "lstm.forget_gate.hidden_transform.bias": b_hf,
            # Cell gate (called cell_gate in Burn)
            "lstm.cell_gate.input_transform.weight": w_ig,  # Transpose!
            "lstm.cell_gate.input_transform.bias": b_ig,
            "lstm.cell_gate.hidden_transform.weight": w_hg,  # Transpose!
            "lstm.cell_gate.hidden_transform.bias": b_hg,
            # Output gate
            "lstm.output_gate.input_transform.weight": w_io,  # Transpose!
            "lstm.output_gate.input_transform.bias": b_io,
            "lstm.output_gate.hidden_transform.weight": w_ho,  # Transpose!
            "lstm.output_gate.hidden_transform.bias": b_ho,
        }
    )

    json_burn_weights.update(
        {
            # Input gate
            "lstm.input_gate.input_transform.weight": w_ii.transpose(0, 1),  # Transpose!
            "lstm.input_gate.input_transform.bias": b_ii,
            "lstm.input_gate.hidden_transform.weight": w_hi.transpose(0, 1),  # Transpose!
            "lstm.input_gate.hidden_transform.bias": b_hi,
            # Forget gate
            "lstm.forget_gate.input_transform.weight": w_if.transpose(0, 1),  # Transpose!
            "lstm.forget_gate.input_transform.bias": b_if,
            "lstm.forget_gate.hidden_transform.weight": w_hf.transpose(0, 1),  # Transpose!
            "lstm.forget_gate.hidden_transform.bias": b_hf,
            # Cell gate (called cell_gate in Burn)
            "lstm.cell_gate.input_transform.weight": w_ig.transpose(0, 1),  # Transpose!
            "lstm.cell_gate.input_transform.bias": b_ig,
            "lstm.cell_gate.hidden_transform.weight": w_hg.transpose(0, 1),  # Transpose!
            "lstm.cell_gate.hidden_transform.bias": b_hg,
            # Output gate
            "lstm.output_gate.input_transform.weight": w_io.transpose(0, 1),  # Transpose!
            "lstm.output_gate.input_transform.bias": b_io,
            "lstm.output_gate.hidden_transform.weight": w_ho.transpose(0, 1),  # Transpose!
            "lstm.output_gate.hidden_transform.bias": b_ho,
        }
    )

    # Add the linear head weights (check if these need transposing too)
    # Linear layers in PyTorch are [out_features, in_features]
    # Burn expects [in_features, out_features]
    burn_weights["head.weight"] = state_dict["head.net.0.weight"]  # Transpose!
    burn_weights["head.bias"] = state_dict["head.net.0.bias"]

    print("\nConverted Burn keys:")
    for key, tensor in burn_weights.items():
        print(f"  {key}: {tensor.shape}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Also save metadata for verification
    metadata = {
        "hidden_size": hidden_size,
        "input_size": weight_ih.shape[1],
        "output_size": burn_weights["head.weight"].shape[1],
        "input_names": cfg["dynamic_inputs"] + cfg["static_attributes"],
        "output_names": cfg["target_variables"],
    }
    json_weights_path = Path(output_path).parent / "weights.json"
    for key, value in metadata.items():
        json_burn_weights[key] = value
    with open(json_weights_path, "w") as f:
        json.dump(json_burn_weights, f, cls=EncodeTensor)

    # Save in PyTorch format that Burn can read
    torch.save(burn_weights, output_path)
    print(f"\nSaved converted weights to {output_path}")

    # Also save metadata for verification
    metadata = {
        "hidden_size": hidden_size,
        "input_size": weight_ih.shape[1],
        "output_size": burn_weights["head.weight"].shape[1],
        "original_keys": list(state_dict.keys()),
        "burn_keys": list(burn_weights.keys()),
        "input_names": cfg["dynamic_inputs"] + cfg["static_attributes"],
        "output_names": cfg["target_variables"],
    }

    metadata_path = Path(output_path).with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def export_training_scalars(cfg_path: Path, output_path: Path = None):
    """Export training scalars to JSON for Rust implementation"""
    if output_path is None:
        output_path = Path(cfg_path).parent / "burn" / "train_data_scaler.json"

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    cfg_path = Path(cfg_path)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # fix relative paths to be relative to the working directory of the script
    cwd = Path.cwd()
    os.chdir(cfg_path.parent.parent)
    cfg["run_dir"] = Path(cfg["run_dir"]).resolve()
    os.chdir(cwd)
    hidden_size = cfg["hidden_size"]

    # Load training scaler file
    scaler_file = cfg["run_dir"] / "train_data/train_data_scaler.yml"
    with open(scaler_file, "r") as f:
        train_data_scaler = yaml.safe_load(f)

    # Extract means and stds
    input_mean = []
    input_std = []

    # Dynamic inputs
    for name in cfg["dynamic_inputs"]:
        input_mean.append(train_data_scaler["xarray_feature_center"]["data_vars"][name]["data"])
        input_std.append(train_data_scaler["xarray_feature_scale"]["data_vars"][name]["data"])

    # Static attributes
    for name in cfg["static_attributes"]:
        input_mean.append(train_data_scaler["attribute_means"][name])
        input_std.append(train_data_scaler["attribute_stds"][name])

    # Output scalars
    output_var = cfg["target_variables"][0]
    output_mean = train_data_scaler["xarray_feature_center"]["data_vars"][output_var]["data"]
    output_std = train_data_scaler["xarray_feature_scale"]["data_vars"][output_var]["data"]

    scalars = {
        "input_mean": input_mean,
        "input_std": input_std,
        "output_mean": output_mean,
        "output_std": output_std,
    }

    with open(output_path, "w") as f:
        json.dump(scalars, f, indent=2)

    print(f"Training scalars exported to {output_path}")
    return hidden_size


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_weights.py input.pt model_config.yml")
        sys.exit(1)

    model_path = sys.argv[1]
    config_path = sys.argv[2]
    hidden_size = export_training_scalars(config_path)

    convert_pytorch_to_burn(model_path, hidden_size, config_path)
"#;

    let script_path = std::env::temp_dir().join("convert_weights.py");
    fs::write(&script_path, script)?;
    dbg!(&weights_path);
    dbg!(&config_path);
    dbg!(&script_path);
    let output = Command::new("uv")
        .arg("run")
        .arg("-p")
        .arg("3.9")
        .arg("--with")
        .arg("pyyaml")
        .arg("--with")
        .arg("numpy")
        .arg("--with")
        .arg("torch")
        .arg("--extra-index-url")
        .arg("https://download.pytorch.org/whl/cpu")
        .arg(&script_path)
        .arg(&weights_path)
        .arg(&config_path)
        .output()?;

    if !output.status.success() {
        anyhow::bail!(
            "Weight conversion failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("Weight conversion successful");
    Ok(())
}
