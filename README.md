# bmi-burn-lstm

A Rust implementation of a BMI (Basic Model Interface) adapter for LSTM-based streamflow prediction, providing a drop-in replacement for the Python BMI LSTM implementations found in [CIROH-UA/lstm](https://github.com/CIROH-UA/lstm), [NOAA-OWP/lstm](https://github.com/NOAA-OWP/lstm), and [jmframe/lstm](https://github.com/jmframe/lstm).

## Status

**⚠️ Active Development** - This package is currently under active development. The outputs produced match those of the Python BMI LSTM implementation and it runs faster, but the code quality is not yet at the same level as the Python implementation.

## Acknowledgments

This project would not have been possible without:
- Contributors to [NOAA-OWP/lstm](https://github.com/NOAA-OWP/lstm) and [jmframe/lstm](https://github.com/jmframe/lstm)
- [@aaraney](https://github.com/aaraney) for the excellent [bmi-rs](https://github.com/aaraney/bmi-rs) Rust BMI bindings
- The NextGen and CIROH communities

## Features

- Native Rust implementation using the [Burn](https://burn.dev/) deep learning framework
- Compatible with existing PyTorch model weights from jmframe/lstm and NextGen-In-A-Box
- Drop-in replacement for Python BMI LSTM adapter
- Supports ensemble model configurations
- Built on [bmi-rs](https://github.com/aaraney/bmi-rs) Rust BMI bindings

## Dependencies

- Rust 2024 edition
- Local clone of [bmi-rs](https://github.com/aaraney/bmi-rs)
- [Astral UV](https://docs.astral.sh/uv/) installed and accessible on the system path

## Usage

The adapter is designed to work with NextGen framework configurations:

```bash
# Install Astral UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install rust
curl https://sh.rustup.rs -sSf | bash -s -- -y && . "$HOME/.cargo/env"

git clone https://github.com/aaraney/bmi-rs
git clone https://github.com/ciroh-ua/bmi-burn-lstm

cd bmi-burn-lstm
cargo build --release
# the shared object is located at target/release/libbmi_burn_lstm.so
```

## Model Compatibility

This implementation works with:
- Trained models from jmframe/lstm
- Models included in NextGen-In-A-Box
- Any PyTorch LSTM model following the [neuralhydrology 1 layer lstm + 1 linear layer architecture](https://github.com/CIROH-UA/lstm/blob/b6fdbfb2cae1528f4a0588ab0fd29d99103fc0b6/lstm/nextgen_cuda_lstm.py#L13)

Weight conversion from PyTorch to Burn format is handled automatically on first run.

## Nextgen realization
To use the model update your NextGen configuration file with the module like this:
```json
{
  "global": {
    "formulations": [
      {
        "name": "bmi_multi",
        "params": {
          "name": "bmi_multi",
          "model_type_name": "lstm",
          "forcing_file": "",
          "init_config": "",
          "allow_exceed_end_time": true,
          "main_output_variable": "land_surface_water__runoff_depth",
          "modules": [
            {
              "name": "bmi_c",
              "params": {
                "name": "bmi_c",
                "model_type_name": "bmi_rust",
                "init_config": "./config/cat_config/lstm/{{id}}.yml",
                "allow_exceed_end_time": true,
                "main_output_variable": "land_surface_water__runoff_depth",
                "uses_forcing_file": false,
                "registration_function": "register_bmi_lstm",
                "library_file": "/dmod/shared_libs/libbmi_burn_lstm.so"
              }
            }
          ]
        }
      }
    ],
    "forcing": {
      "path": "./forcings/forcings.nc",
      "provider": "NetCDF",
      "enable_cache": false
    }
  },
  "time": {
    "start_time": "2010-01-01 00:00:00",
    "end_time": "2011-01-01 00:00:00",
    "output_interval": 3600
  },
  "routing": {
    "t_route_config_file_with_path": "./config/troute.yaml"
  },
  "remotes_enabled": false,
  "output_root": "./outputs/ngen"
}
```

## NGIAB patch
To patch this into nextgen in a box, run this Dockerfile
```Dockerfile
FROM awiciroh/ciroh-ngen-image AS build
RUN dnf install -y gcc clang git
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
WORKDIR /build
RUN git clone https://github.com/aaraney/bmi-rs
RUN git clone https://github.com/ciroh-ua/bmi-burn-lstm
WORKDIR /build/bmi-burn-lstm
RUN cargo build --release

FROM awiciroh/ciroh-ngen-image AS final
COPY --from=build /build/bmi-burn-lstm/target/release/libbmi_burn_lstm.so /dmod/shared_libs/libbmi_burn_lstm.so
```
```bash
docker build -t ngiab -f Dockerfile .
```

## License

[License information pending]
