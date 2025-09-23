use bmi_rs::bmi::{Bmi, BmiResult, Location, RefValues, ValueType, Values, register_model};
use burn::nn::LstmState;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use glob::glob;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

mod nextgen_lstm;
mod python;
use nextgen_lstm::{NextgenLstm, vec_to_tensor};
use python::convert_model;

#[derive(Debug, Serialize, Deserialize)]
struct ModelMetadata {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainingScalars {
    input_mean: Vec<f32>,
    input_std: Vec<f32>,
    output_mean: f32,
    output_std: f32,
}

// Macro to handle variable lookup
macro_rules! match_var {
    ($name:expr, $($pattern:pat => $result:expr),+ $(,)?) => {
        match $name {
            $($pattern => $result,)+
            _ => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", $name)
            )))
        }
    };
}

pub struct LstmBmi<B: Backend> {
    // Model components
    model: Option<NextgenLstm<B>>,
    device: B::Device,
    metadata: Option<ModelMetadata>,
    scalars: Option<TrainingScalars>,
    lstm_state: Option<LstmState<B, 2>>,

    // Configuration
    config_path: String,
    area_sqkm: f32,
    output_scale_factor_cms: f32,

    // BMI variables stored as simple HashMap
    variables: HashMap<String, Vec<f64>>,

    // Variable metadata
    input_var_names: Vec<&'static str>,
    output_var_names: Vec<&'static str>,

    // Time tracking
    current_time: f64,
    start_time: f64,
    end_time: f64,
    time_step: f64,
}

impl<B: Backend> LstmBmi<B> {
    pub fn new(device: B::Device) -> Self {
        // Define all BMI variable names
        let input_vars = vec![
            "atmosphere_water__liquid_equivalent_precipitation_rate",
            "land_surface_air__temperature",
            "land_surface_radiation~incoming~longwave__energy_flux",
            "land_surface_air__pressure",
            "atmosphere_air_water~vapor__relative_saturation",
            "land_surface_radiation~incoming~shortwave__energy_flux",
            "land_surface_wind__x_component_of_velocity",
            "land_surface_wind__y_component_of_velocity",
            // "basin__mean_of_elevation",
            // "basin__mean_of_slope",
        ];

        let output_vars = vec![
            "land_surface_water__runoff_volume_flux",
            "land_surface_water__runoff_depth",
        ];

        // Initialize variables HashMap
        let mut variables = HashMap::new();
        for var in &input_vars {
            variables.insert(var.to_string(), vec![0.0]);
        }
        // add variables here so bmi can't see them
        variables.insert("basin__mean_of_elevation".to_string(), vec![0.0]);
        variables.insert("basin__mean_of_slope".to_string(), vec![0.0]);

        for var in &output_vars {
            variables.insert(var.to_string(), vec![0.0]);
        }

        LstmBmi {
            model: None,
            device,
            metadata: None,
            scalars: None,
            lstm_state: None,
            config_path: String::new(),
            area_sqkm: 0.0,
            output_scale_factor_cms: 0.0,
            variables,
            input_var_names: input_vars,
            output_var_names: output_vars,
            current_time: 0.0,
            start_time: 0.0,
            end_time: 36000.0, // 10 hours default
            time_step: 3600.0, // 1 hour
        }
    }

    fn internal_to_external_name(&self, internal: &str) -> String {
        let mapping = [
            (
                "DLWRF_surface",
                "land_surface_radiation~incoming~longwave__energy_flux",
            ),
            ("PRES_surface", "land_surface_air__pressure"),
            (
                "SPFH_2maboveground",
                "atmosphere_air_water~vapor__relative_saturation",
            ),
            (
                "APCP_surface",
                "atmosphere_water__liquid_equivalent_precipitation_rate",
            ),
            (
                "DSWRF_surface",
                "land_surface_radiation~incoming~shortwave__energy_flux",
            ),
            ("TMP_2maboveground", "land_surface_air__temperature"),
            (
                "UGRD_10maboveground",
                "land_surface_wind__x_component_of_velocity",
            ),
            (
                "VGRD_10maboveground",
                "land_surface_wind__y_component_of_velocity",
            ),
            ("elev_mean", "basin__mean_of_elevation"),
            ("slope_mean", "basin__mean_of_slope"),
        ];

        mapping
            .iter()
            .find(|(k, _)| *k == internal)
            .map(|(_, v)| v.to_string())
            .unwrap_or_else(|| internal.to_string())
    }

    fn run_model(&mut self) -> BmiResult<()> {
        let model = self.model.as_ref().ok_or("Model not initialized")?;
        let metadata = self.metadata.as_ref().ok_or("Metadata not loaded")?;
        let scalars = self.scalars.as_ref().ok_or("Scalars not loaded")?;

        // Gather inputs in the correct order
        let mut inputs = Vec::new();
        for name in &metadata.input_names {
            let bmi_name = self.internal_to_external_name(name);
            let value = self
                .variables
                .get(&bmi_name)
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(0.0) as f32;
            inputs.push(value);
        }

        // dbg!(&inputs);

        // Scale inputs
        let scaled_inputs: Vec<f32> = inputs
            .iter()
            .zip(&scalars.input_mean)
            .zip(&scalars.input_std)
            .map(
                |((val, mean), std)| {
                    if *std != 0.0 { (val - mean) / std } else { 0.0 }
                },
            )
            .collect();

        // dbg!(&scaled_inputs);

        // Create input tensor
        let input_tensor_data = vec_to_tensor(&scaled_inputs, vec![1, 1, metadata.input_size]);
        let input_tensor = Tensor::from_data(input_tensor_data, &self.device);

        // Forward pass
        let (output, new_state) = model.forward(input_tensor, self.lstm_state.take());
        // dbg!(&new_state.hidden);
        // dbg!(&new_state.cell);
        self.lstm_state = Some(new_state);

        // Process output
        let output_vec: Vec<f32> = output.into_data().to_vec().unwrap();
        let output_value = output_vec[0];

        // Denormalize
        let surface_runoff_mm = (output_value * scalars.output_std + scalars.output_mean).max(0.0);

        // Convert to output units
        let surface_runoff_m = surface_runoff_mm / 1000.0;
        let surface_runoff_volume_m3_s = surface_runoff_mm * self.output_scale_factor_cms;

        // Set outputs
        self.variables.insert(
            "land_surface_water__runoff_depth".to_string(),
            vec![surface_runoff_m as f64],
        );
        self.variables.insert(
            "land_surface_water__runoff_volume_flux".to_string(),
            vec![surface_runoff_volume_m3_s as f64],
        );

        Ok(())
    }
}

impl<B: Backend> Bmi for LstmBmi<B> {
    fn initialize(&mut self, config_file: &str) -> BmiResult<()> {
        println!("Initializing LSTM BMI with config: {}", config_file);
        self.config_path = config_file.to_string();

        // Load configuration
        let config_path = Path::new(config_file);
        let config_str = fs::read_to_string(config_path)?;
        let config: serde_yaml::Value = serde_yaml::from_str(&config_str)?;

        // Get training config path
        let training_config_path = Path::new(
            config["train_cfg_file"]
                .get(0)
                .ok_or("Missing train_cfg_file")?
                .as_str()
                .ok_or("train_cfg_file not a string")?,
        );

        let training_config = fs::read_to_string(training_config_path)?;
        let training_config: serde_yaml::Value = serde_yaml::from_str(&training_config)?;

        // Find model path
        let model_dir = training_config["run_dir"]
            .as_str()
            .ok_or("Missing run_dir")?
            .replace(
                "..",
                training_config_path
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .to_str()
                    .unwrap(),
            );

        let model_path = glob(&format!("{}/model_*.pt", model_dir))?
            .next()
            .ok_or("No model file found")??;

        // Convert weights if needed
        let model_folder = model_path.parent().unwrap();
        let converted_path = model_folder
            .join("burn")
            .join(model_path.file_name().unwrap());

        // if !converted_path.exists() {
        println!("Converting PyTorch weights to Burn format...");
        convert_model(&model_path, &training_config_path)?;
        // }

        // Load metadata
        let metadata_str = fs::read_to_string(converted_path.with_extension("json"))?;
        self.metadata = Some(serde_json::from_str(&metadata_str)?);

        // Load scalars
        let scalars_str =
            fs::read_to_string(model_folder.join("burn").join("train_data_scaler.json"))?;
        self.scalars = Some(serde_json::from_str(&scalars_str)?);

        // Load model
        let metadata = self.metadata.as_ref().unwrap();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(converted_path.into(), &self.device)?;

        let mut model = NextgenLstm::init(
            &self.device,
            metadata.input_size,
            metadata.hidden_size,
            metadata.output_size,
        );
        model = model.load_record(record);
        model.load_json_weights(
            &self.device,
            model_folder
                .join("burn")
                .join("weights.json")
                .to_str()
                .unwrap(),
        );

        self.model = Some(model);

        // Get area from config
        self.area_sqkm = config
            .get("area_sqkm")
            .ok_or("Missing area_sqkm")?
            .as_f64()
            .ok_or("area_sqkm not a number")? as f32;

        self.output_scale_factor_cms =
            (1.0 / 1000.0) * (self.area_sqkm * 1000.0 * 1000.0) * (1.0 / 3600.0);

        // Set static inputs from config
        let elevation = config
            .get("elev_mean")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let slope = config
            .get("slope_mean")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        self.variables
            .insert("basin__mean_of_elevation".to_string(), vec![elevation]);
        self.variables
            .insert("basin__mean_of_slope".to_string(), vec![slope]);

        // Reset time
        self.current_time = self.start_time;

        println!("LSTM BMI initialized successfully");
        Ok(())
    }

    fn update(&mut self) -> BmiResult<()> {
        self.run_model()?;
        self.current_time += self.time_step;
        Ok(())
    }

    fn update_until(&mut self, then: f64) -> BmiResult<()> {
        if then < self.current_time {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Target time {} is before current time {}",
                    then, self.current_time
                ),
            )));
        }

        while self.current_time < then {
            self.update()?;
            if self.current_time > then {
                self.current_time = then;
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> BmiResult<()> {
        println!("Finalizing LSTM BMI");
        self.model = None;
        self.lstm_state = None;
        Ok(())
    }

    fn get_component_name(&self) -> &str {
        "NextGen LSTM BMI"
    }

    fn get_input_item_count(&self) -> u32 {
        self.input_var_names.len() as u32
    }

    fn get_output_item_count(&self) -> u32 {
        self.output_var_names.len() as u32
    }

    fn get_input_var_names(&self) -> &[&str] {
        println!("Getting input variable names");
        &self.input_var_names
    }

    fn get_output_var_names(&self) -> &[&str] {
        println!("Getting output variable names");
        &self.output_var_names
    }

    fn get_var_grid(&self, name: &str) -> BmiResult<i32> {
        if self.variables.contains_key(name) {
            Ok(0) // Scalar grid
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", name),
            )))
        }
    }

    fn get_var_type(&self, name: &str) -> BmiResult<ValueType> {
        if self.variables.contains_key(name) {
            Ok(ValueType::F64)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", name),
            )))
        }
    }

    fn get_var_units(&self, name: &str) -> BmiResult<&str> {
        match_var!(name,
            "atmosphere_water__liquid_equivalent_precipitation_rate" => Ok("mm h-1"),
            "land_surface_air__temperature" => Ok("degK"),
            "land_surface_radiation~incoming~longwave__energy_flux" => Ok("W m-2"),
            "land_surface_air__pressure" => Ok("Pa"),
            "atmosphere_air_water~vapor__relative_saturation" => Ok("kg kg-1"),
            "land_surface_radiation~incoming~shortwave__energy_flux" => Ok("W m-2"),
            "land_surface_wind__x_component_of_velocity" => Ok("m s-1"),
            "land_surface_wind__y_component_of_velocity" => Ok("m s-1"),
            "basin__mean_of_elevation" => Ok("m"),
            "basin__mean_of_slope" => Ok("m km-1"),
            "land_surface_water__runoff_volume_flux" => Ok("m3 s-1"),
            "land_surface_water__runoff_depth" => Ok("m")
        )
    }

    fn get_var_location(&self, name: &str) -> BmiResult<Location> {
        if self.variables.contains_key(name) {
            Ok(Location::Node)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", name),
            )))
        }
    }

    fn get_current_time(&self) -> f64 {
        self.current_time
    }

    fn get_start_time(&self) -> f64 {
        self.start_time
    }

    fn get_end_time(&self) -> f64 {
        self.end_time
    }

    fn get_time_units(&self) -> &str {
        "seconds"
    }

    fn get_time_step(&self) -> f64 {
        self.time_step
    }

    fn get_value(&self, name: &str) -> BmiResult<Values> {
        Ok(self
            .variables
            .get(name)
            .map(|v| Values::F64(v.clone()))
            .ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Variable {} not found", name),
                ))
            })?)
    }

    fn get_value_ptr(&self, name: &str) -> BmiResult<RefValues<'_>> {
        Ok(self
            .variables
            .get(name)
            .map(|v| RefValues::F64(v))
            .ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Variable {} not found", name),
                ))
            })?)
    }

    fn get_value_at_indices(&self, name: &str, inds: &[u32]) -> BmiResult<Values> {
        let values = self.variables.get(name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", name),
            )
        })?;

        let mut result = Vec::with_capacity(inds.len());
        for &idx in inds {
            if (idx as usize) >= values.len() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Index {} out of bounds", idx),
                )));
            }
            result.push(values[idx as usize]);
        }
        Ok(Values::F64(result))
    }

    fn set_value(&mut self, name: &str, src: RefValues) -> BmiResult<()> {
        if let RefValues::F64(values) = src {
            if let Some(var) = self.variables.get_mut(name) {
                if values.len() != var.len() {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Source array size mismatch",
                    )));
                }
                var.copy_from_slice(values);
                Ok(())
            } else {
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Variable {} not found", name),
                )))
            }
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Type mismatch: expected F64",
            )))
        }
    }

    fn set_value_at_indices(&mut self, name: &str, inds: &[u32], src: RefValues) -> BmiResult<()> {
        if let RefValues::F64(values) = src {
            if values.len() != inds.len() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Source array size doesn't match indices count",
                )));
            }

            let var = self.variables.get_mut(name).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Variable {} not found", name),
                )
            })?;

            for (i, &idx) in inds.iter().enumerate() {
                if (idx as usize) >= var.len() {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Index {} out of bounds", idx),
                    )));
                }
                var[idx as usize] = values[i];
            }
            Ok(())
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Type mismatch: expected F64",
            )))
        }
    }
}

// Export function for C binding
#[unsafe(no_mangle)]
pub extern "C" fn register_bmi_lstm(handle: *mut ffi::Bmi) -> *mut ffi::Bmi {
    // type Backend = burn::backend::NdArray;
    type Backend = burn::backend::Candle;
    let device = Default::default();

    let model = LstmBmi::<Backend>::new(device);
    register_model(handle, model);
    handle
}
