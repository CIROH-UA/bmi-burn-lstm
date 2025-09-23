use bmi_rs::bmi::{Bmi, BmiResult, Location, RefValues, ValueType, Values, register_model};
use ffi;

/// Simple test model that outputs timestep * 2
pub struct TestModel {
    // Model state
    current_time: f64,
    start_time: f64,
    end_time: f64,
    time_step: f64,
    time_units: String,

    // Input/Output variables
    input_value: Vec<f64>,
    output_value: Vec<f64>,

    // Variable metadata
    component_name: String,
    input_var_names: Vec<&'static str>,
    output_var_names: Vec<&'static str>,
}

// Macro to handle variable lookup with error
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

// Macro to get mutable reference to variable by name
macro_rules! get_var_mut {
    ($self:expr, $name:expr) => {
        match $name {
            "input_value" => Ok(&mut $self.input_value),
            "output_value" => Ok(&mut $self.output_value),
            _ => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", $name),
            ))),
        }
    };
}

// Macro to get immutable reference to variable by name
macro_rules! get_var_ref {
    ($self:expr, $name:expr) => {
        match $name {
            "input_value" => &$self.input_value,
            "output_value" => &$self.output_value,
            _ => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Variable {} not found", $name),
                )))
            }
        }
    };
}

// Macro to validate RefValues type and extract f64 slice
macro_rules! expect_f64 {
    ($src:expr) => {
        match $src {
            RefValues::F64(values) => Ok(values),
            _ => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Type mismatch: expected F64",
            ))),
        }
    };
}

impl TestModel {
    pub fn new() -> Self {
        TestModel {
            current_time: 0.0,
            start_time: 0.0,
            end_time: 100.0,
            time_step: 1.0,
            time_units: "seconds".to_string(),

            input_value: vec![0.0],
            output_value: vec![0.0],

            component_name: "TestModel".to_string(),
            input_var_names: vec!["input_value"],
            output_var_names: vec!["output_value"],
        }
    }

    /// Internal update logic
    fn calculate_output(&mut self) {
        // Simple logic: output = current_time * 2
        self.output_value[0] = self.current_time * 2.0;
    }
}

impl Bmi for TestModel {
    /* Initialize, run, finalize (IRF) */
    fn initialize(&mut self, config_file: &str) -> BmiResult<()> {
        println!("Initializing TestModel with config: {}", config_file);

        // Reset to initial state
        self.current_time = self.start_time;
        self.input_value = vec![0.0];
        self.output_value = vec![0.0];

        // Calculate initial output
        self.calculate_output();

        Ok(())
    }

    fn update(&mut self) -> BmiResult<()> {
        self.current_time += self.time_step;
        self.calculate_output();
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

            // Prevent overshooting
            if self.current_time > then {
                self.current_time = then;
                self.calculate_output();
            }
        }

        Ok(())
    }

    fn finalize(&mut self) -> BmiResult<()> {
        println!("Finalizing TestModel");
        Ok(())
    }

    /* Exchange items */
    fn get_component_name(&self) -> &str {
        &self.component_name
    }

    fn get_input_item_count(&self) -> u32 {
        self.input_var_names.len() as u32
    }

    fn get_output_item_count(&self) -> u32 {
        self.output_var_names.len() as u32
    }

    fn get_input_var_names(&self) -> &[&str] {
        &self.input_var_names
    }

    fn get_output_var_names(&self) -> &[&str] {
        &self.output_var_names
    }

    /* Variable information */
    fn get_var_grid(&self, name: &str) -> BmiResult<i32> {
        // All variables are on grid 0 (scalar grid)
        match_var!(name,
            "input_value" | "output_value" => Ok(0)
        )
    }

    fn get_var_type(&self, name: &str) -> BmiResult<ValueType> {
        match_var!(name,
            "input_value" | "output_value" => Ok(ValueType::F64)
        )
    }

    fn get_var_units(&self, name: &str) -> BmiResult<&str> {
        match_var!(name,
            "input_value" => Ok("dimensionless"),
            "output_value" => Ok("dimensionless")
        )
    }

    fn get_var_location(&self, name: &str) -> BmiResult<Location> {
        match_var!(name,
            "input_value" | "output_value" => Ok(Location::Node)
        )
    }

    /* Time information */
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
        &self.time_units
    }

    fn get_time_step(&self) -> f64 {
        self.time_step
    }

    /* Getters */
    fn get_value(&self, name: &str) -> BmiResult<Values> {
        match_var!(name,
            "input_value" => Ok(Values::F64(self.input_value.clone())),
            "output_value" => Ok(Values::F64(self.output_value.clone()))
        )
    }

    fn get_value_ptr(&self, name: &str) -> BmiResult<RefValues<'_>> {
        match_var!(name,
            "input_value" => Ok(RefValues::F64(&self.input_value)),
            "output_value" => Ok(RefValues::F64(&self.output_value))
        )
    }

    fn get_value_at_indices(&self, name: &str, inds: &[u32]) -> BmiResult<Values> {
        let full_values = get_var_ref!(self, name);

        // Extract values at indices
        let mut result = Vec::with_capacity(inds.len());
        for &idx in inds {
            if (idx as usize) >= full_values.len() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Index {} out of bounds", idx),
                )));
            }
            result.push(full_values[idx as usize]);
        }

        Ok(Values::F64(result))
    }

    /* Setters */
    fn set_value(&mut self, name: &str, src: RefValues) -> BmiResult<()> {
        let values = expect_f64!(src)?;
        let target = get_var_mut!(self, name)?;

        if values.len() != target.len() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Source array size mismatch",
            )));
        }

        target.copy_from_slice(values);
        Ok(())
    }

    fn set_value_at_indices(&mut self, name: &str, inds: &[u32], src: RefValues) -> BmiResult<()> {
        let values = expect_f64!(src)?;
        let target = get_var_mut!(self, name)?;

        if values.len() != inds.len() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Source array size doesn't match indices count",
            )));
        }

        for (i, &idx) in inds.iter().enumerate() {
            if (idx as usize) >= target.len() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Index {} out of bounds", idx),
                )));
            }
            target[idx as usize] = values[i];
        }
        Ok(())
    }
}

// Export function for C binding
#[unsafe(no_mangle)]
pub extern "C" fn register_bmi_test_model(handle: *mut ffi::Bmi) -> *mut ffi::Bmi {
    let model = TestModel::new();
    register_model(handle, model);
    handle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize() {
        let mut model = TestModel::new();
        assert!(model.initialize("test.config").is_ok());
        assert_eq!(model.get_current_time(), 0.0);
    }

    #[test]
    fn test_update() {
        let mut model = TestModel::new();
        model.initialize("test.config").unwrap();

        // Initial state: time=0, output=0
        assert_eq!(model.get_current_time(), 0.0);
        if let Ok(Values::F64(output)) = model.get_value("output_value") {
            assert_eq!(output[0], 0.0);
        }

        // After update: time=1, output=2
        model.update().unwrap();
        assert_eq!(model.get_current_time(), 1.0);
        if let Ok(Values::F64(output)) = model.get_value("output_value") {
            assert_eq!(output[0], 2.0);
        }

        // After another update: time=2, output=4
        model.update().unwrap();
        assert_eq!(model.get_current_time(), 2.0);
        if let Ok(Values::F64(output)) = model.get_value("output_value") {
            assert_eq!(output[0], 4.0);
        }
    }

    #[test]
    fn test_update_until() {
        let mut model = TestModel::new();
        model.initialize("test.config").unwrap();

        model.update_until(5.0).unwrap();
        assert_eq!(model.get_current_time(), 5.0);
        if let Ok(Values::F64(output)) = model.get_value("output_value") {
            assert_eq!(output[0], 10.0); // 5.0 * 2
        }
    }

    #[test]
    fn test_set_get_value() {
        let mut model = TestModel::new();
        model.initialize("test.config").unwrap();

        // Set input value
        let new_input = vec![42.0];
        model
            .set_value("input_value", RefValues::F64(&new_input))
            .unwrap();

        // Get input value
        if let Ok(Values::F64(input)) = model.get_value("input_value") {
            assert_eq!(input[0], 42.0);
        }
    }
}
