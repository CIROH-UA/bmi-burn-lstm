use anyhow::Result;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Convert PyTorch weights to Burn format using Python script
pub fn convert_model(weights_path: &Path, config_path: &Path) -> Result<()> {
    // Python script content
    const PYTHON_SCRIPT: &str = include_str!("../scripts/convert.py");

    let script_path = std::env::temp_dir().join("convert_weights.py");
    fs::write(&script_path, PYTHON_SCRIPT)?;
    // dbg!(&weights_path);
    // dbg!(&config_path);
    // dbg!(&script_path);
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
