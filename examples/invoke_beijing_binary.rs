use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> anyhow::Result<()> {
    // Read the example JSON request (using new Extract format)
    let request_json = fs::read_to_string("examples/beijing_training_request_new.json")?;

    println!("Invoking Beijing training binary with request:");
    println!("{}", request_json);
    println!("\n---\n");

    // Spawn the training binary
    let mut child = Command::new("./target/release/beijing")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Send the request JSON via stdin
    {
        let mut stdin = child.stdin.take().ok_or_else(|| {
            anyhow::anyhow!("Failed to open stdin")
        })?;
        stdin.write_all(request_json.as_bytes())?;
    }

    // Wait for the binary to finish and collect output
    let output = child.wait_with_output()?;

    // Print stderr (logs)
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        println!("=== STDERR (logs) ===\n{}", stderr);
    }

    // Print stdout (response)
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("=== STDOUT (response) ===\n{}", stdout);

    // Check if successful
    if output.status.success() {
        println!("\n✓ Binary completed successfully");
        let response: serde_json::Value = serde_json::from_str(&stdout)?;
        println!("Fitness value: {}", response["fitness"]);
        println!("Genotype ID: {}", response["genotype_id"]);
    } else {
        println!("\n✗ Binary failed with exit code: {}", output.status);
    }

    Ok(())
}
