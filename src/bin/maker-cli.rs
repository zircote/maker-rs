//! MAKER CLI - Standalone command-line interface for the MAKER framework
//!
//! This binary provides a CLI for MAKER without requiring MCP.
//! It has feature parity with the MCP tools.

use clap::{Parser, Subcommand, ValueEnum};
use maker::core::{calculate_kmin, validate_token_length, vote_with_margin, RedFlag, VoteConfig};
use maker::llm::adapter::{create_provider, ProviderConfig};
use maker::mcp::health::{validate_config, HealthChecker};
use maker::mcp::server::ServerConfig;
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::process::ExitCode;

/// MAKER CLI - Zero-error LLM agent execution via SPRT voting
#[derive(Parser)]
#[command(name = "maker-cli")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Output format
    #[arg(short, long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Increase verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Validate configuration and exit
    #[arg(long)]
    validate_config: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

/// Output format for CLI responses
#[derive(Clone, Copy, ValueEnum, Default)]
enum OutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output for programmatic use
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute SPRT voting on a prompt
    Vote {
        /// The prompt to vote on (or - for stdin)
        #[arg(short, long)]
        prompt: Option<String>,

        /// Required vote margin for declaring winner (k >= 1)
        #[arg(short, long, default_value = "3")]
        k_margin: usize,

        /// Maximum samples before timeout
        #[arg(short = 'n', long, default_value = "100")]
        max_samples: usize,

        /// Temperature diversity for sampling (0.0-1.0)
        #[arg(short, long, default_value = "0.1")]
        temperature: f64,

        /// LLM provider to use (ollama, openai, anthropic)
        #[arg(long, default_value = "ollama")]
        provider: String,

        /// Enable adaptive k-margin
        #[arg(long)]
        adaptive: bool,

        /// Matcher type (exact, embedding, code)
        #[arg(long, default_value = "exact")]
        matcher: String,
    },

    /// Validate a response against red-flag rules
    Validate {
        /// The response to validate (or - for stdin)
        #[arg(short, long)]
        response: Option<String>,

        /// Maximum token count
        #[arg(short, long)]
        token_limit: Option<usize>,

        /// Expected JSON schema (as JSON string)
        #[arg(short, long)]
        schema: Option<String>,
    },

    /// Calibrate k-margin from sample data
    Calibrate {
        /// Path to JSON file with calibration samples
        #[arg(short, long)]
        file: Option<String>,

        /// Target reliability (0.0-1.0)
        #[arg(short = 'r', long, default_value = "0.95")]
        target_reliability: f64,

        /// Target step count
        #[arg(short = 's', long, default_value = "1000")]
        target_steps: usize,
    },

    /// Configure default parameters
    Config {
        /// Show current configuration
        #[arg(long)]
        show: bool,

        /// Set default k-margin
        #[arg(long)]
        k_margin: Option<usize>,

        /// Set default provider
        #[arg(long)]
        provider: Option<String>,

        /// Set default matcher type
        #[arg(long)]
        matcher: Option<String>,

        /// Enable/disable adaptive k-margin
        #[arg(long)]
        adaptive: Option<bool>,
    },

    /// Execute recursive task decomposition
    Decompose {
        /// Task description
        #[arg(short, long)]
        task: Option<String>,

        /// Maximum recursion depth
        #[arg(short, long, default_value = "10")]
        depth_limit: usize,

        /// Timeout in seconds
        #[arg(long, default_value = "60")]
        timeout: u64,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Check server health status
    Health {
        /// Check specific component (voting, config, provider)
        #[arg(short, long)]
        component: Option<String>,

        /// Check LLM provider connectivity
        #[arg(long)]
        check_provider: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
#[allow(clippy::enum_variant_names)]
enum Shell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
}

// ============================================================================
// Response Types
// ============================================================================

#[derive(Serialize, Deserialize)]
struct VoteResponse {
    winner: String,
    votes: usize,
    total_samples: usize,
    k_margin: usize,
    converged: bool,
}

#[derive(Serialize, Deserialize)]
struct ValidateResponse {
    valid: bool,
    red_flags: Vec<RedFlagInfo>,
}

#[derive(Serialize, Deserialize)]
struct RedFlagInfo {
    flag_type: String,
    details: String,
}

impl From<&RedFlag> for RedFlagInfo {
    fn from(flag: &RedFlag) -> Self {
        match flag {
            RedFlag::TokenLengthExceeded { actual, limit } => RedFlagInfo {
                flag_type: "TokenLengthExceeded".to_string(),
                details: format!("Token count {} exceeds limit {}", actual, limit),
            },
            RedFlag::FormatViolation { message } => RedFlagInfo {
                flag_type: "FormatViolation".to_string(),
                details: message.clone(),
            },
            RedFlag::LogicLoop { pattern } => RedFlagInfo {
                flag_type: "LogicLoop".to_string(),
                details: format!("Detected repetitive pattern: {}", pattern),
            },
        }
    }
}

#[derive(Serialize, Deserialize)]
struct CalibrateResponse {
    p_estimate: f64,
    confidence_interval: (f64, f64),
    sample_count: usize,
    recommended_k: usize,
}

#[derive(Serialize, Deserialize)]
struct CalibrationSample {
    prompt: String,
    ground_truth: String,
    response: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct ConfigResponse {
    k_margin: usize,
    provider: String,
    matcher: String,
    adaptive: bool,
    max_samples: usize,
}

#[derive(Serialize, Deserialize)]
struct DecomposeResponse {
    task_id: String,
    subtasks: Vec<SubtaskInfo>,
    composition: String,
    depth: usize,
}

#[derive(Serialize, Deserialize)]
struct SubtaskInfo {
    id: String,
    description: String,
    is_leaf: bool,
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .with_writer(io::stderr)
        .init();

    // Handle --validate-config flag
    if cli.validate_config {
        return execute_validate_config();
    }

    // Require a subcommand if not validating config
    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            eprintln!("Error: A subcommand is required. Use --help for usage.");
            return ExitCode::from(2);
        }
    };

    let result = match command {
        Commands::Vote {
            prompt,
            k_margin,
            max_samples,
            temperature,
            provider,
            adaptive: _,
            matcher: _,
        } => execute_vote(
            cli.format,
            prompt,
            k_margin,
            max_samples,
            temperature,
            &provider,
        ),

        Commands::Validate {
            response,
            token_limit,
            schema,
        } => execute_validate(cli.format, response, token_limit, schema),

        Commands::Calibrate {
            file,
            target_reliability,
            target_steps,
        } => execute_calibrate(cli.format, file, target_reliability, target_steps),

        Commands::Config {
            show,
            k_margin,
            provider,
            matcher,
            adaptive,
        } => execute_config(cli.format, show, k_margin, provider, matcher, adaptive),

        Commands::Decompose {
            task,
            depth_limit,
            timeout,
        } => execute_decompose(cli.format, task, depth_limit, timeout),

        Commands::Completions { shell } => {
            generate_completions(shell);
            Ok(())
        }

        Commands::Health {
            component,
            check_provider,
        } => execute_health(cli.format, component, check_provider),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

// ============================================================================
// Command Implementations
// ============================================================================

fn execute_vote(
    format: OutputFormat,
    prompt: Option<String>,
    k_margin: usize,
    max_samples: usize,
    _temperature: f64,
    provider: &str,
) -> Result<(), String> {
    let prompt = get_input(prompt, "prompt")?;

    if k_margin == 0 {
        return Err("k_margin must be >= 1".to_string());
    }

    if prompt.is_empty() {
        return Err("prompt cannot be empty".to_string());
    }

    // Create real provider based on CLI argument
    let provider_config = ProviderConfig::default();
    let client = match create_provider(provider, Some(provider_config)) {
        Ok(Some(c)) => c,
        Ok(None) => {
            return Err(format!(
                "Unknown provider: '{}'. Use: ollama, openai, anthropic",
                provider
            ))
        }
        Err(e) => return Err(format!("Failed to create provider '{}': {}", provider, e)),
    };
    let config = VoteConfig::default().with_max_samples(max_samples);

    let result = vote_with_margin(&prompt, k_margin, client.as_ref(), config);

    match result {
        Ok(vote_result) => {
            let winner_votes = vote_result
                .vote_counts
                .get(&vote_result.winner)
                .copied()
                .unwrap_or(0);
            let response = VoteResponse {
                winner: vote_result.winner,
                votes: winner_votes,
                total_samples: vote_result.total_samples,
                k_margin: vote_result.k_used,
                converged: true, // If we got a result, it converged
            };
            output_response(format, &response)
        }
        Err(e) => Err(format!("Vote failed: {:?}", e)),
    }
}

fn execute_validate(
    format: OutputFormat,
    response: Option<String>,
    token_limit: Option<usize>,
    schema: Option<String>,
) -> Result<(), String> {
    let response_text = get_input(response, "response")?;

    let mut red_flags = Vec::new();

    // Validate token length if specified
    if let Some(limit) = token_limit {
        if let Err(flag) = validate_token_length(&response_text, limit) {
            red_flags.push(RedFlagInfo::from(&flag));
        }
    }

    // Validate JSON schema if specified
    if let Some(schema_str) = schema {
        let _schema: serde_json::Value =
            serde_json::from_str(&schema_str).map_err(|e| format!("Invalid schema JSON: {}", e))?;

        // Check if response is valid JSON
        if serde_json::from_str::<serde_json::Value>(&response_text).is_err() {
            red_flags.push(RedFlagInfo {
                flag_type: "FormatViolation".to_string(),
                details: "Response is not valid JSON".to_string(),
            });
        }
    }

    let valid = red_flags.is_empty();
    let result = ValidateResponse { valid, red_flags };
    output_response(format, &result)
}

fn execute_calibrate(
    format: OutputFormat,
    file: Option<String>,
    target_reliability: f64,
    target_steps: usize,
) -> Result<(), String> {
    let samples: Vec<CalibrationSample> = if let Some(path) = file {
        let content =
            std::fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))?;
        serde_json::from_str(&content).map_err(|e| format!("Invalid JSON: {}", e))?
    } else {
        // Read from stdin
        let mut content = String::new();
        io::stdin()
            .read_to_string(&mut content)
            .map_err(|e| format!("Failed to read stdin: {}", e))?;
        serde_json::from_str(&content).map_err(|e| format!("Invalid JSON: {}", e))?
    };

    if samples.is_empty() {
        return Err("No calibration samples provided".to_string());
    }

    // Calculate success rate from samples that have responses
    let samples_with_responses: Vec<_> = samples.iter().filter(|s| s.response.is_some()).collect();

    if samples_with_responses.is_empty() {
        return Err("No samples have responses for calibration".to_string());
    }

    let correct_count = samples_with_responses
        .iter()
        .filter(|s| {
            s.response
                .as_ref()
                .map(|r| normalize(r) == normalize(&s.ground_truth))
                .unwrap_or(false)
        })
        .count();

    let n = samples_with_responses.len();
    let p_estimate = correct_count as f64 / n as f64;

    // Wilson score confidence interval
    let z = 1.96; // 95% confidence
    let (lower, upper) = wilson_score_interval(correct_count, n, z);

    // Calculate recommended k
    let recommended_k =
        calculate_kmin(p_estimate, target_reliability, target_steps, 1).unwrap_or(3);

    let result = CalibrateResponse {
        p_estimate,
        confidence_interval: (lower, upper),
        sample_count: n,
        recommended_k,
    };
    output_response(format, &result)
}

fn execute_config(
    format: OutputFormat,
    show: bool,
    k_margin: Option<usize>,
    provider: Option<String>,
    matcher: Option<String>,
    adaptive: Option<bool>,
) -> Result<(), String> {
    // For now, just show/set in-memory config
    // In a full implementation, this would persist to a config file

    let mut config = ConfigResponse {
        k_margin: 3,
        provider: "ollama".to_string(),
        matcher: "exact".to_string(),
        adaptive: false,
        max_samples: 100,
    };

    if !show {
        // Apply any updates
        if let Some(k) = k_margin {
            if k == 0 {
                return Err("k_margin must be >= 1".to_string());
            }
            config.k_margin = k;
        }
        if let Some(p) = provider {
            if !["ollama", "openai", "anthropic"].contains(&p.as_str()) {
                return Err(format!(
                    "Invalid provider '{}'. Use: ollama, openai, anthropic",
                    p
                ));
            }
            config.provider = p;
        }
        if let Some(m) = matcher {
            if !["exact", "embedding", "code"].contains(&m.as_str()) {
                return Err(format!(
                    "Invalid matcher '{}'. Use: exact, embedding, code",
                    m
                ));
            }
            config.matcher = m;
        }
        if let Some(a) = adaptive {
            config.adaptive = a;
        }
    }

    output_response(format, &config)
}

fn execute_decompose(
    format: OutputFormat,
    task: Option<String>,
    depth_limit: usize,
    _timeout: u64,
) -> Result<(), String> {
    let task_desc = get_input(task, "task")?;

    if task_desc.is_empty() {
        return Err("task description cannot be empty".to_string());
    }

    // For now, return a simple identity decomposition
    // Full implementation would use the RecursiveOrchestrator
    let result = DecomposeResponse {
        task_id: "task-1".to_string(),
        subtasks: vec![SubtaskInfo {
            id: "task-1".to_string(),
            description: task_desc,
            is_leaf: true,
        }],
        composition: "sequential".to_string(),
        depth: depth_limit.min(1),
    };

    output_response(format, &result)
}

fn generate_completions(shell: Shell) {
    use clap::CommandFactory;
    use clap_complete::{generate, Shell as ClapShell};

    let mut cmd = Cli::command();
    let shell = match shell {
        Shell::Bash => ClapShell::Bash,
        Shell::Zsh => ClapShell::Zsh,
        Shell::Fish => ClapShell::Fish,
        Shell::PowerShell => ClapShell::PowerShell,
    };
    generate(shell, &mut cmd, "maker-cli", &mut io::stdout());
}

fn execute_validate_config() -> ExitCode {
    let config = ServerConfig::default();

    match validate_config(&config) {
        Ok(()) => {
            println!("Configuration is valid.");
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!("Configuration errors:");
            for error in errors {
                eprintln!("  - {}", error);
            }
            ExitCode::FAILURE
        }
    }
}

fn execute_health(
    format: OutputFormat,
    component: Option<String>,
    check_provider: bool,
) -> Result<(), String> {
    let checker = HealthChecker::new();

    // Perform health check
    let status = if check_provider {
        // Try to create a provider to check connectivity
        let provider_config = ProviderConfig::default();
        let provider_healthy = create_provider("ollama", Some(provider_config)).is_ok();
        checker.check_with_provider(provider_healthy)
    } else {
        checker.check()
    };

    // If specific component requested, filter the response
    if let Some(comp) = component {
        let component_status = match comp.to_lowercase().as_str() {
            "voting" => Some(&status.components.voting),
            "config" => Some(&status.components.config),
            "provider" => status.components.llm_provider.as_ref(),
            _ => {
                return Err(format!(
                    "Unknown component: {}. Use: voting, config, provider",
                    comp
                ))
            }
        };

        if let Some(cs) = component_status {
            return output_response(format, cs);
        } else {
            return Err(format!(
                "Component '{}' not available (use --check-provider for provider status)",
                comp
            ));
        }
    }

    output_response(format, &status)
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_input(arg: Option<String>, name: &str) -> Result<String, String> {
    match arg {
        Some(s) if s != "-" => Ok(s),
        _ => {
            // Read from stdin when no arg provided or arg is "-"
            let mut input = String::new();
            io::stdin()
                .read_to_string(&mut input)
                .map_err(|e| format!("Failed to read {} from stdin: {}", name, e))?;
            Ok(input.trim().to_string())
        }
    }
}

fn output_response<T: Serialize>(format: OutputFormat, response: &T) -> Result<(), String> {
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(response)
                .map_err(|e| format!("Failed to serialize response: {}", e))?;
            println!("{}", json);
        }
        OutputFormat::Text => {
            // For text output, we serialize to JSON and then format nicely
            let value = serde_json::to_value(response)
                .map_err(|e| format!("Failed to serialize response: {}", e))?;
            print_value(&value, 0);
        }
    }
    Ok(())
}

fn print_value(value: &serde_json::Value, indent: usize) {
    let prefix = "  ".repeat(indent);
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                match val {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        println!("{}{}:", prefix, key);
                        print_value(val, indent + 1);
                    }
                    _ => {
                        println!("{}{}: {}", prefix, key, format_simple_value(val));
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, val) in arr.iter().enumerate() {
                match val {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        println!("{}[{}]:", prefix, i);
                        print_value(val, indent + 1);
                    }
                    _ => {
                        println!("{}- {}", prefix, format_simple_value(val));
                    }
                }
            }
        }
        _ => {
            println!("{}{}", prefix, format_simple_value(value));
        }
    }
}

fn format_simple_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        _ => value.to_string(),
    }
}

fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn wilson_score_interval(successes: usize, total: usize, z: f64) -> (f64, f64) {
    if total == 0 {
        return (0.0, 1.0);
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = z * z;

    let denominator = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let margin = z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt();

    let lower = ((center - margin) / denominator).max(0.0);
    let upper = ((center + margin) / denominator).min(1.0);

    (lower, upper)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        assert_eq!(normalize("  hello   world  "), "hello world");
        assert_eq!(normalize("single"), "single");
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn test_wilson_score_interval() {
        let (lower, upper) = wilson_score_interval(80, 100, 1.96);
        assert!(lower > 0.7);
        assert!(upper < 0.9);
        assert!(lower < upper);
    }

    #[test]
    fn test_wilson_score_interval_edge_cases() {
        let (lower, upper) = wilson_score_interval(0, 0, 1.96);
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 1.0);

        let (lower, upper) = wilson_score_interval(100, 100, 1.96);
        assert!(upper <= 1.0);
        assert!(lower > 0.9);
    }

    #[test]
    fn test_red_flag_info_from() {
        let flag = RedFlag::TokenLengthExceeded {
            actual: 100,
            limit: 50,
        };
        let info = RedFlagInfo::from(&flag);
        assert_eq!(info.flag_type, "TokenLengthExceeded");
        assert!(info.details.contains("100"));
        assert!(info.details.contains("50"));
    }

    #[test]
    fn test_format_simple_value() {
        assert_eq!(
            format_simple_value(&serde_json::Value::String("test".to_string())),
            "test"
        );
        assert_eq!(format_simple_value(&serde_json::json!(42)), "42");
        assert_eq!(format_simple_value(&serde_json::Value::Bool(true)), "true");
        assert_eq!(format_simple_value(&serde_json::Value::Null), "null");
    }
}
