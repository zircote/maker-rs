//! MAKER CLI - Standalone command-line interface for the MAKER framework
//!
//! This binary provides a CLI for MAKER without requiring MCP.
//! It has feature parity with the MCP tools.

use clap::{Parser, Subcommand, ValueEnum};
use maker::core::adaptive::{KEstimator, KEstimatorConfig};
use maker::core::decomposition::{DecompositionAgent, LlmAgentConfig, LlmDecompositionAgent};
use maker::core::matchers::create_matcher_from_string;
use maker::core::{
    calculate_kmin, validate_token_length, vote_with_margin, vote_with_margin_adaptive, RedFlag,
    VoteConfig,
};
use maker::llm::adapter::{setup_provider_client, valid_providers_str, VALID_PROVIDERS};
use maker::mcp::health::{validate_config, HealthChecker};
use maker::mcp::server::ServerConfig;
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

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

        /// Matcher type (exact, embedding, code) or preset name
        #[arg(long)]
        matcher: Option<String>,

        /// Use a matcher preset (code_generation, summarization, chat, extraction, classification, reasoning, creative)
        #[arg(long, conflicts_with = "matcher")]
        preset: Option<String>,
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

        /// LLM provider to use (ollama, openai, anthropic)
        #[arg(long, default_value = "ollama")]
        provider: String,

        /// Model name override
        #[arg(long)]
        model: Option<String>,
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

/// Response from the vote command containing the voting result.
#[derive(Serialize, Deserialize)]
struct VoteResponse {
    /// The winning response content that achieved k-margin lead
    winner: String,
    /// Number of votes the winner received
    votes: usize,
    /// Total number of samples collected during voting
    total_samples: usize,
    /// The k-margin used for this vote (may differ from requested if adaptive)
    k_margin: usize,
    /// Whether voting converged (true if winner declared)
    converged: bool,
}

/// Response from the validate command with validation results.
#[derive(Serialize, Deserialize)]
struct ValidateResponse {
    /// Whether the response passed all validation checks
    valid: bool,
    /// List of triggered red-flag violations (empty if valid)
    red_flags: Vec<RedFlagInfo>,
}

/// Information about a single red-flag validation failure.
#[derive(Serialize, Deserialize)]
struct RedFlagInfo {
    /// Type of red-flag (TokenLengthExceeded, FormatViolation, LogicLoop)
    flag_type: String,
    /// Human-readable description of the violation
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

/// Response from the calibrate command with estimated parameters.
#[derive(Serialize, Deserialize)]
struct CalibrateResponse {
    /// Estimated per-step success probability (0.0-1.0)
    p_estimate: f64,
    /// 95% Wilson score confidence interval for p_estimate as (lower, upper)
    confidence_interval: (f64, f64),
    /// Number of samples used for estimation
    sample_count: usize,
    /// Recommended k-margin based on p_estimate and target reliability
    recommended_k: usize,
}

/// A single calibration sample with prompt, expected answer, and optional response.
#[derive(Serialize, Deserialize)]
struct CalibrationSample {
    /// The prompt that was sent to the LLM
    prompt: String,
    /// The expected correct response
    ground_truth: String,
    /// The actual LLM response (if available)
    response: Option<String>,
}

/// Response from the config command showing current configuration.
#[derive(Serialize, Deserialize)]
struct ConfigResponse {
    /// Current default k-margin for voting
    k_margin: usize,
    /// Current default LLM provider (ollama, openai, anthropic)
    provider: String,
    /// Current default matcher type (exact, embedding, code, or preset name)
    matcher: String,
    /// Whether adaptive k-margin is enabled by default
    adaptive: bool,
    /// Maximum samples before voting timeout
    max_samples: usize,
}

/// Response from the decompose command with task breakdown.
#[derive(Serialize, Deserialize)]
struct DecomposeResponse {
    /// Unique identifier for the root task
    task_id: String,
    /// List of subtasks from decomposition
    subtasks: Vec<SubtaskInfo>,
    /// Composition strategy (sequential, parallel, conditional)
    composition: String,
    /// Depth of decomposition (0 for atomic, 1+ for decomposed)
    depth: usize,
}

/// Information about a single subtask in a decomposition.
#[derive(Serialize, Deserialize)]
struct SubtaskInfo {
    /// Unique identifier for this subtask
    id: String,
    /// Human-readable description of what this subtask does
    description: String,
    /// Whether this is a leaf node (atomic, cannot be further decomposed)
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
            adaptive,
            matcher,
            preset,
        } => execute_vote(
            cli.format,
            prompt,
            k_margin,
            max_samples,
            temperature,
            &provider,
            adaptive,
            matcher,
            preset,
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
            provider,
            model,
        } => execute_decompose(cli.format, task, depth_limit, timeout, &provider, model),

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

#[allow(clippy::too_many_arguments)]
fn execute_vote(
    format: OutputFormat,
    prompt: Option<String>,
    k_margin: usize,
    max_samples: usize,
    temperature: f64,
    provider: &str,
    adaptive: bool,
    matcher: Option<String>,
    preset: Option<String>,
) -> Result<(), String> {
    let prompt_text = get_input(prompt, "prompt")?;

    if k_margin == 0 {
        return Err("k_margin must be >= 1".to_string());
    }

    if prompt_text.is_empty() {
        return Err("prompt cannot be empty".to_string());
    }

    // Create matcher based on preset or matcher type
    // If preset is provided, use it; otherwise use matcher type
    let matcher_spec = preset.as_deref().or(matcher.as_deref());
    let matcher_arc =
        create_matcher_from_string(matcher_spec, &prompt_text).map_err(|e| e.to_string())?;

    // Create real provider based on CLI argument
    let client = setup_provider_client(provider, None)?;

    let config = VoteConfig::default()
        .with_max_samples(max_samples)
        .with_matcher(matcher_arc)
        .with_diversity_temperature(temperature);

    // Execute voting - adaptive or static
    let result = if adaptive {
        let mut estimator = KEstimator::new(KEstimatorConfig::default());
        vote_with_margin_adaptive(
            &prompt_text,
            &mut estimator,
            0.95,          // target reliability
            k_margin * 10, // remaining_steps estimate
            client.as_ref(),
            config,
        )
    } else {
        vote_with_margin(&prompt_text, k_margin, client.as_ref(), config)
    };

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
            if !VALID_PROVIDERS.contains(&p.as_str()) {
                return Err(format!(
                    "Invalid provider '{}'. Valid options: {}",
                    p,
                    valid_providers_str()
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
    timeout: u64,
    provider: &str,
    model: Option<String>,
) -> Result<(), String> {
    let task_desc = get_input(task, "task")?;

    if task_desc.is_empty() {
        return Err("task description cannot be empty".to_string());
    }

    // Create provider
    let client = setup_provider_client(provider, model)?;

    // Create LLM decomposition agent
    let agent_config = LlmAgentConfig::default().with_max_subtasks(depth_limit);
    let agent = LlmDecompositionAgent::new(Arc::from(client), agent_config);

    // Set timeout context
    let start = std::time::Instant::now();
    let timeout_duration = Duration::from_secs(timeout);

    // Propose decomposition
    let task_id = format!(
        "task_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );

    let proposal = agent
        .propose_decomposition(&task_id, &task_desc, &serde_json::Value::Null, 0)
        .map_err(|e| format!("Decomposition failed: {}", e))?;

    // Check timeout
    if start.elapsed() > timeout_duration {
        return Err(format!("Decomposition timed out after {} seconds", timeout));
    }

    // Convert to response
    let subtasks: Vec<SubtaskInfo> = proposal
        .subtasks
        .iter()
        .map(|st| SubtaskInfo {
            id: st.task_id.clone(),
            description: st.description.clone(),
            is_leaf: st.is_leaf,
        })
        .collect();

    let composition = match proposal.composition_fn {
        maker::core::decomposition::CompositionFunction::Sequential => "sequential",
        maker::core::decomposition::CompositionFunction::Parallel { .. } => "parallel",
        maker::core::decomposition::CompositionFunction::Conditional { .. } => "conditional",
        maker::core::decomposition::CompositionFunction::Custom { ref name, .. } => name.as_str(),
    };

    let result = DecomposeResponse {
        task_id: proposal.proposal_id,
        subtasks,
        composition: composition.to_string(),
        depth: if proposal.subtasks.is_empty() { 0 } else { 1 },
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
        let provider_healthy = setup_provider_client("ollama", None).is_ok();
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

/// Get input from command argument or stdin.
///
/// If `arg` is `Some(value)` and not "-", returns the value directly.
/// If `arg` is `None` or "-", reads from stdin until EOF and returns trimmed content.
///
/// # Arguments
/// * `arg` - Optional command-line argument value
/// * `name` - Name of the input for error messages (e.g., "prompt", "response")
///
/// # Returns
/// * `Ok(String)` - The input content
/// * `Err(String)` - Error message if stdin read fails
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

/// Output a response in the specified format.
///
/// Serializes the response to JSON (pretty-printed) or human-readable text format.
///
/// # Arguments
/// * `format` - Output format (Json or Text)
/// * `response` - Any serializable response struct
///
/// # Returns
/// * `Ok(())` - Response printed to stdout
/// * `Err(String)` - Serialization error
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

/// Recursively print a JSON value with indentation for human-readable output.
///
/// Objects print as "key: value" pairs, arrays as indexed or bulleted items.
/// Nested structures increase indentation by 2 spaces per level.
///
/// # Arguments
/// * `value` - The JSON value to print
/// * `indent` - Current indentation level (0 for root)
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

/// Format a simple JSON value as a string for display.
///
/// Strings are returned as-is, numbers/bools converted to string representation.
/// Complex values (objects/arrays) fall back to JSON serialization.
fn format_simple_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        _ => value.to_string(),
    }
}

/// Normalize whitespace in a string for comparison.
///
/// Collapses all whitespace sequences (spaces, tabs, newlines) into single spaces
/// and trims leading/trailing whitespace. Used for comparing LLM responses.
///
/// # Example
/// ```ignore
/// assert_eq!(normalize("  hello   world  "), "hello world");
/// ```
fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Calculate the Wilson score confidence interval for a proportion.
///
/// Uses the Wilson score method which provides better coverage for small samples
/// and extreme proportions compared to the normal approximation.
///
/// # Arguments
/// * `successes` - Number of successful outcomes
/// * `total` - Total number of trials
/// * `z` - Z-score for desired confidence level (1.96 for 95%)
///
/// # Returns
/// Tuple of (lower_bound, upper_bound) for the confidence interval.
/// Returns (0.0, 1.0) if total is 0.
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
