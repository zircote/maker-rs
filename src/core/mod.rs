//! Core MAKER algorithms
//!
//! This module contains the fundamental algorithms for the MAKER framework:
//! - `kmin`: Calculation of minimum k-margin for target reliability
//! - `voting`: First-to-ahead-by-k voting protocol
//! - `redflag`: Red-flagging parsers for malformed outputs
//! - `orchestration`: Microagent orchestration (m=1 constraint)
//! - `executor`: Sample-validate-vote integration loop
//! - `decomposition`: Recursive decomposition framework (v0.3.0)

pub mod adaptive;
pub mod async_executor;
pub mod decomposition;
pub mod executor;
pub mod kmin;
pub mod matcher;
pub mod matchers;
pub mod orchestration;
pub mod redflag;
pub mod voting;

pub use executor::{
    vote_with_margin, vote_with_margin_adaptive, CostMetrics, LlmClient, LlmResponse,
    MockLlmClient, VoteConfig, VoteError as ExecutorVoteError, VoteResult,
};
pub use kmin::{calculate_kmin, KminError};
pub use matcher::{default_matcher, CandidateMatcher, ExactMatcher};
pub use matchers::embedding::{
    cosine_similarity, EmbeddingClient, EmbeddingMatcher, MockEmbeddingClient,
};
pub use matchers::ollama_embedding::OllamaEmbeddingClient;
pub use matchers::openai_embedding::OpenAiEmbeddingClient;
pub use orchestration::{
    AgentOutput, AgentOutputError, MicroagentConfig, OrchestratorExt, ProposalDecomposer, State,
    Subtask, TaskDecomposer, TaskOrchestrator,
};
pub use redflag::{
    validate_agent_output, validate_json_schema, validate_token_length, RedFlag, RedFlagValidator,
    StrictAgentOutput,
};

#[cfg(feature = "code-matcher")]
pub use matchers::code::CodeMatcher;
pub use voting::{CandidateId, VoteCheckResult, VoteError, VoteEvent, VoteRace};

pub use adaptive::{KAdjusted, KEstimator, KEstimatorConfig, VoteObservation};

// Decomposition framework (v0.3.0)
pub use decomposition::{
    vote_on_decomposition, CompositionFunction, DecompositionAgent, DecompositionConfig,
    DecompositionDiscriminator, DecompositionError, DecompositionProposal, DecompositionSubtask,
    DecompositionVoteResult, IdentityDecomposer, MergeStrategy, ProposalMatcher, TaskId,
};

// Solver framework (v0.3.0)
pub use decomposition::{
    ExecutionError, ExecutionMetrics, ExecutorConfig, LeafNodeExecutor, ProposalResult,
    SubtaskResult,
};

// Aggregator framework (v0.3.0)
pub use decomposition::{
    compose_recursive, AggregatedResult, AggregationError, AggregationMetrics, AggregatorConfig,
    SolutionDiscriminator,
};

// Orchestrator framework (v0.3.0)
pub use decomposition::{
    CancellationToken, OrchestrationError, OrchestrationMetrics, OrchestrationResult,
    OrchestratorConfig, RecursiveOrchestrator,
};

// Async executor framework (v0.3.0)
pub use async_executor::{
    vote_with_margin_adaptive_async, vote_with_margin_async, vote_with_margin_concurrent,
    AsyncLlmClient, AsyncVoteConfig, AsyncVotingExecutor,
    CancellationToken as AsyncCancellationToken, MockAsyncLlmClient,
};
