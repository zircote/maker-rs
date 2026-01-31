//! Core MAKER algorithms
//!
//! This module contains the fundamental algorithms for the MAKER framework:
//! - `kmin`: Calculation of minimum k-margin for target reliability
//! - `voting`: First-to-ahead-by-k voting protocol
//! - `redflag`: Red-flagging parsers for malformed outputs
//! - `orchestration`: Microagent orchestration (m=1 constraint)
//! - `executor`: Sample-validate-vote integration loop

pub mod executor;
pub mod kmin;
pub mod orchestration;
pub mod redflag;
pub mod voting;

pub use executor::{
    vote_with_margin, CostMetrics, LlmClient, LlmResponse, MockLlmClient, VoteConfig,
    VoteError as ExecutorVoteError, VoteResult,
};
pub use kmin::{calculate_kmin, KminError};
pub use orchestration::{
    AgentOutput, AgentOutputError, MicroagentConfig, State, Subtask, TaskDecomposer,
    TaskOrchestrator,
};
pub use redflag::{
    validate_agent_output, validate_json_schema, validate_token_length, RedFlag, RedFlagValidator,
    StrictAgentOutput,
};
pub use voting::{CandidateId, VoteCheckResult, VoteError, VoteEvent, VoteRace};
