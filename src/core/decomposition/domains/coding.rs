//! Coding Domain Decomposer
//!
//! AST-based code decomposition using tree-sitter. Splits coding tasks into
//! atomic subtasks respecting syntactic boundaries (functions, blocks, statements).
//!
//! # Features
//!
//! - **Multi-language support**: Rust, Python, JavaScript
//! - **Decomposition strategies**: Function-level, Block-level, Line-level
//! - **Syntax validation**: Red-flags for parse errors before decomposition
//! - **Context preservation**: Maintains imports, dependencies, and scope
//!
//! # Example
//!
//! ```ignore
//! use maker::core::decomposition::domains::CodingDecomposer;
//! use maker::core::matchers::CodeLanguage;
//!
//! let decomposer = CodingDecomposer::new(CodeLanguage::Rust)
//!     .with_strategy(CodeDecompositionStrategy::FunctionLevel);
//!
//! let proposal = decomposer.propose_decomposition(
//!     "refactor-auth",
//!     "Refactor the authentication module",
//!     &context,
//!     0,
//! )?;
//! ```

use crate::core::decomposition::{
    CompositionFunction, DecompositionAgent, DecompositionError, DecompositionProposal,
    DecompositionSubtask,
};
use crate::core::matchers::CodeLanguage;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tree_sitter::{Node, Parser, Tree};

/// Decomposition strategy for code tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CodeDecompositionStrategy {
    /// Decompose at function/method boundaries
    ///
    /// Creates one subtask per function/method. Best for refactoring or
    /// reviewing entire functions independently.
    #[default]
    FunctionLevel,

    /// Decompose at block boundaries (loops, conditionals, etc.)
    ///
    /// Creates subtasks for each logical block within functions.
    /// Good for complex functions that need finer-grained analysis.
    BlockLevel,

    /// Decompose at statement/line level
    ///
    /// Creates subtasks for individual statements or logical lines.
    /// Useful for very detailed code review or line-by-line generation.
    LineLevel,

    /// Automatic strategy selection based on code complexity
    ///
    /// Chooses the appropriate level based on AST analysis:
    /// - Small functions → LineLevel
    /// - Medium functions → BlockLevel
    /// - Large codebases → FunctionLevel
    Auto,
}

/// Result of syntax validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxValidationResult {
    /// Whether the code is syntactically valid
    pub is_valid: bool,

    /// List of syntax errors found
    pub errors: Vec<SyntaxError>,

    /// Language detected/used for parsing
    pub language: String,

    /// Total number of AST nodes
    pub node_count: usize,
}

/// A syntax error detected during validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxError {
    /// Error message
    pub message: String,

    /// Line number (1-indexed)
    pub line: usize,

    /// Column number (1-indexed)
    pub column: usize,

    /// The problematic code snippet
    pub snippet: Option<String>,
}

/// Coding domain decomposer using tree-sitter AST analysis
///
/// Decomposes coding tasks by parsing source code into an AST and creating
/// subtasks at appropriate syntactic boundaries. All leaf subtasks have m=1.
#[derive(Debug, Clone)]
pub struct CodingDecomposer {
    /// Target programming language
    language: CodeLanguage,

    /// Decomposition strategy
    strategy: CodeDecompositionStrategy,

    /// Minimum lines for a subtask (avoids tiny fragments)
    min_lines: usize,

    /// Maximum lines for a subtask (forces further decomposition)
    max_lines: usize,

    /// Whether to include context (imports, surrounding code) in subtasks
    include_context: bool,
}

impl CodingDecomposer {
    /// Create a new CodingDecomposer for the given language
    pub fn new(language: CodeLanguage) -> Self {
        Self {
            language,
            strategy: CodeDecompositionStrategy::default(),
            min_lines: 3,
            max_lines: 50,
            include_context: true,
        }
    }

    /// Set the decomposition strategy
    pub fn with_strategy(mut self, strategy: CodeDecompositionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the minimum lines per subtask
    pub fn with_min_lines(mut self, min_lines: usize) -> Self {
        self.min_lines = min_lines.max(1);
        self
    }

    /// Set the maximum lines per subtask
    pub fn with_max_lines(mut self, max_lines: usize) -> Self {
        self.max_lines = max_lines.max(self.min_lines);
        self
    }

    /// Enable or disable context inclusion
    pub fn with_context(mut self, include_context: bool) -> Self {
        self.include_context = include_context;
        self
    }

    /// Get the configured language
    pub fn language(&self) -> CodeLanguage {
        self.language
    }

    /// Get the configured strategy
    pub fn strategy(&self) -> CodeDecompositionStrategy {
        self.strategy
    }

    /// Validate syntax of code and return detailed results
    pub fn validate_syntax(&self, code: &str) -> SyntaxValidationResult {
        let tree = match self.parse(code) {
            Some(t) => t,
            None => {
                return SyntaxValidationResult {
                    is_valid: false,
                    errors: vec![SyntaxError {
                        message: "Failed to initialize parser".to_string(),
                        line: 1,
                        column: 1,
                        snippet: None,
                    }],
                    language: self.language.to_string(),
                    node_count: 0,
                }
            }
        };

        let root = tree.root_node();
        let mut errors = Vec::new();

        // Collect syntax errors from the AST
        self.collect_errors(root, code.as_bytes(), &mut errors);

        let node_count = self.count_nodes(root);

        SyntaxValidationResult {
            is_valid: errors.is_empty() && !root.has_error(),
            errors,
            language: self.language.to_string(),
            node_count,
        }
    }

    /// Parse code into a tree-sitter tree
    fn parse(&self, code: &str) -> Option<Tree> {
        let mut parser = Parser::new();
        parser
            .set_language(&self.language.tree_sitter_language())
            .ok()?;
        parser.parse(code, None)
    }

    /// Collect syntax errors from AST nodes
    fn collect_errors(&self, node: Node, source: &[u8], errors: &mut Vec<SyntaxError>) {
        if node.is_error() || node.is_missing() {
            let start = node.start_position();
            let snippet = node
                .utf8_text(source)
                .ok()
                .map(|s| s.chars().take(50).collect());

            errors.push(SyntaxError {
                message: if node.is_missing() {
                    format!("Missing expected syntax element: {}", node.kind())
                } else {
                    format!("Syntax error at {}", node.kind())
                },
                line: start.row + 1,
                column: start.column + 1,
                snippet,
            });
        }

        // Recurse into children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.collect_errors(child, source, errors);
            }
        }
    }

    /// Count total nodes in AST
    fn count_nodes(&self, node: Node) -> usize {
        let mut count = 1;
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                count += self.count_nodes(child);
            }
        }
        count
    }

    /// Determine the effective strategy based on code analysis
    fn effective_strategy(&self, code: &str, tree: &Tree) -> CodeDecompositionStrategy {
        if self.strategy != CodeDecompositionStrategy::Auto {
            return self.strategy;
        }

        let root = tree.root_node();
        let line_count = code.lines().count();
        let function_count = self.count_functions(root);

        // Heuristics for auto-selection
        if function_count == 0 || line_count < 20 {
            CodeDecompositionStrategy::LineLevel
        } else if function_count == 1 && line_count < 100 {
            CodeDecompositionStrategy::BlockLevel
        } else {
            CodeDecompositionStrategy::FunctionLevel
        }
    }

    /// Count function/method definitions in AST
    fn count_functions(&self, node: Node) -> usize {
        let function_kinds = self.function_node_kinds();
        let mut count = 0;

        if function_kinds.contains(&node.kind()) {
            count += 1;
        }

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                count += self.count_functions(child);
            }
        }

        count
    }

    /// Get node kinds that represent functions/methods
    fn function_node_kinds(&self) -> &[&str] {
        match self.language {
            CodeLanguage::Rust => &["function_item", "impl_item", "trait_item"],
            CodeLanguage::Python => &["function_definition", "class_definition"],
            CodeLanguage::JavaScript => &[
                "function_declaration",
                "method_definition",
                "arrow_function",
            ],
        }
    }

    /// Get node kinds that represent blocks
    fn block_node_kinds(&self) -> &[&str] {
        match self.language {
            CodeLanguage::Rust => &[
                "block",
                "if_expression",
                "match_expression",
                "loop_expression",
                "for_expression",
                "while_expression",
            ],
            CodeLanguage::Python => &[
                "block",
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "with_statement",
            ],
            CodeLanguage::JavaScript => &[
                "statement_block",
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "switch_statement",
            ],
        }
    }

    /// Extract subtasks at function level
    fn extract_function_subtasks(
        &self,
        node: Node,
        source: &[u8],
        parent_id: &str,
        subtasks: &mut Vec<DecompositionSubtask>,
        order: &mut usize,
    ) {
        let function_kinds = self.function_node_kinds();

        if function_kinds.contains(&node.kind()) {
            let name = self.extract_function_name(node, source);
            let code = node.utf8_text(source).unwrap_or("").to_string();
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;

            let task_id = format!("{}-fn-{}-L{}", parent_id, name, start_line);

            let subtask =
                DecompositionSubtask::leaf(&task_id, format!("Process function: {}", name))
                    .with_parent(parent_id)
                    .with_order(*order)
                    .with_context(json!({
                        "code": code,
                        "function_name": name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "language": self.language.to_string(),
                        "node_kind": node.kind(),
                    }))
                    .with_metadata("domain", json!("coding"))
                    .with_metadata("strategy", json!("function_level"));

            subtasks.push(subtask);
            *order += 1;
        }

        // Recurse into children to find nested functions
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.extract_function_subtasks(child, source, parent_id, subtasks, order);
            }
        }
    }

    /// Extract subtasks at block level
    fn extract_block_subtasks(
        &self,
        node: Node,
        source: &[u8],
        parent_id: &str,
        subtasks: &mut Vec<DecompositionSubtask>,
        order: &mut usize,
    ) {
        let block_kinds = self.block_node_kinds();
        let function_kinds = self.function_node_kinds();

        // For functions, decompose into their blocks
        if function_kinds.contains(&node.kind()) {
            let fn_name = self.extract_function_name(node, source);

            // Find the block child and decompose it
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if block_kinds.contains(&child.kind()) || child.kind() == "block" {
                        self.extract_block_children(
                            child, source, parent_id, &fn_name, subtasks, order,
                        );
                    }
                }
            }
        } else {
            // Recurse to find functions
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    self.extract_block_subtasks(child, source, parent_id, subtasks, order);
                }
            }
        }
    }

    /// Extract individual blocks from a function body
    fn extract_block_children(
        &self,
        node: Node,
        source: &[u8],
        parent_id: &str,
        fn_name: &str,
        subtasks: &mut Vec<DecompositionSubtask>,
        order: &mut usize,
    ) {
        let block_kinds = self.block_node_kinds();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                let kind = child.kind();

                if block_kinds.contains(&kind) {
                    let code = child.utf8_text(source).unwrap_or("").to_string();
                    let start_line = child.start_position().row + 1;
                    let end_line = child.end_position().row + 1;

                    let task_id = format!("{}-{}-blk-{}-L{}", parent_id, fn_name, kind, start_line);

                    let subtask = DecompositionSubtask::leaf(
                        &task_id,
                        format!("Process {} block in {}", kind, fn_name),
                    )
                    .with_parent(parent_id)
                    .with_order(*order)
                    .with_context(json!({
                        "code": code,
                        "function_name": fn_name,
                        "block_kind": kind,
                        "start_line": start_line,
                        "end_line": end_line,
                        "language": self.language.to_string(),
                    }))
                    .with_metadata("domain", json!("coding"))
                    .with_metadata("strategy", json!("block_level"));

                    subtasks.push(subtask);
                    *order += 1;
                } else {
                    // Recurse for nested blocks
                    self.extract_block_children(child, source, parent_id, fn_name, subtasks, order);
                }
            }
        }

        // If no blocks found, treat the whole function body as one subtask
        if subtasks.is_empty() {
            let code = node.utf8_text(source).unwrap_or("").to_string();
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;

            let task_id = format!("{}-{}-body-L{}", parent_id, fn_name, start_line);

            let subtask =
                DecompositionSubtask::leaf(&task_id, format!("Process body of {}", fn_name))
                    .with_parent(parent_id)
                    .with_order(*order)
                    .with_context(json!({
                        "code": code,
                        "function_name": fn_name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "language": self.language.to_string(),
                    }))
                    .with_metadata("domain", json!("coding"))
                    .with_metadata("strategy", json!("block_level"));

            subtasks.push(subtask);
            *order += 1;
        }
    }

    /// Extract subtasks at line/statement level
    fn extract_line_subtasks(
        &self,
        node: Node,
        source: &[u8],
        parent_id: &str,
        subtasks: &mut Vec<DecompositionSubtask>,
        order: &mut usize,
    ) {
        let statement_kinds = self.statement_node_kinds();

        if statement_kinds.contains(&node.kind()) {
            let code = node.utf8_text(source).unwrap_or("").to_string();
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;
            let line_count = end_line - start_line + 1;

            // Skip tiny statements if they're part of a larger group
            if line_count >= self.min_lines || subtasks.is_empty() {
                let task_id = format!("{}-stmt-L{}", parent_id, start_line);
                let kind = node.kind();

                let subtask = DecompositionSubtask::leaf(
                    &task_id,
                    format!("Process {} at line {}", kind, start_line),
                )
                .with_parent(parent_id)
                .with_order(*order)
                .with_context(json!({
                    "code": code,
                    "statement_kind": kind,
                    "start_line": start_line,
                    "end_line": end_line,
                    "language": self.language.to_string(),
                }))
                .with_metadata("domain", json!("coding"))
                .with_metadata("strategy", json!("line_level"));

                subtasks.push(subtask);
                *order += 1;
            }
        }

        // Recurse into children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.extract_line_subtasks(child, source, parent_id, subtasks, order);
            }
        }
    }

    /// Get node kinds that represent statements
    fn statement_node_kinds(&self) -> &[&str] {
        match self.language {
            CodeLanguage::Rust => &[
                "let_declaration",
                "expression_statement",
                "return_expression",
                "if_expression",
                "match_expression",
                "loop_expression",
                "for_expression",
                "while_expression",
                "macro_invocation",
            ],
            CodeLanguage::Python => &[
                "expression_statement",
                "return_statement",
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "with_statement",
                "assert_statement",
                "import_statement",
                "import_from_statement",
                "assignment",
            ],
            CodeLanguage::JavaScript => &[
                "expression_statement",
                "return_statement",
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "switch_statement",
                "variable_declaration",
                "lexical_declaration",
                "import_statement",
                "export_statement",
            ],
        }
    }

    /// Extract function/method name from a function node
    fn extract_function_name(&self, node: Node, source: &[u8]) -> String {
        // Look for identifier child that represents the name
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                let kind = child.kind();
                if kind == "identifier"
                    || kind == "name"
                    || kind == "field_identifier"
                    || kind == "property_identifier"
                {
                    if let Ok(name) = child.utf8_text(source) {
                        return name.to_string();
                    }
                }
            }
        }

        // Fallback: use node position
        format!("anonymous_L{}", node.start_position().row + 1)
    }

    /// Extract imports/preamble for context
    fn extract_preamble(&self, node: Node, source: &[u8]) -> String {
        let import_kinds = match self.language {
            CodeLanguage::Rust => vec!["use_declaration", "mod_item", "extern_crate_declaration"],
            CodeLanguage::Python => vec!["import_statement", "import_from_statement"],
            CodeLanguage::JavaScript => vec!["import_statement", "export_statement"],
        };

        let mut preamble = String::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if import_kinds.contains(&child.kind()) {
                    if let Ok(text) = child.utf8_text(source) {
                        preamble.push_str(text);
                        preamble.push('\n');
                    }
                }
            }
        }

        preamble
    }
}

impl DecompositionAgent for CodingDecomposer {
    fn propose_decomposition(
        &self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        _depth: usize,
    ) -> Result<DecompositionProposal, DecompositionError> {
        // Extract code from context
        let code = context
            .get("code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DecompositionError::AgentError {
                message: "Missing 'code' field in context".to_string(),
            })?;

        // Validate syntax first
        let validation = self.validate_syntax(code);
        if !validation.is_valid {
            let error_msgs: Vec<String> = validation
                .errors
                .iter()
                .map(|e| format!("Line {}: {}", e.line, e.message))
                .collect();

            return Err(DecompositionError::ValidationError {
                message: format!("Syntax errors found: {}", error_msgs.join("; ")),
            });
        }

        // Parse the code
        let tree = self
            .parse(code)
            .ok_or_else(|| DecompositionError::AgentError {
                message: "Failed to parse code".to_string(),
            })?;

        let root = tree.root_node();
        let source = code.as_bytes();

        // Determine effective strategy
        let strategy = self.effective_strategy(code, &tree);

        // Extract subtasks based on strategy
        let mut subtasks = Vec::new();
        let mut order = 0;

        match strategy {
            CodeDecompositionStrategy::FunctionLevel => {
                self.extract_function_subtasks(root, source, task_id, &mut subtasks, &mut order);
            }
            CodeDecompositionStrategy::BlockLevel => {
                self.extract_block_subtasks(root, source, task_id, &mut subtasks, &mut order);
            }
            CodeDecompositionStrategy::LineLevel | CodeDecompositionStrategy::Auto => {
                self.extract_line_subtasks(root, source, task_id, &mut subtasks, &mut order);
            }
        }

        // If no subtasks extracted, create a single leaf for the whole code
        if subtasks.is_empty() {
            let subtask = DecompositionSubtask::leaf(
                format!("{}-whole", task_id),
                format!("Process entire code block: {}", description),
            )
            .with_parent(task_id)
            .with_order(0)
            .with_context(json!({
                "code": code,
                "language": self.language.to_string(),
            }))
            .with_metadata("domain", json!("coding"))
            .with_metadata("strategy", json!(format!("{:?}", strategy)));

            subtasks.push(subtask);
        }

        // Add preamble context to first subtask if enabled
        if self.include_context && !subtasks.is_empty() {
            let preamble = self.extract_preamble(root, source);
            if !preamble.is_empty() {
                if let Some(first) = subtasks.first_mut() {
                    let mut ctx = first.context.clone();
                    if let Some(obj) = ctx.as_object_mut() {
                        obj.insert("preamble".to_string(), json!(preamble));
                    }
                    first.context = ctx;
                }
            }
        }

        // Determine composition function based on strategy
        let composition_fn = match strategy {
            CodeDecompositionStrategy::FunctionLevel => CompositionFunction::Parallel {
                merge_strategy: crate::core::decomposition::MergeStrategy::Concatenate,
            },
            _ => CompositionFunction::Sequential,
        };

        let mut metadata = HashMap::new();
        metadata.insert("language".to_string(), json!(self.language.to_string()));
        metadata.insert("strategy".to_string(), json!(format!("{:?}", strategy)));
        metadata.insert("node_count".to_string(), json!(validation.node_count));

        Ok(DecompositionProposal {
            proposal_id: format!("coding-{}-{}", self.language, task_id),
            source_task_id: task_id.to_string(),
            subtasks,
            composition_fn,
            confidence: if validation.is_valid { 0.9 } else { 0.5 },
            rationale: Some(format!(
                "Code decomposed using {:?} strategy for {} ({} AST nodes)",
                strategy, self.language, validation.node_count
            )),
            metadata,
        })
    }

    fn is_atomic(&self, _task_id: &str, description: &str) -> bool {
        // Consider atomic if description suggests a single-line change
        let atomic_hints = [
            "rename",
            "change type",
            "update value",
            "add import",
            "remove import",
            "fix typo",
        ];

        let desc_lower = description.to_lowercase();
        atomic_hints.iter().any(|h| desc_lower.contains(h))
    }

    fn name(&self) -> &str {
        "coding"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // CodingDecomposer Construction Tests
    // ==========================================

    #[test]
    fn test_new_decomposer() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Rust);
        assert_eq!(decomposer.language(), CodeLanguage::Rust);
        assert_eq!(
            decomposer.strategy(),
            CodeDecompositionStrategy::FunctionLevel
        );
    }

    #[test]
    fn test_builder_methods() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Python)
            .with_strategy(CodeDecompositionStrategy::BlockLevel)
            .with_min_lines(5)
            .with_max_lines(100)
            .with_context(false);

        assert_eq!(decomposer.language(), CodeLanguage::Python);
        assert_eq!(decomposer.strategy(), CodeDecompositionStrategy::BlockLevel);
    }

    // ==========================================
    // Syntax Validation Tests
    // ==========================================

    #[test]
    fn test_validate_valid_rust() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Rust);
        let code = r#"
fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}
"#;
        let result = decomposer.validate_syntax(code);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        assert!(result.node_count > 0);
    }

    #[test]
    fn test_validate_valid_python() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Python);
        let code = r#"
def hello(name):
    return f"Hello, {name}!"
"#;
        let result = decomposer.validate_syntax(code);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_valid_javascript() {
        let decomposer = CodingDecomposer::new(CodeLanguage::JavaScript);
        let code = r#"
function hello(name) {
    return `Hello, ${name}!`;
}
"#;
        let result = decomposer.validate_syntax(code);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_invalid_syntax() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Rust);
        let code = r#"
fn broken( {
    // missing closing paren and function body
"#;
        let result = decomposer.validate_syntax(code);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    // ==========================================
    // Function-Level Decomposition Tests
    // ==========================================

    #[test]
    fn test_rust_function_decomposition() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Rust)
            .with_strategy(CodeDecompositionStrategy::FunctionLevel);

        let code = r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
"#;

        let result = decomposer.propose_decomposition(
            "math-ops",
            "Process math operations",
            &json!({"code": code}),
            0,
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();

        // Should have 2 function subtasks
        assert_eq!(proposal.subtasks.len(), 2);

        // All should be leaf nodes with m=1
        for subtask in &proposal.subtasks {
            assert!(subtask.is_leaf);
            assert_eq!(subtask.m_value, 1);
        }

        // Should use Parallel composition for functions
        assert!(matches!(
            proposal.composition_fn,
            CompositionFunction::Parallel { .. }
        ));
    }

    #[test]
    fn test_python_function_decomposition() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Python)
            .with_strategy(CodeDecompositionStrategy::FunctionLevel);

        let code = r#"
def greet(name):
    return f"Hello, {name}!"

def farewell(name):
    return f"Goodbye, {name}!"
"#;

        let result = decomposer.propose_decomposition(
            "greetings",
            "Process greeting functions",
            &json!({"code": code}),
            0,
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();
        assert_eq!(proposal.subtasks.len(), 2);
    }

    // ==========================================
    // Line-Level Decomposition Tests
    // ==========================================

    #[test]
    fn test_line_level_decomposition() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Python)
            .with_strategy(CodeDecompositionStrategy::LineLevel)
            .with_min_lines(1); // Allow single-line statements

        let code = r#"
x = 1
y = 2
z = x + y
print(z)
"#;

        let result = decomposer.propose_decomposition(
            "simple-calc",
            "Process simple calculation",
            &json!({"code": code}),
            0,
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();

        // Should have at least one subtask
        assert!(!proposal.subtasks.is_empty());

        // All should be leaves with m=1
        for subtask in &proposal.subtasks {
            assert!(subtask.is_leaf);
            assert_eq!(subtask.m_value, 1);
        }

        // Should use Sequential composition
        assert!(matches!(
            proposal.composition_fn,
            CompositionFunction::Sequential
        ));
    }

    // ==========================================
    // Auto Strategy Tests
    // ==========================================

    #[test]
    fn test_auto_strategy_small_code() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Python)
            .with_strategy(CodeDecompositionStrategy::Auto);

        let code = "x = 1\ny = 2\n";

        let tree = decomposer.parse(code).unwrap();
        let strategy = decomposer.effective_strategy(code, &tree);

        assert_eq!(strategy, CodeDecompositionStrategy::LineLevel);
    }

    // ==========================================
    // Error Handling Tests
    // ==========================================

    #[test]
    fn test_missing_code_in_context() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Rust);

        let result = decomposer.propose_decomposition(
            "test",
            "Test task",
            &json!({"not_code": "something"}),
            0,
        );

        assert!(result.is_err());
        assert!(matches!(result, Err(DecompositionError::AgentError { .. })));
    }

    #[test]
    fn test_syntax_error_red_flag() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Rust);

        let code = "fn broken( {";

        let result =
            decomposer.propose_decomposition("test", "Test task", &json!({"code": code}), 0);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(DecompositionError::ValidationError { .. })
        ));
    }

    // ==========================================
    // DecompositionAgent Trait Tests
    // ==========================================

    #[test]
    fn test_agent_name() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Rust);
        assert_eq!(decomposer.name(), "coding");
    }

    #[test]
    fn test_is_atomic() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Rust);

        assert!(decomposer.is_atomic("task-1", "Rename the function"));
        assert!(decomposer.is_atomic("task-2", "Change type of parameter"));
        assert!(decomposer.is_atomic("task-3", "Fix typo in comment"));
        assert!(!decomposer.is_atomic("task-4", "Refactor the entire module"));
    }

    // ==========================================
    // m=1 Enforcement Tests
    // ==========================================

    #[test]
    fn test_all_subtasks_have_m1() {
        let decomposer = CodingDecomposer::new(CodeLanguage::Rust);

        let code = r#"
fn a() { println!("a"); }
fn b() { println!("b"); }
fn c() { println!("c"); }
"#;

        let result = decomposer.propose_decomposition(
            "funcs",
            "Process functions",
            &json!({"code": code}),
            0,
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();

        // Verify all subtasks are valid leaves with m=1
        for subtask in &proposal.subtasks {
            assert!(
                subtask.is_leaf,
                "Subtask {} should be a leaf",
                subtask.task_id
            );
            assert_eq!(
                subtask.m_value, 1,
                "Subtask {} should have m_value=1",
                subtask.task_id
            );
            assert!(subtask.validate().is_ok());
        }

        // Verify proposal validates
        assert!(proposal.validate().is_ok());
    }

    // ==========================================
    // Strategy Serialization Tests
    // ==========================================

    #[test]
    fn test_strategy_serialization() {
        let strategies = vec![
            CodeDecompositionStrategy::FunctionLevel,
            CodeDecompositionStrategy::BlockLevel,
            CodeDecompositionStrategy::LineLevel,
            CodeDecompositionStrategy::Auto,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let parsed: CodeDecompositionStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, parsed);
        }
    }

    #[test]
    fn test_validation_result_serialization() {
        let result = SyntaxValidationResult {
            is_valid: true,
            errors: vec![],
            language: "rust".to_string(),
            node_count: 42,
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: SyntaxValidationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.is_valid, parsed.is_valid);
        assert_eq!(result.language, parsed.language);
        assert_eq!(result.node_count, parsed.node_count);
    }
}
