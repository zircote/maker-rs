//! Multi-file Orchestration
//!
//! Provides state management and coordination for decomposition tasks
//! that span multiple files. Supports file-level locking, cross-file
//! dependency tracking, and atomic multi-file commits.
//!
//! # Features
//!
//! - **FileSystemState**: Track state across multiple files
//! - **File locking**: Prevent concurrent modification conflicts
//! - **Dependency tracking**: Enforce file operation ordering
//! - **Atomic commits**: All-or-nothing multi-file changes
//!
//! # Example
//!
//! ```ignore
//! use maker::core::decomposition::filesystem::{FileSystemState, FileOperation};
//!
//! let mut state = FileSystemState::new();
//! state.add_file("src/main.rs", FileContent::new("fn main() {}"));
//! state.add_dependency("src/lib.rs", "src/main.rs");
//!
//! let commit = state.prepare_commit()?;
//! commit.execute()?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Content of a file being tracked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContent {
    /// Original content (before modifications)
    pub original: Option<String>,

    /// Current/modified content
    pub current: String,

    /// File metadata
    #[serde(default)]
    pub metadata: FileMetadata,
}

impl FileContent {
    /// Create new file content
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            original: None,
            current: content.into(),
            metadata: FileMetadata::default(),
        }
    }

    /// Create file content with original preserved
    pub fn with_original(original: impl Into<String>, current: impl Into<String>) -> Self {
        Self {
            original: Some(original.into()),
            current: current.into(),
            metadata: FileMetadata::default(),
        }
    }

    /// Check if file has been modified
    pub fn is_modified(&self) -> bool {
        match &self.original {
            Some(orig) => orig != &self.current,
            None => true, // New file
        }
    }

    /// Get the diff between original and current (simple line-based)
    pub fn diff(&self) -> Vec<DiffLine> {
        let original_lines: Vec<&str> = self
            .original
            .as_ref()
            .map(|s| s.lines().collect())
            .unwrap_or_default();

        let current_lines: Vec<&str> = self.current.lines().collect();

        compute_diff(&original_lines, &current_lines)
    }
}

/// File metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileMetadata {
    /// Language hint (e.g., "rust", "python")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// File encoding (default UTF-8)
    #[serde(default = "default_encoding")]
    pub encoding: String,

    /// Whether file is executable
    #[serde(default)]
    pub executable: bool,

    /// Custom attributes
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, serde_json::Value>,
}

fn default_encoding() -> String {
    "utf-8".to_string()
}

/// A line in a diff
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffLine {
    /// Unchanged line
    Context(String),
    /// Added line
    Added(String),
    /// Removed line
    Removed(String),
}

/// Simple line-based diff computation
fn compute_diff(original: &[&str], current: &[&str]) -> Vec<DiffLine> {
    let mut result = Vec::new();

    // Simple LCS-based diff
    let lcs = compute_lcs(original, current);
    let mut orig_idx = 0;
    let mut curr_idx = 0;
    let mut lcs_idx = 0;

    while orig_idx < original.len() || curr_idx < current.len() {
        if lcs_idx < lcs.len()
            && orig_idx < original.len()
            && curr_idx < current.len()
            && original[orig_idx] == lcs[lcs_idx]
            && current[curr_idx] == lcs[lcs_idx]
        {
            result.push(DiffLine::Context(lcs[lcs_idx].to_string()));
            orig_idx += 1;
            curr_idx += 1;
            lcs_idx += 1;
        } else if orig_idx < original.len()
            && (lcs_idx >= lcs.len() || original[orig_idx] != lcs[lcs_idx])
        {
            result.push(DiffLine::Removed(original[orig_idx].to_string()));
            orig_idx += 1;
        } else if curr_idx < current.len() {
            result.push(DiffLine::Added(current[curr_idx].to_string()));
            curr_idx += 1;
        }
    }

    result
}

/// Compute longest common subsequence
fn compute_lcs<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<&'a str> {
    let m = a.len();
    let n = b.len();

    if m == 0 || n == 0 {
        return Vec::new();
    }

    // Build LCS table
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to find LCS
    let mut result = Vec::new();
    let mut i = m;
    let mut j = n;

    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            result.push(a[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    result.reverse();
    result
}

/// Status of a file lock
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LockStatus {
    /// File is not locked
    Unlocked,
    /// File is locked for reading (shared)
    ReadLocked,
    /// File is locked for writing (exclusive)
    WriteLocked,
}

/// A file lock
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileLock {
    /// Path to the locked file
    pub path: PathBuf,

    /// Lock status
    pub status: LockStatus,

    /// Lock holder identifier
    pub holder: String,

    /// Timestamp when lock was acquired
    pub acquired_at: u64,
}

/// An operation on a file
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FileOperation {
    /// Create a new file
    Create {
        path: PathBuf,
        content: String,
        #[serde(default)]
        metadata: FileMetadata,
    },

    /// Modify an existing file
    Modify {
        path: PathBuf,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        original: Option<String>,
    },

    /// Delete a file
    Delete { path: PathBuf },

    /// Rename/move a file
    Rename { from: PathBuf, to: PathBuf },
}

impl FileOperation {
    /// Get the primary path affected by this operation
    pub fn path(&self) -> &Path {
        match self {
            FileOperation::Create { path, .. } => path,
            FileOperation::Modify { path, .. } => path,
            FileOperation::Delete { path } => path,
            FileOperation::Rename { from, .. } => from,
        }
    }

    /// Get all paths affected by this operation
    pub fn affected_paths(&self) -> Vec<&Path> {
        match self {
            FileOperation::Create { path, .. } => vec![path],
            FileOperation::Modify { path, .. } => vec![path],
            FileOperation::Delete { path } => vec![path],
            FileOperation::Rename { from, to } => vec![from, to],
        }
    }
}

/// Error types for filesystem operations
#[derive(Debug, Clone, PartialEq)]
pub enum FileSystemError {
    /// File is locked by another operation
    FileLocked { path: PathBuf, holder: String },

    /// File not found in state
    FileNotFound { path: PathBuf },

    /// Dependency cycle detected
    DependencyCycle { files: Vec<PathBuf> },

    /// Dependency not satisfied
    DependencyNotSatisfied { file: PathBuf, depends_on: PathBuf },

    /// Commit validation failed
    CommitValidationFailed { message: String },

    /// File already exists
    FileAlreadyExists { path: PathBuf },

    /// Generic I/O error
    IoError { message: String },
}

impl std::fmt::Display for FileSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileLocked { path, holder } => {
                write!(f, "File '{}' is locked by '{}'", path.display(), holder)
            }
            Self::FileNotFound { path } => {
                write!(f, "File '{}' not found in state", path.display())
            }
            Self::DependencyCycle { files } => {
                let paths: Vec<_> = files.iter().map(|p| p.display().to_string()).collect();
                write!(f, "Dependency cycle detected: {}", paths.join(" -> "))
            }
            Self::DependencyNotSatisfied { file, depends_on } => {
                write!(
                    f,
                    "Dependency not satisfied: '{}' depends on '{}'",
                    file.display(),
                    depends_on.display()
                )
            }
            Self::CommitValidationFailed { message } => {
                write!(f, "Commit validation failed: {}", message)
            }
            Self::FileAlreadyExists { path } => {
                write!(f, "File '{}' already exists", path.display())
            }
            Self::IoError { message } => {
                write!(f, "I/O error: {}", message)
            }
        }
    }
}

impl std::error::Error for FileSystemError {}

/// Multi-file state manager
///
/// Tracks the state of multiple files, their dependencies, and locks.
/// Provides atomic commit capability for multi-file changes.
#[derive(Debug)]
pub struct FileSystemState {
    /// Files being tracked
    files: Arc<RwLock<HashMap<PathBuf, FileContent>>>,

    /// File locks
    locks: Arc<RwLock<HashMap<PathBuf, FileLock>>>,

    /// Dependencies (file -> files it depends on)
    dependencies: Arc<RwLock<HashMap<PathBuf, HashSet<PathBuf>>>>,

    /// Pending operations
    pending_ops: Arc<RwLock<Vec<FileOperation>>>,

    /// Current holder ID (for lock tracking)
    holder_id: String,
}

impl Clone for FileSystemState {
    fn clone(&self) -> Self {
        // Share the same underlying Arc data (shallow clone)
        // This allows commits to modify the same state
        Self {
            files: Arc::clone(&self.files),
            locks: Arc::clone(&self.locks),
            dependencies: Arc::clone(&self.dependencies),
            pending_ops: Arc::clone(&self.pending_ops),
            holder_id: self.holder_id.clone(),
        }
    }
}

impl FileSystemState {
    /// Create a deep clone with separate storage
    pub fn deep_clone(&self) -> Self {
        Self {
            files: Arc::new(RwLock::new(self.files.read().unwrap().clone())),
            locks: Arc::new(RwLock::new(self.locks.read().unwrap().clone())),
            dependencies: Arc::new(RwLock::new(self.dependencies.read().unwrap().clone())),
            pending_ops: Arc::new(RwLock::new(self.pending_ops.read().unwrap().clone())),
            holder_id: self.holder_id.clone(),
        }
    }
}

impl Default for FileSystemState {
    fn default() -> Self {
        Self::new()
    }
}

impl FileSystemState {
    /// Create a new empty file system state
    pub fn new() -> Self {
        Self {
            files: Arc::new(RwLock::new(HashMap::new())),
            locks: Arc::new(RwLock::new(HashMap::new())),
            dependencies: Arc::new(RwLock::new(HashMap::new())),
            pending_ops: Arc::new(RwLock::new(Vec::new())),
            holder_id: format!("holder-{}", std::process::id()),
        }
    }

    /// Create with a specific holder ID
    pub fn with_holder_id(holder_id: impl Into<String>) -> Self {
        Self {
            files: Arc::new(RwLock::new(HashMap::new())),
            locks: Arc::new(RwLock::new(HashMap::new())),
            dependencies: Arc::new(RwLock::new(HashMap::new())),
            pending_ops: Arc::new(RwLock::new(Vec::new())),
            holder_id: holder_id.into(),
        }
    }

    /// Get the holder ID
    pub fn holder_id(&self) -> &str {
        &self.holder_id
    }

    /// Add a file to track
    pub fn add_file(
        &self,
        path: impl AsRef<Path>,
        content: FileContent,
    ) -> Result<(), FileSystemError> {
        let path = path.as_ref().to_path_buf();
        let mut files = self.files.write().unwrap();

        if files.contains_key(&path) {
            return Err(FileSystemError::FileAlreadyExists { path });
        }

        files.insert(path, content);
        Ok(())
    }

    /// Update a file's content
    pub fn update_file(
        &self,
        path: impl AsRef<Path>,
        content: String,
    ) -> Result<(), FileSystemError> {
        let path = path.as_ref().to_path_buf();
        let mut files = self.files.write().unwrap();

        let file = files
            .get_mut(&path)
            .ok_or_else(|| FileSystemError::FileNotFound { path: path.clone() })?;

        // Preserve original if not already set
        if file.original.is_none() {
            file.original = Some(file.current.clone());
        }

        file.current = content;
        Ok(())
    }

    /// Get a file's content
    pub fn get_file(&self, path: impl AsRef<Path>) -> Option<FileContent> {
        let files = self.files.read().unwrap();
        files.get(path.as_ref()).cloned()
    }

    /// Check if a file exists in state
    pub fn has_file(&self, path: impl AsRef<Path>) -> bool {
        let files = self.files.read().unwrap();
        files.contains_key(path.as_ref())
    }

    /// Get all tracked file paths
    pub fn file_paths(&self) -> Vec<PathBuf> {
        let files = self.files.read().unwrap();
        files.keys().cloned().collect()
    }

    /// Get count of tracked files
    pub fn file_count(&self) -> usize {
        let files = self.files.read().unwrap();
        files.len()
    }

    /// Add a dependency between files
    pub fn add_dependency(
        &self,
        file: impl AsRef<Path>,
        depends_on: impl AsRef<Path>,
    ) -> Result<(), FileSystemError> {
        let file = file.as_ref().to_path_buf();
        let depends_on = depends_on.as_ref().to_path_buf();

        // Check for cycles
        if self.would_create_cycle(&file, &depends_on) {
            return Err(FileSystemError::DependencyCycle {
                files: vec![file, depends_on],
            });
        }

        let mut deps = self.dependencies.write().unwrap();
        deps.entry(file).or_default().insert(depends_on);

        Ok(())
    }

    /// Check if adding a dependency would create a cycle
    fn would_create_cycle(&self, file: &Path, depends_on: &Path) -> bool {
        // Check if depends_on transitively depends on file
        let deps = self.dependencies.read().unwrap();
        let mut visited = HashSet::new();
        let mut stack = vec![depends_on.to_path_buf()];

        while let Some(current) = stack.pop() {
            if current == file {
                return true;
            }

            if visited.insert(current.clone()) {
                if let Some(current_deps) = deps.get(&current) {
                    for dep in current_deps {
                        stack.push(dep.clone());
                    }
                }
            }
        }

        false
    }

    /// Get dependencies for a file
    pub fn get_dependencies(&self, file: impl AsRef<Path>) -> Vec<PathBuf> {
        let deps = self.dependencies.read().unwrap();
        deps.get(file.as_ref())
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Acquire a write lock on a file
    pub fn acquire_write_lock(&self, path: impl AsRef<Path>) -> Result<(), FileSystemError> {
        let path = path.as_ref().to_path_buf();
        let mut locks = self.locks.write().unwrap();

        if let Some(lock) = locks.get(&path) {
            if lock.holder != self.holder_id {
                return Err(FileSystemError::FileLocked {
                    path,
                    holder: lock.holder.clone(),
                });
            }
        }

        locks.insert(
            path.clone(),
            FileLock {
                path,
                status: LockStatus::WriteLocked,
                holder: self.holder_id.clone(),
                acquired_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            },
        );

        Ok(())
    }

    /// Release a lock on a file
    pub fn release_lock(&self, path: impl AsRef<Path>) -> Result<(), FileSystemError> {
        let path = path.as_ref().to_path_buf();
        let mut locks = self.locks.write().unwrap();

        if let Some(lock) = locks.get(&path) {
            if lock.holder != self.holder_id {
                return Err(FileSystemError::FileLocked {
                    path,
                    holder: lock.holder.clone(),
                });
            }
        }

        locks.remove(&path);
        Ok(())
    }

    /// Check if a file is locked
    pub fn is_locked(&self, path: impl AsRef<Path>) -> bool {
        let locks = self.locks.read().unwrap();
        locks.contains_key(path.as_ref())
    }

    /// Get lock status for a file
    pub fn lock_status(&self, path: impl AsRef<Path>) -> LockStatus {
        let locks = self.locks.read().unwrap();
        locks
            .get(path.as_ref())
            .map(|l| l.status)
            .unwrap_or(LockStatus::Unlocked)
    }

    /// Add a pending operation
    pub fn add_operation(&self, op: FileOperation) -> Result<(), FileSystemError> {
        // Validate the operation
        match &op {
            FileOperation::Create { path, .. } => {
                if self.has_file(path) {
                    return Err(FileSystemError::FileAlreadyExists { path: path.clone() });
                }
            }
            FileOperation::Modify { path, .. } | FileOperation::Delete { path } => {
                if !self.has_file(path) {
                    return Err(FileSystemError::FileNotFound { path: path.clone() });
                }
            }
            FileOperation::Rename { from, to } => {
                if !self.has_file(from) {
                    return Err(FileSystemError::FileNotFound { path: from.clone() });
                }
                if self.has_file(to) {
                    return Err(FileSystemError::FileAlreadyExists { path: to.clone() });
                }
            }
        }

        let mut ops = self.pending_ops.write().unwrap();
        ops.push(op);
        Ok(())
    }

    /// Get pending operations
    pub fn pending_operations(&self) -> Vec<FileOperation> {
        let ops = self.pending_ops.read().unwrap();
        ops.clone()
    }

    /// Clear pending operations
    pub fn clear_pending(&self) {
        let mut ops = self.pending_ops.write().unwrap();
        ops.clear();
    }

    /// Prepare a commit of all pending operations
    pub fn prepare_commit(&self) -> Result<FileCommit, FileSystemError> {
        let ops = self.pending_ops.read().unwrap().clone();

        if ops.is_empty() {
            return Err(FileSystemError::CommitValidationFailed {
                message: "No pending operations".to_string(),
            });
        }

        // Verify all dependencies are satisfied
        self.verify_dependencies(&ops)?;

        // Acquire locks for all affected files
        let mut locked_paths = Vec::new();
        for op in &ops {
            for path in op.affected_paths() {
                if !locked_paths.contains(&path.to_path_buf()) {
                    self.acquire_write_lock(path)?;
                    locked_paths.push(path.to_path_buf());
                }
            }
        }

        Ok(FileCommit {
            operations: ops,
            state: self.clone(),
            locked_paths,
        })
    }

    /// Verify all dependencies are satisfied for the given operations
    fn verify_dependencies(&self, ops: &[FileOperation]) -> Result<(), FileSystemError> {
        let deps = self.dependencies.read().unwrap();

        // Build set of files being modified
        let modified: HashSet<PathBuf> = ops
            .iter()
            .flat_map(|op| op.affected_paths())
            .map(|p| p.to_path_buf())
            .collect();

        // Check each modified file's dependencies
        for path in &modified {
            if let Some(file_deps) = deps.get(path) {
                for dep in file_deps {
                    // Dependency must either be in this commit or already exist
                    if !modified.contains(dep) && !self.has_file(dep) {
                        return Err(FileSystemError::DependencyNotSatisfied {
                            file: path.clone(),
                            depends_on: dep.clone(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Get modified files
    pub fn modified_files(&self) -> Vec<PathBuf> {
        let files = self.files.read().unwrap();
        files
            .iter()
            .filter(|(_, content)| content.is_modified())
            .map(|(path, _)| path.clone())
            .collect()
    }

    /// Rollback all changes (restore original content)
    pub fn rollback(&self) {
        let mut files = self.files.write().unwrap();

        for (_, content) in files.iter_mut() {
            if let Some(original) = &content.original {
                content.current = original.clone();
            }
        }

        // Clear pending operations
        self.clear_pending();
    }
}

/// A prepared commit ready for execution
#[derive(Debug)]
pub struct FileCommit {
    /// Operations to execute
    operations: Vec<FileOperation>,

    /// Reference to the state
    state: FileSystemState,

    /// Paths that were locked
    locked_paths: Vec<PathBuf>,
}

impl FileCommit {
    /// Get the operations in this commit
    pub fn operations(&self) -> &[FileOperation] {
        &self.operations
    }

    /// Execute the commit (applies all operations)
    pub fn execute(self) -> Result<CommitResult, FileSystemError> {
        let mut applied = Vec::new();

        for op in &self.operations {
            match op {
                FileOperation::Create {
                    path,
                    content,
                    metadata,
                } => {
                    let file_content = FileContent {
                        original: None,
                        current: content.clone(),
                        metadata: metadata.clone(),
                    };
                    // Force add (we've already validated)
                    let mut files = self.state.files.write().unwrap();
                    files.insert(path.clone(), file_content);
                }
                FileOperation::Modify { path, content, .. } => {
                    let mut files = self.state.files.write().unwrap();
                    if let Some(file) = files.get_mut(path) {
                        if file.original.is_none() {
                            file.original = Some(file.current.clone());
                        }
                        file.current = content.clone();
                    }
                }
                FileOperation::Delete { path } => {
                    let mut files = self.state.files.write().unwrap();
                    files.remove(path);
                }
                FileOperation::Rename { from, to } => {
                    let mut files = self.state.files.write().unwrap();
                    if let Some(content) = files.remove(from) {
                        files.insert(to.clone(), content);
                    }
                }
            }
            applied.push(op.clone());
        }

        // Release all locks
        for path in &self.locked_paths {
            let _ = self.state.release_lock(path);
        }

        // Clear pending operations
        self.state.clear_pending();

        Ok(CommitResult {
            operations_applied: applied.len(),
            files_affected: self.locked_paths.clone(),
        })
    }

    /// Abort the commit (release locks without applying)
    pub fn abort(self) {
        for path in &self.locked_paths {
            let _ = self.state.release_lock(path);
        }
    }
}

/// Result of a successful commit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResult {
    /// Number of operations applied
    pub operations_applied: usize,

    /// Files that were affected
    pub files_affected: Vec<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // FileContent Tests
    // ==========================================

    #[test]
    fn test_file_content_new() {
        let content = FileContent::new("hello");
        assert_eq!(content.current, "hello");
        assert!(content.original.is_none());
    }

    #[test]
    fn test_file_content_with_original() {
        let content = FileContent::with_original("original", "modified");
        assert_eq!(content.original, Some("original".to_string()));
        assert_eq!(content.current, "modified");
    }

    #[test]
    fn test_file_content_is_modified() {
        let new_file = FileContent::new("content");
        assert!(new_file.is_modified()); // New file is considered modified

        let unchanged = FileContent::with_original("same", "same");
        assert!(!unchanged.is_modified());

        let changed = FileContent::with_original("original", "modified");
        assert!(changed.is_modified());
    }

    #[test]
    fn test_file_content_diff() {
        let content = FileContent::with_original("line1\nline2\nline3", "line1\nmodified\nline3");
        let diff = content.diff();

        assert!(diff
            .iter()
            .any(|d| matches!(d, DiffLine::Context(s) if s == "line1")));
        assert!(diff
            .iter()
            .any(|d| matches!(d, DiffLine::Removed(s) if s == "line2")));
        assert!(diff
            .iter()
            .any(|d| matches!(d, DiffLine::Added(s) if s == "modified")));
    }

    // ==========================================
    // FileSystemState Tests
    // ==========================================

    #[test]
    fn test_state_new() {
        let state = FileSystemState::new();
        assert_eq!(state.file_count(), 0);
    }

    #[test]
    fn test_state_with_holder_id() {
        let state = FileSystemState::with_holder_id("test-holder");
        assert_eq!(state.holder_id(), "test-holder");
    }

    #[test]
    fn test_add_file() {
        let state = FileSystemState::new();
        let result = state.add_file("test.rs", FileContent::new("fn main() {}"));
        assert!(result.is_ok());
        assert!(state.has_file("test.rs"));
    }

    #[test]
    fn test_add_duplicate_file() {
        let state = FileSystemState::new();
        state.add_file("test.rs", FileContent::new("v1")).unwrap();
        let result = state.add_file("test.rs", FileContent::new("v2"));
        assert!(matches!(
            result,
            Err(FileSystemError::FileAlreadyExists { .. })
        ));
    }

    #[test]
    fn test_update_file() {
        let state = FileSystemState::new();
        state.add_file("test.rs", FileContent::new("v1")).unwrap();
        state.update_file("test.rs", "v2".to_string()).unwrap();

        let content = state.get_file("test.rs").unwrap();
        assert_eq!(content.current, "v2");
        assert_eq!(content.original, Some("v1".to_string()));
    }

    #[test]
    fn test_update_nonexistent_file() {
        let state = FileSystemState::new();
        let result = state.update_file("nonexistent.rs", "content".to_string());
        assert!(matches!(result, Err(FileSystemError::FileNotFound { .. })));
    }

    #[test]
    fn test_file_paths() {
        let state = FileSystemState::new();
        state.add_file("a.rs", FileContent::new("a")).unwrap();
        state.add_file("b.rs", FileContent::new("b")).unwrap();

        let paths = state.file_paths();
        assert_eq!(paths.len(), 2);
    }

    // ==========================================
    // Dependency Tests
    // ==========================================

    #[test]
    fn test_add_dependency() {
        let state = FileSystemState::new();
        let result = state.add_dependency("main.rs", "lib.rs");
        assert!(result.is_ok());

        let deps = state.get_dependencies("main.rs");
        assert_eq!(deps.len(), 1);
        assert!(deps.contains(&PathBuf::from("lib.rs")));
    }

    #[test]
    fn test_dependency_cycle_detection() {
        let state = FileSystemState::new();
        state.add_dependency("a.rs", "b.rs").unwrap();
        state.add_dependency("b.rs", "c.rs").unwrap();

        // c.rs -> a.rs would create a cycle
        let result = state.add_dependency("c.rs", "a.rs");
        assert!(matches!(
            result,
            Err(FileSystemError::DependencyCycle { .. })
        ));
    }

    // ==========================================
    // Lock Tests
    // ==========================================

    #[test]
    fn test_acquire_write_lock() {
        let state = FileSystemState::new();
        state.add_file("test.rs", FileContent::new("")).unwrap();

        let result = state.acquire_write_lock("test.rs");
        assert!(result.is_ok());
        assert!(state.is_locked("test.rs"));
        assert_eq!(state.lock_status("test.rs"), LockStatus::WriteLocked);
    }

    #[test]
    fn test_release_lock() {
        let state = FileSystemState::new();
        state.add_file("test.rs", FileContent::new("")).unwrap();
        state.acquire_write_lock("test.rs").unwrap();

        let result = state.release_lock("test.rs");
        assert!(result.is_ok());
        assert!(!state.is_locked("test.rs"));
    }

    #[test]
    fn test_lock_conflict() {
        let state1 = FileSystemState::with_holder_id("holder1");

        // Create state2 sharing the same files, locks, and dependencies maps
        let state2 = FileSystemState {
            files: state1.files.clone(),
            locks: state1.locks.clone(),
            dependencies: state1.dependencies.clone(),
            pending_ops: Arc::new(RwLock::new(Vec::new())),
            holder_id: "holder2".to_string(),
        };

        state1.add_file("test.rs", FileContent::new("")).unwrap();
        state1.acquire_write_lock("test.rs").unwrap();

        let result = state2.acquire_write_lock("test.rs");
        assert!(matches!(result, Err(FileSystemError::FileLocked { .. })));
    }

    // ==========================================
    // Operation Tests
    // ==========================================

    #[test]
    fn test_add_create_operation() {
        let state = FileSystemState::new();
        let op = FileOperation::Create {
            path: PathBuf::from("new.rs"),
            content: "fn new() {}".to_string(),
            metadata: FileMetadata::default(),
        };

        let result = state.add_operation(op);
        assert!(result.is_ok());
        assert_eq!(state.pending_operations().len(), 1);
    }

    #[test]
    fn test_add_modify_operation() {
        let state = FileSystemState::new();
        state.add_file("test.rs", FileContent::new("v1")).unwrap();

        let op = FileOperation::Modify {
            path: PathBuf::from("test.rs"),
            content: "v2".to_string(),
            original: None,
        };

        let result = state.add_operation(op);
        assert!(result.is_ok());
    }

    #[test]
    fn test_add_operation_file_not_found() {
        let state = FileSystemState::new();
        let op = FileOperation::Modify {
            path: PathBuf::from("nonexistent.rs"),
            content: "content".to_string(),
            original: None,
        };

        let result = state.add_operation(op);
        assert!(matches!(result, Err(FileSystemError::FileNotFound { .. })));
    }

    // ==========================================
    // Commit Tests
    // ==========================================

    #[test]
    fn test_prepare_commit() {
        let state = FileSystemState::new();
        state.add_file("test.rs", FileContent::new("v1")).unwrap();

        let op = FileOperation::Modify {
            path: PathBuf::from("test.rs"),
            content: "v2".to_string(),
            original: None,
        };
        state.add_operation(op).unwrap();

        let commit = state.prepare_commit();
        assert!(commit.is_ok());
    }

    #[test]
    fn test_prepare_commit_empty() {
        let state = FileSystemState::new();
        let result = state.prepare_commit();
        assert!(matches!(
            result,
            Err(FileSystemError::CommitValidationFailed { .. })
        ));
    }

    #[test]
    fn test_execute_commit() {
        let state = FileSystemState::new();
        state.add_file("test.rs", FileContent::new("v1")).unwrap();

        let op = FileOperation::Modify {
            path: PathBuf::from("test.rs"),
            content: "v2".to_string(),
            original: None,
        };
        state.add_operation(op).unwrap();

        let commit = state.prepare_commit().unwrap();
        let result = commit.execute();
        assert!(result.is_ok());

        let content = state.get_file("test.rs").unwrap();
        assert_eq!(content.current, "v2");
    }

    #[test]
    fn test_abort_commit() {
        let state = FileSystemState::new();
        state.add_file("test.rs", FileContent::new("v1")).unwrap();

        let op = FileOperation::Modify {
            path: PathBuf::from("test.rs"),
            content: "v2".to_string(),
            original: None,
        };
        state.add_operation(op).unwrap();

        let commit = state.prepare_commit().unwrap();
        commit.abort();

        // File should not be modified
        let content = state.get_file("test.rs").unwrap();
        assert_eq!(content.current, "v1");
    }

    // ==========================================
    // Rollback Tests
    // ==========================================

    #[test]
    fn test_rollback() {
        let state = FileSystemState::new();
        state
            .add_file("test.rs", FileContent::new("original"))
            .unwrap();
        state
            .update_file("test.rs", "modified".to_string())
            .unwrap();

        state.rollback();

        let content = state.get_file("test.rs").unwrap();
        assert_eq!(content.current, "original");
    }

    // ==========================================
    // Modified Files Tests
    // ==========================================

    #[test]
    fn test_modified_files() {
        let state = FileSystemState::new();
        state
            .add_file("unchanged.rs", FileContent::with_original("same", "same"))
            .unwrap();
        state.add_file("new.rs", FileContent::new("new")).unwrap();
        state
            .add_file("changed.rs", FileContent::with_original("v1", "v2"))
            .unwrap();

        let modified = state.modified_files();
        assert_eq!(modified.len(), 2);
        assert!(modified.contains(&PathBuf::from("new.rs")));
        assert!(modified.contains(&PathBuf::from("changed.rs")));
    }

    // ==========================================
    // FileOperation Tests
    // ==========================================

    #[test]
    fn test_operation_path() {
        let create = FileOperation::Create {
            path: PathBuf::from("new.rs"),
            content: "".to_string(),
            metadata: FileMetadata::default(),
        };
        assert_eq!(create.path(), Path::new("new.rs"));

        let rename = FileOperation::Rename {
            from: PathBuf::from("old.rs"),
            to: PathBuf::from("new.rs"),
        };
        assert_eq!(rename.path(), Path::new("old.rs"));
    }

    #[test]
    fn test_operation_affected_paths() {
        let rename = FileOperation::Rename {
            from: PathBuf::from("old.rs"),
            to: PathBuf::from("new.rs"),
        };
        let affected = rename.affected_paths();
        assert_eq!(affected.len(), 2);
    }

    // ==========================================
    // Error Display Tests
    // ==========================================

    #[test]
    fn test_error_display() {
        let errors = vec![
            FileSystemError::FileLocked {
                path: PathBuf::from("test.rs"),
                holder: "other".to_string(),
            },
            FileSystemError::FileNotFound {
                path: PathBuf::from("test.rs"),
            },
            FileSystemError::DependencyCycle {
                files: vec![PathBuf::from("a.rs"), PathBuf::from("b.rs")],
            },
            FileSystemError::DependencyNotSatisfied {
                file: PathBuf::from("a.rs"),
                depends_on: PathBuf::from("b.rs"),
            },
            FileSystemError::CommitValidationFailed {
                message: "test".to_string(),
            },
            FileSystemError::FileAlreadyExists {
                path: PathBuf::from("test.rs"),
            },
            FileSystemError::IoError {
                message: "test".to_string(),
            },
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
        }
    }

    // ==========================================
    // Serialization Tests
    // ==========================================

    #[test]
    fn test_file_content_serialization() {
        let content = FileContent::with_original("orig", "curr");
        let json = serde_json::to_string(&content).unwrap();
        let parsed: FileContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content.original, parsed.original);
        assert_eq!(content.current, parsed.current);
    }

    #[test]
    fn test_file_operation_serialization() {
        let ops = vec![
            FileOperation::Create {
                path: PathBuf::from("new.rs"),
                content: "fn new()".to_string(),
                metadata: FileMetadata::default(),
            },
            FileOperation::Modify {
                path: PathBuf::from("test.rs"),
                content: "modified".to_string(),
                original: Some("original".to_string()),
            },
            FileOperation::Delete {
                path: PathBuf::from("old.rs"),
            },
            FileOperation::Rename {
                from: PathBuf::from("a.rs"),
                to: PathBuf::from("b.rs"),
            },
        ];

        for op in ops {
            let json = serde_json::to_string(&op).unwrap();
            let _parsed: FileOperation = serde_json::from_str(&json).unwrap();
        }
    }
}
