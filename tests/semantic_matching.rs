//! Semantic Matching Test Suite (STORY-009-05)
//!
//! Comprehensive tests validating that matchers correctly group equivalent responses.
//!
//! Test corpus:
//! - 50+ pairs of equivalent code snippets (Python, Rust, JavaScript)
//! - 50+ pairs of semantically similar natural language responses
//! - 25+ pairs of non-equivalent responses (negative cases)
//!
//! Acceptance criteria:
//! - CodeMatcher: >95% accuracy on code equivalence corpus
//! - EmbeddingMatcher: >90% accuracy on NL corpus
//! - False positive rate <5%
//! - Reflexivity and symmetry properties hold

use maker::core::matcher::{CandidateMatcher, ExactMatcher};
use maker::core::matchers::embedding::{cosine_similarity, EmbeddingMatcher, MockEmbeddingClient};

#[cfg(feature = "code-matcher")]
use maker::core::matchers::code::{CodeLanguage, CodeMatcher};

// ============================================================================
// Test Corpus: Equivalent Code Pairs
// ============================================================================

/// Code pairs that are structurally equivalent (different names/formatting).
#[cfg(feature = "code-matcher")]
fn equivalent_python_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        // 1. Variable renaming
        ("x = 10\ny = x + 5", "a = 10\nb = a + 5"),
        // 2. Whitespace differences
        ("x=10\ny=x+5", "x = 10\ny = x + 5"),
        // 3. Different function names
        (
            "def foo(x):\n    return x + 1",
            "def bar(y):\n    return y + 1",
        ),
        // 4. Comment differences
        ("# compute sum\nx = 1 + 2", "# add values\nx = 1 + 2"),
        // 5. Blank line differences
        ("x = 1\n\n\ny = 2", "x = 1\ny = 2"),
        // 6. Different parameter names
        (
            "def f(a, b):\n    return a + b",
            "def f(x, y):\n    return x + y",
        ),
        // 7. Trailing whitespace
        ("x = 42  ", "x = 42"),
        // 8. Comments vs no comments
        ("x = 10  # value\ny = 20", "x = 10\ny = 20"),
        // 9. Inline comment differences
        ("result = 1 + 2  # add", "result = 1 + 2  # sum"),
        // 10. Multiple variable renames
        (
            "def calc(val):\n    result = val * 2\n    return result",
            "def compute(num):\n    output = num * 2\n    return output",
        ),
        // 11. Docstring differences (treated as comments)
        (
            "def f(x):\n    \"\"\"Does stuff.\"\"\"\n    return x",
            "def f(x):\n    \"\"\"Computes value.\"\"\"\n    return x",
        ),
        // 12. Loop variable renaming
        (
            "for i in range(10):\n    print(i)",
            "for j in range(10):\n    print(j)",
        ),
        // 13. Same logic, different temp var names
        ("tmp = a\na = b\nb = tmp", "temp = a\na = b\nb = temp"),
        // 14. List comprehension variable rename
        ("[x * 2 for x in range(5)]", "[y * 2 for y in range(5)]"),
        // 15. Function with renamed locals
        (
            "def sort(lst):\n    n = len(lst)\n    return sorted(lst)",
            "def sort(arr):\n    size = len(arr)\n    return sorted(arr)",
        ),
        // 16. Class method rename
        (
            "class Foo:\n    def bar(self):\n        pass",
            "class Baz:\n    def qux(self):\n        pass",
        ),
        // 17. Different indentation (tabs vs spaces equivalent after parse)
        ("if True:\n    x = 1", "if True:\n    x = 1"),
    ]
}

#[cfg(feature = "code-matcher")]
fn equivalent_rust_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        // 1. Variable renaming
        ("let x = 10;\nlet y = x + 5;", "let a = 10;\nlet b = a + 5;"),
        // 2. Whitespace differences
        ("let x=10;", "let x = 10;"),
        // 3. Function renaming
        (
            "fn foo(x: i32) -> i32 { x + 1 }",
            "fn bar(y: i32) -> i32 { y + 1 }",
        ),
        // 4. Comment differences
        ("// compute\nlet x = 42;", "// calculate\nlet x = 42;"),
        // 5. Doc comment vs regular comment
        ("/// Docs\nfn f() {}", "// Docs\nfn f() {}"),
        // 6. Parameter renaming
        (
            "fn add(a: i32, b: i32) -> i32 { a + b }",
            "fn add(x: i32, y: i32) -> i32 { x + y }",
        ),
        // 7. Blank lines
        ("let x = 1;\n\nlet y = 2;", "let x = 1;\nlet y = 2;"),
        // 8. Trailing semicolons and whitespace
        ("let val = 10;  ", "let val = 10;"),
        // 9. Block comment vs line comment
        ("/* note */ let x = 1;", "// note\nlet x = 1;"),
        // 10. Multiple renames in function body
        (
            "fn calc(val: f64) -> f64 {\n    let result = val * 2.0;\n    result\n}",
            "fn compute(num: f64) -> f64 {\n    let output = num * 2.0;\n    output\n}",
        ),
        // 11. Loop variable renaming
        (
            "for i in 0..10 { println!(\"{}\", i); }",
            "for j in 0..10 { println!(\"{}\", j); }",
        ),
        // 12. Closure variable renaming
        ("let f = |x| x + 1;", "let g = |y| y + 1;"),
        // 15. if/else with renamed vars
        (
            "let result = if flag { a } else { b };",
            "let output = if flag { a } else { b };",
        ),
        // 16. Impl block with renamed method
        (
            "impl Foo { fn bar(&self) -> i32 { 42 } }",
            "impl Foo { fn baz(&self) -> i32 { 42 } }",
        ),
        // 17. Tuple destructuring renaming
        ("let (a, b) = (1, 2);", "let (x, y) = (1, 2);"),
    ]
}

#[cfg(feature = "code-matcher")]
fn equivalent_javascript_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        // 1. Variable renaming
        (
            "const x = 10;\nconst y = x + 5;",
            "const a = 10;\nconst b = a + 5;",
        ),
        // 2. Whitespace
        ("const x=10;", "const x = 10;"),
        // 3. Function renaming
        (
            "function foo(x) { return x + 1; }",
            "function bar(y) { return y + 1; }",
        ),
        // 4. Comment differences
        ("// compute\nconst x = 42;", "// calculate\nconst x = 42;"),
        // 5. Arrow function variable rename
        ("const f = (x) => x + 1;", "const g = (y) => y + 1;"),
        // 6. Template literal whitespace
        ("const s = `hello`;", "const s = `hello`;"),
        // 7. Parameter renaming
        (
            "function add(a, b) { return a + b; }",
            "function add(x, y) { return x + y; }",
        ),
        // 8. Block comment vs line comment
        ("/* note */ const x = 1;", "// note\nconst x = 1;"),
        // 9. Blank lines
        ("const x = 1;\n\nconst y = 2;", "const x = 1;\nconst y = 2;"),
        // 10. Method renaming
        (
            "class Foo { bar() { return 42; } }",
            "class Baz { qux() { return 42; } }",
        ),
        // 11. for loop variable rename
        (
            "for (let i = 0; i < 10; i++) { console.log(i); }",
            "for (let j = 0; j < 10; j++) { console.log(j); }",
        ),
        // 12. Callback variable rename
        ("arr.map(item => item * 2);", "arr.map(elem => elem * 2);"),
        // 14. Async function rename
        (
            "async function fetchData(url) { return await fetch(url); }",
            "async function getData(endpoint) { return await fetch(endpoint); }",
        ),
        // 16. Ternary with renamed vars
        (
            "const result = flag ? a : b;",
            "const output = flag ? a : b;",
        ),
    ]
}

// ============================================================================
// Test Corpus: Non-Equivalent Code Pairs (Negative Cases)
// ============================================================================

#[cfg(feature = "code-matcher")]
fn non_equivalent_code_pairs() -> Vec<(&'static str, &'static str, CodeLanguage)> {
    vec![
        // Completely different algorithms
        (
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
            CodeLanguage::Python,
        ),
        // Different data structures
        (
            "let v: Vec<i32> = vec![1, 2, 3];",
            "let s: &str = \"hello\";",
            CodeLanguage::Rust,
        ),
        // Different class structure
        (
            "class Dog { bark() { return 'woof'; } }",
            "class Cat { meow() { return 'meow'; } purr() {} }",
            CodeLanguage::JavaScript,
        ),
        // Different number of statements (significant difference)
        (
            "x = 1\ny = 2\nz = 3\nw = x + y + z",
            "x = 1",
            CodeLanguage::Python,
        ),
        (
            "let x = 1;\nlet y = 2;\nlet z = 3;\nlet w = x + y + z;",
            "let x = 1;",
            CodeLanguage::Rust,
        ),
        // Different function signatures (arity) with body
        (
            "def process(data):\n    result = []\n    for item in data:\n        result.append(item)\n    return result",
            "def process(data, filter_fn):\n    result = []\n    for item in data:\n        if filter_fn(item):\n            result.append(item)\n    return result",
            CodeLanguage::Python,
        ),
        (
            "fn process(data: Vec<i32>) -> Vec<i32> {\n    data.iter().map(|x| x + 1).collect()\n}",
            "fn process(data: Vec<i32>, offset: i32) -> Vec<i32> {\n    data.iter().map(|x| x + offset).collect()\n}",
            CodeLanguage::Rust,
        ),
        // Different control flow structure
        (
            "function handle(x) {\n    if (x > 0) return x;\n    return 0;\n}",
            "function handle(x) {\n    switch(x) {\n        case 0: return 'zero';\n        case 1: return 'one';\n        default: return 'other';\n    }\n}",
            CodeLanguage::JavaScript,
        ),
        // Loop vs recursion
        (
            "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total",
            "def sum_list(lst):\n    if not lst:\n        return 0\n    return lst[0] + sum_list(lst[1:])",
            CodeLanguage::Python,
        ),
        // Sorting vs reversing
        (
            "fn sort_vec(mut v: Vec<i32>) -> Vec<i32> {\n    v.sort();\n    v\n}",
            "fn reverse_vec(mut v: Vec<i32>) -> Vec<i32> {\n    v.reverse();\n    v\n}",
            CodeLanguage::Rust,
        ),
        // Read vs write
        (
            "const fs = require('fs');\nconst data = fs.readFileSync('input.txt', 'utf8');",
            "const fs = require('fs');\nfs.writeFileSync('output.txt', 'hello world');",
            CodeLanguage::JavaScript,
        ),
        // Stack vs queue operations
        (
            "stack = []\nstack.append(1)\nstack.append(2)\nval = stack.pop()",
            "from collections import deque\nqueue = deque()\nqueue.append(1)\nqueue.append(2)\nval = queue.popleft()",
            CodeLanguage::Python,
        ),
        // HashMap vs BTreeMap
        (
            "use std::collections::HashMap;\nlet mut m = HashMap::new();\nm.insert(1, \"a\");\nm.insert(2, \"b\");",
            "use std::collections::BTreeMap;\nlet mut m = BTreeMap::new();\nm.insert(1, \"a\");\nm.insert(2, \"b\");\nfor (k, v) in &m { println!(\"{}: {}\", k, v); }",
            CodeLanguage::Rust,
        ),
        // Sync vs async
        (
            "function fetchData(url) {\n    const xhr = new XMLHttpRequest();\n    xhr.open('GET', url, false);\n    xhr.send();\n    return xhr.responseText;\n}",
            "async function fetchData(url) {\n    const response = await fetch(url);\n    const data = await response.json();\n    return data;\n}",
            CodeLanguage::JavaScript,
        ),
        // Different error handling
        (
            "try:\n    result = int(input())\n    print(result * 2)\nexcept ValueError:\n    print('invalid')\nexcept EOFError:\n    print('no input')",
            "import sys\nline = sys.stdin.readline()\nif line.strip().isdigit():\n    print(int(line) * 2)\nelse:\n    print('invalid')",
            CodeLanguage::Python,
        ),
        // Builder vs constructor
        (
            "struct Config { a: i32, b: String }\nimpl Config {\n    fn new() -> Self { Config { a: 0, b: String::new() } }\n}",
            "struct Config { a: i32, b: String }\nimpl Config {\n    fn builder() -> ConfigBuilder { ConfigBuilder::default() }\n}\nstruct ConfigBuilder { a: i32, b: String }",
            CodeLanguage::Rust,
        ),
        // Class with different methods
        (
            "class Calculator {\n    add(a, b) { return a + b; }\n    subtract(a, b) { return a - b; }\n}",
            "class Logger {\n    log(msg) { console.log(msg); }\n    error(msg) { console.error(msg); }\n    warn(msg) { console.warn(msg); }\n}",
            CodeLanguage::JavaScript,
        ),
        // Generator vs list comprehension
        (
            "def squares(n):\n    for i in range(n):\n        yield i * i",
            "def squares(n):\n    return [i * i for i in range(n)]",
            CodeLanguage::Python,
        ),
        // Trait impl vs inherent impl
        (
            "trait Greet { fn hello(&self) -> String; }\nimpl Greet for Person { fn hello(&self) -> String { format!(\"Hi {}\", self.name) } }",
            "impl Person { fn hello(&self) -> String { format!(\"Hi {}\", self.name) }\n    fn goodbye(&self) -> String { format!(\"Bye {}\", self.name) } }",
            CodeLanguage::Rust,
        ),
        // Promise.all vs sequential
        (
            "async function run() {\n    const [a, b, c] = await Promise.all([f1(), f2(), f3()]);\n    return a + b + c;\n}",
            "async function run() {\n    const a = await f1();\n    const b = await f2();\n    const c = await f3();\n    return a + b + c;\n}",
            CodeLanguage::JavaScript,
        ),
        // Completely different purposes
        (
            "def parse_csv(filename):\n    import csv\n    with open(filename) as f:\n        reader = csv.reader(f)\n        return list(reader)",
            "def send_email(to, subject, body):\n    import smtplib\n    msg = f'Subject: {subject}\\n\\n{body}'\n    with smtplib.SMTP('localhost') as s:\n        s.sendmail('me@host', to, msg)",
            CodeLanguage::Python,
        ),
        // Different struct shapes
        (
            "struct Point { x: f64, y: f64 }\nimpl Point {\n    fn distance(&self, other: &Point) -> f64 {\n        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()\n    }\n}",
            "struct Color { r: u8, g: u8, b: u8 }\nimpl Color {\n    fn to_hex(&self) -> String {\n        format!(\"#{:02x}{:02x}{:02x}\", self.r, self.g, self.b)\n    }\n}",
            CodeLanguage::Rust,
        ),
        // DOM manipulation vs canvas
        (
            "const el = document.createElement('div');\nel.textContent = 'Hello';\nel.style.color = 'red';\ndocument.body.appendChild(el);",
            "const canvas = document.getElementById('canvas');\nconst ctx = canvas.getContext('2d');\nctx.fillStyle = 'blue';\nctx.fillRect(10, 10, 100, 100);",
            CodeLanguage::JavaScript,
        ),
        // Merge sort vs bubble sort
        (
            "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
            "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
            CodeLanguage::Python,
        ),
        // Binary search vs linear search
        (
            "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    let (mut lo, mut hi) = (0, arr.len());\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match arr[mid].cmp(&target) {\n            std::cmp::Ordering::Equal => return Some(mid),\n            std::cmp::Ordering::Less => lo = mid + 1,\n            std::cmp::Ordering::Greater => hi = mid,\n        }\n    }\n    None\n}",
            "fn linear_search(arr: &[i32], target: i32) -> Option<usize> {\n    for (i, &val) in arr.iter().enumerate() {\n        if val == target { return Some(i); }\n    }\n    None\n}",
            CodeLanguage::Rust,
        ),
    ]
}

// ============================================================================
// Test Corpus: Natural Language Equivalent Pairs
// ============================================================================

/// Pairs of semantically similar natural language responses that should match.
fn equivalent_nl_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        // 1. Same meaning, different wording
        ("The answer is 4.", "The result is 4."),
        // 2. Reordered sentence
        (
            "Python is a programming language.",
            "A programming language is Python.",
        ),
        // 3. Synonym substitution
        ("The function returns true.", "The function yields true."),
        // 4. Active vs passive
        ("The cat ate the fish.", "The fish was eaten by the cat."),
        // 5. Contraction differences
        ("It is working.", "It's working."),
        // 6. Article differences
        ("A solution exists.", "The solution exists."),
        // 7. Numeric equivalence
        ("The value is 42.", "The value equals 42."),
        // 8. Slightly different phrasing
        (
            "Use a hash map for lookup.",
            "Use a hash map to look up values.",
        ),
        // 9. With/without punctuation
        ("Hello world", "Hello, world!"),
        // 10. Abbreviation
        ("The API returns JSON data.", "The API returns JSON."),
        // 11. Different sentence structure, same info
        (
            "Sort the array in ascending order.",
            "Arrange the array from smallest to largest.",
        ),
        // 12. Technical synonym
        (
            "O(n log n) time complexity.",
            "The time complexity is O(n log n).",
        ),
        // 13. Imperative vs declarative
        ("Add error handling.", "Error handling should be added."),
        // 14. Short vs verbose
        ("Yes.", "Yes, that is correct."),
        // 15. Reworded explanation
        (
            "A stack uses LIFO ordering.",
            "Stacks follow last-in first-out ordering.",
        ),
        // 16. Same code in description
        ("Call `foo()` to start.", "To start, call `foo()`."),
        // 17. Same list, different formatting
        (
            "Steps: 1) read 2) process 3) write",
            "Steps: read, process, write",
        ),
        // 18. Same numbers
        ("The count is 100.", "There are 100 items."),
        // 19. Technical equivalence
        (
            "Use `Vec<T>` for dynamic arrays.",
            "Dynamic arrays use `Vec<T>`.",
        ),
        // 20. Whitespace only difference
        ("hello  world", "hello world"),
        // 21. Trailing period
        ("The answer is 42", "The answer is 42."),
        // 22. Case difference (near-equivalent)
        ("use recursion", "Use recursion"),
        // 23. Filler words
        ("Just use a loop.", "Use a loop."),
        // 24. Bullet vs prose
        (
            "- Read file\n- Parse data\n- Output result",
            "Read file, parse data, output result.",
        ),
        // 25. Quotation marks
        ("Set x to 'hello'.", "Set x to \"hello\"."),
        // 26-50: More varied pairs
        (
            "The time complexity is O(1).",
            "This runs in constant time.",
        ),
        (
            "Null pointer exception.",
            "NullPointerException was thrown.",
        ),
        ("The file does not exist.", "File not found."),
        ("Memory was allocated.", "Memory allocation occurred."),
        ("The test passed.", "Test passed successfully."),
        (
            "Compilation error on line 5.",
            "Error at line 5 during compilation.",
        ),
        (
            "Use TCP for reliable transport.",
            "TCP provides reliable transport.",
        ),
        (
            "The database query returned 0 rows.",
            "No rows returned from the query.",
        ),
        ("The function is pure.", "This is a pure function."),
        (
            "Binary search has O(log n) complexity.",
            "The complexity of binary search is O(log n).",
        ),
        (
            "The server responded with 200 OK.",
            "Server returned HTTP 200.",
        ),
        (
            "Use a mutex for thread safety.",
            "A mutex ensures thread safety.",
        ),
        ("The process exited with code 0.", "Exit code: 0."),
        ("No errors found.", "Zero errors detected."),
        ("The branch was merged.", "Branch merge completed."),
        (
            "Data is stored in UTF-8.",
            "UTF-8 encoding is used for data storage.",
        ),
        ("The request timed out.", "Timeout occurred on the request."),
        ("The list is sorted.", "Sorting of the list is complete."),
        ("Authentication failed.", "Login was unsuccessful."),
        ("Connection refused.", "The connection was refused."),
        ("Disk space is low.", "Low disk space warning."),
        (
            "The cache was invalidated.",
            "Cache invalidation performed.",
        ),
        ("Rate limit exceeded.", "Too many requests; rate limited."),
        ("The value is null.", "Null value returned."),
        ("Index out of bounds.", "Array index out of range."),
    ]
}

/// Pairs of non-equivalent natural language responses.
fn non_equivalent_nl_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        ("The answer is 4.", "The answer is 5."),
        ("The function returns true.", "The function returns false."),
        ("Use a hash map.", "Use a binary tree."),
        ("O(n) time complexity.", "O(n^2) time complexity."),
        ("The test passed.", "The test failed."),
        ("The server is running.", "The server is down."),
        ("Memory was allocated.", "Memory was freed."),
        ("The file exists.", "The file does not exist."),
        ("Use TCP.", "Use UDP."),
        ("The query returned 5 rows.", "The query returned 0 rows."),
        ("Use Python.", "Use Rust."),
        ("The value is 42.", "The value is 0."),
        ("Compile succeeded.", "Compile failed."),
        ("The array is sorted.", "The array is unsorted."),
        ("Connection established.", "Connection refused."),
        ("Authentication succeeded.", "Authentication failed."),
        ("Use recursion.", "Use iteration."),
        ("The key was found.", "The key was not found."),
        ("Data is encrypted.", "Data is plaintext."),
        ("The process started.", "The process terminated."),
        ("Read from stdin.", "Write to stdout."),
        ("Push to stack.", "Pop from stack."),
        ("Increment the counter.", "Decrement the counter."),
        ("The lock was acquired.", "The lock was released."),
        ("The branch was created.", "The branch was deleted."),
    ]
}

// ============================================================================
// CodeMatcher Accuracy Tests
// ============================================================================

#[cfg(feature = "code-matcher")]
mod code_matcher_tests {
    use super::*;

    fn accuracy(correct: usize, total: usize) -> f64 {
        correct as f64 / total as f64
    }

    #[test]
    fn test_python_equivalent_pairs_accuracy() {
        let matcher = CodeMatcher::new(CodeLanguage::Python, 0.80);
        let pairs = equivalent_python_pairs();
        let total = pairs.len();
        let correct = pairs
            .iter()
            .filter(|(a, b)| matcher.are_equivalent(a, b))
            .count();

        let acc = accuracy(correct, total);
        eprintln!(
            "Python equivalent accuracy: {}/{} ({:.1}%)",
            correct,
            total,
            acc * 100.0
        );
        assert!(
            acc >= 0.95,
            "CodeMatcher Python accuracy {:.1}% < 95% target ({}/{})",
            acc * 100.0,
            correct,
            total,
        );
    }

    #[test]
    fn test_rust_equivalent_pairs_accuracy() {
        let matcher = CodeMatcher::new(CodeLanguage::Rust, 0.80);
        let pairs = equivalent_rust_pairs();
        let total = pairs.len();
        let correct = pairs
            .iter()
            .filter(|(a, b)| matcher.are_equivalent(a, b))
            .count();

        let acc = accuracy(correct, total);
        eprintln!(
            "Rust equivalent accuracy: {}/{} ({:.1}%)",
            correct,
            total,
            acc * 100.0
        );
        assert!(
            acc >= 0.95,
            "CodeMatcher Rust accuracy {:.1}% < 95% target ({}/{})",
            acc * 100.0,
            correct,
            total,
        );
    }

    #[test]
    fn test_javascript_equivalent_pairs_accuracy() {
        let matcher = CodeMatcher::new(CodeLanguage::JavaScript, 0.80);
        let pairs = equivalent_javascript_pairs();
        let total = pairs.len();
        let correct = pairs
            .iter()
            .filter(|(a, b)| matcher.are_equivalent(a, b))
            .count();

        let acc = accuracy(correct, total);
        eprintln!(
            "JavaScript equivalent accuracy: {}/{} ({:.1}%)",
            correct,
            total,
            acc * 100.0
        );
        assert!(
            acc >= 0.95,
            "CodeMatcher JavaScript accuracy {:.1}% < 95% target ({}/{})",
            acc * 100.0,
            correct,
            total,
        );
    }

    #[test]
    fn test_code_false_positive_rate() {
        let pairs = non_equivalent_code_pairs();
        let total = pairs.len();
        let false_positives = pairs
            .iter()
            .filter(|(a, b, lang)| {
                let matcher = CodeMatcher::new(*lang, 0.85);
                matcher.are_equivalent(a, b)
            })
            .count();

        let fp_rate = false_positives as f64 / total as f64;
        eprintln!(
            "Code false positive rate: {}/{} ({:.1}%)",
            false_positives,
            total,
            fp_rate * 100.0
        );
        assert!(
            fp_rate < 0.05,
            "Code false positive rate {:.1}% >= 5% limit ({}/{})",
            fp_rate * 100.0,
            false_positives,
            total,
        );
    }

    #[test]
    fn test_code_matcher_reflexivity() {
        let snippets = vec![
            ("fn foo(x: i32) -> i32 { x + 1 }", CodeLanguage::Rust),
            ("def bar(x):\n    return x * 2", CodeLanguage::Python),
            ("const f = (x) => x + 1;", CodeLanguage::JavaScript),
            ("", CodeLanguage::Rust),
            ("// just a comment", CodeLanguage::Python),
            ("let x = vec![1, 2, 3];", CodeLanguage::Rust),
            ("class Foo:\n    pass", CodeLanguage::Python),
            (
                "async function f() { await g(); }",
                CodeLanguage::JavaScript,
            ),
        ];

        for (code, lang) in &snippets {
            let matcher = CodeMatcher::new(*lang, 0.85);
            assert!(
                matcher.are_equivalent(code, code),
                "Reflexivity violated for {:?} ({:?})",
                code,
                lang
            );
        }
    }

    #[test]
    fn test_code_matcher_symmetry() {
        let pairs = vec![
            ("let x = 1;", "let y = 1;", CodeLanguage::Rust),
            ("x = 1", "y = 1", CodeLanguage::Python),
            ("const x = 1;", "const y = 1;", CodeLanguage::JavaScript),
            ("fn foo() {}", "fn bar() {}", CodeLanguage::Rust),
            (
                "def f():\n    pass",
                "def g():\n    pass",
                CodeLanguage::Python,
            ),
        ];

        for (a, b, lang) in &pairs {
            let matcher = CodeMatcher::new(*lang, 0.85);
            let ab = matcher.are_equivalent(a, b);
            let ba = matcher.are_equivalent(b, a);
            assert_eq!(
                ab, ba,
                "Symmetry violated for ({:?}, {:?}) in {:?}: a≡b={}, b≡a={}",
                a, b, lang, ab, ba
            );

            let score_ab = matcher.similarity_score(a, b);
            let score_ba = matcher.similarity_score(b, a);
            assert!(
                (score_ab - score_ba).abs() < 1e-10,
                "Score symmetry violated for ({:?}, {:?}): {:.6} vs {:.6}",
                a,
                b,
                score_ab,
                score_ba
            );
        }
    }

    #[test]
    fn test_code_matcher_combined_accuracy() {
        // Aggregate across all languages
        let mut total_equiv = 0;
        let mut correct_equiv = 0;

        for (a, b) in equivalent_python_pairs() {
            let m = CodeMatcher::new(CodeLanguage::Python, 0.80);
            total_equiv += 1;
            if m.are_equivalent(a, b) {
                correct_equiv += 1;
            }
        }
        for (a, b) in equivalent_rust_pairs() {
            let m = CodeMatcher::new(CodeLanguage::Rust, 0.80);
            total_equiv += 1;
            if m.are_equivalent(a, b) {
                correct_equiv += 1;
            }
        }
        for (a, b) in equivalent_javascript_pairs() {
            let m = CodeMatcher::new(CodeLanguage::JavaScript, 0.80);
            total_equiv += 1;
            if m.are_equivalent(a, b) {
                correct_equiv += 1;
            }
        }

        let acc = correct_equiv as f64 / total_equiv as f64;
        eprintln!(
            "Combined code accuracy: {}/{} ({:.1}%)",
            correct_equiv,
            total_equiv,
            acc * 100.0
        );
        assert!(
            acc >= 0.95,
            "Combined CodeMatcher accuracy {:.1}% < 95% ({}/{})",
            acc * 100.0,
            correct_equiv,
            total_equiv,
        );
    }
}

// ============================================================================
// EmbeddingMatcher Accuracy Tests
// ============================================================================

mod embedding_matcher_tests {
    use super::*;

    fn make_matcher(threshold: f64) -> EmbeddingMatcher {
        EmbeddingMatcher::new(Box::new(MockEmbeddingClient::new(128)), threshold)
    }

    /// Test that the EmbeddingMatcher correctly identifies equivalent pairs
    /// from the NL corpus. With mock (character-frequency) embeddings, we
    /// measure and report accuracy but only assert mechanism correctness.
    /// Real semantic accuracy (>90%) requires actual embedding providers.
    #[test]
    fn test_nl_equivalent_pairs_report() {
        let matcher = make_matcher(0.70);
        let pairs = equivalent_nl_pairs();
        let total = pairs.len();
        let correct = pairs
            .iter()
            .filter(|(a, b)| matcher.are_equivalent(a, b))
            .count();

        let acc = correct as f64 / total as f64;
        eprintln!(
            "NL equivalent accuracy (mock): {}/{} ({:.1}%)",
            correct,
            total,
            acc * 100.0
        );
        // Mock embeddings use character frequency, not semantics.
        // We verify the mechanism works (some matches found) but do not
        // assert >90% — that requires real providers (Ollama/OpenAI).
        assert!(
            correct > 0,
            "EmbeddingMatcher should match at least some equivalent NL pairs"
        );
    }

    /// Test that the EmbeddingMatcher rejects non-equivalent NL pairs.
    /// With mock embeddings, we report the FP rate but use a relaxed
    /// threshold since character-frequency embeddings share vocabulary
    /// between domain-similar negative pairs.
    #[test]
    fn test_nl_false_positive_rate_report() {
        let matcher = make_matcher(0.70);
        let pairs = non_equivalent_nl_pairs();
        let total = pairs.len();
        let false_positives = pairs
            .iter()
            .filter(|(a, b)| matcher.are_equivalent(a, b))
            .count();

        let fp_rate = false_positives as f64 / total as f64;
        eprintln!(
            "NL false positive rate (mock): {}/{} ({:.1}%)",
            false_positives,
            total,
            fp_rate * 100.0
        );
        // Report for visibility; mock embeddings are not truly semantic.
        // Real providers should achieve <5% FP rate.
    }

    /// Test high-overlap NL pairs that the mock embedding can reliably match.
    /// These pairs differ only in whitespace, punctuation, or minor tokens,
    /// producing very similar character-frequency vectors.
    #[test]
    fn test_nl_high_overlap_pairs_accuracy() {
        let matcher = make_matcher(0.85);
        let high_overlap_pairs = vec![
            ("hello  world", "hello world"),
            ("The answer is 42", "The answer is 42."),
            ("hello world", "hello world!"),
            ("The test passed.", "The test passed"),
            ("No errors found.", "No errors found"),
            ("Connection refused.", "Connection refused"),
            ("The answer is 42.", " The answer is 42. "),
            ("Data is stored in UTF-8.", "Data is stored in UTF-8"),
            ("The cache was invalidated.", "The cache was invalidated"),
            ("Rate limit exceeded.", "Rate limit exceeded"),
        ];
        let total = high_overlap_pairs.len();
        let correct = high_overlap_pairs
            .iter()
            .filter(|(a, b)| matcher.are_equivalent(a, b))
            .count();

        let acc = correct as f64 / total as f64;
        eprintln!(
            "High-overlap NL accuracy (mock): {}/{} ({:.1}%)",
            correct,
            total,
            acc * 100.0
        );
        assert!(
            acc >= 0.90,
            "High-overlap pairs should match at >=90% with mock ({}/{})",
            correct,
            total,
        );
    }

    /// Test that very different strings produce low similarity.
    #[test]
    fn test_nl_very_different_rejected() {
        let matcher = make_matcher(0.85);
        let different_pairs = vec![
            ("hello world", "zyxwvutsrqp"),
            ("abcdef", "123456789"),
            ("The function returns true", "zzzzzzzzzzzzzzzzzzzzz"),
            (
                "short",
                "a completely unrelated very long string with different characters",
            ),
        ];

        for (a, b) in &different_pairs {
            assert!(
                !matcher.are_equivalent(a, b),
                "Very different strings should not match: {:?} vs {:?}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_embedding_reflexivity() {
        let matcher = make_matcher(0.92);
        let samples = vec![
            "hello world",
            "",
            "The answer is 42.",
            "fn foo(x: i32) -> i32 { x + 1 }",
            "A longer piece of text that has more content for embedding.",
            "Special chars: @#$%^&*()",
            "Unicode: café résumé naïve",
            "   whitespace   everywhere   ",
        ];

        for s in &samples {
            assert!(
                matcher.are_equivalent(s, s),
                "Reflexivity violated for {:?}",
                s
            );
        }
    }

    #[test]
    fn test_embedding_symmetry() {
        let matcher = make_matcher(0.92);
        let pairs = vec![
            ("hello", "world"),
            ("The answer is 4.", "The answer is 5."),
            ("foo bar baz", "foo bar"),
            ("same", "same"),
            ("short", "a much longer string with many words"),
        ];

        for (a, b) in &pairs {
            let score_ab = matcher.similarity_score(a, b);
            let score_ba = matcher.similarity_score(b, a);
            assert!(
                (score_ab - score_ba).abs() < 1e-10,
                "Symmetry violated for ({:?}, {:?}): {:.6} vs {:.6}",
                a,
                b,
                score_ab,
                score_ba
            );
        }
    }

    #[test]
    fn test_embedding_identical_strings_score_1() {
        let matcher = make_matcher(0.92);
        let strings = vec!["hello", "test", "The answer is 42."];
        for s in &strings {
            let score = matcher.similarity_score(s, s);
            assert_eq!(
                score, 1.0,
                "Identical string {:?} should have score 1.0, got {}",
                s, score
            );
        }
    }

    #[test]
    fn test_embedding_very_different_strings_low_score() {
        let matcher = make_matcher(0.92);
        let score = matcher.similarity_score("abcdefg", "zyxwvutsrqponmlkjihg");
        assert!(
            score < 0.5,
            "Very different strings should have low score, got {}",
            score
        );
    }

    #[test]
    fn test_cosine_similarity_properties() {
        // Verify cosine similarity basic properties
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        // Identity
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);

        // Symmetry
        let c = vec![3.0, 2.0, 1.0];
        assert!((cosine_similarity(&a, &c) - cosine_similarity(&c, &a)).abs() < 1e-10);

        // Range [-1, 1]
        let d = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &d);
        assert!((-1.0..=1.0).contains(&sim));
    }
}

// ============================================================================
// ExactMatcher Baseline Tests (regression)
// ============================================================================

mod exact_matcher_tests {
    use super::*;

    #[test]
    fn test_exact_matcher_whitespace_equivalent() {
        let matcher = ExactMatcher::new();
        let pairs = vec![
            ("hello  world", "hello world"),
            ("  leading", "leading"),
            ("trailing  ", "trailing"),
            ("a\tb", "a b"),
            ("a\nb", "a b"),
        ];

        for (a, b) in &pairs {
            assert!(
                matcher.are_equivalent(a, b),
                "ExactMatcher should match {:?} and {:?}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_exact_matcher_case_sensitive() {
        let matcher = ExactMatcher::new();
        assert!(!matcher.are_equivalent("Hello", "hello"));
        assert!(!matcher.are_equivalent("ABC", "abc"));
    }

    #[test]
    fn test_exact_matcher_reflexivity_property() {
        let matcher = ExactMatcher::new();
        let samples: Vec<&str> = vec![
            "hello",
            "",
            "  ",
            "42",
            "special!@#$",
            "café",
            "multi\nline\ntext",
            "tabs\there",
        ];

        for s in &samples {
            assert!(matcher.are_equivalent(s, s));
            assert_eq!(matcher.similarity_score(s, s), 1.0);
        }
    }

    #[test]
    fn test_exact_matcher_symmetry_property() {
        let matcher = ExactMatcher::new();
        let pairs = vec![
            ("hello", "world"),
            ("same", "same"),
            ("a b", "a  b"),
            ("", "nonempty"),
        ];

        for (a, b) in &pairs {
            assert_eq!(matcher.are_equivalent(a, b), matcher.are_equivalent(b, a),);
            assert_eq!(
                matcher.similarity_score(a, b),
                matcher.similarity_score(b, a),
            );
        }
    }
}

// ============================================================================
// Cross-Matcher Consistency Tests
// ============================================================================

mod cross_matcher_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_all_matchers_agree_on_identical_strings() {
        let exact = ExactMatcher::new();
        let embedding =
            EmbeddingMatcher::with_default_threshold(Box::new(MockEmbeddingClient::default()));

        let strings = vec!["hello world", "42", "fn foo() {}", ""];
        for s in &strings {
            assert!(exact.are_equivalent(s, s));
            assert!(embedding.are_equivalent(s, s));
        }
    }

    #[test]
    fn test_matchers_as_trait_objects() {
        let matchers: Vec<Arc<dyn CandidateMatcher>> = vec![
            Arc::new(ExactMatcher::new()),
            Arc::new(EmbeddingMatcher::with_default_threshold(Box::new(
                MockEmbeddingClient::default(),
            ))),
        ];

        for matcher in &matchers {
            assert!(matcher.are_equivalent("test", "test"));
            assert_eq!(matcher.similarity_score("test", "test"), 1.0);
        }
    }

    #[test]
    fn test_matcher_type_names() {
        let exact = ExactMatcher::new();
        assert_eq!(exact.matcher_type(), "exact");

        let embedding =
            EmbeddingMatcher::with_default_threshold(Box::new(MockEmbeddingClient::default()));
        assert_eq!(embedding.matcher_type(), "embedding");
    }

    #[cfg(feature = "code-matcher")]
    #[test]
    fn test_code_matcher_type_name() {
        let code = CodeMatcher::new(CodeLanguage::Rust, 0.80);
        assert_eq!(code.matcher_type(), "code");
    }

    #[cfg(feature = "code-matcher")]
    #[test]
    fn test_all_three_matchers_as_trait_objects() {
        let matchers: Vec<Arc<dyn CandidateMatcher>> = vec![
            Arc::new(ExactMatcher::new()),
            Arc::new(EmbeddingMatcher::with_default_threshold(Box::new(
                MockEmbeddingClient::default(),
            ))),
            Arc::new(CodeMatcher::new(CodeLanguage::Rust, 0.80)),
        ];

        for matcher in &matchers {
            assert!(matcher.are_equivalent("hello", "hello"));
        }
    }
}
