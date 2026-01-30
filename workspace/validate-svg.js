/**
 * Validate SVG files for well-formed XML and structural correctness.
 *
 * Parses each SVG through an XML parser, reports errors, and checks for
 * required SVG structural elements. Exit code 1 on any failure.
 *
 * Usage:
 *   node validate-svg.js                          # validate all .github/*.svg
 *   node validate-svg.js path/to/file.svg [...]   # validate specific files
 */

const { DOMParser } = require("@xmldom/xmldom");
const fs = require("fs");
const path = require("path");
const glob = require("glob");

const GITHUB_DIR = path.resolve(__dirname, "..", ".github");

function collectFiles(args) {
  if (args.length > 0) {
    return args;
  }
  const pattern = path.join(GITHUB_DIR, "*.svg");
  const files = glob.sync(pattern);
  if (files.length === 0) {
    console.log("No SVG files found in .github/");
    process.exit(0);
  }
  return files;
}

function validateXml(content) {
  const errors = [];
  const parser = new DOMParser({
    onError: (level, msg) => {
      if (level !== "warning") {
        errors.push(`[${level}] ${msg}`);
      }
    },
  });

  let doc = null;
  try {
    doc = parser.parseFromString(content, "image/svg+xml");
  } catch (err) {
    errors.push(`[fatal] ${err.message || String(err)}`);
  }
  return { doc, errors };
}

function validateSvgStructure(doc) {
  const issues = [];

  const root = doc.documentElement;
  if (!root || root.tagName !== "svg") {
    issues.push("Missing <svg> root element");
    return issues;
  }

  const xmlns = root.getAttribute("xmlns");
  if (!xmlns || !xmlns.includes("w3.org/2000/svg")) {
    issues.push(
      'Missing or incorrect xmlns attribute (expected "http://www.w3.org/2000/svg")',
    );
  }

  const hasViewBox = root.hasAttribute("viewBox");
  const hasDimensions =
    root.hasAttribute("width") && root.hasAttribute("height");
  if (!hasViewBox && !hasDimensions) {
    issues.push("Missing viewBox or width/height attributes on <svg>");
  }

  return issues;
}

function validateEntityEncoding(content) {
  const issues = [];

  const ampRegex = /&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[\da-fA-F]+);)/g;
  let match;
  while ((match = ampRegex.exec(content)) !== null) {
    const line = content.substring(0, match.index).split("\n").length;
    issues.push(`Unescaped '&' at line ${line}`);
  }

  if (/<[^>]*</.test(content)) {
    issues.push("Potential nested/unclosed tag detected (< inside tag)");
  }

  return issues;
}

function main() {
  const args = process.argv.slice(2);
  const files = collectFiles(args);

  let totalErrors = 0;
  const results = [];

  for (const filePath of files) {
    const relPath = path.relative(process.cwd(), filePath);
    const fileErrors = [];

    if (!fs.existsSync(filePath)) {
      console.error(`  MISSING: ${relPath}`);
      totalErrors++;
      continue;
    }

    const content = fs.readFileSync(filePath, "utf-8");

    // 1. Raw content checks (pre-parse)
    fileErrors.push(...validateEntityEncoding(content));

    // 2. XML parse
    const { doc, errors: parseErrors } = validateXml(content);
    fileErrors.push(...parseErrors);

    // 3. SVG structure (only if parse succeeded)
    if (parseErrors.length === 0 && doc) {
      fileErrors.push(...validateSvgStructure(doc));
    }

    const status = fileErrors.length > 0 ? "FAIL" : "PASS";
    totalErrors += fileErrors.length;
    results.push({ file: relPath, status, errors: fileErrors });
  }

  // Report
  console.log("\nSVG Validation Report");
  console.log("=".repeat(60));

  for (const r of results) {
    console.log(`\n  [${r.status}] ${r.file}`);
    for (const err of r.errors) {
      console.log(`         - ${err}`);
    }
  }

  console.log(`\n${"=".repeat(60)}`);
  const passed = results.filter((r) => r.status === "PASS").length;
  console.log(
    `  ${passed}/${results.length} files passed, ${totalErrors} error(s)\n`,
  );

  process.exit(totalErrors > 0 ? 1 : 0);
}

main();
