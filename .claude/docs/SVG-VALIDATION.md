# SVG Post-Write Validation

After writing any SVG file, **always** run the XML linter to catch malformed markup:

```bash
cd workspace && node validate-svg.js ../.github/<file>.svg
```

If validation fails:
1. Read the error report for line numbers and descriptions
2. Fix the identified issues (unescaped entities, unclosed tags, missing attributes)
3. Re-write the corrected SVG
4. Re-run validation until it passes

## Common Pitfalls

- **Unescaped `&`** in text content -- use `&amp;`
- **Unescaped `<` or `>`** in text -- use `&lt;` / `&gt;`
- **Missing `xmlns`** -- always include `xmlns="http://www.w3.org/2000/svg"`
- **Unclosed tags** -- every `<tag>` needs `</tag>` or self-close with `/>`
- **Attribute quoting** -- all attribute values must be quoted
