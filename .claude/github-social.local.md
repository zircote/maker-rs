---
# Image generation provider (default: svg)
# Options: svg, dalle-3, gemini, manual
provider: svg

# SVG-specific settings (only used when provider: svg)
svg_style: illustrated

# Dark mode support (default: false)
# false = light mode only, true = dark mode only, both = generate both variants
dark_mode: both

# Output settings
output_path: .github/social-preview.svg
dimensions: 1280x640
include_text: true
colors: auto

# README infographic settings
infographic_output: .github/readme-infographic.svg
infographic_style: hybrid       # architecture | features | hybrid

# Upload to repository (requires gh CLI or GITHUB_TOKEN)
upload_to_repo: false
---

# GitHub Social Plugin Configuration

This configuration was created by `/github-social:setup`.

## Provider Options

### SVG (Default - Recommended)
Claude generates clean, minimal SVG graphics directly. No API key required.
- **Pros**: Free, instant, editable, small file size (10-50KB)
- **Best for**: Professional, predictable results

### DALL-E 3
OpenAI's image generation. Requires `OPENAI_API_KEY`.
- **Pros**: Artistic, creative, varied styles
- **Cost**: ~$0.08 per image

### Gemini
Google's image generation via Gemini API. Requires `GEMINI_API_KEY`.
- **Models**: `gemini-2.5-flash-image` (fast), `gemini-3-pro-image-preview` (quality)
- **Cost**: ~$0.039 per image

### Manual
Outputs optimized prompts for Midjourney or other tools.

## SVG Style Options

**Minimal** (default): Clean, simple design with project name and subtle geometric accents. Maximum 3-5 shapes, generous whitespace.

**Geometric**: More complex arrangements with 8-15 geometric shapes representing domain metaphors abstractly.

**Illustrated**: Hand-drawn aesthetic using organic SVG paths with warm colors.

## Dark Mode

Set `dark_mode: both` to generate light and dark variants automatically:
- `.github/social-preview.svg` (light)
- `.github/social-preview-dark.svg` (dark)

## Command Overrides

Override any setting via command flags:
```bash
/social-preview --provider=dalle-3 --dark-mode
/readme-enhance --provider=gemini
```
