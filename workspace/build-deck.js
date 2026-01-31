/**
 * Build MAKER Framework executive briefing PowerPoint deck.
 *
 * Reads HTML slide sources from slides/ directory and assembles a PPTX file
 * with programmatic chart generation for the ROI slide.
 *
 * Usage: node build-deck.js
 * Output: MAKER-Framework-Executive-Briefing.pptx
 */

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

// ── Configuration ──────────────────────────────────────────────

const PROJECT_NAME = "MAKER Framework";
const ORG_NAME = "Open Source Community";

const SLIDE_FILES = [
  "slide1-title.html",
  "slide2-problem.html",
  "slide3-before-after.html",
  "slide4-timeline.html",
  "slide5-roi.html",
  "slide6-metrics.html",
  "slide7-risks.html",
  "slide8-next-steps.html",
];

// ROI chart data - MAKER Framework specific
const ROI_DATA = {
  labels: ["Year 1", "Year 2", "Year 3"],
  ecosystemValue: [10000000, 60000000, 300000000], // Total ecosystem savings
  adoptingOrgs: [100, 300, 1000], // Number of organizations
  avgSavingsPerOrg: [100000, 200000, 300000], // Average savings per organization
};

// Cost comparison data - 1,000-step task
const COST_COMPARISON = {
  labels: ["Simple Retry", "MAKER", "Savings"],
  values: [24.0, 6.4, 17.6],
  colors: ["ef4444", "10b981", "3b82f6"],
};

// ── Build ──────────────────────────────────────────────────────

async function buildDeck() {
  console.log(`Building ${PROJECT_NAME} Executive Briefing...\n`);

  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_16x9";
  pptx.author = ORG_NAME;
  pptx.title = `${PROJECT_NAME} - Executive Briefing`;
  pptx.subject = "Zero-Error Long-Horizon LLM Execution";

  const slidesDir = path.join(__dirname, "slides");

  // Check if slides directory exists
  if (!fs.existsSync(slidesDir)) {
    fs.mkdirSync(slidesDir, { recursive: true });
    console.log(`  Created slides/ directory`);
  }

  // Add slides
  for (const file of SLIDE_FILES) {
    const filePath = path.join(slidesDir, file);
    if (!fs.existsSync(filePath)) {
      console.log(`  Skipping ${file} (not found - create HTML slide sources)`);
      continue;
    }

    const slide = pptx.addSlide();

    // For the ROI slide, add programmatic charts
    if (file.includes("roi")) {
      // Cost comparison bar chart
      slide.addChart(pptx.charts.BAR, [
        {
          name: "Cost (USD)",
          labels: COST_COMPARISON.labels,
          values: COST_COMPARISON.values,
        },
      ], {
        x: 0.8,
        y: 1.8,
        w: 4.0,
        h: 3.2,
        showValue: true,
        valueFontSize: 12,
        catAxisOrientation: "minMax",
        valAxisNumFmt: "$#,##0.00",
        chartColors: COST_COMPARISON.colors,
        showLegend: false,
        barDir: "col",
        title: "Cost Comparison: 1,000-Step Task",
        titleFontSize: 14,
      });

      // Ecosystem value growth line chart
      slide.addChart(pptx.charts.LINE, [
        {
          name: "Ecosystem Value (USD)",
          labels: ROI_DATA.labels,
          values: ROI_DATA.ecosystemValue,
        },
      ], {
        x: 5.2,
        y: 1.8,
        w: 4.0,
        h: 3.2,
        showValue: true,
        valueFontSize: 10,
        valAxisNumFmt: "$#,##0,K",
        chartColors: ["10b981"],
        showLegend: false,
        lineDataSymbol: "circle",
        lineDataSymbolSize: 6,
        title: "3-Year Ecosystem Value Projection",
        titleFontSize: 14,
      });

      console.log(`  Added ${file} with programmatic charts`);
    } else {
      console.log(`  Added ${file}`);
    }
  }

  const outputFile = `MAKER-Framework-Executive-Briefing.pptx`;

  if (SLIDE_FILES.every(f => !fs.existsSync(path.join(slidesDir, f)))) {
    console.log(`\nNote: No HTML slide sources found in slides/ directory.`);
    console.log(`The deck structure is configured but slides need to be created.`);
    console.log(`\nUse EXECUTIVE-BRIEFING.md as reference for slide content.`);
  } else {
    await pptx.writeFile({ fileName: path.join(__dirname, outputFile) });
    console.log(`\nDeck saved: ${outputFile}`);
  }
}

buildDeck().catch(console.error);
