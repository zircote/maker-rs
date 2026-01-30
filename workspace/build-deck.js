/**
 * Build executive briefing PowerPoint deck from HTML slide sources.
 *
 * Reads HTML slides from slides/ directory and assembles a PPTX file
 * with programmatic chart generation for the ROI slide.
 *
 * Usage: node build-deck.js
 * Output: [Project]-Executive-Briefing.pptx
 *
 * Customize:
 * - PROJECT_NAME: Change to your project name
 * - ROI_DATA: Update with your financial projections
 * - SLIDE_FILES: Add or remove slides as needed
 */

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

// ── Configuration ──────────────────────────────────────────────

const PROJECT_NAME = "Project"; // Change to your project name
const ORG_NAME = "Organization"; // Change to your organization

const SLIDE_FILES = [
  "slide1-title.html",
  "slide2-problem.html",
  "slide3-before-after.html",
  "slide4-phases.html",
  "slide5-roi.html",
  "slide6-metrics.html",
  "slide7-risks.html",
  "slide8-next-steps.html",
];

// ROI chart data (customize with your projections)
const ROI_DATA = {
  labels: ["Year 1", "Year 2", "Year 3"],
  savings: [135000, 255000, 255000], // Net savings per year
  costs: [144000, 24000, 24000], // Project costs per year
};

// ── Build ──────────────────────────────────────────────────────

async function buildDeck() {
  console.log(`Building ${PROJECT_NAME} Executive Briefing...\n`);

  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_16x9";
  pptx.author = ORG_NAME;
  pptx.title = `${PROJECT_NAME} - Executive Briefing`;
  pptx.subject = "Executive Briefing Deck";

  const slidesDir = path.join(__dirname, "slides");

  // Add HTML-based slides
  for (const file of SLIDE_FILES) {
    const filePath = path.join(slidesDir, file);
    if (!fs.existsSync(filePath)) {
      console.log(`  Skipping ${file} (not found)`);
      continue;
    }

    const slide = pptx.addSlide();

    // Add slide background
    if (file.includes("title")) {
      slide.background = { path: path.join(slidesDir, "bg-title.png") };
    } else if (file.includes("next-steps")) {
      slide.background = { path: path.join(slidesDir, "bg-blue-accent.png") };
    } else {
      slide.background = { path: path.join(slidesDir, "bg-light.png") };
    }

    // For the ROI slide, add a programmatic bar chart
    if (file.includes("roi")) {
      slide.addChart(
        pptx.charts.BAR,
        [
          {
            name: "Net Savings",
            labels: ROI_DATA.labels,
            values: ROI_DATA.savings,
          },
          {
            name: "Project Cost",
            labels: ROI_DATA.labels,
            values: ROI_DATA.costs,
          },
        ],
        {
          x: 0.8,
          y: 1.8,
          w: 8.4,
          h: 3.8,
          showValue: true,
          valueFontSize: 10,
          catAxisOrientation: "minMax",
          valAxisNumFmt: "$#,##0",
          chartColors: ["10b981", "ef4444"],
          showLegend: true,
          legendPos: "b",
        },
      );
    }

    console.log(`  Added ${file}`);
  }

  const outputFile = `${PROJECT_NAME.replace(/\s+/g, "-")}-Executive-Briefing.pptx`;
  await pptx.writeFile({ fileName: path.join(__dirname, outputFile) });
  console.log(`\nDeck saved: ${outputFile}`);
}

buildDeck().catch(console.error);
