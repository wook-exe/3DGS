import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const reportsDir = process.argv[2] ?? "security-reports";
const summaryPath = join(reportsDir, "dependency-security-summary.md");
const severityOrder = ["info", "low", "moderate", "high", "critical"];
const blockingSeverities = new Set(["moderate", "high", "critical"]);

function readJson(fileName) {
  const filePath = join(reportsDir, fileName);
  if (!existsSync(filePath)) {
    return null;
  }

  return JSON.parse(readFileSync(filePath, "utf8").replace(/^\uFEFF/, ""));
}

function countBlockingFromNpmAudit(report) {
  const metadata = report?.metadata?.vulnerabilities;
  if (!metadata) {
    return 0;
  }

  return [...blockingSeverities].reduce((total, severity) => total + (metadata[severity] ?? 0), 0);
}

function countBlockingFromSnyk(report) {
  const vulnerabilities = report?.vulnerabilities;
  if (!Array.isArray(vulnerabilities)) {
    return 0;
  }

  return vulnerabilities.filter((item) => blockingSeverities.has(item.severity)).length;
}

function npmAuditLines(report) {
  const metadata = report?.metadata?.vulnerabilities;
  if (!metadata) {
    return ["npm audit report was not available."];
  }

  return [
    "| Severity | Count |",
    "| --- | ---: |",
    ...severityOrder.map((severity) => `| ${severity} | ${metadata[severity] ?? 0} |`)
  ];
}

function snykLines(report) {
  if (report?.skipped) {
    return ["Snyk scan skipped because `SNYK_TOKEN` is not configured."];
  }

  const vulnerabilities = report?.vulnerabilities;
  if (!Array.isArray(vulnerabilities)) {
    return ["Snyk report was not available."];
  }

  if (vulnerabilities.length === 0) {
    return ["No Snyk vulnerabilities reported."];
  }

  return [
    "| Package | Severity | Title |",
    "| --- | --- | --- |",
    ...vulnerabilities
      .slice(0, 20)
      .map((item) => `| ${item.packageName ?? item.name ?? "unknown"} | ${item.severity} | ${item.title ?? item.id ?? "n/a"} |`)
  ];
}

const npmAudit = readJson("npm-audit.json");
const snyk = readJson("snyk.json");
const blockingCount = countBlockingFromNpmAudit(npmAudit) + countBlockingFromSnyk(snyk);
const generatedAt = new Date().toISOString();

const summary = [
  "# Dependency Security Report",
  "",
  `Generated at: ${generatedAt}`,
  "",
  `Blocking vulnerability count (moderate or higher): ${blockingCount}`,
  "",
  "## npm audit",
  "",
  ...npmAuditLines(npmAudit),
  "",
  "## Snyk",
  "",
  ...snykLines(snyk),
  ""
].join("\n");

writeFileSync(summaryPath, summary, "utf8");

if (process.env.GITHUB_STEP_SUMMARY) {
  appendFileSync(process.env.GITHUB_STEP_SUMMARY, summary);
}

if (process.env.GITHUB_OUTPUT) {
  appendFileSync(process.env.GITHUB_OUTPUT, `blocking_count=${blockingCount}\n`);
  appendFileSync(process.env.GITHUB_OUTPUT, `summary_path=${summaryPath}\n`);
}
