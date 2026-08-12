const playwrightModule = process.env.PLAYWRIGHT_MODULE || "playwright";
const { chromium } = await import(playwrightModule);

const baseUrl = process.env.ATLAS_BASE_URL || "http://127.0.0.1:8765";
const executablePath = process.env.BROWSER_EXECUTABLE;
const browser = await chromium.launch({
  headless: true,
  ...(executablePath ? { executablePath } : {}),
});
const page = await browser.newPage({
  viewport: { width: 1600, height: 1200 },
  deviceScaleFactor: 1,
});
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("response", (response) => {
  if (response.status() >= 400 && !response.url().endsWith("/favicon.ico")) {
    errors.push(`${response.status()} ${response.url()}`);
  }
});

const dataQuery = "../../concept/09_in_context_execution_calibration_design_inspector.data.json";
await page.goto(
  `${baseUrl}/auxiliary/atlas_app/architecture_atlas.html?data=${dataQuery}`,
  { waitUntil: "networkidle" },
);

const designButtons = page.locator('rect[role="button"]');
if ((await designButtons.count()) !== 8) {
  throw new Error(`candidate Inspector must render 8 design buttons, got ${await designButtons.count()}`);
}

const titles = [
  "Perturbation Condition",
  "Frozen Planner",
  "Frozen Tracker",
  "Support Rollout",
  "Context Encoder",
  "Latent Calibration",
  "Calibration Learning",
  "Frozen Query Execution",
];
for (const title of titles) {
  const button = page.getByRole("button", { name: `查看 ${title} 的 Inspector 卡片` });
  if ((await button.count()) !== 1) throw new Error(`missing or ambiguous card button: ${title}`);
  await button.click();
  const visibleText = (await page.locator("svg").textContent()).replace(/\s+/g, "");
  if (!visibleText.includes(title.replace(/\s+/g, ""))) {
    throw new Error(`selected card title is not visible: ${title}`);
  }
}

const allText = (await page.locator("svg").textContent()).replace(/\s+/g, "");
for (const required of [
  "采样一个隐藏执行条件ξ",
  "执行并记录第一次SupportRollout",
  "整段编码一次并产生Δzξ",
  "固定Δzξ执行QueryRollout",
  "只更新ContextEncoder",
]) {
  if (!allText.includes(required)) throw new Error(`shared spine is missing: ${required}`);
}

await page.screenshot({ path: "/tmp/in_context_calibration_design_inspector_desktop.png", fullPage: true });
await page.setViewportSize({ width: 390, height: 844 });
await page.reload({ waitUntil: "networkidle" });
if ((await page.locator('rect[role="button"]').count()) !== 8) {
  throw new Error("mobile candidate Inspector lost design buttons");
}
await page.screenshot({ path: "/tmp/in_context_calibration_design_inspector_mobile.png", fullPage: true });

await browser.close();
if (errors.length) throw new Error(`browser errors: ${errors.join(" | ")}`);
console.log(
  "candidate calibration Design Inspector visual: PASS cards=8 screenshots=/tmp/in_context_calibration_design_inspector_{desktop,mobile}.png",
);
