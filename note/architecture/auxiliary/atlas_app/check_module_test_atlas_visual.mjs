const playwrightModule = process.env.PLAYWRIGHT_MODULE || "playwright";
const { chromium } = await import(playwrightModule);

const baseUrl = process.env.ATLAS_BASE_URL || "http://127.0.0.1:8766";
const executablePath = process.env.BROWSER_EXECUTABLE;
const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("response", (response) => {
 if (response.status() >= 400 && !response.url().endsWith("/favicon.ico")) {
  errors.push(`${response.status()} ${response.url()}`);
 }
});

await page.goto(`${baseUrl}/05_frontres_module_test_atlas.html`, { waitUntil: "networkidle" });
const frame = page.frameLocator("iframe");
const moduleButtons = frame.locator('rect[role="button"][data-inspector-kind="module-test"]');
if (await moduleButtons.count() !== 22) throw new Error("Test Atlas must render 22 module buttons");

const formalStageButton = frame.locator('rect[role="button"][data-inspector-kind="stage-reading"]');
if (await formalStageButton.count() !== 1) throw new Error("Test Atlas must render one Formal Runtime Audit stage card");
await formalStageButton.click();
const formalStageText = (await frame.locator("svg").textContent()).replace(/\s+/g, "");
for (const required of ["PhaseA：Method-CodeAlignment", "PhaseB：FormalRuntimeAudit", "模块自身算错，退回ModuleTest", "不证明策略已经有效"]) {
 if (!formalStageText.includes(required)) throw new Error(`Formal Runtime Audit card missing visible text: ${required}`);
}
await page.screenshot({ path: "/tmp/frontres_formal_runtime_stage_desktop.png", fullPage: true });

for (let index = 0; index < 22; index += 1) {
 await moduleButtons.nth(index).click();
 const selectedText = (await frame.locator("svg").textContent()).replace(/\s+/g, "");
 for (const required of ["要验证的设计规则", "伪样本测试", "正确结果", "证明什么"]) {
  if (!selectedText.includes(required)) throw new Error(`test card ${index + 1} missing visible pseudo-sample field: ${required}`);
 }
}

const gainButton = frame.getByRole("button", { name: "查看 Repair Gain 的 Inspector 卡片" });
if (await gainButton.count() !== 1) throw new Error("Repair Gain button is missing or ambiguous");
await gainButton.click();
const frameText = await frame.locator("svg").textContent();
const compactFrameText = frameText.replace(/\s+/g, "");
for (const required of [
"Repair更接近Clean时Gain提高",
"缺失证据不能补零",
"伪样本测试",
]) {
if (!compactFrameText.includes(required)) throw new Error(`selected Repair Gain card missing visible text: ${required}`);
}
await page.screenshot({ path: "/tmp/frontres_module_test_atlas_desktop.png", fullPage: true });

await page.setViewportSize({ width: 390, height: 844 });
await page.reload({ waitUntil: "networkidle" });
const mobileFrame = page.frameLocator("iframe");
if (await mobileFrame.locator('rect[role="button"][data-inspector-kind="module-test"]').count() !== 22
 || await mobileFrame.locator('rect[role="button"][data-inspector-kind="stage-reading"]').count() !== 1) {
 throw new Error("mobile Test Atlas lost module buttons");
}
await page.screenshot({ path: "/tmp/frontres_module_test_atlas_mobile.png", fullPage: true });

await page.setViewportSize({ width: 1440, height: 1000 });
await page.goto(`${baseUrl}/02_frontres_design_inspector.html`, { waitUntil: "networkidle" });
if ((await page.title()) !== "02 FrontRES Design Inspector") {
 throw new Error(`Design Inspector title drift: ${await page.title()}`);
}
if (await page.frameLocator("iframe").locator('rect[role="button"]').count() !== 10) {
 throw new Error("Design Inspector must preserve ten design-point buttons");
}
await page.screenshot({ path: "/tmp/frontres_design_inspector_desktop.png", fullPage: true });

await page.goto(`${baseUrl}/`, { waitUntil: "networkidle" });
const atlasOrder = (await page.locator("section.grid strong").allTextContents()).slice(0, 5);
const expectedAtlasOrder = [
 "01 FrontRES Method Figure",
 "02 FrontRES Design Inspector",
 "03 FEMR Module Inspector",
 "04 Code Quality Evidence Atlas",
 "05 FrontRES Module Test Atlas",
];
if (atlasOrder.join("|") !== expectedAtlasOrder.join("|")) {
 throw new Error(`top-level Atlas order drift: ${atlasOrder.join(" | ")}`);
}
for (const [path, expectedTitle] of [
 ["/01_frontres_method_figure.html", "01 FrontRES Method Figure"],
 ["/02_frontres_design_inspector.html", "02 FrontRES Design Inspector"],
 ["/03_femr_module_inspector.html", "03 FEMR Module Inspector"],
 ["/04_code_quality_evidence_atlas.html", "04 FEMR Code Quality Evidence Atlas"],
 ["/05_frontres_module_test_atlas.html", "05 FrontRES Module Test Atlas"],
]) {
 await page.goto(`${baseUrl}${path}`, { waitUntil: "networkidle" });
 if ((await page.title()) !== expectedTitle) throw new Error(`${path} title drift: ${await page.title()}`);
}

await browser.close();
if (errors.length) throw new Error(`browser errors: ${errors.join(" | ")}`);
console.log("visual inspector OK test_cards=22 formal_stage_cards=1 design_cards=10 desktop+mobile screenshots=/tmp/frontres_*inspector*.png");
