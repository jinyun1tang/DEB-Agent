import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

import { buildSlide02 } from "./templates/slide-02.mjs";
import { buildSlide05 } from "./templates/slide-05.mjs";
import { buildSlide13 } from "./templates/slide-13.mjs";
import { buildSlide17 } from "./templates/slide-17.mjs";
import { buildSlide26 } from "./templates/slide-26.mjs";

const TMP_DIR = "/Users/jinyuntang/work/github/DEB-Agent/outputs/crabtree-model-deck-build";
const FINAL_PPTX = "/Users/jinyuntang/work/github/DEB-Agent/outputs/crabtree-model-presentation.pptx";
const RENDER_DIR = path.join(TMP_DIR, "rendered");
const ASSET_DIR = path.join(TMP_DIR, "assets");

const BLACK = "#000000";
const PANEL = "#F2F2F2";
const RULE = "#B8BCC4";
const BLUE = "#3D8DFF";
const CYAN = "#6DCBF4";
const PALE = "#EDEDED";

function token(text, fontSize = 24, opts = {}) {
  return {
    runs: [
      {
        run: text,
        textStyle: {
          fontSize: `${fontSize}px`,
          typeface: "Helvetica Neue",
          color: opts.color ?? BLACK,
          bold: opts.bold ?? false,
        },
      },
    ],
    spaceAfter: opts.spaceAfter ?? 700,
    paragraphStyle: { lineSpacingPercent: opts.lineSpacingPercent ?? 98000 },
  };
}

function textBlock(title, body, titleSize = 28, bodySize = 21.33) {
  return {
    titleGoesHere: token(title, titleSize, { bold: false, spaceAfter: 650 }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: token(body, bodySize, {
      spaceAfter: 200,
      lineSpacingPercent: 110000,
    }),
  };
}

function textBlockHere(title, body, titleSize = 28, bodySize = 21.33) {
  return {
    titleHere: token(title, titleSize, { bold: false, spaceAfter: 650 }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: token(body, bodySize, {
      spaceAfter: 200,
      lineSpacingPercent: 110000,
    }),
  };
}

function tableIntro(topic, body) {
  return {
    topic: token(topic, 21.33, { spaceAfter: 500 }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: token(body, 21.33, {
      lineSpacingPercent: 108000,
    }),
  };
}

function footer(n) {
  return token(String(n), 13.33);
}

function notes(slide, sourceLines, presenterNote) {
  slide.speakerNotes.textFrame.setText(
    [
      "[Sources]",
      ...sourceLines.map((line) => `- ${line}`),
      "",
      "[Presenter note]",
      presenterNote,
    ].join("\n"),
  );
  slide.speakerNotes.setVisible(true);
}

async function imageBytes(name) {
  const p = path.join(ASSET_DIR, name);
  const bytes = await fs.readFile(p);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addText(slide, name, text, position, fontSize = 22, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize,
    color: opts.color ?? BLACK,
    bold: opts.bold ?? false,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "top",
  };
  return box;
}

function addBaseSlide(presentation, titleText, slideNumber) {
  const slide = presentation.slides.add();
  addText(slide, "slide-title", titleText, {
    left: 41.33,
    top: 36.12,
    width: 1197.33,
    height: 109.97,
  }, 38.67);
  addText(slide, "slide-number", String(slideNumber), {
    left: 1184.18,
    top: 659.24,
    width: 54.48,
    height: 25.33,
  }, 13.33, { alignment: "right", verticalAlignment: "bottom" });
  return slide;
}

function addCard(slide, name, title, body, position, fontSize = 24) {
  slide.shapes.add({
    geometry: "roundRect",
    name: `${name}-surface`,
    position,
    fill: PANEL,
    line: { style: "solid", fill: PANEL, width: 0 },
    borderRadius: 6,
  });
  addText(slide, `${name}-text`, `${title}\n${body}`, {
    left: position.left + 30,
    top: position.top + 28,
    width: position.width - 60,
    height: position.height - 56,
  }, fontSize);
}

function addCoveringPanel(slide, position, fill = "#FFFFFF") {
  return slide.shapes.add({
    geometry: "rect",
    name: "cover-existing-region",
    position,
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function addNode(slide, name, label, sublabel, position, fill = "#FFFFFF") {
  const node = slide.shapes.add({
    geometry: "roundRect",
    name,
    position,
    fill,
    line: { style: "solid", fill: RULE, width: 1.3 },
    borderRadius: 6,
  });
  node.text = `${label}\n${sublabel}`;
  node.text.style = {
    fontSize: 19,
    color: BLACK,
    alignment: "center",
    verticalAlignment: "middle",
  };
  return node;
}

function connect(slide, from, to, fromSide = "right", toSide = "left", color = RULE, width = 2) {
  return slide.shapes.connect(from, to, {
    kind: "elbow",
    fromSide,
    toSide,
    line: { style: "solid", fill: color, width },
    tail: { type: "arrow", width: "sm", length: "sm" },
  });
}

function addArchitectureDiagram(slide) {
  const g = addNode(slide, "glucose-node", "Glucose G", "primary substrate", {
    left: 680,
    top: 205,
    width: 170,
    height: 78,
  }, "#FFFFFF");
  const reserve = addNode(slide, "reserve-node", "C reserve", "x_C and j_E,C", {
    left: 895,
    top: 205,
    width: 170,
    height: 78,
  }, "#FFFFFF");
  const ferm = addNode(slide, "fermentation-node", "Fermentation", "ethanol + ATP", {
    left: 785,
    top: 330,
    width: 170,
    height: 78,
  }, "#F2F2F2");
  const resp = addNode(slide, "respiration-node", "Respiration", "ATP + O2 demand", {
    left: 1005,
    top: 330,
    width: 185,
    height: 78,
  }, "#F2F2F2");
  const etoh = addNode(slide, "ethanol-node", "Ethanol", "product, then substrate", {
    left: 785,
    top: 470,
    width: 170,
    height: 78,
  }, "#FFFFFF");
  const biomass = addNode(slide, "biomass-node", "Living V", "growth minus mortality", {
    left: 1005,
    top: 470,
    width: 185,
    height: 78,
  }, "#FFFFFF");

  connect(slide, g, reserve, "right", "left", RULE, 2);
  connect(slide, reserve, ferm, "bottom", "top", "#A65F4A", 2.4);
  connect(slide, reserve, resp, "right", "top", BLUE, 2.4);
  connect(slide, ferm, etoh, "bottom", "top", "#A65F4A", 2.4);
  connect(slide, resp, biomass, "bottom", "top", BLUE, 2.4);
  connect(slide, etoh, resp, "right", "left", "#A65F4A", 2);

  addText(slide, "diagram-caption", "Glucose drives overflow fermentation; ethanol can later feed respiration when glucose repression weakens.", {
    left: 682,
    top: 575,
    width: 510,
    height: 52,
  }, 18, { color: "#333333" });
}

function addMetricTable(slide) {
  const values = [
    ["Observation target", "Metric", "Expected batch signature", "Interpretation"],
    ["Glucose", "depletion time; uptake rate", "Monotone decline during first phase", "Constrains uptake and reserve formation"],
    ["Ethanol", "peak; production / use slopes", "Accumulation during glucose, decline after diauxic shift", "Tests overflow and ethanol-as-substrate dynamics"],
    ["Dissolved O2", "minimum; recovery time", "Demand can depress O2 but should not require an impossible negative pool", "Tests respiratory oxygen limitation"],
    ["Biomass", "living V and total V(1+x_C)", "Total biomass plateaus; viable structure may decline later", "Separates stationary OD from mortality"],
    ["Allocation", "theta_F timing and hysteresis", "High-G fermentation; relaxation after G depletion", "Tests dynamic splitter timescale"],
  ];
  const tbl = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 48,
    top: 244,
    width: 1184,
    height: 372,
    values,
    columnWidths: [236, 230, 390, 328],
  });
  tbl.styleOptions = { headerRow: true, bandedRows: true };
  tbl.borders.assign({ style: "solid", fill: RULE, width: 1 });
  tbl.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 4 }).assign({
    fill: PALE,
    textStyle: { fontSize: 16, bold: true, color: BLACK },
  });
  tbl.cells.block({ row: 1, column: 0, rowCount: values.length - 1, columnCount: 4 }).assign({
    textStyle: { fontSize: 16, color: BLACK },
  });
}

function addImageFrame(slide, position, name) {
  slide.shapes.add({
    geometry: "rect",
    name: `${name}-frame`,
    position,
    fill: "#FFFFFF",
    line: { style: "solid", fill: RULE, width: 1 },
  });
}

function addSoftmaxTransitionGraph(slide) {
  const xs = Array.from({ length: 13 }, (_, i) => i - 6);
  const thetaF = xs.map((x) => 1 / (1 + Math.exp(-x)));
  const thetaR = thetaF.map((y) => 1 - y);
  const categories = xs.map((x) => (x > 0 ? `+${x}` : String(x)));

  addText(slide, "softmax-graph-title", "Transition captured by softmax", {
    left: 680,
    top: 173,
    width: 535,
    height: 40,
  }, 28);
  addText(slide, "softmax-graph-caption", "When U_F - U_R crosses zero, allocation shifts smoothly from respiration to fermentation.", {
    left: 680,
    top: 214,
    width: 535,
    height: 62,
  }, 18, { color: "#333333" });

  slide.charts.add("line", {
    position: { left: 674, top: 295, width: 555, height: 305 },
    categories,
    series: [
      {
        name: "theta_F*",
        values: thetaF,
        line: { style: "solid", fill: "#A65F4A", width: 3 },
        marker: { symbol: "none" },
      },
      {
        name: "theta_R*",
        values: thetaR,
        line: { style: "solid", fill: BLUE, width: 3 },
        marker: { symbol: "none" },
      },
    ],
    lineOptions: { smooth: true },
    legend: {
      position: "bottom",
      overlay: false,
      textStyle: { fill: BLACK, fontSize: 12 },
    },
    xAxis: {
      title: { text: "score difference U_F - U_R", textStyle: { fill: BLACK, fontSize: 12 } },
      textStyle: { fill: "#4B5563", fontSize: 11 },
      line: { style: "solid", fill: RULE, width: 1 },
      majorGridlines: null,
    },
    yAxis: {
      min: 0,
      max: 1,
      majorUnit: 0.25,
      numberFormatCode: "0%",
      title: { text: "allocation weight", textStyle: { fill: BLACK, fontSize: 12 } },
      textStyle: { fill: "#4B5563", fontSize: 11 },
      line: { style: "solid", fill: RULE, width: 1 },
      majorGridlines: { style: "solid", fill: "#E1E4E8", width: 0.8 },
    },
    plotAreaFill: "#FFFFFF",
    plotAreaLine: { style: "solid", fill: "#FFFFFF", width: 0 },
    chartLine: { style: "solid", fill: "#FFFFFF", width: 0 },
  });

  addText(slide, "softmax-threshold-note", "At equal scores: theta_F* = theta_R* = 0.5", {
    left: 772,
    top: 604,
    width: 366,
    height: 28,
  }, 16, { color: "#333333", alignment: "center" });
}

async function buildDeck() {
  await fs.mkdir(RENDER_DIR, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  const s1 = buildSlide02(presentation, {
    title: token("Crabtree effect", 24),
    title2: token("Model sketch", 24),
    title3: token("A dynamic carbon-allocation model\nfor aerobic overflow fermentation", 68, {
      lineSpacingPercent: 90000,
    }),
  });
  notes(s1, [
    "Model source: references/crabtree-effect-model-sketch.html",
    "Executable source: references/crabtree-effect-mdeb-model.ipynb",
  ], "Open by framing the deck as a testable model sketch, not a fitted yeast model.");

  const s2 = buildSlide13(presentation, {
    footer1: footer(2),
    title: token("Crabtree effect: ethanol can appear even when oxygen is available", 38.67),
    body1: textBlock("Observed phenotype", "High glucose pushes Crabtree-positive yeast toward aerobic fermentation, so ethanol accumulates even before oxygen is exhausted."),
    body2: textBlock("Core tradeoff", "Respiration is high yield; fermentation is lower yield but can be the shorter, faster pathway under high glycolytic supply."),
    body3: textBlock("Batch sequence", "Glucose drawdown, ethanol overflow, diauxic transition, ethanol use, stationary phase, then loss of viability."),
    body4: textBlock("Modeling task", "Represent both the fast overflow decision and the slower regulatory return to respiratory metabolism."),
  });
  notes(s2, [
    "Tang et al. 2021, 2021-8 copy.pdf, section 3.5, describes Warburg/Crabtree fermentation despite oxygen availability.",
    "Model source: crabtree-effect-model-sketch.html, regime descriptions and ethanol-as-second-substrate update.",
  ], "Keep the biology simple: the deck asks what mechanism can make fermentation rational under oxygen availability.");

  const s3 = buildSlide13(presentation, {
    footer1: footer(3),
    title: token("Modeling hypothesis: overflow is a regulated pathway allocation problem", 38.67),
    body1: textBlock("Reserve memory", "mDEB reserve density x_C separates external glucose history from immediate catabolic supply."),
    body2: textBlock("Pathway economics", "Ohm's-law returns compare fermentation and respiration as yield divided by effective resistance."),
    body3: textBlock("Softmax splitter", "Pathway scores U_F and U_R become allocation weights theta_F and theta_R, with finite regulatory lag."),
    body4: textBlock("Second substrate", "Ethanol is both product and substrate: glucose represses ethanol uptake until the diauxic phase."),
  });
  notes(s3, [
    "Tang and Riley 2025, bg-22-1809-2025.pdf, develops mDEB reserve and structural biomass equations.",
    "Tang et al. 2021, 2021-8 copy.pdf, motivates Ohm's-law pathway comparisons.",
    "Model source: crabtree-effect-model-sketch.html, dynamic splitter and ethanol substrate equations.",
  ], "This is the central claim: do not present the splitter as a fitted black box; present it as an allocation hypothesis.");

  const s4 = addBaseSlide(presentation, "Evaluate the model against batch-culture signatures", 4);
  addText(s4, "metrics-intro", "Evaluation metrics\nThe model should be judged by qualitative sequence and quantitative event timing: glucose exhaustion, ethanol peak, oxygen drawdown and recovery, biomass plateau or viability loss, and allocation hysteresis.", {
    left: 42.09,
    top: 111.02,
    width: 1197.33,
    height: 106.27,
  }, 21.33);
  addMetricTable(s4);
  notes(s4, [
    "Notebook source: crabtree-effect-mdeb-model.ipynb, default metrics and structural-versus-total biomass distinction.",
    "Model source: crabtree-effect-model-sketch.html, ethanol-as-second-substrate equations.",
  ], "Use this slide as the scorecard before showing equations or plots.");

  const s5 = buildSlide05(presentation, {
    footer1: footer(5),
    title: token("State variables separate reserve, allocation, and biomass", 38.67),
    body1: textBlockHere("State variables", "G: glucose substrate\nEtOH: ethanol product and substrate\nO2: dissolved oxygen\nx_C: internal carbon reserve\nV: living structural biomass\ntheta_F: fermentation allocation", 28, 21.33),
    body2: textBlockHere("", "", 28, 21.33),
  });
  addCoveringPanel(s5, { left: 656.86, top: 190, width: 581.33, height: 440 }, "#FFFFFF");
  addArchitectureDiagram(s5);
  notes(s5, [
    "Model source: crabtree-effect-model-sketch.html, state and flux diagram updated with glucose and ethanol substrates.",
    "Notebook source: crabtree-effect-mdeb-model.ipynb, living structural biomass V and total biomass proxy V(1+x_C).",
  ], "Emphasize that total biomass and living structural biomass are intentionally separate readouts.");

  const s6 = buildSlide05(presentation, {
    footer1: footer(6),
    title: token("Glucose feeds reserve; ethanol feeds oxygen-dependent respiration", 38.67),
    body1: textBlockHere("Glucose reserve input", "j_A,G = j_G,max G/(K_G + G)\n\n dx_C/dt = j_A,G - j_E,C - mu x_C\n\n j_E,C = v_E,C x_C/(K_E,C + x_C)", 28, 21.33),
    body2: textBlockHere("Ethanol substrate gate", "I_G(G) = 1/(1 + (G/K_G,E)^n)\n\n j_A,EtOH = j_EtOH,max EtOH/(K_EtOH+EtOH) * O2/(K_O,EtOH+O2) * I_G\n\nEthanol uptake is weak while glucose is high.", 28, 19.5),
  });
  notes(s6, [
    "Model source: crabtree-effect-model-sketch.html, external glucose, ethanol, and oxygen equations.",
    "Tang and Riley 2025, bg-22-1809-2025.pdf, reserve dynamics basis.",
  ], "This slide introduces the diauxic mechanism in the updated HTML: glucose repression is not symmetric competition.");

  const s7 = addBaseSlide(presentation, "Softmax weights make the pathway switch continuous", 7);
  addText(s7, "pathway-scores-title", "Pathway scores", {
    left: 52,
    top: 173,
    width: 560,
    height: 40,
  }, 28);
  addText(s7, "pathway-scores-body", "Pi_F = Y_ATP,F/(1/f_G + r_F)\n\nPi_R = Y_ATP,R/(1/f_G + r_TCA + r_ETC + 1/f_O2)\n\nU_F = beta Pi_F + beta_a ln(a_G/a_C)\nU_R = beta Pi_R", {
    left: 52,
    top: 225,
    width: 565,
    height: 205,
  }, 21);
  addText(s7, "allocation-rule-title", "Softmax allocation", {
    left: 52,
    top: 462,
    width: 560,
    height: 40,
  }, 28);
  addText(s7, "allocation-rule-body", "theta_i* = exp(U_i) / [exp(U_F)+exp(U_R)]\n\ntheta_F* = 1 / [1 + exp(-(U_F - U_R))]\n\ndtheta_F/dt = (theta_F* - theta_F)/tau_reg", {
    left: 52,
    top: 512,
    width: 565,
    height: 122,
  }, 21);
  addSoftmaxTransitionGraph(s7);
  notes(s7, [
    "Model source: crabtree-effect-model-sketch.html, softmax splitter, explicit sigmoid form, and attention analogy.",
    "Tang et al. 2021, 2021-8 copy.pdf, Ohm's-law pathway return framing.",
  ], "The attention analogy is mathematical, not biological: scores become normalized allocation weights; the graph shows the smooth transition.");

  const s8 = addBaseSlide(presentation, "Notebook run captures overflow, not diauxic ethanol use", 8);
  addImageFrame(s8, { left: 44, top: 160, width: 570, height: 452 }, "state-plot");
  s8.images.add({
    blob: await imageBytes("notebook-plot-1-white.png"),
    contentType: "image/png",
    alt: "Default high-glucose pulse simulation from notebook showing glucose, oxygen, reserve, living structural biomass, total biomass, fermentation allocation, and ethanol.",
    fit: "contain",
    position: { left: 48, top: 167, width: 562, height: 438 },
  });
  addCard(s8, "overflow-card", "Overflow appears", "theta_F peaks near 0.97 and ethanol peaks near 40.5 in the illustrative high-glucose pulse.", {
    left: 657.68,
    top: 158,
    width: 580.99,
    height: 138,
  }, 21);
  addCard(s8, "biomass-card", "Biomass distinction matters", "Total biomass V(1+x_C) peaks at 10.75 while living structural V peaks at 3.45.", {
    left: 657.68,
    top: 326,
    width: 580.99,
    height: 138,
  }, 21);
  addCard(s8, "gap-card", "Known gap", "The notebook trajectory still lacks oxygen-limited ethanol uptake after glucose depletion.", {
    left: 657.68,
    top: 494,
    width: 580.99,
    height: 138,
  }, 21);
  notes(s8, [
    "Notebook source: crabtree-effect-mdeb-model.ipynb, default run outputs and embedded state plot.",
    "Model source: crabtree-effect-model-sketch.html, updated ethanol-as-second-substrate equations identify the next implementation step.",
  ], "Be explicit that this plot is a structure test, not a finished batch fit.");

  const s9 = buildSlide17(presentation, {
    footer1: footer(9),
    title: token("A realistic batch fit should reproduce the full diauxic sequence", 38.67),
    label1: token("Glucose phase", 18.67),
    label2: token("Diauxic switch", 18.67),
    label3: token("Stationary / death", 18.67),
    body1: textBlockHere("Overflow", "High G drives theta_F upward; ethanol accumulates while glucose is consumed.", 21.33, 20),
    body2: textBlockHere("Ethanol use", "Glucose repression relaxes; ethanol uptake and respiration require O2.", 21.33, 20),
    body3: textBlockHere("Biomass readout", "Total biomass can plateau while living V declines under maintenance deficit.", 21.33, 20),
  });
  notes(s9, [
    "Model source: crabtree-effect-model-sketch.html, glucose depletion and ethanol substrate regime.",
    "Notebook source: crabtree-effect-mdeb-model.ipynb, mortality and living-versus-total biomass treatment.",
  ], "This slide turns model gaps into testable next requirements.");

  const s10 = buildSlide26(presentation, {
    title: token("Next", 24),
    title2: token("Use the sketch as a falsifiable batch model", 72),
    title3: {
      loremIpsumDetails: token("Implement ethanol uptake", 26.67),
      loremIpsumDetails2: token("Cap respiratory O2 demand", 26.67),
      loremIpsumDetails3: token("Fit the batch metrics", 26.67),
    },
  });
  notes(s10, [
    "Source ledger: outputs/crabtree-model-deck-build/source-notes.txt",
    "Model source: references/crabtree-effect-model-sketch.html and references/crabtree-effect-mdeb-model.ipynb.",
  ], "Close by making the next work concrete: implement, constrain, then calibrate.");

  const inspect = await presentation.inspect({
    kind: "slide,textbox,shape,image,table,chart,notes",
    maxChars: 30000,
  });
  await fs.writeFile(path.join(TMP_DIR, "deck-inspect.ndjson"), inspect.ndjson);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(RENDER_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(RENDER_DIR, `${stem}.layout.json`), await layout.text());
  }

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(RENDER_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(FINAL_PPTX);
}

buildDeck().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
