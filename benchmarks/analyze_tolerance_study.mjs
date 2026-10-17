import fs from "node:fs";
import path from "node:path";

const root = path.resolve(process.argv[2] ?? "out/tolerance-study/sweep-10-steps");
const referenceLabel = "outer_1e-10__cg_1e-12";

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function readSnapshot(file) {
  const buffer = fs.readFileSync(file);
  if (buffer.subarray(0, 8).toString("ascii") !== "VANKATOL") {
    throw new Error(`Invalid snapshot magic: ${file}`);
  }
  const version = buffer.readUInt32LE(8);
  const nx = buffer.readUInt32LE(12);
  const ny = buffer.readUInt32LE(16);
  const nz = buffer.readUInt32LE(20);
  const cells = Number(buffer.readBigUInt64LE(24));
  if (version !== 1 || cells !== nx * ny * nz || buffer.length !== 32 + cells * 40) {
    throw new Error(`Invalid snapshot header or length: ${file}`);
  }
  return { buffer, nx, ny, nz, cells };
}

function compareSnapshots(candidate, reference) {
  if (
    candidate.nx !== reference.nx ||
    candidate.ny !== reference.ny ||
    candidate.nz !== reference.nz
  ) {
    throw new Error("Snapshot dimensions do not match");
  }
  const sumDiff2 = [0, 0, 0, 0, 0];
  const sumRef2 = [0, 0, 0, 0, 0];
  const maxAbs = [0, 0, 0, 0, 0];
  for (let cell = 0; cell < reference.cells; ++cell) {
    const base = 32 + cell * 40;
    for (let field = 0; field < 5; ++field) {
      const offset = base + field * 8;
      const a = candidate.buffer.readDoubleLE(offset);
      const b = reference.buffer.readDoubleLE(offset);
      const difference = a - b;
      sumDiff2[field] += difference * difference;
      sumRef2[field] += b * b;
      maxAbs[field] = Math.max(maxAbs[field], Math.abs(difference));
    }
  }
  const relative = sumDiff2.map((value, index) =>
    sumRef2[index] > 0 ? Math.sqrt(value / sumRef2[index]) : Math.sqrt(value / reference.cells),
  );
  const velocityDifference2 = sumDiff2[0] + sumDiff2[1] + sumDiff2[2];
  const velocityReference2 = sumRef2[0] + sumRef2[1] + sumRef2[2];
  return {
    delta_velocity_rms_abs: Math.sqrt(velocityDifference2 / (3 * reference.cells)),
    delta_pressure_rms_abs: Math.sqrt(sumDiff2[3] / reference.cells),
    delta_phi_rms_abs: Math.sqrt(sumDiff2[4] / reference.cells),
    delta_velocity_rel_l2: Math.sqrt(velocityDifference2 / velocityReference2),
    delta_pressure_rel_l2: relative[3],
    delta_phi_rel_l2: relative[4],
    delta_u_rel_l2: relative[0],
    delta_v_rel_l2: relative[1],
    delta_w_rel_l2: relative[2],
    delta_u_max_abs: maxAbs[0],
    delta_v_max_abs: maxAbs[1],
    delta_w_max_abs: maxAbs[2],
    delta_pressure_max_abs: maxAbs[3],
    delta_phi_max_abs: maxAbs[4],
  };
}

const directories = fs
  .readdirSync(root, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)
  .filter((name) =>
    fs.existsSync(path.join(root, name, "summary.json")) &&
    fs.existsSync(path.join(root, name, "tolerance_snapshot.bin")),
  );

const reference = readSnapshot(path.join(root, referenceLabel, "tolerance_snapshot.bin"));
const referenceSummary = readJson(path.join(root, referenceLabel, "summary.json"));
const results = [];
let repeat = null;

for (const label of directories) {
  const summary = readJson(path.join(root, label, "summary.json"));
  const snapshot = readSnapshot(path.join(root, label, "tolerance_snapshot.bin"));
  const differences = compareSnapshots(snapshot, reference);
  const result = {
    ...summary,
    speedup: referenceSummary.mean_time_s / summary.mean_time_s,
    ...differences,
    abs_vof_mass_error: Math.abs(summary.final_vof_mass_error),
  };
  if (summary.repeat) repeat = result;
  else results.push(result);
}

results.sort((a, b) => a.outer_tol - b.outer_tol || b.cg_tol - a.cg_tol);

const tightestByOuter = new Map();
for (const row of results) {
  const current = tightestByOuter.get(row.outer_tol);
  if (!current || row.cg_tol < current.cg_tol) tightestByOuter.set(row.outer_tol, row);
}
for (const row of results) {
  row.mean_cg_iterations_per_step = row.total_cg_iterations / row.timesteps;
  const tight = tightestByOuter.get(row.outer_tol);
  const candidateSnapshot = readSnapshot(path.join(root, row.label, "tolerance_snapshot.bin"));
  const tightSnapshot = readSnapshot(path.join(root, tight.label, "tolerance_snapshot.bin"));
  const pcgEffect = compareSnapshots(candidateSnapshot, tightSnapshot);
  row.delta_velocity_rms_abs_vs_tight_cg = pcgEffect.delta_velocity_rms_abs;
  row.delta_pressure_rms_abs_vs_tight_cg = pcgEffect.delta_pressure_rms_abs;
}

const csvColumns = [
  "outer_tol",
  "cg_tol",
  "mean_outer_iterations",
  "cg_iterations_per_outer",
  "mean_cg_iterations_per_step",
  "mean_time_s",
  "speedup",
  "final_outer_residual_u",
  "final_outer_residual_v",
  "final_outer_residual_w",
  "final_outer_residual_p",
  "final_time_residual_u",
  "final_time_residual_v",
  "final_time_residual_w",
  "final_time_residual_p",
  "delta_velocity_rms_abs",
  "delta_pressure_rms_abs",
  "delta_phi_rms_abs",
  "delta_velocity_rms_abs_vs_tight_cg",
  "delta_pressure_rms_abs_vs_tight_cg",
  "delta_velocity_rel_l2",
  "delta_pressure_rel_l2",
  "delta_phi_rel_l2",
  "delta_u_max_abs",
  "delta_v_max_abs",
  "delta_w_max_abs",
  "delta_pressure_max_abs",
  "delta_phi_max_abs",
  "final_vof_mass_error",
  "final_divergence_rms",
  "final_divergence_max",
];
const csv = [
  csvColumns.join(","),
  ...results.map((row) => csvColumns.map((column) => row[column]).join(",")),
].join("\n");

fs.writeFileSync(path.join(root, "results.json"), JSON.stringify({ reference: referenceSummary, repeat, results }, null, 2));
fs.writeFileSync(path.join(root, "results.csv"), `${csv}\n`);

if (repeat) {
  const drift = (repeat.mean_time_s / referenceSummary.mean_time_s - 1) * 100;
  console.log(`Reference timing repeat drift: ${drift.toFixed(2)}%`);
}
console.log(`Analyzed ${results.length} tolerance pairs in ${root}`);
