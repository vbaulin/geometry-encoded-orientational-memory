#!/usr/bin/env python3
"""Shared-target capillary memory with matched and separately written releases.

Energies are in kBT and time in 1/Dr. Every release is field-free. The three
Hamiltonians use the same positional graph and the same target. Checkpoints
contain the angular state, sampled trajectory and RNG state, not just metrics.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

try:
    from .rotating_colloids_capillary_pair import make_caged_graph, _torque, _graph_tensors
except ImportError:
    from rotating_colloids_capillary_pair import make_caged_graph, _torque, _graph_tensors


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("physical", "no_capillary", "equal_weight_relative")
RELEASE_ARMS = ("matched_physical", "matched_no_capillary", "matched_equal_weight_relative",
                "own_write_no_capillary", "own_write_equal_weight_relative")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def record_schedule(steps: int, samples: int) -> np.ndarray:
    return np.unique(np.r_[0, np.rint(np.geomspace(1, steps, min(samples, steps))).astype(int), steps])


def coefficients(graph, j: float, g: float):
    a, b = j * graph.alignment_weight, g * graph.capillary_weight
    return np.stack((a, a, a + b)), np.stack((b, np.zeros_like(b), np.zeros_like(b)))


def evolve(graph, initial, a, b, *, dt, steps, seed, device, path,
           axis=None, field=0.0, samples=160, checkpoint_steps=10000, stop_after=None):
    """Vectorize Hamiltonian arms while retaining the tested pair torque.

    Samples include the exact initial and terminal states. All phases advance
    from the true terminal state rather than the final plotting sample.
    """
    import torch

    initial = np.asarray(initial, dtype=np.float32)
    arms, replicas, nodes = initial.shape
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != (arms, len(graph.src)) or b.shape != a.shape:
        raise ValueError("Interaction coefficients do not match arm/edge layout")
    if steps < 1 or dt <= 0 or checkpoint_steps < 1:
        raise ValueError("Positive steps, dt and checkpoint interval required")
    schedule = record_schedule(steps, samples)
    spec = digest({"initial": array_digest(initial), "a": array_digest(a), "b": array_digest(b),
                   "phi": array_digest(graph.phi), "src": array_digest(graph.src),
                   "tgt": array_digest(graph.tgt), "dt": dt, "steps": steps,
                   "seed": seed, "axis": None if axis is None else array_digest(axis),
                   "field": field, "schedule": schedule.tolist(), "device_type": device.type,
                   "core_sha256": hashlib.sha256((Path(__file__).parent / "rotating_colloids_capillary_pair.py").read_bytes()).hexdigest()})
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["spec"]) != spec:
                raise ValueError(f"Existing stage uses different parameters: {path}")
            return {key: saved[key].copy() for key in saved.files}
    partial = path.with_name(path.stem + ".partial.npz")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    state = torch.as_tensor(initial.reshape(-1, nodes), device=device).clone()
    trajectory = np.empty((len(schedule), arms, replicas, nodes), dtype=np.float32)
    trajectory[0] = initial
    step, recorded = 0, 1
    if partial.exists():
        with np.load(partial, allow_pickle=False) as saved:
            if str(saved["spec"]) != spec:
                raise ValueError(f"Checkpoint uses different parameters: {partial}")
            step, recorded = int(saved["step"]), int(saved["recorded"])
            trajectory[:recorded] = saved["theta"]
            state = torch.as_tensor(saved["state"], device=device).clone()
            generator.set_state(torch.from_numpy(saved["rng_state"]).cpu())
    tensors = _graph_tensors(graph, device)
    tensors["wa"] = torch.as_tensor(np.repeat(a, replicas, axis=0), dtype=torch.float32, device=device)
    tensors["wc"] = torch.as_tensor(np.repeat(b, replicas, axis=0), dtype=torch.float32, device=device)
    axis_tensor = None if axis is None else torch.as_tensor(axis, dtype=torch.float32, device=device)
    start_time = time.monotonic()

    def save_checkpoint():
        atomic_npz(partial, spec=spec, step=step, recorded=recorded,
                   theta=trajectory[:recorded], state=state.cpu().numpy(),
                   rng_state=generator.get_state().cpu().numpy())

    with torch.no_grad():
        while step < steps:
            torque = _torque(state, tensors, 1.0, 1.0, write_axis=axis_tensor, write_field=field)
            state += dt * torque + math.sqrt(2 * dt) * torch.randn(
                state.shape, generator=generator, device=device)
            state.remainder_(math.pi)
            step += 1
            if recorded < len(schedule) and step == schedule[recorded]:
                trajectory[recorded] = state.cpu().numpy().reshape(arms, replicas, nodes)
                recorded += 1
            if step % checkpoint_steps == 0 or step == steps:
                if not bool(torch.isfinite(state).all()):
                    raise FloatingPointError(f"Nonfinite angles in {path}, step {step}")
                save_checkpoint()
                print(json.dumps({"event": "stage_progress", "stage": path.stem,
                                  "step": step, "steps": steps,
                                  "elapsed_this_process_s": round(time.monotonic() - start_time, 2)}), flush=True)
            if stop_after is not None and step >= stop_after and step < steps:
                save_checkpoint()
                return None
    result = {"spec": np.asarray(spec), "time": schedule * dt, "theta": trajectory,
              "terminal_theta": state.cpu().numpy().reshape(arms, replicas, nodes),
              "rng_state": generator.get_state().cpu().numpy()}
    atomic_npz(path, **result)
    partial.unlink(missing_ok=True)
    return result


def overlaps(theta, target, unrelated):
    z = np.exp(2j * np.asarray(theta, dtype=np.float64))
    z_target = np.exp(2j * np.asarray(target, dtype=np.float64))
    m, mt = z.mean(axis=-1), z_target.mean()
    q = (z * z_target.conj()).mean(axis=-1)
    connected = q - m * mt.conjugate()
    other = np.exp(2j * np.asarray(unrelated, dtype=np.float64))
    return {
        "Q": q.real, "Q_connected": connected.real,
        "Q_rotation_aligned": abs(q), "Q_connected_rotation_aligned": abs(connected),
        "S": abs(m), "mean_z_real": m.real, "mean_z_imag": m.imag,
        "Q_unrelated": (z * other.conj()).mean(axis=-1).real,
        "target_S": float(abs(mt)),
    }


def spatial_curve(graph, angles):
    """Equal-time correlator over all pairs, retaining replica uncertainties."""
    src, tgt = np.triu_indices(len(graph.positions), 1)
    delta = graph.positions[tgt] - graph.positions[src]
    delta -= graph.box * np.round(delta / graph.box)
    distance = np.linalg.norm(delta, axis=1)
    edges = np.arange(0, 8.5, 0.5)
    bins = np.digitize(distance, edges) - 1
    keep = (bins >= 0) & (bins < len(edges) - 1)
    src, tgt, bins = src[keep], tgt[keep], bins[keep]
    counts = np.bincount(bins, minlength=len(edges) - 1)
    curves = []
    for row in angles:
        corr = np.cos(2 * (row[src] - row[tgt]))
        total = np.bincount(bins, weights=corr, minlength=len(counts))
        mean = np.full(len(counts), np.nan)
        np.divide(total, counts, out=mean, where=counts > 0)
        curves.append(mean - abs(np.exp(2j * row).mean()) ** 2)
    return {"r_over_a": (edges[:-1] + edges[1:]) / 2, "pair_count": counts,
            "connected": np.asarray(curves)}


def case_seed(config, name):
    return int(digest({"config": config, "stage": name})[:8], 16)


def run_case(config: dict, out: Path, device, checkpoint_steps: int):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = out / "manifest.json"
        current_manifest = {"config": config, "description": __doc__, "arms": RELEASE_ARMS,
                               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "core_sha256": hashlib.sha256((Path(__file__).parent / "rotating_colloids_capillary_pair.py").read_bytes()).hexdigest(),
                               "noise": "independent across Hamiltonian arms and replicas"}
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            for key in ("config", "code_sha256", "core_sha256"):
                if previous[key] != current_manifest[key]:
                    raise ValueError(f"Do not mix simulation code or settings in {out}: {key}")
        else:
            atomic_json(manifest, current_manifest)
        if (out / "summary.json").exists():
            if not all((out / name).exists() for name in ("release.npz", "observables.npz", "preparation.npz")):
                raise ValueError(f"Completed case is missing its angular source: {out}")
            return
        graph = make_caged_graph(config["n"], disorder=config["disorder"], cutoff=2.6,
                                 alignment_range=1.35, alignment_decay=0.2, seed=config["graph_seed"])
        n, reps = len(graph.positions), config["replicas"]
        j, g = 4 * config["coupling_scale"], 5 * config["coupling_scale"]
        a, b = coefficients(graph, j, g)
        # The two target patterns share exactly the same field-free energy.
        parent_config = {k: v for k, v in config.items() if k != "target"}
        prep_seed = case_seed(parent_config, "target")
        prep = evolve(graph, np.random.default_rng(prep_seed).uniform(0, np.pi, (1, 1, n)),
                      a[:1], b[:1], dt=config["dt"], steps=config["equilibration_steps"],
                      seed=prep_seed, device=device, path=out / "target_relaxation.npz",
                      samples=2, checkpoint_steps=checkpoint_steps)
        target = prep["terminal_theta"][0, 0].copy()
        if config["target"] == "quarter_turn":
            target = np.remainder(target + np.pi / 2, np.pi)
        seed = case_seed(config, "write")
        rng = np.random.default_rng(seed)
        common_initial = rng.uniform(0, np.pi, (reps, n)).astype(np.float32)
        unrelated = rng.uniform(0, np.pi, n).astype(np.float32)
        write = evolve(graph, np.repeat(common_initial[None], 3, axis=0), a, b,
                       dt=config["dt"], steps=config["write_steps"], seed=seed, device=device,
                       path=out / "write.npz", axis=target, field=config["write_field"],
                       checkpoint_steps=checkpoint_steps)
        written = write["terminal_theta"]
        release_initial = np.stack((written[0], written[0], written[0], written[1], written[2]))
        np.testing.assert_array_equal(release_initial[:3], np.repeat(written[0][None], 3, axis=0))
        indices = [0, 1, 2, 1, 2]
        atomic_npz(out / "preparation.npz", target=target, unrelated_target=unrelated,
                   release_initial=release_initial, positions=graph.positions,
                   box=graph.box, src=graph.src, tgt=graph.tgt, phi=graph.phi,
                   wa=graph.alignment_weight, wc=graph.capillary_weight)
        release = evolve(graph, release_initial, a[indices], b[indices], dt=config["dt"],
                         steps=config["release_steps"], seed=case_seed(config, "release"),
                         device=device, path=out / "release.npz", checkpoint_steps=checkpoint_steps)
        obs = overlaps(release["theta"], target, unrelated)
        atomic_npz(out / "observables.npz", time=release["time"], **obs)
        spatial = spatial_curve(graph, release["terminal_theta"][0])
        atomic_npz(out / "spatial_final.npz", **spatial)
        summary = {"config": config, "matched_initial_states_identical": True,
                   "target_sha256": array_digest(target), "target_S": obs["target_S"],
                   "graph": graph.metadata, "release_time": float(release["time"][-1]), "arms": {}}
        for i, name in enumerate(RELEASE_ARMS):
            summary["arms"][name] = {
                key: {"mean": float(value[-1, i].mean()),
                      "replica_sd": float(value[-1, i].std(ddof=1)) if reps > 1 else 0.0,
                      "per_replica": value[-1, i].tolist()}
                for key, value in obs.items() if isinstance(value, np.ndarray)
            }
        atomic_json(out / "summary.json", summary)
        print(json.dumps({"event": "case_complete", "case": str(out), "release_time": summary["release_time"]}), flush=True)


def analyze(directory: Path):
    from collections import defaultdict

    groups = defaultdict(list)
    for path in sorted(directory.rglob("summary.json")):
        row = json.loads(path.read_text())
        c = row["config"]
        for arm, obs in row["arms"].items():
            groups[(c["n"], c["disorder"], c["coupling_scale"], c["target"], arm,
                    row["release_time"])].append((c["graph_seed"], obs))
    table = []
    for key, rows in sorted(groups.items()):
        seeds = [seed for seed, _ in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate graph realization: {key}")
        entry = dict(zip(("n", "disorder", "coupling_scale", "target", "arm", "release_time"), key))
        entry["graph_count"] = len(rows)
        entry["graph_seeds"] = seeds
        for metric in ("Q", "Q_connected", "Q_rotation_aligned", "S", "Q_unrelated"):
            values = np.asarray([obs[metric]["mean"] for _, obs in rows])
            entry[metric] = {"mean": float(values.mean()), "graph_sem":
                             float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None}
        table.append(entry)
    atomic_json(directory / "matched_preparation_report.json", {"groups": table,
                "interpretation": "Shared targets; identical release starts in matched arms; independent Brownian noise. The equal-weight control preserves edge magnitudes, not target energy or torque."})
    print(json.dumps({"event": "analysis_complete", "groups": len(table), "output_dir": str(directory)}))


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sizes", default="32")
    parser.add_argument("--graph-seeds", default="17,29,43,71,97")
    parser.add_argument("--disorders", default="0.11,0.16")
    parser.add_argument("--coupling-scales", default="1")
    parser.add_argument("--targets", default="relaxed,quarter_turn")
    parser.add_argument("--replicas", type=int, default=48)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--equilibration-steps", type=int, default=50000)
    parser.add_argument("--write-steps", type=int, default=50000)
    parser.add_argument("--release-steps", type=int, default=250000)
    parser.add_argument("--write-field", type=float, default=1.5)
    parser.add_argument("--checkpoint-steps", type=int, default=10000)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(args.output_dir)
        return
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for n in map(int, args.sizes.split(",")):
        for seed in map(int, args.graph_seeds.split(",")):
            for sigma in map(float, args.disorders.split(",")):
                for scale in map(float, args.coupling_scales.split(",")):
                    for target in args.targets.split(","):
                        if target not in ("relaxed", "quarter_turn"):
                            raise ValueError(f"Unknown target {target}")
                        config = {"schema": 1, "n": n, "graph_seed": seed, "disorder": sigma,
                                  "coupling_scale": scale, "target": target,
                                  **{k: getattr(args, k) for k in ("replicas", "dt", "equilibration_steps",
                                      "write_steps", "release_steps", "write_field")}}
                        name = f"n{n}_sigma{sigma:g}_lambda{scale:g}_seed{seed}_{target}"
                        run_case(config, args.output_dir / name, device, args.checkpoint_steps)
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
