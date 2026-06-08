# nca-digits

Conditional Growing Neural Cellular Automata — grows digits 0–9 from a single seed cell. Hand-written BPTT + Adam in Rust, no ML framework.

```
seed (one-hot digit) → 64 CA steps → 28×28 digit
```

~16.7k params, gradient-checked.

---

## Quick start

```bash
make build          # compile
make test           # verify BPTT (must pass)
make train-robust   # train (saves weights_robust.bin)
make eval           # score accuracy + self-repair
make viz            # watch digits grow
```

`make help` for all targets.

---

## Training modes

| Mode | Accuracy | Self-repair | File |
|------|:--------:|:-----------:|------|
| `overfit-all` | 10/10 | 3/10 | `weights_all.bin` |
| `train-robust` | 8/10 | 10/10 | `weights_robust.bin` |
| `train` | 7/10 | 10/10 | `weights.bin` |

All modes checkpoint every 1000 steps and support `RESUME=1`.

---

## Commands

| Command | Description |
|---------|-------------|
| `check` | print gradient-check error |
| `train [steps] [resume]` | pool + truncated BPTT |
| `train-robust [steps] [resume]` | full BPTT + pool + damage aug |
| `overfit-all [iters]` | sharpest accuracy, no pool |
| `gen [digit] [weights]` | grow a digit, print ASCII |
| `eval [weights]` | loss / accuracy / self-repair |

---

## Visualizer

```bash
make viz   # or: cargo run --release --features viz --bin viz
```

| Key | Action |
|-----|--------|
| `0-9` / `↑↓` | switch digit |
| `Space` / `N` | pause / step |
| `R` | reset |
| `D` | damage 30% (self-repair demo) |
| `L` | toggle auto-loop |
| `-` `=` | speed |
| `Esc` | quit |

---

## Architecture

- **Grid:** 28×28, 16 channels — RGB (0-2), alpha (3), digit one-hot (4-13), hidden (14-15)
- **Perception:** identity ⊕ Sobel-x ⊕ Sobel-y → 48-d input
- **Network:** `Linear(48→256) + ReLU → Linear(256→16)`
- **Training:** truncated BPTT (k=16), MSE vs 5×7-font targets, Adam lr=2e-4

---

## Layout

```
src/
  constants.rs   hyperparams + channel layout
  digits.rs      5×7 font + 28×28 targets
  grid.rs        Grid type, seed_init, alive mask
  perception.rs  Sobel + perceive()
  network.rs     NCANet, Grads, save/load
  backward.rs    StepCache, rollouts, BPTT
  optimizer.rs   Adam
  replay.rs      experience-replay pool
  main.rs        CLI
  bin/viz.rs     macroquad visualizer
```

Requires stable Rust (edition 2024). See `CONTEXT.md` for the full design spec.
