use nca_digits::backward::{backward, mse_grad, mse_loss, train_rollout};
use nca_digits::constants::*;
use nca_digits::digits::target_rgba;
use nca_digits::gradient_check;
use nca_digits::network::{Grads, NCANet};
use nca_digits::optimizer::Adam;
use nca_digits::render::print_grid;
use nca_digits::replay::ReplayPool;
use nca_digits::grid::{seed_init, Grid};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("train");

    match cmd {
        "check" => {
            let worst = gradient_check();
            println!("gradient check: worst relative error = {worst:.2e}");
            println!("{}", if worst < 5e-2 { "PASS" } else { "FAIL" });
        }
        "train" => {
            let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10_000);
            let resume = args.get(3).map(String::as_str) == Some("resume");
            train(steps, resume);
        }
        // full BPTT + pool + damage aug -> crisp and self-repairing
        "train-robust" => {
            let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8000);
            let resume = args.get(3).map(String::as_str) == Some("resume");
            train_robust(steps, resume);
        }
        "overfit" => {
            let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
            let digit: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
            overfit(digit, iters, false);
        }
        // discriminator: solid all-on target (loss -> 0 proves the loop is correct)
        "overfit-solid" => {
            let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
            overfit(3, iters, true);
        }
        // all 10 digits, full BPTT, no pool — sharpest (isolates capacity vs schedule)
        "overfit-all" => {
            let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1500);
            overfit_all(iters);
        }
        // `gen <digit> [weights_path]` — inference
        "gen" => {
            let digit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
            let path = args.get(3).map(String::as_str).unwrap_or("weights.bin");
            generate(digit, path);
        }
        // `eval [weights_path]` — per-digit loss, accuracy, self-repair
        "eval" => eval(args.get(2).map(String::as_str).unwrap_or("weights.bin")),
        other => eprintln!(
            "unknown command: {other} (use `check` | `train [steps] [resume]` | `overfit [iters] [digit]` | `gen [digit]` | `eval`)"
        ),
    }
}

// Single-digit overfit, full BPTT, no pool (debug / discriminator).
fn overfit(digit: usize, iters: usize, solid: bool) {
    use nca_digits::grid::seed_init;
    use ndarray::Array3;
    let mut rng = StdRng::seed_from_u64(1);
    let mut net = NCANet::init(&mut rng);
    let mut opt = Adam::new();
    let seed = seed_init(digit);
    let target = if solid {
        Array3::from_elem((H, W, RGBA), 1.0) // all-on target
    } else {
        target_rgba(digit)
    };
    println!("overfit digit {digit}, {iters} iters, full BPTT k={STEPS}");

    let mut final_grid = seed.clone();
    for it in 0..iters {
        let (fg, caches) = train_rollout(&net, &seed, 0, STEPS, digit, &mut rng);
        let loss = mse_loss(&fg, &target);
        let d_final = mse_grad(&fg, &target);
        let (g, _) = backward(&net, &caches, d_final, digit);
        let gnorm = (g.w1.iter().chain(g.b1.iter()).chain(g.w2.iter()).chain(g.b2.iter())
            .map(|x| x * x).sum::<f32>()).sqrt();
        let anorm = final_alpha_sum(&fg);
        opt.step(&mut net, &g);
        final_grid = fg;
        if it % 20 == 0 {
            println!("iter {it:4}  loss {loss:.4}  |grad| {gnorm:.3e}  sum_alpha {anorm:.2}");
        }
    }
    let loss = mse_loss(&final_grid, &target);
    print_grid(&final_grid, digit, STEPS, loss);
}

// Total alpha mass (growth vs collapse).
fn final_alpha_sum(g: &nca_digits::grid::Grid) -> f32 {
    let mut s = 0.0;
    for r in 0..H {
        for c in 0..W {
            s += g[[r, c, ALPHA]];
        }
    }
    s
}

// All 10 digits, full BPTT, no pool — each iter updates every digit once (parallel).
fn overfit_all(iters: usize) {
    let mut rng = StdRng::seed_from_u64(2);
    let mut net = NCANet::init(&mut rng);
    let mut opt = Adam::new();
    let targets: Vec<_> = (0..N_DIGITS).map(target_rgba).collect();
    println!("overfit-all: {iters} iters x {N_DIGITS} digits, full BPTT k={STEPS} (no pool)");

    for it in 0..iters {
        // cosine LR decay LR -> 0.1*LR
        let frac = it as f32 / iters as f32;
        opt.lr = LR * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * frac).cos()));

        let net_ref = &net;
        let targets_ref = &targets;
        let res: Vec<(f32, Grads)> = (0..N_DIGITS)
            .into_par_iter()
            .map(|d| {
                let mut rng = StdRng::seed_from_u64(it as u64 * 100 + d as u64);
                let (fg, caches) = train_rollout(net_ref, &seed_init(d), 0, STEPS, d, &mut rng);
                let loss = mse_loss(&fg, &targets_ref[d]);
                let d_final = mse_grad(&fg, &targets_ref[d]);
                let (g, _) = backward(net_ref, &caches, d_final, d);
                (loss, g)
            })
            .collect();

        let mut grads = Grads::zero();
        let mut losses = [0.0f32; N_DIGITS];
        for (d, (l, g)) in res.iter().enumerate() {
            grads.add(g);
            losses[d] = *l;
        }
        grads.scale(1.0 / N_DIGITS as f32);
        opt.step(&mut net, &grads);

        if it % 100 == 0 {
            let avg: f32 = losses.iter().sum::<f32>() / N_DIGITS as f32;
            let per: Vec<String> = losses.iter().map(|l| format!("{l:.2}")).collect();
            println!("it {it:5}  avg {avg:.4}  [{}]", per.join(" "));
        }
    }

    net.save("weights_all.bin").ok();
    let mut rng = StdRng::seed_from_u64(7);
    println!("\nfinal per-digit loss (fresh grow):");
    for d in 0..N_DIGITS {
        let g = grow(&net, d, &mut rng);
        println!("  {d}: {:.4}", mse_loss(&g, &targets[d]));
    }
    println!("\nsaved weights_all.bin (copy to weights.bin to viz/eval it)");
    for d in [0usize, 3, 8] {
        let g = grow(&net, d, &mut rng);
        print_grid(&g, d, STEPS, mse_loss(&g, &targets[d]));
        println!();
    }
}

// Pool + full BPTT (crisp/persistent) + damage augmentation (regeneration) + LR decay.
fn train_robust(n_steps: usize, resume: bool) {
    let mut rng = StdRng::seed_from_u64(0);
    // resume fine-tunes at lower LR; fresh starts from full LR
    let (mut net, lr_hi, lr_lo) = if resume {
        match NCANet::load("weights_robust.bin") {
            Ok(n) => {
                println!("resumed from weights_robust.bin (fine-tune LR)");
                (n, 0.3 * LR, 0.05 * LR)
            }
            Err(e) => {
                eprintln!("resume failed ({e}); starting fresh");
                (NCANet::init(&mut rng), LR, 0.1 * LR)
            }
        }
    } else {
        (NCANet::init(&mut rng), LR, 0.1 * LR)
    };
    let mut opt = Adam::new();
    let mut pool = ReplayPool::new(POOL_SIZE);
    let n_damage = (BATCH / 4).max(1); // damaged samples per batch
    println!(
        "train-robust: {n_steps} steps, full BPTT k={STEPS}, batch {BATCH}, {n_damage} damaged/batch"
    );

    for step in 0..n_steps {
        let frac = step as f32 / n_steps as f32;
        let cos = 0.5 * (1.0 + (std::f32::consts::PI * frac).cos());
        opt.lr = lr_lo + (lr_hi - lr_lo) * cos;

        let idx = pool.sample(BATCH, &mut rng);
        let mut samples: Vec<(usize, Grid)> = idx
            .iter()
            .map(|&i| (pool.items[i].0, pool.items[i].1.clone()))
            .collect();
        // damage augmentation
        for j in 0..n_damage {
            damage_region(&mut samples[j].1, W / 2, &mut rng);
        }

        let net_ref = &net;
        let results: Vec<(f32, Grads, Grid)> = samples
            .par_iter()
            .enumerate()
            .map(|(pos, (digit, grid0))| {
                let mut rng = StdRng::seed_from_u64(step as u64 * 1_000_003 + pos as u64);
                let (final_grid, caches) = train_rollout(net_ref, grid0, 0, STEPS, *digit, &mut rng);
                let target = target_rgba(*digit);
                let loss = mse_loss(&final_grid, &target);
                let d_final = mse_grad(&final_grid, &target);
                let (g, _) = backward(net_ref, &caches, d_final, *digit);
                (loss, g, final_grid)
            })
            .collect();

        let mut grads = Grads::zero();
        let mut batch_loss = 0.0f32;
        let mut worst = (0usize, f32::NEG_INFINITY);
        let mut finals = Vec::with_capacity(BATCH);
        for (pos, (loss, g, fg)) in results.into_iter().enumerate() {
            grads.add(&g);
            batch_loss += loss;
            if loss > worst.1 {
                worst = (pos, loss);
            }
            finals.push(fg);
        }
        grads.scale(1.0 / BATCH as f32);
        batch_loss /= BATCH as f32;
        opt.step(&mut net, &grads);

        for (pos, &i) in idx.iter().enumerate() {
            if pos == worst.0 {
                pool.reset(i);
            } else {
                pool.put(i, std::mem::take(&mut finals[pos]));
            }
        }

        if step % 100 == 0 {
            println!("step {step:6}  loss {batch_loss:.4}");
        }
        if step > 0 && step % 1000 == 0 {
            net.save("weights_robust.bin").ok();
        }
    }
    net.save("weights_robust.bin").ok();
    println!("\nsaved weights_robust.bin  —  eval with: cargo run --release -- eval weights_robust.bin");
}

fn train(n_steps: usize, resume: bool) {
    const WARMUP: usize = STEPS - BPTT_K; // un-cached steps before the BPTT window
    let mut rng = StdRng::seed_from_u64(0);
    // resume loads weights.bin (Adam + pool re-warm); mismatch -> fresh
    let mut net = if resume {
        match NCANet::load("weights.bin") {
            Ok(n) => {
                println!("resumed from weights.bin");
                n
            }
            Err(e) => {
                eprintln!("resume failed ({e}); starting fresh");
                NCANet::init(&mut rng)
            }
        }
    } else {
        NCANet::init(&mut rng)
    };
    let mut opt = Adam::new();
    let mut pool = ReplayPool::new(POOL_SIZE);

    println!("training {n_steps} steps (batch {BATCH}, {STEPS} fwd steps, BPTT_K {BPTT_K})");

    for step in 0..n_steps {
        let idx = pool.sample(BATCH, &mut rng);
        // snapshot so the parallel closure doesn't borrow pool
        let samples: Vec<(usize, Grid)> = idx
            .iter()
            .map(|&i| (pool.items[i].0, pool.items[i].1.clone()))
            .collect();

        // samples are independent (own grads/d_in) -> safe to parallelize
        let net_ref = &net;
        let results: Vec<(f32, Grads, Grid)> = samples
            .par_iter()
            .enumerate()
            .map(|(pos, (digit, grid0))| {
                let mut rng = StdRng::seed_from_u64(step as u64 * 1_000_003 + pos as u64);
                let (final_grid, caches) =
                    train_rollout(net_ref, grid0, WARMUP, BPTT_K, *digit, &mut rng);
                let target = target_rgba(*digit);
                let loss = mse_loss(&final_grid, &target);
                let d_final = mse_grad(&final_grid, &target);
                let (g, _) = backward(net_ref, &caches, d_final, *digit);
                (loss, g, final_grid)
            })
            .collect();

        let mut grads = Grads::zero();
        let mut batch_loss = 0.0f32;
        let mut worst = (0usize, f32::NEG_INFINITY); // (sample position, loss)
        let mut finals = Vec::with_capacity(BATCH);
        for (pos, (loss, g, final_grid)) in results.into_iter().enumerate() {
            grads.add(&g);
            batch_loss += loss;
            if loss > worst.1 {
                worst = (pos, loss);
            }
            finals.push(final_grid);
        }

        grads.scale(1.0 / BATCH as f32);
        batch_loss /= BATCH as f32;
        opt.step(&mut net, &grads);

        // write evolved states back; reset the worst to a fresh seed
        for (pos, &i) in idx.iter().enumerate() {
            if pos == worst.0 {
                pool.reset(i);
            } else {
                pool.put(i, std::mem::take(&mut finals[pos]));
            }
        }

        if step % 100 == 0 {
            println!("step {step:6}  loss {batch_loss:.4}");
        }
        if step > 0 && step % 1000 == 0 {
            if let Err(e) = net.save("weights.bin") {
                eprintln!("save failed: {e}");
            }
        }
    }

    net.save("weights.bin").ok();
    println!("\nfinal samples:");
    sample_render(&net, &mut rng);
}

// Grow a few digits from fresh seeds and print them.
fn sample_render(net: &NCANet, rng: &mut StdRng) {
    for digit in [0usize, 3, 8] {
        let (final_grid, _) = train_rollout(net, &seed_init(digit), STEPS, 0, digit, rng);
        let target = target_rgba(digit);
        let loss = mse_loss(&final_grid, &target);
        print_grid(&final_grid, digit, STEPS, loss);
        println!();
    }
}

// Grow one digit STEPS steps from a fresh seed.
fn grow(net: &NCANet, digit: usize, rng: &mut StdRng) -> Grid {
    train_rollout(net, &seed_init(digit), STEPS, 0, digit, rng).0
}

fn load_or_warn(path: &str) -> Option<NCANet> {
    match NCANet::load(path) {
        Ok(n) => Some(n),
        Err(e) => {
            eprintln!("could not load {path} ({e}) — train first");
            None
        }
    }
}

// Inference: grow `digit` from the given weights and render it.
fn generate(digit: usize, path: &str) {
    let Some(net) = load_or_warn(path) else { return };
    let mut rng = StdRng::seed_from_u64(7);
    let g = grow(&net, digit, &mut rng);
    let loss = mse_loss(&g, &target_rgba(digit));
    print_grid(&g, digit, STEPS, loss);
}

// Zero a random square region (training damage).
fn damage_region(g: &mut Grid, side: usize, rng: &mut StdRng) {
    let r0 = rng.gen_range(0..H);
    let c0 = rng.gen_range(0..W);
    for r in r0..(r0 + side).min(H) {
        for c in c0..(c0 + side).min(W) {
            for ch in 0..C {
                g[[r, c, ch]] = 0.0;
            }
        }
    }
}

// Zero a random fraction of cells (self-repair test damage).
fn damage(g: &Grid, frac: f32, rng: &mut StdRng) -> Grid {
    let mut d = g.clone();
    for r in 0..H {
        for c in 0..W {
            if rng.r#gen::<f32>() < frac {
                for ch in 0..C {
                    d[[r, c, ch]] = 0.0;
                }
            }
        }
    }
    d
}

// Metrics: per-digit recon MSE, classification accuracy, self-repair.
fn eval(path: &str) {
    let Some(net) = load_or_warn(path) else { return };
    println!("eval {path}");
    let mut rng = StdRng::seed_from_u64(7);

    let grids: Vec<Grid> = (0..N_DIGITS).map(|d| grow(&net, d, &mut rng)).collect();
    let targets: Vec<_> = (0..N_DIGITS).map(target_rgba).collect();

    println!("digit | recon_loss | predicted");
    let mut sum_loss = 0.0f32;
    let mut correct = 0;
    for d in 0..N_DIGITS {
        let loss = mse_loss(&grids[d], &targets[d]);
        // classify: which target is the grown image closest to?
        let mut best = (0usize, f32::INFINITY);
        for t in 0..N_DIGITS {
            let l = mse_loss(&grids[d], &targets[t]);
            if l < best.1 {
                best = (t, l);
            }
        }
        if best.0 == d {
            correct += 1;
        }
        println!(
            "  {d}   |   {loss:.4}   |   {} {}",
            best.0,
            if best.0 == d { "ok" } else { "MISS" }
        );
        sum_loss += loss;
    }
    println!(
        "avg recon loss {:.4}  |  accuracy {correct}/{N_DIGITS}",
        sum_loss / N_DIGITS as f32
    );

    println!("\nself-repair (erase 30%, +20 steps):");
    for d in 0..N_DIGITS {
        let grown = mse_loss(&grids[d], &targets[d]);
        let damaged = damage(&grids[d], 0.30, &mut rng);
        let dmg = mse_loss(&damaged, &targets[d]);
        let repaired = train_rollout(&net, &damaged, 20, 0, d, &mut rng).0;
        let after = mse_loss(&repaired, &targets[d]);
        let verdict = if after < dmg { "recovered" } else { "no" };
        println!("  {d}: grown {grown:.3} -> damaged {dmg:.3} -> repaired {after:.3}  [{verdict}]");
    }
}
