// Forward-with-cache, MSE loss, truncated BPTT.
// One step: new = life * (grid + mask*delta), then seed re-injection.
// The network is applied to all H*W cells at once (see `update`).

use crate::constants::*;
use crate::digits::seed_cells;
use crate::grid::{alive_mask, new_grid, Grid};
use crate::network::{Grads, NCANet};
use crate::perception::{perceive_all, SOBEL_X, SOBEL_Y};
use ndarray::{Array2, Array3, Axis, Zip};
use rand::Rng;

// Pin each seed cell's alpha + one-hot (a permanent live source; stop-grad).
fn reinject_seed(g: &mut Grid, digit: usize) {
    for (r, c) in seed_cells(digit) {
        g[[r, c, ALPHA]] = 1.0;
        for ch in COND_LO..COND_HI {
            g[[r, c, ch]] = if ch == COND_LO + digit { 1.0 } else { 0.0 };
        }
    }
}

fn seed_mask(digit: usize) -> Vec<bool> {
    let mut m = vec![false; H * W];
    for (r, c) in seed_cells(digit) {
        m[r * W + c] = true;
    }
    m
}

pub fn random_mask<R: Rng>(rng: &mut R) -> Vec<bool> {
    (0..H * W).map(|_| rng.gen_bool(0.5)).collect()
}

pub struct StepCache {
    pub masks: Vec<bool>,
    pub perceptions: Array2<f32>, // [H*W, PERC]
    pub hidden: Array2<f32>,      // [H*W, HIDDEN], post-ReLU
    pub life: Array2<f32>,        // frozen alive gate
}

fn is_cond(ch: usize) -> bool {
    (COND_LO..COND_HI).contains(&ch)
}

// grid + mask * delta with the conditioning channels frozen, before the alive
// gate. The network runs once over all cells (two matrix products). Also returns
// the perception and hidden matrices, which backward needs.
fn update(net: &NCANet, grid: &Grid, masks: &[bool]) -> (Grid, Array2<f32>, Array2<f32>) {
    let perceptions = perceive_all(grid);
    let (hidden, delta) = net.forward_batch(&perceptions);
    let mut new = grid.clone();
    for idx in 0..H * W {
        if !masks[idx] {
            continue;
        }
        let (r, c) = (idx / W, idx % W);
        for ch in 0..C {
            if !is_cond(ch) {
                new[[r, c, ch]] += delta[[idx, ch]];
            }
        }
    }
    (new, perceptions, hidden)
}

fn apply_life(g: &mut Grid, life: &Array2<f32>) {
    for r in 0..H {
        for c in 0..W {
            let l = life[[r, c]];
            for ch in 0..C {
                g[[r, c, ch]] *= l;
            }
        }
    }
}

pub fn step_cache(net: &NCANet, grid: &Grid, masks: &[bool], digit: usize) -> (Grid, StepCache) {
    let (mut new, perceptions, hidden) = update(net, grid, masks);
    let life = alive_mask(&new);
    apply_life(&mut new, &life);
    reinject_seed(&mut new, digit);
    (
        new,
        StepCache {
            masks: masks.to_vec(),
            perceptions,
            hidden,
            life,
        },
    )
}

// Forward step without caching.
pub fn step_only(net: &NCANet, grid: &Grid, masks: &[bool], digit: usize) -> Grid {
    let (mut new, _, _) = update(net, grid, masks);
    let life = alive_mask(&new);
    apply_life(&mut new, &life);
    reinject_seed(&mut new, digit);
    new
}

pub fn rollout_cache(
    net: &NCANet,
    grid0: &Grid,
    masks: &[Vec<bool>],
    digit: usize,
) -> (Grid, Vec<StepCache>) {
    let mut g = grid0.clone();
    let mut caches = Vec::with_capacity(masks.len());
    for m in masks {
        let (ng, cache) = step_cache(net, &g, m, digit);
        caches.push(cache);
        g = ng;
    }
    (g, caches)
}

pub fn rollout(net: &NCANet, grid0: &Grid, masks: &[Vec<bool>], digit: usize) -> Grid {
    let mut g = grid0.clone();
    for m in masks {
        g = step_only(net, &g, m, digit);
    }
    g
}

// step_only with a provided (frozen) life gate — for the gradient check, so the
// finite difference matches the analytic stop-grad treatment of the alive gate.
pub fn step_only_frozen(
    net: &NCANet,
    grid: &Grid,
    masks: &[bool],
    life: &Array2<f32>,
    digit: usize,
) -> Grid {
    let (mut new, _, _) = update(net, grid, masks);
    apply_life(&mut new, life);
    reinject_seed(&mut new, digit);
    new
}

pub fn rollout_frozen(
    net: &NCANet,
    grid0: &Grid,
    masks: &[Vec<bool>],
    lifes: &[Array2<f32>],
    digit: usize,
) -> Grid {
    let mut g = grid0.clone();
    for (m, life) in masks.iter().zip(lifes) {
        g = step_only_frozen(net, &g, m, life, digit);
    }
    g
}

// `warmup` un-cached steps, then `k` cached steps (the BPTT window).
pub fn train_rollout<R: Rng>(
    net: &NCANet,
    grid0: &Grid,
    warmup: usize,
    k: usize,
    digit: usize,
    rng: &mut R,
) -> (Grid, Vec<StepCache>) {
    let mut g = grid0.clone();
    for _ in 0..warmup {
        let m = random_mask(rng);
        g = step_only(net, &g, &m, digit);
    }
    let mut caches = Vec::with_capacity(k);
    for _ in 0..k {
        let m = random_mask(rng);
        let (ng, cache) = step_cache(net, &g, &m, digit);
        caches.push(cache);
        g = ng;
    }
    (g, caches)
}

// MSE over the RGBA channels.
pub fn mse_loss(grid: &Grid, target: &Array3<f32>) -> f32 {
    let n = (H * W * RGBA) as f32;
    let mut s = 0.0;
    for r in 0..H {
        for c in 0..W {
            for ch in 0..RGBA {
                let d = grid[[r, c, ch]] - target[[r, c, ch]];
                s += d * d;
            }
        }
    }
    s / n
}

// f64 MSE — for the gradient check (f32 cancels two near-equal losses).
pub fn mse_loss_f64(grid: &Grid, target: &Array3<f32>) -> f64 {
    let n = (H * W * RGBA) as f64;
    let mut s = 0.0f64;
    for r in 0..H {
        for c in 0..W {
            for ch in 0..RGBA {
                let d = grid[[r, c, ch]] as f64 - target[[r, c, ch]] as f64;
                s += d * d;
            }
        }
    }
    s / n
}

pub fn mse_grad(grid: &Grid, target: &Array3<f32>) -> Grid {
    let mut d = new_grid();
    let n = (H * W * RGBA) as f32;
    for r in 0..H {
        for c in 0..W {
            for ch in 0..RGBA {
                d[[r, c, ch]] = 2.0 * (grid[[r, c, ch]] - target[[r, c, ch]]) / n;
            }
        }
    }
    d
}

// Backprop one step: accumulate weight grads, return d_in (grad wrt the input).
// Mirrors `update`: the two layers are backpropagated for all cells at once.
pub fn backward_step(
    net: &NCANet,
    cache: &StepCache,
    d_out: &Grid,
    grads: &mut Grads,
    seed: &[bool],
) -> Grid {
    let mut d_in = new_grid();
    // grad wrt the network's delta, one row per cell; zero where the cell did
    // not fire and on the frozen conditioning channels
    let mut d_delta = Array2::<f32>::zeros((H * W, C));
    for idx in 0..H * W {
        let (r, c) = (idx / W, idx % W);
        let l = cache.life[[r, c]]; // frozen alive gate
        for ch in 0..C {
            // seed cells: alpha + one-hot are pinned -> zero upstream grad
            let pinned = seed[idx] && (ch == ALPHA || is_cond(ch));
            let d = if pinned { 0.0 } else { d_out[[r, c, ch]] * l };
            d_in[[r, c, ch]] = d; // residual passthrough
            if cache.masks[idx] && !is_cond(ch) {
                d_delta[[idx, ch]] = d;
            }
        }
    }
    // layer2
    grads.b2 += &d_delta.sum_axis(Axis(0));
    grads.w2 += &d_delta.t().dot(&cache.hidden);
    // ReLU
    let mut d_pre = d_delta.dot(&net.w2);
    Zip::from(&mut d_pre).and(&cache.hidden).for_each(|d, &h| {
        if h <= 0.0 {
            *d = 0.0;
        }
    });
    // layer1
    grads.b1 += &d_pre.sum_axis(Axis(0));
    grads.w1 += &d_pre.t().dot(&cache.perceptions);
    let d_perc = d_pre.dot(&net.w1);
    // scatter d_perc back through perception
    for idx in 0..H * W {
        if !cache.masks[idx] {
            continue; // row is all zero
        }
        let (r, c) = (idx / W, idx % W);
        // identity -> self
        for ch in 0..C {
            d_in[[r, c, ch]] += d_perc[[idx, ch]];
        }
        // sobel -> 3x3 neighbors
        for ki in 0..3 {
            for kj in 0..3 {
                let nr = r as i32 + ki as i32 - 1;
                let nc = c as i32 + kj as i32 - 1;
                if nr < 0 || nr >= H as i32 || nc < 0 || nc >= W as i32 {
                    continue;
                }
                let (nr, nc) = (nr as usize, nc as usize);
                let wx = SOBEL_X[ki][kj];
                let wy = SOBEL_Y[ki][kj];
                for ch in 0..C {
                    d_in[[nr, nc, ch]] +=
                        d_perc[[idx, C + ch]] * wx + d_perc[[idx, 2 * C + ch]] * wy;
                }
            }
        }
    }
    d_in
}

// Returns (grads, d_grid0). d_grid0 is used by the gradient check, ignored in training.
pub fn backward(net: &NCANet, caches: &[StepCache], d_final: Grid, digit: usize) -> (Grads, Grid) {
    let seed = seed_mask(digit);
    let mut grads = Grads::zero();
    let mut d = d_final;
    for cache in caches.iter().rev() {
        d = backward_step(net, cache, &d, &mut grads, &seed);
    }
    (grads, d)
}
