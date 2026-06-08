// Forward-with-cache, MSE loss, truncated BPTT.
// One step: new = life * (grid + mask*delta), then seed re-injection.

use crate::constants::*;
use crate::digits::seed_cells;
use crate::grid::{alive_mask, new_grid, Grid};
use crate::network::{Grads, NCANet};
use crate::perception::{perceive, SOBEL_X, SOBEL_Y};
use ndarray::{Array2, Array3};
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
    pub perceptions: Vec<[f32; PERC]>,
    pub pre_relu: Vec<[f32; HIDDEN]>,
    pub life: Array2<f32>, // frozen alive gate
}

pub fn step_cache(net: &NCANet, grid: &Grid, masks: &[bool], digit: usize) -> (Grid, StepCache) {
    let mut new = grid.clone();
    let mut perceptions = vec![[0.0f32; PERC]; H * W];
    let mut pre_relu = vec![[0.0f32; HIDDEN]; H * W];
    for r in 0..H {
        for c in 0..W {
            let idx = r * W + c;
            let p = perceive(grid, r, c);
            let (pre, mut delta) = net.forward(&p);
            perceptions[idx] = p;
            pre_relu[idx] = pre;
            for ch in COND_LO..COND_HI {
                delta[ch] = 0.0; // freeze conditioning
            }
            if masks[idx] {
                for ch in 0..C {
                    new[[r, c, ch]] += delta[ch];
                }
            }
        }
    }
    let life = alive_mask(&new);
    for r in 0..H {
        for c in 0..W {
            let l = life[[r, c]];
            for ch in 0..C {
                new[[r, c, ch]] *= l;
            }
        }
    }
    reinject_seed(&mut new, digit);
    (
        new,
        StepCache {
            masks: masks.to_vec(),
            perceptions,
            pre_relu,
            life,
        },
    )
}

// Forward step without caching.
pub fn step_only(net: &NCANet, grid: &Grid, masks: &[bool], digit: usize) -> Grid {
    let mut new = grid.clone();
    for r in 0..H {
        for c in 0..W {
            let idx = r * W + c;
            let p = perceive(grid, r, c);
            let (_, mut delta) = net.forward(&p);
            for ch in COND_LO..COND_HI {
                delta[ch] = 0.0;
            }
            if masks[idx] {
                for ch in 0..C {
                    new[[r, c, ch]] += delta[ch];
                }
            }
        }
    }
    let life = alive_mask(&new);
    for r in 0..H {
        for c in 0..W {
            let l = life[[r, c]];
            for ch in 0..C {
                new[[r, c, ch]] *= l;
            }
        }
    }
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
    let mut new = grid.clone();
    for r in 0..H {
        for c in 0..W {
            let idx = r * W + c;
            let p = perceive(grid, r, c);
            let (_, mut delta) = net.forward(&p);
            for ch in COND_LO..COND_HI {
                delta[ch] = 0.0;
            }
            if masks[idx] {
                for ch in 0..C {
                    new[[r, c, ch]] += delta[ch];
                }
            }
        }
    }
    for r in 0..H {
        for c in 0..W {
            let l = life[[r, c]];
            for ch in 0..C {
                new[[r, c, ch]] *= l;
            }
        }
    }
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
pub fn backward_step(
    net: &NCANet,
    cache: &StepCache,
    d_out: &Grid,
    grads: &mut Grads,
    seed: &[bool],
) -> Grid {
    let mut d_in = new_grid();
    for r in 0..H {
        for c in 0..W {
            let idx = r * W + c;
            let l = cache.life[[r, c]]; // frozen alive gate
            let mut d_new = [0.0f32; C];
            for ch in 0..C {
                d_new[ch] = d_out[[r, c, ch]] * l;
            }
            // seed cells: alpha + one-hot are pinned -> zero upstream grad
            if seed[idx] {
                d_new[ALPHA] = 0.0;
                for ch in COND_LO..COND_HI {
                    d_new[ch] = 0.0;
                }
            }
            // residual passthrough
            for ch in 0..C {
                d_in[[r, c, ch]] += d_new[ch];
            }
            if !cache.masks[idx] {
                continue;
            }
            let mut d_delta = d_new;
            for ch in COND_LO..COND_HI {
                d_delta[ch] = 0.0; // conditioning frozen
            }
            let pre = &cache.pre_relu[idx];
            // layer2
            let mut dh = [0.0f32; HIDDEN];
            for k in 0..C {
                let dd = d_delta[k];
                grads.b2[k] += dd;
                let w2row = net.w2.row(k);
                for i in 0..HIDDEN {
                    let h = if pre[i] > 0.0 { pre[i] } else { 0.0 };
                    grads.w2[[k, i]] += dd * h;
                    dh[i] += w2row[i] * dd;
                }
            }
            // ReLU
            let mut d_pre = [0.0f32; HIDDEN];
            for i in 0..HIDDEN {
                d_pre[i] = if pre[i] > 0.0 { dh[i] } else { 0.0 };
            }
            // layer1
            let p = &cache.perceptions[idx];
            let mut d_perc = [0.0f32; PERC];
            for i in 0..HIDDEN {
                let dp = d_pre[i];
                grads.b1[i] += dp;
                let w1row = net.w1.row(i);
                for j in 0..PERC {
                    grads.w1[[i, j]] += dp * p[j];
                    d_perc[j] += w1row[j] * dp;
                }
            }
            // scatter d_perc back through perception: identity -> self
            for ch in 0..C {
                d_in[[r, c, ch]] += d_perc[ch];
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
                        d_in[[nr, nc, ch]] += d_perc[C + ch] * wx + d_perc[2 * C + ch] * wy;
                    }
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
