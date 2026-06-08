pub mod backward;
pub mod constants;
pub mod digits;
pub mod grid;
pub mod network;
pub mod optimizer;
pub mod perception;
pub mod render;
pub mod replay;

use backward::{backward, mse_grad, mse_loss_f64, rollout_cache, rollout_frozen};
use constants::*;
use grid::seed_init;
use ndarray::Array3;
use network::NCANet;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// Finite-difference gradient check (random weights, frozen masks + life gate,
// f64 loss, no truncation). Returns the worst relative error over the samples.
pub fn gradient_check() -> f32 {
    const K: usize = 6;
    let mut rng = StdRng::seed_from_u64(42);

    let net = NCANet::init_random(&mut rng);
    let digit = 3;
    let grid0 = seed_init(digit);

    let masks: Vec<Vec<bool>> = (0..K)
        .map(|_| (0..H * W).map(|_| rng.gen_bool(0.5)).collect())
        .collect();
    let target = Array3::from_shape_fn((H, W, RGBA), |_| rng.r#gen::<f32>());

    let (final_grid, caches) = rollout_cache(&net, &grid0, &masks, digit);
    let d_final = mse_grad(&final_grid, &target);
    let (grads, d_grid0) = backward(&net, &caches, d_final, digit);
    let lifes: Vec<_> = caches.iter().map(|c| c.life.clone()).collect();

    let eps = 1e-3f32;
    let mut worst = 0.0f32;

    let mut samples: Vec<(&str, usize, usize)> = Vec::new();
    for i in (0..HIDDEN).step_by(31) {
        for j in (0..PERC).step_by(17) {
            samples.push(("w1", i, j));
        }
    }
    for i in (0..HIDDEN).step_by(29) {
        samples.push(("b1", i, 0));
    }
    for k in 0..C {
        for i in (0..HIDDEN).step_by(37) {
            samples.push(("w2", k, i));
        }
    }
    for k in 0..C {
        samples.push(("b2", k, 0));
    }

    for (tensor, a, b) in samples {
        let mut np = net.clone();
        let mut nm = net.clone();
        let analytic = match tensor {
            "w1" => {
                np.w1[[a, b]] += eps;
                nm.w1[[a, b]] -= eps;
                grads.w1[[a, b]]
            }
            "b1" => {
                np.b1[a] += eps;
                nm.b1[a] -= eps;
                grads.b1[a]
            }
            "w2" => {
                np.w2[[a, b]] += eps;
                nm.w2[[a, b]] -= eps;
                grads.w2[[a, b]]
            }
            "b2" => {
                np.b2[a] += eps;
                nm.b2[a] -= eps;
                grads.b2[a]
            }
            _ => unreachable!(),
        };
        let lp = mse_loss_f64(&rollout_frozen(&np, &grid0, &masks, &lifes, digit), &target);
        let lm = mse_loss_f64(&rollout_frozen(&nm, &grid0, &masks, &lifes, digit), &target);
        let numeric = ((lp - lm) / (2.0 * eps as f64)) as f32;

        if analytic.abs() < 1e-5 {
            continue; // below the f32 finite-diff floor
        }
        let denom = numeric.abs() + analytic.abs();
        let rel = (numeric - analytic).abs() / denom;
        if rel > 0.05 {
            eprintln!(
                "  {tensor}[{a},{b}] num={numeric:+.3e} ana={analytic:+.3e} rel={rel:.2e}"
            );
        }
        if rel > worst {
            worst = rel;
        }
    }

    // d_in check: perturb grid0 cells (cross-cell sobel scatter). Alpha excluded
    // (it would flip the frozen alive gate).
    let cr = H / 2;
    let cc = W / 2;
    let cells = [
        (cr, cc),
        (cr, cc + 1),
        (cr - 1, cc),
        (cr + 1, cc - 1),
        (cr + 2, cc + 2),
        (cr - 3, cc + 1),
        (cr, cc + 4),
    ];
    let channels = [0usize, 1, 2, 7, 14, 15]; // RGB, one conditioning, both hidden
    for (r, c) in cells {
        for ch in channels {
            let analytic = d_grid0[[r, c, ch]];
            if analytic.abs() < 5e-4 {
                continue; // cross-cell grads are noise-dominated below this
            }
            let mut gp = grid0.clone();
            let mut gm = grid0.clone();
            gp[[r, c, ch]] += eps;
            gm[[r, c, ch]] -= eps;
            let lp = mse_loss_f64(&rollout_frozen(&net, &gp, &masks, &lifes, digit), &target);
            let lm = mse_loss_f64(&rollout_frozen(&net, &gm, &masks, &lifes, digit), &target);
            let numeric = ((lp - lm) / (2.0 * eps as f64)) as f32;
            let rel = (numeric - analytic).abs() / (numeric.abs() + analytic.abs());
            if rel > 0.05 {
                eprintln!(
                    "  d_in[{r},{c},{ch}] num={numeric:+.3e} ana={analytic:+.3e} rel={rel:.2e}"
                );
            }
            if rel > worst {
                worst = rel;
            }
        }
    }

    worst
}

#[cfg(test)]
mod tests {
    use super::gradient_check;

    #[test]
    fn backward_matches_finite_difference() {
        let worst = gradient_check();
        assert!(worst < 5e-2, "gradient check worst rel error = {worst:.2e}");
    }
}
