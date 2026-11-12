// NCANet: Linear(PERC->HIDDEN) + ReLU -> Linear(HIDDEN->C). Perception -> delta.

use crate::constants::*;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::io::{Read, Write};

#[derive(Clone)]
pub struct NCANet {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

#[derive(Clone)]
pub struct Grads {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

impl Grads {
    pub fn zero() -> Self {
        Grads {
            w1: Array2::zeros((HIDDEN, PERC)),
            b1: Array1::zeros(HIDDEN),
            w2: Array2::zeros((C, HIDDEN)),
            b2: Array1::zeros(C),
        }
    }

    pub fn add(&mut self, o: &Grads) {
        self.w1 += &o.w1;
        self.b1 += &o.b1;
        self.w2 += &o.w2;
        self.b2 += &o.b2;
    }

    pub fn scale(&mut self, s: f32) {
        self.w1 *= s;
        self.b1 *= s;
        self.w2 *= s;
        self.b2 *= s;
    }
}

impl NCANet {
    // layer1 He-ish, layer2 small-random (pure-zero w2 deadlocks: no growth gradient).
    pub fn init<R: Rng>(rng: &mut R) -> Self {
        let s1 = (2.0 / PERC as f32).sqrt();
        let s2 = 0.03; // larger blows up the 64-step rollout
        NCANet {
            w1: Array2::from_shape_fn((HIDDEN, PERC), |_| (rng.r#gen::<f32>() * 2.0 - 1.0) * s1),
            b1: Array1::zeros(HIDDEN),
            w2: Array2::from_shape_fn((C, HIDDEN), |_| (rng.r#gen::<f32>() * 2.0 - 1.0) * s2),
            b2: Array1::zeros(C),
        }
    }

    // Gradient-check init: all layers small-random (w2 != 0, or the check is trivial).
    pub fn init_random<R: Rng>(rng: &mut R) -> Self {
        let mk = |rng: &mut R, rows, cols, s: f32| {
            Array2::from_shape_fn((rows, cols), |_| (rng.r#gen::<f32>() * 2.0 - 1.0) * s)
        };
        // w2 ~ 1/HIDDEN keeps delta in the near-linear regime the check can verify
        let s2 = 16.0 / HIDDEN as f32;
        NCANet {
            w1: mk(rng, HIDDEN, PERC, 0.3),
            b1: Array1::from_shape_fn(HIDDEN, |_| (rng.r#gen::<f32>() * 2.0 - 1.0) * 0.05),
            w2: mk(rng, C, HIDDEN, s2),
            b2: Array1::from_shape_fn(C, |_| (rng.r#gen::<f32>() * 2.0 - 1.0) * 0.05),
        }
    }

    // All cells at once: `p` is [N, PERC], one row per cell. Returns
    // (hidden [N, HIDDEN], delta [N, C]); hidden is the post-ReLU activation,
    // cached for backward (hidden > 0 <=> the ReLU passed gradient).
    pub fn forward_batch(&self, p: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let mut hidden = p.dot(&self.w1.t()) + &self.b1;
        hidden.mapv_inplace(|x| if x > 0.0 { x } else { 0.0 }); // ReLU
        let delta = hidden.dot(&self.w2.t()) + &self.b2;
        (hidden, delta)
    }

    pub fn add_scaled(&mut self, g: &Grads, scale: f32) {
        self.w1.scaled_add(scale, &g.w1);
        self.b1.scaled_add(scale, &g.b1);
        self.w2.scaled_add(scale, &g.w2);
        self.b2.scaled_add(scale, &g.b2);
    }

    // Raw little-endian f32: w1, b1, w2, b2.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        for &v in self
            .w1
            .iter()
            .chain(self.b1.iter())
            .chain(self.w2.iter())
            .chain(self.b2.iter())
        {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<NCANet> {
        let mut f = std::fs::File::open(path)?;
        // reject a checkpoint from a different architecture
        let expected = (HIDDEN * PERC + HIDDEN + C * HIDDEN + C) * 4;
        let actual = f.metadata()?.len() as usize;
        if actual != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("weights size {actual} != expected {expected} (HIDDEN mismatch?)"),
            ));
        }
        let mut read_n = |n: usize| -> std::io::Result<Vec<f32>> {
            let mut out = Vec::with_capacity(n);
            let mut buf = [0u8; 4];
            for _ in 0..n {
                f.read_exact(&mut buf)?;
                out.push(f32::from_le_bytes(buf));
            }
            Ok(out)
        };
        let w1 = Array2::from_shape_vec((HIDDEN, PERC), read_n(HIDDEN * PERC)?).unwrap();
        let b1 = Array1::from_vec(read_n(HIDDEN)?);
        let w2 = Array2::from_shape_vec((C, HIDDEN), read_n(C * HIDDEN)?).unwrap();
        let b2 = Array1::from_vec(read_n(C)?);
        Ok(NCANet { w1, b1, w2, b2 })
    }
}
