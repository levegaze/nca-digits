// Adam; gradients clamped element-wise to [-GRAD_CLIP, GRAD_CLIP].

use crate::constants::*;
use crate::network::{Grads, NCANet};
use ndarray::{Array1, Array2, Zip};

pub struct Adam {
    m: Grads,
    v: Grads,
    t: i32,
    pub lr: f32, // settable for LR scheduling (defaults to LR)
}

impl Adam {
    pub fn new() -> Self {
        Adam {
            m: Grads::zero(),
            v: Grads::zero(),
            t: 0,
            lr: LR,
        }
    }

    pub fn step(&mut self, net: &mut NCANet, g: &Grads) {
        self.t += 1;
        let bc1 = 1.0 - BETA1.powi(self.t);
        let bc2 = 1.0 - BETA2.powi(self.t);
        let lr = self.lr;
        update2(&mut net.w1, &g.w1, &mut self.m.w1, &mut self.v.w1, bc1, bc2, lr);
        update1(&mut net.b1, &g.b1, &mut self.m.b1, &mut self.v.b1, bc1, bc2, lr);
        update2(&mut net.w2, &g.w2, &mut self.m.w2, &mut self.v.w2, bc1, bc2, lr);
        update1(&mut net.b2, &g.b2, &mut self.m.b2, &mut self.v.b2, bc1, bc2, lr);
    }
}

fn adam_elem(w: &mut f32, g: f32, m: &mut f32, v: &mut f32, bc1: f32, bc2: f32, lr: f32) {
    let g = g.clamp(-GRAD_CLIP, GRAD_CLIP);
    *m = BETA1 * *m + (1.0 - BETA1) * g;
    *v = BETA2 * *v + (1.0 - BETA2) * g * g;
    let mhat = *m / bc1;
    let vhat = *v / bc2;
    *w -= lr * mhat / (vhat.sqrt() + EPS);
}

fn update2(w: &mut Array2<f32>, g: &Array2<f32>, m: &mut Array2<f32>, v: &mut Array2<f32>, bc1: f32, bc2: f32, lr: f32) {
    Zip::from(w).and(g).and(m).and(v).for_each(|w, &g, m, v| adam_elem(w, g, m, v, bc1, bc2, lr));
}

fn update1(w: &mut Array1<f32>, g: &Array1<f32>, m: &mut Array1<f32>, v: &mut Array1<f32>, bc1: f32, bc2: f32, lr: f32) {
    Zip::from(w).and(g).and(m).and(v).for_each(|w, &g, m, v| adam_elem(w, g, m, v, bc1, bc2, lr));
}
