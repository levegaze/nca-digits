// Experience-replay pool of (digit, grid) states. Training samples states from
// the pool, evolves them, and writes the results back — so the model learns to
// grow from both fresh seeds and partially-grown states (stability).

use crate::constants::*;
use crate::grid::{seed_init, Grid};
use rand::Rng;

pub struct ReplayPool {
    pub items: Vec<(usize, Grid)>, // (digit, state)
}

impl ReplayPool {
    // Fill with fresh seeds, digits assigned round-robin.
    pub fn new(size: usize) -> Self {
        let items = (0..size).map(|i| {
            let d = i % N_DIGITS;
            (d, seed_init(d))
        }).collect();
        ReplayPool { items }
    }

    // Sample `batch` distinct pool indices.
    pub fn sample<R: Rng>(&self, batch: usize, rng: &mut R) -> Vec<usize> {
        let mut idx = Vec::with_capacity(batch);
        while idx.len() < batch {
            let i = rng.gen_range(0..self.items.len());
            if !idx.contains(&i) {
                idx.push(i);
            }
        }
        idx
    }

    // Reset a slot to a fresh seed (keeps the same digit).
    pub fn reset(&mut self, i: usize) {
        let d = self.items[i].0;
        self.items[i].1 = seed_init(d);
    }

    // Write an evolved state back into a slot.
    pub fn put(&mut self, i: usize, grid: Grid) {
        self.items[i].1 = grid;
    }
}
