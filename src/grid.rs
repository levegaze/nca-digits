// Grid type plus seed initialization and the alive mask.

use crate::constants::*;
use crate::digits::seed_cells;
use ndarray::{Array2, Array3};

// [H, W, C] dense state. RGBA in 0..4, one-hot in 4..14, hidden in 14..16.
pub type Grid = Array3<f32>;

pub fn new_grid() -> Grid {
    Array3::zeros((H, W, C))
}

// Seed the glyph-mass cells: RGB=1, alpha=1, digit one-hot.
pub fn seed_init(digit: usize) -> Grid {
    let mut g = new_grid();
    for (r, c) in seed_cells(digit) {
        g[[r, c, 0]] = 1.0; // R
        g[[r, c, 1]] = 1.0; // G
        g[[r, c, 2]] = 1.0; // B
        g[[r, c, ALPHA]] = 1.0;
        g[[r, c, COND_LO + digit]] = 1.0;
    }
    g
}

// 0/1 mask: alive if any 3x3 neighbor has alpha >= ALIVE_THRESH (max-pool).
pub fn alive_mask(g: &Grid) -> Array2<f32> {
    let mut m = Array2::zeros((H, W));
    for r in 0..H {
        for c in 0..W {
            let mut alive = false;
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < H as i32 && nc >= 0 && nc < W as i32 {
                        if g[[nr as usize, nc as usize, ALPHA]] >= ALIVE_THRESH {
                            alive = true;
                        }
                    }
                }
            }
            m[[r, c]] = if alive { 1.0 } else { 0.0 };
        }
    }
    m
}
