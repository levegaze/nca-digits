// Per-cell perception: identity + Sobel-x + Sobel-y over all channels.
// Out-of-bounds neighbors are treated as zero (zero padding).

use crate::constants::*;
use crate::grid::Grid;

// Sobel kernels scaled by 1/8 (the Growing-CA convention).
pub const SOBEL_X: [[f32; 3]; 3] = [
    [-1.0 / 8.0, 0.0, 1.0 / 8.0],
    [-2.0 / 8.0, 0.0, 2.0 / 8.0],
    [-1.0 / 8.0, 0.0, 1.0 / 8.0],
];
pub const SOBEL_Y: [[f32; 3]; 3] = [
    [-1.0 / 8.0, -2.0 / 8.0, -1.0 / 8.0],
    [0.0, 0.0, 0.0],
    [1.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0],
];

// perception layout: [0..C] = self, [C..2C] = d/dx, [2C..3C] = d/dy.
pub fn perceive(g: &Grid, r: usize, c: usize) -> [f32; PERC] {
    let mut p = [0.0f32; PERC];
    // identity: this cell's channels
    for ch in 0..C {
        p[ch] = g[[r, c, ch]];
    }
    // sobel filters over the 3x3 neighborhood
    for ki in 0..3 {
        for kj in 0..3 {
            let nr = r as i32 + ki as i32 - 1;
            let nc = c as i32 + kj as i32 - 1;
            if nr < 0 || nr >= H as i32 || nc < 0 || nc >= W as i32 {
                continue; // zero padding
            }
            let (nr, nc) = (nr as usize, nc as usize);
            let wx = SOBEL_X[ki][kj];
            let wy = SOBEL_Y[ki][kj];
            for ch in 0..C {
                let v = g[[nr, nc, ch]];
                p[C + ch] += wx * v;
                p[2 * C + ch] += wy * v;
            }
        }
    }
    p
}
