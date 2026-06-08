// ASCII rendering of a grid. Alpha channel (ch 3) decides alive/dead; the R
// channel decides shade.

use crate::constants::*;
use crate::grid::Grid;

const SHADES: [char; 5] = [' ', '.', ':', '#', '@'];

pub fn print_grid(grid: &Grid, digit: usize, step: usize, loss: f32) {
    println!("digit: {digit}  |  step: {step}  |  loss: {loss:.4}");
    for r in 0..H {
        let mut line = String::with_capacity(W * 2);
        for c in 0..W {
            let alive = grid[[r, c, ALPHA]] >= ALIVE_THRESH;
            let ch = if !alive {
                SHADES[0]
            } else {
                let shade = (grid[[r, c, 0]].clamp(0.0, 1.0) * 4.0).round() as usize;
                SHADES[shade.min(4)]
            };
            line.push(ch);
            line.push(ch); // double-width for roughly square aspect
        }
        println!("{line}");
    }
}
