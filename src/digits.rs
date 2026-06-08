// Digit targets from a built-in 5x7 font, upscaled 4x to 20x28 and centered
// horizontally in the 28x28 grid (stroke width ~4px).

use crate::constants::*;
use ndarray::Array3;

// Each digit is 7 rows; each row uses the low 5 bits, MSB = leftmost pixel.
pub const FONT: [[u8; 7]; N_DIGITS] = [
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110], // 0
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // 1
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111], // 2
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110], // 3
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010], // 4
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110], // 5
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110], // 6
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000], // 7
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110], // 8
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100], // 9
];

const SCALE: usize = 4;
const GW: usize = 5 * SCALE; // glyph width  = 20
const GH: usize = 7 * SCALE; // glyph height = 28
const COL0: usize = (W - GW) / 2; // 4
const ROW0: usize = (H - GH) / 2; // 0

// Number of on-pixels nearest the center used as the seed mass.
pub const SEED_K: usize = 9;

// Glyph-mass seed: the SEED_K on-pixels nearest the center (on-target -> growth
// gradient; a single center cell is off-target for 0/8/middle-of-3).
pub fn seed_cells(digit: usize) -> Vec<(usize, usize)> {
    let (cr, cc) = (H / 2, W / 2);
    let mut on: Vec<(usize, usize)> = Vec::new();
    for r in 0..H {
        for c in 0..W {
            if pixel_on(digit, r, c) {
                on.push((r, c));
            }
        }
    }
    let d2 = |&(r, c): &(usize, usize)| {
        let dr = r as i32 - cr as i32;
        let dc = c as i32 - cc as i32;
        (dr * dr + dc * dc) as i64
    };
    on.sort_by_key(d2);
    on.truncate(SEED_K);
    on
}

// True if the upscaled glyph for `digit` is on at grid cell (r, c).
fn pixel_on(digit: usize, r: usize, c: usize) -> bool {
    if r < ROW0 || r >= ROW0 + GH || c < COL0 || c >= COL0 + GW {
        return false;
    }
    let fr = (r - ROW0) / SCALE; // 0..7
    let fc = (c - COL0) / SCALE; // 0..5
    let row = FONT[digit][fr];
    (row >> (4 - fc)) & 1 == 1
}

// Target RGBA grid for a digit: on pixels -> (1,1,1,1), off -> 0. Shape [H,W,RGBA].
pub fn target_rgba(digit: usize) -> Array3<f32> {
    let mut t = Array3::zeros((H, W, RGBA));
    for r in 0..H {
        for c in 0..W {
            if pixel_on(digit, r, c) {
                for ch in 0..RGBA {
                    t[[r, c, ch]] = 1.0;
                }
            }
        }
    }
    t
}
