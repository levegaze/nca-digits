// Visualizer: grow a digit one CA step per tick; D damages it (self-repair).
// Run: cargo run --release --features viz --bin viz  (loads weights.bin)

use macroquad::prelude::*;
use nca_digits::backward::{random_mask, step_only};
use nca_digits::constants::*;
use nca_digits::digits::target_rgba;
use nca_digits::grid::{seed_init, Grid};
use nca_digits::network::NCANet;
use ndarray::Array3;
// `::rand` — macroquad's prelude re-exports its own `rand`, so disambiguate.
use ::rand::rngs::StdRng;
use ::rand::{Rng, SeedableRng};

const CELL: f32 = 18.0; // pixels per grid cell
const MARGIN: f32 = 40.0;
const HUD_H: f32 = 70.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "NCA digits — growing".to_owned(),
        window_width: (W as f32 * CELL + MARGIN * 2.0) as i32,
        window_height: (H as f32 * CELL + MARGIN * 2.0 + HUD_H) as i32,
        window_resizable: false,
        ..Default::default()
    }
}

struct Sim {
    net: NCANet,
    grid: Grid,
    target: Array3<f32>,
    digit: usize,
    step: usize,
    rng: StdRng,
}

impl Sim {
    fn reset(&mut self, digit: usize) {
        self.digit = digit;
        self.grid = seed_init(digit);
        self.target = target_rgba(digit);
        self.step = 0;
    }

    fn tick(&mut self) {
        let m = random_mask(&mut self.rng);
        self.grid = step_only(&self.net, &self.grid, &m, self.digit);
        self.step += 1;
    }

    // Erase ~30% of cells to demonstrate self-repair.
    fn damage(&mut self) {
        for r in 0..H {
            for c in 0..W {
                if self.rng.r#gen::<f32>() < 0.30 {
                    for ch in 0..C {
                        self.grid[[r, c, ch]] = 0.0;
                    }
                }
            }
        }
    }

    fn alive_count(&self) -> usize {
        let mut n = 0;
        for r in 0..H {
            for c in 0..W {
                if self.grid[[r, c, ALPHA]] >= ALIVE_THRESH {
                    n += 1;
                }
            }
        }
        n
    }
}

const DIGIT_KEYS: [KeyCode; 10] = [
    KeyCode::Key0, KeyCode::Key1, KeyCode::Key2, KeyCode::Key3, KeyCode::Key4,
    KeyCode::Key5, KeyCode::Key6, KeyCode::Key7, KeyCode::Key8, KeyCode::Key9,
];

#[macroquad::main(window_conf)]
async fn main() {
    let net = match NCANet::load("weights.bin") {
        Ok(n) => n,
        Err(e) => {
            loop {
                clear_background(Color::new(0.04, 0.05, 0.08, 1.0));
                draw_text(
                    &format!("could not load weights.bin ({e})"),
                    20.0, 60.0, 24.0, RED,
                );
                draw_text("train first:  cargo run --release -- train 10000", 20.0, 95.0, 22.0, WHITE);
                if is_key_pressed(KeyCode::Escape) {
                    return;
                }
                next_frame().await;
            }
        }
    };

    let mut sim = Sim {
        net,
        grid: seed_init(0),
        target: target_rgba(0),
        digit: 0,
        step: 0,
        rng: StdRng::seed_from_u64(7),
    };

    let mut paused = false;
    let mut looping = true; // auto-replay the growth after it settles
    let mut speed = 8.0f32; // CA steps per second
    let mut acc = 0.0f32;
    let mut hold = 0.0f32; // time held after reaching STEPS
    const HOLD_SECS: f32 = 1.5;

    loop {
        // ---- input ----
        for (d, &k) in DIGIT_KEYS.iter().enumerate() {
            if is_key_pressed(k) {
                sim.reset(d);
            }
        }
        if is_key_pressed(KeyCode::Up) {
            sim.reset((sim.digit + 1) % N_DIGITS);
        }
        if is_key_pressed(KeyCode::Down) {
            sim.reset((sim.digit + N_DIGITS - 1) % N_DIGITS);
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::N) {
            sim.tick(); // manual step, any time
        }
        if is_key_pressed(KeyCode::R) {
            sim.reset(sim.digit);
        }
        if is_key_pressed(KeyCode::L) {
            looping = !looping;
        }
        if is_key_pressed(KeyCode::D) {
            sim.damage();
        }
        if is_key_pressed(KeyCode::Minus) {
            speed = (speed - 1.0).max(1.0);
        }
        if is_key_pressed(KeyCode::Equal) {
            speed = (speed + 1.0).min(60.0);
        }
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        // ---- step ----
        // grow only to STEPS (past it the no-persistence net drifts), then hold/loop
        if !paused {
            if sim.step < STEPS {
                acc += get_frame_time();
                let dur = 1.0 / speed;
                while acc >= dur && sim.step < STEPS {
                    sim.tick();
                    acc -= dur;
                }
            } else {
                acc = 0.0;
                hold += get_frame_time();
                if looping && hold >= HOLD_SECS {
                    sim.reset(sim.digit);
                    hold = 0.0;
                }
            }
        }

        draw(&sim, paused, speed, looping);
        next_frame().await;
    }
}

fn draw(sim: &Sim, paused: bool, speed: f32, looping: bool) {
    clear_background(Color::new(0.04, 0.05, 0.08, 1.0));

    for r in 0..H {
        for c in 0..W {
            let x = MARGIN + c as f32 * CELL;
            let y = MARGIN + r as f32 * CELL;

            // faint target ghost
            if sim.target[[r, c, ALPHA]] > 0.5 {
                draw_rectangle_lines(x, y, CELL, CELL, 1.0, Color::new(0.2, 0.25, 0.32, 1.0));
            }

            let a = sim.grid[[r, c, ALPHA]];
            if a < ALIVE_THRESH {
                continue;
            }
            // R intensity -> cyan/white; alpha -> opacity
            let i = sim.grid[[r, c, 0]].clamp(0.0, 1.0);
            let col = Color::new(0.25 * i, 0.85 * i, i.max(0.15), a.clamp(0.0, 1.0));
            draw_rectangle(x, y, CELL - 1.0, CELL - 1.0, col);
        }
    }

    // border around the field
    draw_rectangle_lines(
        MARGIN - 1.0, MARGIN - 1.0,
        W as f32 * CELL + 2.0, H as f32 * CELL + 2.0,
        2.0, Color::new(0.2, 0.5, 0.7, 1.0),
    );

    // ---- HUD ----
    let hud_y = MARGIN + H as f32 * CELL + 24.0;
    let phase = if sim.step < STEPS {
        "growing"
    } else if paused {
        "paused"
    } else if looping {
        "settled (loop)"
    } else {
        "settled"
    };
    let info = format!(
        "digit {} | step {}/{} | alive {} | {} | {:.0} steps/s",
        sim.digit,
        sim.step,
        STEPS,
        sim.alive_count(),
        phase,
        speed,
    );
    draw_text(&info, MARGIN, hud_y, 24.0, Color::new(0.8, 0.9, 1.0, 1.0));
    let help = "0-9/Up/Down:digit  Space:pause  N:step  R:reset  L:loop  D:damage  -/=:speed  Esc:quit";
    draw_text(help, MARGIN, hud_y + 26.0, 17.0, Color::new(0.55, 0.62, 0.72, 1.0));
}
