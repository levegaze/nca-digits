// All grid dimensions, network sizes, and training hyperparameters.

pub const H: usize = 28; // grid height
pub const W: usize = 28; // grid width
pub const C: usize = 16; // channels per cell
pub const PERC: usize = C * 3; // perception vector = 48 (identity + sobel_x + sobel_y)
pub const HIDDEN: usize = 256; // NN hidden layer size (was 128; raised for multi-digit capacity)
// Parameters: 48*256 + 256 + 256*16 + 16 = 16,656

pub const N_DIGITS: usize = 10;
pub const STEPS: usize = 64; // forward steps per training iteration
pub const BPTT_K: usize = 16; // truncated BPTT depth (k=32 gave only a marginal gain at 2x
                              // cost; capacity is the bigger lever, so keep k=16 for throughput)
pub const POOL_SIZE: usize = 1024; // experience replay pool
pub const BATCH: usize = 8; // seeds per training step

pub const LR: f32 = 2e-4;
pub const BETA1: f32 = 0.9;
pub const BETA2: f32 = 0.999;
pub const EPS: f32 = 1e-8;
pub const GRAD_CLIP: f32 = 1.0;
pub const ALIVE_THRESH: f32 = 0.1; // alpha channel cutoff

// Channel roles: 0-2 RGB, 3 alpha, 4..14 digit one-hot, 14-15 hidden.
pub const ALPHA: usize = 3;
pub const COND_LO: usize = 4; // first conditioning channel
pub const COND_HI: usize = 14; // one past last conditioning channel
pub const RGBA: usize = 4; // R,G,B,A are the visible/target channels
