#![allow(non_snake_case)]
#![windows_subsystem = "windows"]

mod problems {
    pub mod naProblem;
    pub mod cartpoleProblem;
    pub mod sphereContinuousProblem;
    pub mod sphereDiscreteProblem;
}
mod utils {
    pub mod interval;
    pub mod mathUtils;
}
mod mutationAlgorithms {
    pub mod mutationAlgorithm;
    pub mod gaussianMutation;
    pub mod localOnePlusOneNA;
    pub mod SAOnePlusOneNA;
    pub mod SALocalMutation;
    pub mod onePlusOneNA;
}

pub mod ui;

mod nn {
    pub mod ann;
}
use druid::{AppLauncher,WindowDesc};
use ui::{AppState, build_ui};

static mut R: f64 = 240.0;
static mut OPTIMUM: f64 = 1.0;

// Creates the graphical user interface window and its initial state
pub fn main() {
    let window = WindowDesc::new(build_ui())
        .window_size((1000., 720.))
        .resizable(false)
        .title(
            "Neuroevolution Testing Framework - Master's Thesis Project",
        );
    let appState = AppState::default();

    AppLauncher::with_window(window)
        .log_to_console()
        .launch(appState)
        .expect("launch failed");
}


