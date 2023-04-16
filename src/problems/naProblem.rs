use std::fmt::Display;

use crate::nn::ann::NANN;
use ndarray::*;

/*
    Base trait for all problem types.
 */
pub trait NAProblem: Display {
    fn evaluate(&self, nn: &NANN) -> (bool, f64, Array2<f64>);
}