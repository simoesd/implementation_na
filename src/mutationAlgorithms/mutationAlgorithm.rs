use std::fmt;

use crate::{nn::ann::NANN, problems::naProblem::NAProblem};
/*
    Base trait for all mutation algorithms
 */
pub trait MutationAlgorithm: fmt::Display {
    fn getProblem(&self) -> &Box<dyn NAProblem>;
    
    fn mutate(&mut self, nn: NANN, originalScore: f64) -> NANN;
}
