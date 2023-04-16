use std::fmt;

use crate::mutationAlgorithms::mutationAlgorithm::MutationAlgorithm;
use crate::problems::naProblem::NAProblem;
use crate::nn::ann::NANN;
use crate::utils::mathUtils;
use rand::{Rng, distributions};
use rand_distr::{Bernoulli, Uniform};

pub struct OnePlusOneNA {
    problem: Box<dyn NAProblem>,
    resolutionParameter: f64,
    numberOfNeurons: u32
}

impl OnePlusOneNA {
    pub fn new(nn: &NANN, problem: Box<dyn NAProblem>, resolutionParameter: f64) ->  Box<dyn MutationAlgorithm> {
        Box::new(OnePlusOneNA {
            problem,
            resolutionParameter,
            numberOfNeurons: nn.layers.iter().fold(0, |total, l| total + l.getBiases().len()) as u32
        })
    }
}

impl fmt::Display for OnePlusOneNA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(1+1)NA")
    }
}

impl MutationAlgorithm for OnePlusOneNA{
    fn getProblem(&self) -> &Box<dyn NAProblem> {
        &self.problem
    }
    
    fn mutate(&mut self, nn: NANN, originalScore: f64) -> NANN {
        let mut randGen = rand::thread_rng();
        let mut mutatedNN = nn.clone();
        let bernoulli = Bernoulli::from_ratio(1, self.numberOfNeurons).unwrap();
        let uniform = Uniform::new_inclusive(-1.0, 1.0);

        // At least one parameter is always mutated
        loop {
            // Draw what parameter is mutated. i is the layer, x and y correspond to the coordinates in that layer's matrix
            let i = randGen.gen_range(0..mutatedNN.layers.len());
            let x = randGen.gen_range(0..mutatedNN.layers[i].weights.nrows());
            let y = randGen.gen_range(0..mutatedNN.layers[i].weights.ncols());
            mutatedNN.layers[i].weights[(x, y)] += (mathUtils::harmonicDistribution() / self.resolutionParameter) * randGen.sample::<f64, Uniform<f64>>(uniform).signum();
            if mutatedNN.isUsingBias() {
                mutatedNN.layers[i].biases[(y)] += (mathUtils::harmonicDistribution() / self.resolutionParameter) * randGen.sample::<f64, Uniform<f64>>(uniform).signum();
            }
            
            if !randGen.sample(bernoulli) {
                break;
            }
        }

        let (_, mutatedScore, _) = self.problem.evaluate(&mutatedNN);
        if mutatedScore >= originalScore {
            mutatedNN
        } else {
            nn
        }
    }
}