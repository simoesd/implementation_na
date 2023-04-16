use crate::{mutationAlgorithms::mutationAlgorithm::MutationAlgorithm, utils::mathUtils};
use crate::problems::naProblem::NAProblem;
use crate::nn::ann::NANN;
use ndarray::{Array2, Array1};
use rand::Rng;
use rand_distr::{Uniform, Bernoulli, Distribution};
use std::fmt;

pub struct SAOnePlusOneNA {
    problem: Box<dyn NAProblem>,
    resolutionParameter: f64,
    numberOfNeurons: u32,
    weightStepSizes: Vec<Array2<f64>>,
    biasStepSizes: Vec<Array1<f64>>,
    successAdaptation: f64,
    failureAdaptation: f64
}

impl SAOnePlusOneNA {
    pub fn new(nn: &NANN, problem: Box<dyn NAProblem>, resolutionParameter: f64, successAdaptation: f64, failureAdaptation: f64) -> Box<dyn MutationAlgorithm> {
        assert!(successAdaptation > 1.0);
        assert!(failureAdaptation > 0.0 && failureAdaptation < 1.0);
        let mut weightStepSizes: Vec<Array2<f64>> = vec![];
        let mut biasStepSizes: Vec<Array1<f64>> = vec![];
        for layer in &nn.layers {
            weightStepSizes.push(Array2::from_elem(layer.weights.raw_dim(), resolutionParameter/8.0));
            if nn.isUsingBias() {
                biasStepSizes.push(Array1::from_elem(layer.getBiases().raw_dim(), resolutionParameter/8.0));
            }
        }
        Box::new(SAOnePlusOneNA {
            problem,
            weightStepSizes,
            resolutionParameter,
            numberOfNeurons: nn.layers.iter().fold(0, |total, l| total + l.getBiases().len()) as u32,
            biasStepSizes,
            successAdaptation,
            failureAdaptation
        })
    }
}

impl fmt::Display for SAOnePlusOneNA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Self Adaptive (1+1)NA")
    }
}

impl MutationAlgorithm for SAOnePlusOneNA {

    fn getProblem(&self) -> &Box<dyn NAProblem> {
        &self.problem
    }
    
    fn mutate(&mut self, nn: NANN, originalScore: f64) -> NANN {
        let mut randGen = rand::thread_rng();
        
        let mut mutatedNN = nn.clone();

        let mut mutatedParameters: Vec<(usize, usize, usize)> = vec![];
        let bernoulli = Bernoulli::from_ratio(1, self.numberOfNeurons).unwrap();
        let uniform = Uniform::new_inclusive(-1.0, 1.0);

        // At least one parameter is always mutated
        loop {
            // Draw what parameter is mutated. i is the layer, x and y correspond to the coordinates in that layer's matrix
            let i = randGen.gen_range(0..mutatedNN.layers.len());
            let x = randGen.gen_range(0..self.weightStepSizes[i].nrows());
            let y = randGen.gen_range(0..self.weightStepSizes[i].ncols());
            mutatedNN.layers[i].weights[(x, y)] += (mathUtils::harmonicDistribution() * (self.weightStepSizes[i][(x, y)] * randGen.sample::<f64, Uniform<f64>>(uniform).signum())) / self.resolutionParameter;
            if mutatedNN.isUsingBias() {
                mutatedNN.layers[i].biases[(y)] += (mathUtils::harmonicDistribution() *(self.biasStepSizes[i][(y)] * randGen.sample::<f64, Uniform<f64>>(uniform).signum())) / self.resolutionParameter;
            }

            // Store the mutated parameters for self-adaptation
            mutatedParameters.push((i, x, y));
                                    
            if !randGen.sample(bernoulli) {
                break;
            }
        }

        let (_, mutatedScore, _) = self.problem.evaluate(&mutatedNN);
        // Go through each mutated parameter and adjust their mutation strength
        mutatedParameters.iter().for_each(|(i, x, y)| {
            let adaptationStrength = if mutatedScore > originalScore { self.successAdaptation } else { self.failureAdaptation };
            self.weightStepSizes[*i][(*x, *y)] = f64::max(1.0, self.weightStepSizes[*i][(*x, *y)] * adaptationStrength);
            if nn.isUsingBias() {
                self.biasStepSizes[*i][(*y)] = f64::max(1.0, self.biasStepSizes[*i][(*y)] * adaptationStrength);
            }
        });
        if mutatedScore >= originalScore {
            mutatedNN
        } else {
            nn
        }
    }
}