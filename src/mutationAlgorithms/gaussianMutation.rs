use crate::mutationAlgorithms::mutationAlgorithm::MutationAlgorithm;
use crate::nn::ann::NANN;
use rand_distr::{Normal, Distribution};
use crate::problems::naProblem::NAProblem;
use std::fmt;

pub struct GaussianMutation {
    problem: Box<dyn NAProblem>
}

impl GaussianMutation {
    pub fn new(problem: Box<dyn NAProblem>) ->  Box<dyn MutationAlgorithm> {
        Box::new(GaussianMutation {
            problem
        })
    }
}

impl fmt::Display for GaussianMutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gaussian Mutation")
    }
}

impl MutationAlgorithm for GaussianMutation {
    fn getProblem(&self) -> &Box<dyn NAProblem> {
        &self.problem
    }
    
    /*
     * For each parameter, with a probability of 1/number of neurons in the layer, mutate the parameter by a random value from a Standard Normal Distribution
     */
    fn mutate(&mut self, nn: NANN, originalScore: f64) -> NANN {
        let mut randGen = rand::thread_rng();
        let gaussGenerator = Normal::<f64>::new(0.0, 0.5).unwrap();
        let mut mutatedNN = nn.clone();
        for i in 0..mutatedNN.layers.len() {
            let layer = mutatedNN.layers[i].clone();
            let probOfMutation = 1.0 / layer.biases.dim() as f32;
            mutatedNN.layers[i].weights = layer.weights.map(|x| if rand::random::<f32>() < probOfMutation { x + gaussGenerator.sample(&mut randGen) } else { *x });
            if mutatedNN.isUsingBias() {
                mutatedNN.layers[i].biases = layer.biases.map(|x| if rand::random::<f32>() < probOfMutation {  x + gaussGenerator.sample(&mut randGen)} else {*x});
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