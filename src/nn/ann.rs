use std::f64::consts;

use ndarray::{Array2, Array1, Array};
use rand_distr::Uniform;
use crate::{problems::sphereContinuousProblem::SphereContinuousNAProblem, mutationAlgorithms::mutationAlgorithm::MutationAlgorithm, utils::mathUtils};
use ndarray_rand::RandomExt;

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    activationFunction: fn(Array2<f64>) -> Array2<f64>
}

impl Layer {
    // Creates a layer and initializes it's parameter's values.
    pub fn new(dimInput: usize, dimOutput: usize, activationFunction: fn(Array2<f64>) -> Array2<f64> , resolutionParameter: f64, usingBias: bool) -> Layer {
        let mut weights = Array2::<f64>::random((dimInput, dimOutput), Uniform::new_inclusive(-0.1 * resolutionParameter, 0.1 * resolutionParameter));
        weights.mapv_inplace(|w| w.round()/resolutionParameter);
        let mut biases: Array1<f64>;
        if usingBias {
            biases = Array::random(dimOutput, Uniform::new_inclusive(-0.1 * resolutionParameter, 0.1 * resolutionParameter));
            biases.mapv_inplace(|b| b.round()/resolutionParameter);
        } else {
            biases = Array::zeros(dimOutput);
        }
        Layer {
            weights,
            biases,
            activationFunction
        }
    }

    pub fn getBiases(&self) -> &Array1<f64> {
        &self.biases
    }

    // Single layer forward pass
    pub fn forward(&self, inputs: Array1<f64>) -> Array1<f64> {
        let mut outputs : Array1<f64>;
        match (inputs.dim(), self.weights.dim()) {
            (x, (y, _)) if x == y => outputs = inputs.dot(&self.weights),
            (x, (_, y)) if x == y => outputs = self.weights.dot(&inputs),
            (x, (m, n)) => panic!("Matrices cannot be multiplied: ({x}) and ({m}, {n})"),
        }
        outputs = outputs + self.getBiases();
        return outputs;
    }
}

#[derive(Debug, Clone)]
pub struct NANN {
    pub layers: Vec<Layer>,
    usingBias: bool,
}

impl NANN {
    // Creates a neural network with the provided specifications. Layer sizes are provided as tuples in the form (inputSize, outputSize)
    pub fn new(layerSizes: Vec<(usize, usize)>, activationFunction: fn(Array2<f64>) -> Array2<f64>, resolutionParameter: f64, usingBias: bool) -> NANN {
        let layers = layerSizes.iter().map(
            |(inputSize, outputSize)|
                Layer::new(*inputSize, *outputSize, activationFunction,  resolutionParameter, usingBias)
        ).collect::<Vec<Layer>>();
        NANN {
            layers,
            usingBias
        }
    }

    pub fn isUsingBias(&self) -> bool {
        self.usingBias
    }

    pub fn forward(self, inputs: Array2<f64>) -> Array2<f64> {
        inputs.rows().into_iter().fold(Array2::from_elem((0, self.layers[self.layers.len()-1].weights.dim().1), 15.0), |mut acc, i| {
            acc.push_row(
                self.layers.iter().fold(i.to_owned(), |output, layer| {
                    layer.forward(output)
                }).view()
            );
            acc
        })
    }
    
}

pub fn run(mut nn: NANN, mut mutationAlgorithm:  Box<dyn MutationAlgorithm>) -> (i32, i32, f64, Array2<f64>) {
    let mut i = 1;
    let mut success;
    let mut score = 0.0;
    let mut solution = Array2::zeros([1usize, 1usize]);
    let mut maxScoreGeneration = 0;
    let mut maxScore = 0.0;
    while i as f64 <= unsafe { 100.0 * crate::R * crate::R.log2() } {
        (success, score, solution) = mutationAlgorithm.getProblem().evaluate(&nn);
        if score > maxScore {
            maxScore = score;
            maxScoreGeneration = i;
        }
        if success {
            return (i, maxScoreGeneration, score, solution);
        } else {
            nn = mutationAlgorithm.mutate(nn.clone(), score);
        }
        i += 1;
    }
    return (i, maxScoreGeneration, score, solution);
}