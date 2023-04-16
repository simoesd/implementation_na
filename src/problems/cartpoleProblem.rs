
use crate::nn::ann::NANN;
use std::f64::consts;
use std::fmt;
use ndarray::*;
use rand::*;
use rand::SeedableRng;

use super::naProblem::NAProblem;

const GRAVITY_PER_TYPESTEP: f64 = 0.327;
const POLE_LENGTH: f64 = 10.0;
const CART_STEP_SIZE: f64 = 0.6;

pub struct CartpoleProblem {
    directionFunction: fn(f64) -> f64,
    problemName: String,
    seed: u64
}

impl CartpoleProblem {

    /**
     * Creates a new CartpoleProblem instance where the neural network's output will be ran through the provided `directionFunction`.
     * See `CartpoleProblem::Default` and `CartpoleProblem::new*` for examples
     */
    pub fn new(directionFunction: fn(f64) -> f64, problemName: String) -> Box<dyn NAProblem> {
        Box::new(CartpoleProblem {
            directionFunction,
            problemName,
            seed: rand::thread_rng().gen_range(0..10000)
        })
    }

    /**
     * The neural network's output will be directly used as the distance to move in a given direction.
     */
    pub fn default() -> Box<dyn NAProblem> {
        CartpoleProblem::new(
            |x| x,
            String::from("Cartpole continuous")
        )
    }

    /**
     * The neural network's output will be used as the distance to move the cart in a given direction, but rounded to the closest 1/r, discritizing the domain.
     */
    pub fn newDiscretizedContinuous() -> Box<dyn NAProblem> {
        CartpoleProblem::new(
            |x|  x * CART_STEP_SIZE.round() / CART_STEP_SIZE ,
            String::from("Cartpole discrete"),
        )
    }

    /**
     * The neural network's output will be treated as a binary result, determining if the cart moves left or right a single step of size CART_STEP_SIZE
     * With certain step sizes, will get stuck in a local optimum
     */
    pub fn newSingleStep() -> Box<dyn NAProblem> {
        CartpoleProblem::new(
            |x: f64| {
                x.signum() * CART_STEP_SIZE
            },
            String::from("Cartpole single step"),
        )
    }

    /**
     * The neural network's output will be translated into an integer, corresponding to how many steps of a fixed size the cart will move.
     * At least one step will be taken, no matter what.
     */
    pub fn newNSteps() -> Box<dyn NAProblem> {
        CartpoleProblem::new(
            |x| {
                let result = match x {
                    y if -1.0 < y && y < 0.0 => -1.0,
                    y if 0.0 <= y && y < 1.0 => 1.0,
                    _ => x.round()
                };
                result * 10.0 / unsafe { crate::R }
            },
            String::from("Cartpole N Steps"),
        )
    }

}

impl fmt::Display for CartpoleProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.problemName)
    }
}

impl NAProblem for CartpoleProblem {
    fn evaluate(&self, nn: &NANN) -> (bool, f64, Array2<f64>) {
        let mut randGen = rngs::StdRng::seed_from_u64(self.seed);

        let mut cartX: f64 = randGen.gen_range(-5.0..=5.0);
        let mut cartVelocity: f64 = 0.0;
        let mut poleAngle: f64 = randGen.gen_range(-0.01..=0.01);
        let mut poleAngularVelocity: f64 = 0.0;
        let mut poleAngularAcceleration: f64 = 0.0;
        let mut timestep: f64 = 0.0;
        let mut directionHistory: Vec<f64> = vec!();

        let result: (bool, f64) = loop  {
            // Requests an output from the neural network with the current state of the problem.
            // Casts the prediction into a float and runs it through the direction function provided on initialization
            let acceleration: f64 = (self.directionFunction)(*nn.clone().forward(
                Array2::<f64>::from_shape_vec(Ix2(1usize, 5usize),
                vec![poleAngle, poleAngularVelocity, poleAngularAcceleration, cartX, cartVelocity]).unwrap()
            ).get((0, 0)).unwrap() as f64);
            directionHistory.push(acceleration);
            cartVelocity += acceleration;
            cartX += cartVelocity;

            // Equations of motion
            poleAngularAcceleration = (3.0 / (7.0 * (POLE_LENGTH / 2.0))) * ((GRAVITY_PER_TYPESTEP * poleAngle.sin()) - (cartVelocity * poleAngle.cos()));
            poleAngularVelocity += poleAngularAcceleration;
            poleAngle += poleAngularVelocity;
            timestep += 1.0;

            if poleAngle.abs() >= consts::PI as f64 / 16.0 || cartX.abs() >= 50.0 {
                break (false, timestep);
            }
            if timestep >= 900.0 {
                break (true, timestep);
            }
        };
        // Unlike the other problems, doesn't just return the last prediction. This problem has a temporal aspect, and returns all predictions for the current evaluation.
        return (result.0, result.1, Array2::from_shape_vec((1, directionHistory.len()), directionHistory).unwrap());
    }
    
}