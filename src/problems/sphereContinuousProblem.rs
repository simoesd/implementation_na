use crate::problems::naProblem::NAProblem;
use crate::nn::ann::NANN;
use crate::utils::{interval::Interval, mathUtils};
use std::fmt;
use std::f64::consts;
use ndarray::*;

pub struct SphereContinuousNAProblem {
    ranges: Vec<(f64, f64)>,
    problemName: String,
}

impl SphereContinuousNAProblem {
    pub fn new(ranges: Vec<(f64, f64)>, problemName: String) -> Box<dyn NAProblem> {
        Box::new(SphereContinuousNAProblem {
            ranges,
            problemName,
        })
    }

    pub fn newQuarter() -> Box<dyn NAProblem> {
        SphereContinuousNAProblem::new(
            vec![(0.0, consts::PI/2.0)],
            String::from("Sphere Continuous Quarter"))
    }

    pub fn newHalf() -> Box<dyn NAProblem> {
        SphereContinuousNAProblem::new(
            vec![(0.0, consts::PI)],
            String::from("Sphere Continuous Half")
        )
    }

    pub fn newTwoQuarters() -> Box<dyn NAProblem> {
        SphereContinuousNAProblem::new(
            vec![
                (0.0, consts::PI/2.0),
                (consts::PI, 3.0*consts::PI/2.0)
            ],
            String::from("Sphere Continuous Two Quarters")
        )
    }

    pub fn newLocalOpt() -> Box<dyn NAProblem> {
        SphereContinuousNAProblem::new( 
            vec![
                (0.0, consts::PI/3.0),
                (2.0*consts::PI/3.0, consts::PI),
                (4.0*consts::PI/3.0, 11.0*consts::PI/6.0)
            ],
            String::from("Sphere Continuous Local Optima")
        )
    }
}

impl fmt::Display for SphereContinuousNAProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.problemName)
    }

}

impl NAProblem for SphereContinuousNAProblem {
    fn evaluate(&self, nn: &NANN) -> (bool, f64, Array2<f64>) {
    
        let inputs: Vec<f64> = self.ranges.iter().fold(vec![], |mut acc, (x_1, x_2)| {acc.append(vec![*x_1, *x_2].as_mut()); acc});
        let prediction: Array2<f64> = (*nn).clone().forward(Array2::<f64>::from_shape_vec(Ix2(1usize, inputs.len()), inputs).unwrap());
        let givenRanges = Interval::fromVec(self.ranges.clone());
        let predictionRanges = Interval::fromVec(prediction.rows().into_iter().fold(vec![], |mut acc, row| {
            acc.extend::<Vec<(f64, f64)>>(row.axis_chunks_iter(Axis(0), 2).map(|pair| {
                let angle: f64 = *pair.index(0);
                let bias: f64 = mathUtils::ring(*pair.index(1), 1.0, true);
                
                //Transforming the parameters of the output line into the covered arc in the unit circle
                let start_angle = mathUtils::ring(angle - bias.acos(), 2.0*consts::PI, false);
                let end_angle = mathUtils::ring(bias.acos() + angle, 2.0*consts::PI, false);
                (start_angle, end_angle)
            }).collect());
            acc
        }));
        
        // Positives
        let mut correctPredictionRanges = predictionRanges.intersection(&givenRanges);
        
        // Negatives
        let givenComplement = &givenRanges.complement();
        let predictedComplement = &predictionRanges.complement();
        let complementIntersect = givenComplement.intersection(predictedComplement);
        correctPredictionRanges = correctPredictionRanges.union(&complementIntersect);
        
        let correctPredictionArea: f64 = correctPredictionRanges.ranges.iter().fold(0.0, |acc, (x, y)| acc + (y - x));
        let success = unsafe { correctPredictionArea / (2.0*consts::PI) >= crate::OPTIMUM - (1.0 / crate::R) };
        let score = correctPredictionArea / (2.0*consts::PI);
        return (success, score, prediction);
    }
}