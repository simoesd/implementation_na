use crate::problems::naProblem::NAProblem;
use crate::nn::ann::NANN;
use crate::utils::mathUtils;
use std::fmt;
use std::f64::consts;
use rand_distr::Distribution;
use ndarray::*;

pub struct SphereDiscreteNAProblem {
    ranges: Vec<Vec<(f64, f64)>>,
    points: Vec<(Vec<f64>, bool)>,
    numPoints: u32,
    problemName: String
}


impl SphereDiscreteNAProblem {
    pub fn new(ranges: Vec<Vec<(f64, f64)>>, numPoints: u32, problemName: String) -> Box<dyn NAProblem> {
        // Creates the points to be used through out this execution, evaluates whether they are positive or negative, and converts them to a polar representation.
        let points = mathUtils::nSpherePointGeneration(numPoints, ranges[0].len() + 1).iter().map(
            |p| {
                (mathUtils::cartesianToPolar(p), mathUtils::inRange(&mathUtils::cartesianToPolar(p), &ranges))
            }).collect();
        Box::new(SphereDiscreteNAProblem {
            ranges,
            points,
            numPoints,
            problemName
        })
    }

    pub fn newQuarter(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new(
            vec![vec![(0.0, consts::PI/2.0)]],
            numPoints,
            String::from("Sphere Discrete Quarter")
        )
    }


    pub fn newHalf(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new(
             vec![vec![(0.0, consts::PI)]],
            numPoints,
            String::from("Sphere Discrete Half")
        )
    }

    pub fn newTwoQuarters(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new(
            vec![
                vec![(0.0, consts::PI/2.0)],
                vec![(consts::PI, 3.0*consts::PI/2.0)]
            ],
            numPoints,
            String::from("Sphere Discrete Two Quarters")
        )
    }

    pub fn newLocalOpt(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new( 
            vec![
                vec![(0.0, consts::PI/3.0)],
                vec![(2.0*consts::PI/3.0, consts::PI)],
                vec![(4.0*consts::PI/3.0, 11.0*consts::PI/6.0)]
            ],
            numPoints,
            String::from("Sphere Discrete Local Optima")
        )
    }

    
    pub fn newCorner3D(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new(
            vec![vec![(0.0, consts::PI/2.0), (0.0, consts::PI/2.0)]],
            numPoints,
            String::from("Sphere 3D Corner")
        )
    }

    pub fn newHalf3D(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new(
            vec![vec![(0.0, consts::PI), (0.0, consts::PI)]],
            numPoints,
            String::from("Sphere 3D Half")
        )
    }

    pub fn newSlice3D(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new(
            vec![vec![(0.0, consts::PI), (0.0, consts::PI/2.0)]],
            numPoints,
            String::from("Sphere 3D Slice")
        )
    }

    pub fn newTwoSlices3D(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new( 
            vec![
                vec![ (0.0, consts::PI/2.0), (0.0, consts::PI),],
                vec![(consts::PI, 3.0*consts::PI/2.0), (consts::PI, consts::PI*2.0)]
            ],
            numPoints,
            String::from("Sphere 3D Two Slices")
        )
    }
    
    pub fn newQuarter4D(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new(
            vec![
                vec![(0.0, consts::PI), (0.0, consts::PI), (0.0, consts::PI)]
            ],
            numPoints,
            String::from("Sphere 4D Quarter")
        )
    }
        
    pub fn newHalf4D(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new(
            vec![
                vec![(0.0, 2.0*consts::PI), (0.0, consts::PI), (0.0, consts::PI)]
            ],
            numPoints,
            String::from("Sphere 4D Half")
        )
    }
        
    pub fn newTwoQuarters4D(numPoints: u32) -> Box<dyn NAProblem> {
        SphereDiscreteNAProblem::new(
            vec![
                vec![(0.0, consts::PI), (0.0, consts::PI), (0.0, consts::PI)],
                vec![(consts::PI, 2.0*consts::PI), (0.0, consts::PI), (0.0, consts::PI)]
            ],
            numPoints,
            String::from("Sphere 4D Two Quarters")
        )
    }
}

impl fmt::Display for SphereDiscreteNAProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.problemName)
    }

}

impl NAProblem for SphereDiscreteNAProblem {
    fn evaluate(&self, nn: &NANN) -> (bool, f64, Array2<f64>) {

        // Shapes the ranges and requests a prediction from the network
        let inputs: Vec<f64> = self.ranges.iter().fold(vec![], |mut acc, range| {
            acc.extend(range.iter().fold(vec![], |mut r: Vec<f64>, (x_1, x_2)| {r.extend([x_1, x_2]); r}));
            acc
        });
        let prediction: Array2<f64> = (*nn).clone().forward(Array2::<f64>::from_shape_vec(Ix2(self.ranges.len(), self.ranges[0].len()*2), inputs).unwrap());

        // Transforms the prediction into a vector with angles in [0, 2pi] and bias in [-1, 1]
        let normalVectors: Vec<Vec<f64>> = prediction.rows().into_iter().map(|row| {
            row.into_iter().enumerate().map(|(i, &value)| {
                if i < row.len() - 1 {
                    mathUtils::ring(value, consts::PI*2.0, false)
                } else {
                    mathUtils::ring(value, 1.0, true)
                }
            }).collect()
        }).collect();

        // Points in a positive range should be classified as positive by at least one of the output ranges.
        // Points in a negative range should be classified as negative by all output ranges. This prevents a single positive point contributing to the score through multiple output ranges
        let accurate_predictions: i32 = self.points.iter().fold(0, |acc, (point, correctClassification)| {
            if *correctClassification {
                if normalVectors.iter().any(|vector| mathUtils::abovePlane(point, vector)) {
                    return acc + 1;
                }
            } else {
                if normalVectors.iter().all(|vector| !mathUtils::abovePlane(point, vector)) {
                    return acc + 1;
                }
            }
            acc
        });
        return (unsafe { (accurate_predictions as f64 / self.numPoints as f64) >= crate::OPTIMUM - (1.0 / crate::R)}, accurate_predictions as f64 / self.numPoints as f64, prediction);

    }
}