use std::f64::consts;

use rand_distr::{Normal, Distribution};
use ndarray::*;
use rand::*;

/*
 * Extended modulo function. Values will loop around the domain until they are inside it.
 * Implemented to correctly work with negative values (ring(-0.2, 0, 1) = -0.8) and allow symmetrical limits ([-10, 10] for example)
 */
pub fn ring(value: f64, limit: f64, allowNegativeResults: bool) -> f64 {
    if allowNegativeResults {
        if value >= 0.0 {
            return ((value+limit) % (limit*2.0)) - limit;
        } else {
            return ((((value.abs()+limit)*value.signum() % (limit*2.0)))) + limit;
        }
    } else {
        if value >= 0.0 {
            return value%limit;
        } else {
            return limit + (value%limit);
        }
    }
}

// Generates points of the specified dimension using a method proposed by Muller and Marsaglia
pub fn nSpherePointGeneration(numberOfPoints: u32, dimensions: usize) -> Vec<Vec<f64>> {
    let mut randGen = rand::thread_rng();
    let gaussGenerator = Normal::<f64>::new(0.0, 1.0).unwrap();
    let mut points: Vec<Vec<f64>> = vec![];
    for _ in 0..numberOfPoints {
        let generatedPoint: Vec<f64> = gaussGenerator.sample_iter(&mut randGen).take(dimensions).collect();
        let sum = generatedPoint.iter().map(|p| p.powi(2)).sum::<f64>().sqrt();
        points.push(generatedPoint.iter().map(|x| x/sum).collect());
    }
    return points;
}

// Evaluates if a point (in polar coordinates) is inside a range of angles. Supports high dimensions.
pub fn inRange(point: &Vec<f64>, ranges: &Vec<Vec<(f64, f64)>>) -> bool {
    ranges.iter().any(|range|
        point.iter().zip(range).all(
            |(p, (r_1, r_2))| {
                let p_a = &ring(*p, consts::PI*2.0, false);
                p_a >= r_1 && p_a <= r_2
            }
        )
    )
}

// Evaluates if a point (in polar coordinates) is above the plane represented by a normal vector. Supports high dimensions.
pub fn abovePlane(point: &Vec<f64>, normalVector: &Vec<f64>) -> bool {
    let hyperplaneMatrix: Array1<f64> = Array1::from_vec(polarToCartesian(&normalVector[0..normalVector.len()-1].to_vec()));
    let bias = normalVector[normalVector.len()-1];
    let cartesianPoint = polarToCartesian(point);
    let pointMatrix: Array1<f64> = Array1::from_vec(cartesianPoint);
    let matrixProduct = hyperplaneMatrix.dot(&(pointMatrix - hyperplaneMatrix.map(|x| x*bias)));

    return matrixProduct >= 0.0;
}

// Converts points from cartesian to polar. Assumes a radius of 1. Supports up to 4 dimensions.
pub fn cartesianToPolar(point: &Vec<f64>) -> Vec<f64>  {
    match &**point {
        // 2D
        [x_1, x_2] => vec![
            f64::atan2(*x_2, *x_1)
        ],
        // 3D
        [x_1, x_2, x_3] => vec![
            f64::atan2(*x_2, *x_1),
            f64::atan2((x_1.powi(2) + x_2.powi(2)).sqrt(), *x_3),
        ],
        // 4D
        [x_1, x_2, x_3, x_4] => if *x_1 == 0.0 && *x_2 == 0.0 && *x_3 == 0.0 && *x_4 == 0.0 {
            vec![0.0, 0.0, 0.0]
        } else {
            vec![
                f64::atan2(*x_2, *x_1),
                f64::acos(x_3 / (x_1.powi(2) + x_2.powi(2) + x_3.powi(2) + x_4.powi(2)).sqrt()),
                f64::atan2(*x_4, (x_1.powi(2) + x_2.powi(2)).sqrt()),
            ]
        },
        _ => panic!("Dimensions above 3 aren't supported")
    }
}

// Converts points from polar to cartesian. Assumes a radius of 1. Supports up to 4 dimensions.
pub fn polarToCartesian(point: &Vec<f64>) -> Vec<f64>  {
    match &**point {
        // 2D
        [a_1] => vec![
            a_1.cos(),
            a_1.sin()
        ],
        // 3D
        [a_2, a_1] => vec![
            a_1.sin() * a_2.cos(),
            a_1.sin() * a_2.sin(),
            a_1.cos()
        ],
        // 4D
        [a_1, a_2, a_3] => vec![
            a_1.cos() * a_2.sin() * a_3.cos(),
            a_1.sin() * a_2.sin() * a_3.cos(),
            a_2.cos() * a_3.cos(),
            a_2.sin() * a_3.sin()
        ],
        _ => panic!("Dimensions above 3 aren't supported: {}", point.len())
    }
}

pub fn harmonicNumber(n: f64) -> f64 {
    let mut harmNumber = 0.0;
    for i in 1..=n as i64 {
        harmNumber += 1.0/i as f64;
    }
    return harmNumber;
}

// Generates a value according to the Harmonic Distribution
pub fn harmonicDistribution() -> f64 {
    let harmNumber = unsafe { harmonicNumber(crate::R) };
    let u: f64 = rand::thread_rng().gen();
    let mut acc = 0.0;
    let mut result = 1.0;

    loop {
        acc += 1.0 / (result * harmNumber);
        if acc >= u {
            break result;
        }
        result += 1.0;
    }
}