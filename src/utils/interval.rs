use crate::utils::mathUtils;
use std::f64::consts;

#[derive(Debug)]
pub struct Interval {
    pub ranges: Vec<(f64, f64)>
}

/*
    Class used to represent a continuous interval between two angles.
    Includes operations like intersection, union and complement.
*/
impl Interval {

    pub fn new(start: f64, end: f64) -> Interval {
        let start = mathUtils::ring(start, 2.0*consts::PI, false);
        let end = mathUtils::ring(end, 2.0*consts::PI, false);
        Interval {
            ranges: Interval::makeValid(start, end)
        }
    }
    
    pub fn fromVec(ranges: Vec<(f64, f64)>) -> Interval {
        let processedRanges = ranges.iter().fold(vec![], |mut acc, (x, y)| {
            acc.extend(Interval::makeValid(*x, *y));
            acc
        });
        let mut newRangeSet = Interval{ ranges: processedRanges } ;
        newRangeSet.deduplicateRanges();
        newRangeSet
    }

    fn makeValid(start: f64, end: f64) -> Vec<(f64, f64)> {
        let start = mathUtils::ring(start, 2.0*consts::PI, false);
        let end = mathUtils::ring(end, 2.0*consts::PI, false);
        if start <= end {
            vec![(start, end)]
        } else {
            vec![(0.0, end), (start, consts::PI*2.0)]
        }
    }

    pub fn union(&self, range: &Interval) -> Interval {
        let mut newRanges: Vec<(f64, f64)> = self.ranges.clone();
        newRanges.extend(range.ranges.clone());
        Interval::fromVec(newRanges)
    }

    pub fn intersection(&self, range: &Interval) -> Interval {
        let mut intersectingRanges = vec![];
        for (a, b) in &range.ranges {
            for (x, y) in &self.ranges {
                let start_pos = f64::max(*x, *a);
                let end_pos = f64::min(*y, *b);
                if start_pos <= end_pos {
                    intersectingRanges.push((start_pos, end_pos));
                }
            }
        }
        Interval::fromVec(intersectingRanges)
    }

    pub fn complement(&self) -> Interval{
        let initialRange = Interval::fromVec(vec![(0.0, self.ranges[0].0), (self.ranges[0].1, 2.0*consts::PI)]);
        self.ranges.iter().skip(1).fold(initialRange, |acc, (x, y)| {
            acc.intersection(&Interval::fromVec(vec![(0.0, *x), (*y, 2.0*consts::PI)]))
        })
    }

    // Ensures one interval object doesn't include overlapping areas. If this happens, they are joined.
    pub fn deduplicateRanges(&mut self) {
        self.ranges = self.ranges.iter()
            .fold(vec![], |mut acc, (x, y)| {
                for i in 0..acc.len() {
                    let (a, b) = acc[i];
                    if !(*x < a && *y < a) && !(*x > b && *y > b) {
                        // Non-disjoint ranges
                        acc[i] = (f64::min(a, *x), f64::max(b, *y));
                        let mut tmp = Interval::fromVec(acc);
                        tmp.deduplicateRanges();
                        return tmp.ranges;
                    }
                }
                acc.push((*x, *y));
                return acc;
            }
        );
    }
}