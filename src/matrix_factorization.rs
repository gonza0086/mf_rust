use std::{fs, io::Write, time::Instant};

use ndarray::{Array2, Axis};

pub struct MatrixFactorization {
    k_latent: usize,
    threshold: f64,
    alpha: f64,
    max_steps: usize,
}

impl MatrixFactorization {
    pub fn new(
        k_latent: usize,
        threshold: f64,
        alpha: f64,
        max_steps: usize,
    ) -> MatrixFactorization {
        MatrixFactorization {
            k_latent,
            threshold,
            alpha,
            max_steps,
        }
    }

    fn sgd(&self, y: Array2<f64>) -> (Array2<f64>, f64) {
        let mask = y.map(|x| {
            if *x == 0. {
                return 0.;
            };
            return 1.;
        });

        let (m, n) = y.dim();
        let count = &mask.sum();
        let mut r_squared = 0.;

        let mut u = Array2::from_elem((m, self.k_latent), 3.5);
        let mut v = Array2::from_elem((n, self.k_latent), 3.5);

        for i in 0..self.max_steps {
            let (difference, error) = mse(&y, &u, &v, &mask, &count);
            r_squared = 1. - error;

            if error <= self.threshold {
                println!("Threshold reached in {} iterations!", i);
                break;
            }

            let u_delta = self.alpha / count * (difference.dot(&v));
            let v_delta = self.alpha / count * (difference.t().dot(&u));

            u = u + u_delta;
            v = v + v_delta;
        }

        let prediction = u.dot(&v.t());

        return (normalize(&prediction), r_squared);
    }

    pub fn train(&self, y: Array2<f64>) -> Array2<f64> {
        println!("Training model...");
        let now = Instant::now();
        let (prediction, r_squared) = self.sgd(y);

        println!(
            "Model trained in {}s with an R^2 = {}",
            now.elapsed().as_secs(),
            r_squared
        );
        write_to(&prediction, "matrix.data");

        return prediction;
    }
}

fn mse(
    y: &Array2<f64>,
    u: &Array2<f64>,
    v: &Array2<f64>,
    mask: &Array2<f64>,
    count: &f64,
) -> (Array2<f64>, f64) {
    let estimation = u.dot(&v.t());
    let difference = (y - estimation) * mask;
    let error = 0.5 * (difference.map_axis(Axis(1), |x| x.dot(&x))).sum() / count;

    return (difference, error);
}

fn normalize(y: &Array2<f64>) -> Array2<f64> {
    let normalized_prediction = y.map(|x| {
        if *x > 4.5 {
            return 5.0;
        } else if *x > 3.5 {
            return 4.;
        } else if *x > 2.5 {
            return 3.;
        } else if *x > 1.5 {
            return 2.;
        }
        return 1.0;
    });

    return normalized_prediction;
}

pub fn write_to(matrix: &Array2<f64>, path: &str) {
    println!("Writing prediction in file...");
    let now = Instant::now();
    let mut file = fs::File::create(path).unwrap();

    let rows = matrix.axis_iter(Axis(0));

    for row in rows {
        for rating in row {
            let _ = file.write(rating.to_string().as_bytes());
            let _ = file.write(b"   ");
        }
        let _ = file.write(b"\n");
    }

    println!("Prediction written in {}ms", now.elapsed().as_millis());
}
