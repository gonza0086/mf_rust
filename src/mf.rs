use ndarray::Array2;
use std::time::Instant;

pub struct MF {
    alpha: f64,
    max_steps: usize,
    k_latent: usize,
    threshold: f64,
}

impl MF {
    pub fn new(alpha: f64, max_steps: usize, k_latent: usize, threshold: f64) -> MF {
        MF {
            alpha,
            max_steps,
            k_latent,
            threshold,
        }
    }

    fn sgd(&self, y: Array2<f64>) -> (Array2<f64>, f64) {
        let (m, n) = y.dim();
        let mut u = Array2::from_elem((m, self.k_latent), 3.);
        let mut v = Array2::from_elem((n, self.k_latent), 3.);

        let mask = y.map(|x| {
            if *x == 0. {
                return 0.;
            };
            return 1.;
        });

        for epoch in 0..self.max_steps {
            for row in 0..m {
                for col in 0..n {
                    let rating = y[[row, col]];
                    if rating != 0. {
                        let user_row = u.row(row);
                        let movie_col = v.row(col);
                        let prediction = user_row.dot(&movie_col);
                        let e = rating - prediction;

                        let delta_u = self.alpha * e * movie_col.to_owned();
                        let delta_v = self.alpha * e * user_row.to_owned();

                        let u_sum = user_row.to_owned() + delta_u;
                        let v_sum = movie_col.to_owned() + delta_v;

                        u.row_mut(row).assign(&u_sum);
                        v.row_mut(col).assign(&v_sum);
                    }
                }
            }

            let (_diff, _error, rerror) = mse(&y, &u.dot(&v.t()), &mask);
            println!("Relative error = {} in epoch: {}", rerror, epoch);

            if rerror < self.threshold {
                println!("Threshold reached in {} iterations!", epoch);
                break;
            }
        }

        let final_prediction = u.dot(&v.t());
        let (_diff, error, _rerror) = mse(&y, &final_prediction, &mask);
        return (final_prediction, error);
    }

    pub fn train(&self, y: Array2<f64>) {
        let now = Instant::now();
        println!("Training model...");

        let (estimation, error) = self.sgd(y);
        let prediction = round_all(&estimation);

        println!("MSE = {}\n{}", error, prediction);
        println!("Model trained in {}s", now.elapsed().as_secs());
    }
}

fn mse(y: &Array2<f64>, y_hat: &Array2<f64>, mask: &Array2<f64>) -> (Array2<f64>, f64, f64) {
    let difference = (y - y_hat) * mask;
    let squared_difference = &difference * &difference;
    let error = squared_difference.sum() / (mask.sum() as f64);
    let relative_to = (y * y).sum();
    let rerror = squared_difference.sum() / relative_to;

    return (difference, error, rerror);
}

fn round_all(y: &Array2<f64>) -> Array2<f64> {
    y.map(|x| x.round())
}
