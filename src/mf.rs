use ndarray::{Array1, Array2};
use std::time::Instant;

pub struct MF {
    alpha: f64,
    max_steps: usize,
    k_latent: usize,
    threshold: f64,
    kappa: f64,
}

impl MF {
    pub fn new(alpha: f64, max_steps: usize, k_latent: usize, threshold: f64, kappa: f64) -> MF {
        MF {
            alpha,
            max_steps,
            k_latent,
            threshold,
            kappa,
        }
    }

    fn sgd(&self, y: Array2<f64>) -> (Array2<f64>, f64) {
        let (m, n) = y.dim();
        let mut u = Array2::from_elem((m, self.k_latent), 3.);
        let mut v = Array2::from_elem((n, self.k_latent), 3.);

        let non_null_ratings_position = ignore_null_ratings(&y);

        for epoch in 0..self.max_steps {
            for (row, col, rating) in non_null_ratings_position.iter() {
                let user_row = u.row(*row);
                let movie_col = v.row(*col);
                let prediction = user_row.dot(&movie_col);
                let e = rating - prediction;

                let delta_u = self.alpha * (e * movie_col.to_owned());
                let delta_v = self.alpha * (e * user_row.to_owned());

                let u_sum = user_row.to_owned() + delta_u;
                let v_sum = movie_col.to_owned() + delta_v;

                u.row_mut(*row).assign(&u_sum);
                v.row_mut(*col).assign(&v_sum);
            }

            let rmse = rmse(&non_null_ratings_position, &u.dot(&v.t()));
            println!("Error = {} in epoch: {}", rmse, epoch);

            if rmse < self.threshold {
                println!("Threshold reached in {} iterations!", epoch);
                break;
            }
        }

        let final_prediction = u.dot(&v.t());
        let rmse = rmse(&non_null_ratings_position, &final_prediction);
        return (final_prediction, rmse);
    }

    pub fn train(&self, y: Array2<f64>) {
        println!("Training model...");

        let now = Instant::now();
        let (estimation, error) = self.sgd(y);
        println!("Model trained in {}s", now.elapsed().as_secs());

        let prediction = round_all(&estimation);
        println!("RMSE = {}\n{}", error, prediction);
    }
}

fn rmse(ratings: &Vec<(usize, usize, f64)>, y_hat: &Array2<f64>) -> f64 {
    let mut sum = 0.;

    for (row, col, rating) in ratings.iter() {
        let prediction = y_hat[[*row, *col]];
        let e = (rating - prediction).powi(2);
        sum += e;
    }

    return sum / ratings.len() as f64;
}

fn round_all(y: &Array2<f64>) -> Array2<f64> {
    y.map(|x| x.round())
}

fn ignore_null_ratings(y: &Array2<f64>) -> Vec<(usize, usize, f64)> {
    let (m, n) = y.dim();
    let mut tuples: Vec<(usize, usize, f64)> = Vec::new();

    for row in 0..m {
        for col in 0..n {
            let rating = y[[row, col]];

            if rating != 0. {
                tuples.push((row, col, rating));
            }
        }
    }

    return tuples;
}
