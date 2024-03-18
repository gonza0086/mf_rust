use std::time::Instant;

use ndarray::Array2;

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

    // Not yet SGD bud GD
    fn sgd(&self, y: Array2<f64>) -> (Array2<f64>, f64) {
        let mask = y.map(|x| {
            if *x == 0. {
                return 0.;
            };
            return 1.;
        });

        let (m, n) = y.dim();
        let mut u = Array2::from_elem((m, self.k_latent), 3.);
        let mut v = Array2::from_elem((n, self.k_latent), 3.);
        let mut mean_squared_error = 0.;

        for i in 0..self.max_steps {
            let estimation = u.dot(&v.t());
            let (difference, error) = mse(&y, &estimation, &mask);

            mean_squared_error = error;

            if mean_squared_error <= self.threshold {
                println!("Threshold reached in {} iterations!", i);
                break;
            }

            let delta_u = (2. * self.alpha / mask.sum()) * difference.dot(&v);
            let delta_v = (2. * self.alpha / mask.sum()) * difference.t().dot(&u);

            u = u + &delta_u;
            v = v + &delta_v;
        }

        return (u.dot(&v.t()), mean_squared_error);
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

fn mse(y: &Array2<f64>, y_hat: &Array2<f64>, mask: &Array2<f64>) -> (Array2<f64>, f64) {
    let difference = (y - y_hat) * mask;
    let squared_difference = &difference * &difference;
    let error = squared_difference.sum() / (mask.sum() as f64);

    return (difference, error as f64);
}

fn round_all(y: &Array2<f64>) -> Array2<f64> {
    y.map(|x| x.round())
}

fn normalize(y: &Array2<f64>) -> Array2<f64> {
    y.map(|x| {
        if *x > 5. {
            return 5.;
        }
        return *x;
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mse_is_cero_when_estimation_is_equal_to_observed_data() {
        let y = Array2::from_elem((4, 1), 1.);
        let y_hat = Array2::from_elem((4, 1), 1.);
        let mask = Array2::from_elem((4, 1), 1.);

        let (_, error) = mse(&y, &y_hat, &mask);
        assert_eq!(0., error);
    }

    #[test]
    fn test_mse_is_non_zero_when_whole_estimation_is_different_from_observed_data() {
        let y = arr2(&[[1., 1., 1., 1.], [1., 1., 1., 1.]]);
        let y_hat = arr2(&[[0., 0., 0., 0.], [0., 0., 0., 0.]]);
        let mask = Array2::from_elem((2, 4), 1.);

        let (_, error) = mse(&y, &y_hat, &mask);
        assert_eq!(1., error);
    }

    #[test]
    fn test_non_negative_numbers_as_result_of_the_power() {
        let y = arr2(&[[0., 0., 0., 0.], [0., 0., 0., 0.]]);
        let y_hat = arr2(&[[1., 1., 1., 1.], [1., 1., 1., 1.]]);
        let mask = Array2::from_elem((2, 4), 1.);

        let (_, error) = mse(&y, &y_hat, &mask);
        assert_eq!(1., error);
    }

    #[test]
    fn test_mse_is_calculated_correctly_with_complex_example() {
        let y = arr2(&[
            [0., 10., 7., 5.],
            [0., 2., 9., 12.],
            [4., 2., 2., 6.],
            [10., 3., 9., 15.],
        ]);
        let y_hat = arr2(&[
            [2., 4., 10., 5.],
            [0., 1., 6., 1.],
            [3., 6., 2., 6.],
            [21., 3., 14., 14.],
        ]);
        let mask = Array2::from_elem((4, 4), 1.);

        let (_, error) = mse(&y, &y_hat, &mask);
        assert_eq!(21.5, error);
    }

    #[test]
    fn test_masked_values_are_not_considered_for_the_calculation() {
        let y = arr2(&[[1., 1., 1., 1.], [1., 1., 1., 1.]]);
        let y_hat = arr2(&[[1., 0., 0., 1.], [1., 1., 1., 1.]]);
        let mask = arr2(&[[1., 0., 0., 1.], [1., 1., 1., 1.]]);

        let (_, error) = mse(&y, &y_hat, &mask);
        assert_eq!(0., error);
    }
}
