use ndarray::{Array2, Axis};

struct MF {
    alpha: f64,
    max_steps: usize,
    k_latent: usize,
    threshold: f64,
}

impl MF {}

fn mse(y: &Array2<f64>, y_hat: &Array2<f64>, mask: &Array2<f64>) -> f64 {
    let difference = (y - y_hat) * mask;
    let squared_difference = &difference * &difference;
    let error = squared_difference.sum() / (squared_difference.len() as f64);

    return error as f64;
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

        let error = mse(&y, &y_hat, &mask);
        assert_eq!(0., error);
    }

    #[test]
    fn test_mse_is_non_zero_when_whole_estimation_is_different_from_observed_data() {
        let y = arr2(&[[1., 1., 1., 1.], [1., 1., 1., 1.]]);
        let y_hat = arr2(&[[0., 0., 0., 0.], [0., 0., 0., 0.]]);
        let mask = Array2::from_elem((2, 4), 1.);

        let error = mse(&y, &y_hat, &mask);
        assert_eq!(1., error);
    }

    #[test]
    fn test_non_negative_numbers_as_result_of_the_power() {
        let y = arr2(&[[0., 0., 0., 0.], [0., 0., 0., 0.]]);
        let y_hat = arr2(&[[1., 1., 1., 1.], [1., 1., 1., 1.]]);
        let mask = Array2::from_elem((2, 4), 1.);

        let error = mse(&y, &y_hat, &mask);
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

        let error = mse(&y, &y_hat, &mask);
        assert_eq!(21.5, error);
    }

    #[test]
    fn test_masked_values_are_not_considered_for_the_calculation() {
        let y = arr2(&[[1., 1., 1., 1.], [1., 1., 1., 1.]]);
        let y_hat = arr2(&[[1., 0., 0., 1.], [1., 1., 1., 1.]]);
        let mask = arr2(&[[1., 0., 0., 1.], [1., 1., 1., 1.]]);

        let error = mse(&y, &y_hat, &mask);
        assert_eq!(0., error);
    }
}
