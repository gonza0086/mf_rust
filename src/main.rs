use core::f64;

use ndarray::{arr2, Array2, Axis};
// Cargar todos los datos de MovieLens

fn mse(y: Array2<f64>, u: Array2<f64>, v: Array2<f64>) {
    let difference = y - u.dot(&v.t());
    let count = 20.;
    let error = (difference.map_axis(Axis(1), |v| v.dot(&v))).sum() / count;
    println!("Difference =\n{}\nError = {}", difference, error);
}

fn main() {
    let test = arr2(&[
        [4., 1., 2., 0., 5.],
        [5., 2., 2.5, 3.5, 5.],
        [1., 4., 4., 2., 5.],
        [0., 3.5, 3., 1.5, 4.],
    ]);

    let u_initial = arr2(&[[3., 3., 3.], [3., 3., 3.], [3., 3., 3.], [3., 3., 3.]]);

    let v_initial = arr2(&[
        [3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.],
    ]);

    mse(test, u_initial, v_initial);
}
