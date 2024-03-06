use core::f64;
use ndarray::{Array2, Axis};
use std::{fs::read_to_string, time::Instant};

fn mse(
    y: &Array2<f64>,
    u: &Array2<f64>,
    v: &Array2<f64>,
    mask: &Array2<f64>,
    count: &f64,
) -> (Array2<f64>, f64) {
    let difference = (y - u.dot(&v.t())) * mask;
    let error = 0.5 * (difference.map_axis(Axis(1), |x| x.dot(&x))).sum() / count;

    return (difference, error);
}

fn sgd(
    y: Array2<f64>,
    k_latent: usize,
    alpha: f64,
    max_steps: usize,
    threshold: f64,
) -> (Array2<f64>, f64) {
    let mask = y.map(|x| {
        if *x == 0. {
            return 0.;
        };
        return 1.;
    });
    let (m, n) = y.dim();
    let count = &mask.sum();
    let mut r_squared = 0.;

    let mut u = Array2::from_elem((m, k_latent), 3.5);
    let mut v = Array2::from_elem((n, k_latent), 3.5);

    for i in 0..max_steps {
        let (difference, error) = mse(&y, &u, &v, &mask, &count);
        r_squared = 1. - error;

        if error < threshold {
            println!("Threshold reached in {} iterations!", i);
            break;
        }

        let u_delta = alpha / count * (difference.dot(&v));
        let v_delta = alpha / count * (difference.t().dot(&u));

        u = u + u_delta;
        v = v + v_delta;
    }

    return (u.dot(&v.t()), r_squared);
}

fn read_movielens_data(datasource: &str) -> Array2<f64> {
    let mut matrix = Array2::from_elem((943, 1682), 0.);

    let lines: Vec<String> = read_to_string(datasource)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();

    for line in lines {
        let mut iterator = line.split_whitespace().map(parse_to_float);

        let row_num = iterator.next().unwrap() - 1;
        let col_num = iterator.next().unwrap() - 1;
        matrix[(row_num, col_num)] = iterator.next().unwrap() as f64;
    }

    return matrix;
}

fn parse_to_float(s: &str) -> usize {
    return s.parse::<usize>().unwrap();
}

fn main() {
    println!("Execution Started!");
    let now = Instant::now();
    let data = read_movielens_data("u.data");

    println!("Data fetched in {}ms!", now.elapsed().as_millis());

    let (prediction, r_squared) = sgd(data, 5, 0.2, 1000, 0.9);

    println!(
        "Prediction Matrix:\n{}\nR_Squared = {}",
        prediction, r_squared
    );

    println!("Successfully finished in {}s!", now.elapsed().as_secs());
}
