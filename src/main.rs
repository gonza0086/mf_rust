mod datasource_reader;
mod matrix_factorization;
mod mf;

use datasource_reader::DatasourceReader;
use matrix_factorization::MatrixFactorization;
use std::time::Instant;

fn main() {
    println!("Execution Started!");
    let now = Instant::now();

    let movielens_reader = DatasourceReader::new("movielens_ds.data");
    let data = movielens_reader.read_data();

    let mf = MatrixFactorization::new(10, 0.1, 0.1, 1000);
    mf.train(data);

    println!("Successfully finished in {}s!", now.elapsed().as_secs());
}
