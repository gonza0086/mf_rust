mod datasource_reader;
mod matrix_factorization;
mod mf;

use datasource_reader::DatasourceReader;
use mf::MF;
use std::time::Instant;

fn main() {
    println!("Execution Started!");
    let now = Instant::now();

    let movielens_reader = DatasourceReader::new("short_movielens_ds.data");
    let data = movielens_reader.read_data();

    let mf = MF::new(0.2, 10000, 10, 0.1);
    mf.train(data);

    println!("Successfully finished in {}s!", now.elapsed().as_secs());
}
