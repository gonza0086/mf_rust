use ndarray::Array2;
use std::{collections::HashMap, fs::read_to_string, time::Instant, usize};

pub struct DatasourceReader {
    datasource: String,
    rows: usize,
    cols: usize,
    mapped_users: HashMap<usize, usize>,
    mapped_movies: HashMap<usize, usize>,
}

impl DatasourceReader {
    pub fn new(path: &str) -> DatasourceReader {
        let mut users: HashMap<usize, usize> = HashMap::new();
        let mut movies: HashMap<usize, usize> = HashMap::new();
        let mut mapped_user_id = 0;
        let mut mapped_movie_id = 0;

        let lines: Vec<String> = read_to_string(path)
            .unwrap()
            .lines()
            .map(String::from)
            .collect();

        for line in lines {
            let mut iterator = line.split_whitespace().map(parse_to_uzise);

            let user_id = iterator.next().unwrap();
            if !users.contains_key(&user_id) {
                users.insert(user_id, mapped_user_id);
                mapped_user_id += 1;
            }

            let movie_id = iterator.next().unwrap();
            if !movies.contains_key(&movie_id) {
                movies.insert(movie_id, mapped_movie_id);
                mapped_movie_id += 1;
            }
        }

        return DatasourceReader {
            datasource: path.to_string(),
            rows: users.len(),
            cols: movies.len(),
            mapped_users: users,
            mapped_movies: movies,
        };
    }

    pub fn read_data(&self) -> Array2<f64> {
        println!("Reading data from file...");
        let now = Instant::now();
        let mut matrix = Array2::from_elem((self.rows, self.cols), 0.);

        let lines: Vec<String> = read_to_string(&self.datasource)
            .unwrap()
            .lines()
            .map(String::from)
            .collect();

        for line in lines {
            let mut iterator = line.split_whitespace().map(parse_to_uzise);

            let row_num = self.mapped_users.get(&iterator.next().unwrap()).unwrap();
            let col_num = self.mapped_movies.get(&iterator.next().unwrap()).unwrap();
            matrix[(*row_num, *col_num)] = iterator.next().unwrap() as f64;
        }
        println!("Data red in {}ms", now.elapsed().as_millis());

        return matrix;
    }
}

fn parse_to_uzise(s: &str) -> usize {
    return s.parse::<usize>().unwrap();
}
