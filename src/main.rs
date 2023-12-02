use std::{collections::HashMap, fs};

fn main() {
    // day01();
    day02();
}

fn day02() {
    let contents = fs::read_to_string("aoc02.txt").unwrap();
    let games: Vec<Vec<HashMap<String, u64>>> = contents
        .lines()
        .map(|line| line.split_once(": ").unwrap().1)
        .map(|line| {
            line.split("; ")
                .map(|cube_set| {
                    cube_set
                        .split(", ")
                        .map(|count_and_cube| count_and_cube.split_once(" ").unwrap())
                        .map(|(count_str, cube)| {
                            (cube.to_owned(), count_str.parse().unwrap())
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let result: u64 = games
        .iter()
        .enumerate()
        .filter(|(_, cube_sets)| {
            cube_sets.iter().all(|cube_map| {
                cube_map.iter().all(|(color, &count)| {
                    match color.as_str() {
                        "red" => count <= 12,
                        "green" => count <= 13,
                        "blue" => count <= 14,
                        _ => false,
                    }
                })
            })
        })
        .map(|(game_id, _)| (game_id + 1) as u64)
        .sum();

    println!("{result}");

    let result2: u64 = games
        .iter()
        .map(|cube_sets| {
            cube_sets
                .iter()
                .flat_map(|y| y.iter())
                .fold(HashMap::<String, u64>::new(), |mut acc, (color, &count)| {
                    acc.entry(color.clone()).and_modify(|e| { *e = count.max(*e) }).or_insert(count);

                    acc
                })
                .values()
                .product::<u64>()
        })
        .sum();

    println!("{result2}");
}

fn day01() {
    let contents = fs::read_to_string("aoc01.txt").unwrap();
    let calibration_sum: u64 = contents
        .lines()
        .map(|line| {
            let first_digit = line
                .chars()
                .find(|c| c.is_digit(10))
                .unwrap()
                .to_digit(10)
                .unwrap() as u64;
            let last_digit = line
                .chars()
                .rfind(|c| c.is_digit(10))
                .unwrap()
                .to_digit(10)
                .unwrap() as u64;

            first_digit * 10 + last_digit
        })
        .sum();

    println!("{}", calibration_sum);

    let calibration_sum2: u64 = contents
        .lines()
        .map(|line| {
            line.replace("one", "one1one")
                .replace("two", "two2two")
                .replace("three", "three3three")
                .replace("four", "four4four")
                .replace("five", "five5five")
                .replace("six", "six6six")
                .replace("seven", "seven7seven")
                .replace("eight", "eight8eight")
                .replace("nine", "nine9nine")
        })
        .map(|line| {
            let first_digit = line
                .chars()
                .find(|c| c.is_digit(10))
                .unwrap()
                .to_digit(10)
                .unwrap() as u64;
            let last_digit = line
                .chars()
                .rfind(|c| c.is_digit(10))
                .unwrap()
                .to_digit(10)
                .unwrap() as u64;

            first_digit * 10 + last_digit
        })
        .sum();

    println!("{}", calibration_sum2);
}
