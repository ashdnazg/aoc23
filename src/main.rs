#![allow(dead_code)]
use std::{collections::HashMap, fs};

fn main() {
    // day01();
    // day02();
    day03();
}

fn day03() {
    let contents = fs::read_to_string("aoc03.txt").unwrap();
    let grid: Vec<Vec<char>> = contents.lines().map(|line| line.chars().collect()).collect();
    let mut number_positions: HashMap<(usize, usize), u64> = HashMap::new();
    for (y, row) in grid.iter().enumerate() {
        let mut num = 0;
        let mut num_length = 0;
        for (x, c) in row.iter().enumerate() {
            if let Some(digit) = c.to_digit(10) {
                num_length += 1;
                num *= 10;
                num += digit as u64;
            } else {
                for i in (x-num_length)..x {
                    number_positions.insert((i, y), num);
                }
                num = 0;
                num_length = 0;
            }
        }
        for i in (row.len()-num_length)..row.len() {
            number_positions.insert((i, y), num);
        }
    }

    let mut sum = 0;
    let mut number_positions_copy = number_positions.clone();

    for (y, row) in grid.iter().enumerate() {
        for (x, &c) in row.iter().enumerate() {
            if c.is_digit(10) || c == '.' {
                continue;
            }
            {
                let tl = number_positions_copy.remove(&(x - 1, y - 1));
                let t = number_positions_copy.remove(&(x, y - 1));
                let tr = number_positions_copy.remove(&(x + 1, y - 1));
                if let Some(num) = tl {
                    sum += num;
                } else if let Some(num) = t {
                    sum += num;
                }

                if t.is_none() {
                    if let Some(num) = tr {
                        sum += num;
                    }
                }
            }

            {
                if let Some(num) = number_positions_copy.remove(&(x - 1, y)) {
                    sum += num;
                }
                if let Some(num) = number_positions_copy.remove(&(x + 1, y)) {
                    sum += num;
                }
            }

            {
                let bl = number_positions_copy.remove(&(x - 1, y + 1));
                let b = number_positions_copy.remove(&(x, y + 1));
                let br = number_positions_copy.remove(&(x + 1, y + 1));
                if let Some(num) = bl {
                    sum += num;
                } else if let Some(num) = b {
                    sum += num;
                }

                if b.is_none() {
                    if let Some(num) = br {
                        sum += num;
                    }
                }
            }
        }
    }

    println!("{sum}");

    let mut number_positions_copy2 = number_positions.clone();
    let mut sum2 = 0;
    for (y, row) in grid.iter().enumerate() {
        for (x, &c) in row.iter().enumerate() {
            if c != '*' {
                continue;
            }
            let mut product = 1;
            let mut adjacent_numbers = 0;

            {
                let tl = number_positions_copy2.remove(&(x - 1, y - 1));
                let t = number_positions_copy2.remove(&(x, y - 1));
                let tr = number_positions_copy2.remove(&(x + 1, y - 1));
                if let Some(num) = tl {
                    product *= num;
                    adjacent_numbers += 1;
                } else if let Some(num) = t {
                    product *= num;
                    adjacent_numbers += 1;
                }

                if t.is_none() {
                    if let Some(num) = tr {
                        product *= num;
                        adjacent_numbers += 1;
                    }
                }
            }

            {
                if let Some(num) = number_positions_copy2.remove(&(x - 1, y)) {
                    product *= num;
                    adjacent_numbers += 1;
                }
                if let Some(num) = number_positions_copy2.remove(&(x + 1, y)) {
                    product *= num;
                    adjacent_numbers += 1;
                }
            }

            {
                let bl = number_positions_copy2.remove(&(x - 1, y + 1));
                let b = number_positions_copy2.remove(&(x, y + 1));
                let br = number_positions_copy2.remove(&(x + 1, y + 1));
                if let Some(num) = bl {
                    product *= num;
                    adjacent_numbers += 1;
                } else if let Some(num) = b {
                    product *= num;
                    adjacent_numbers += 1;
                }

                if b.is_none() {
                    if let Some(num) = br {
                        product *= num;
                        adjacent_numbers += 1;
                    }
                }
            }

            if adjacent_numbers == 2 {
                sum2 += product;
            }
        }
    }

    println!("{sum2}");
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
