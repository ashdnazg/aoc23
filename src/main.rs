#![allow(dead_code)]
use std::{collections::HashMap, fs, iter::zip, ops::Range};

fn main() {
    // day01();
    // day02();
    // day03();
    // day04();
    // day05();
    day06();
}

#[derive(Debug)]
struct MapEntry {
    dst_start: u64,
    src_start: u64,
    length: u64,
}

fn day06() {
    let contents = fs::read_to_string("aoc06.txt").unwrap();
    let mut data_iter = contents.lines().map(|line| {
        line.split_whitespace()
            .skip(1)
            .map(|num_str| num_str.parse().unwrap())
            .collect::<Vec<u64>>()
    });
    let times = data_iter.next().unwrap();
    let distances = data_iter.next().unwrap();

    let result: u64 = zip(&times, &distances)
        .map(|(&t, &d)| {
            let tf = t as f64;
            let df = d as f64;
            let s = (tf * tf - 4.0 * df).sqrt();
            let x1 = (tf - s) * 0.5;
            let x2 = (tf + s) * 0.5;
            ((x2 - 0.001).floor() - (x1 + 0.001).ceil()) as u64 + 1
        })
        .product();

    println!("{}", result);

    let tf = times
        .iter()
        .fold(0, |acc, n| acc * 10u64.pow(n.ilog10() + 1) + n) as f64;
    let df = distances
        .iter()
        .fold(0, |acc, n| acc * 10u64.pow(n.ilog10() + 1) + n) as f64;
    let s = (tf * tf - 4.0 * df).sqrt();
    let x1 = (tf - s) * 0.5;
    let x2 = (tf + s) * 0.5;
    let result2 = ((x2 - 0.001).floor() - (x1 + 0.001).ceil()) as u64 + 1;

    println!("{}", result2);
}

fn day05() {
    let contents = fs::read_to_string("aoc05.txt").unwrap();
    let (seeds_str, maps_str) = contents.split_once("\n\n").unwrap();
    let seeds: Vec<u64> = seeds_str
        .trim()
        .split_once(": ")
        .unwrap()
        .1
        .split(" ")
        .map(|num_str| num_str.parse().unwrap())
        .collect();

    let mut maps: Vec<Vec<MapEntry>> = maps_str
        .split("\n\n")
        .map(|map_str| {
            map_str
                .lines()
                .skip(1)
                .map(|entry_str| entry_str.trim().split_once(" ").unwrap())
                .map(|(dst_start_str, rest_str)| {
                    (
                        dst_start_str.parse().unwrap(),
                        rest_str.split_once(" ").unwrap(),
                    )
                })
                .map(|(dst_start, (src_start_str, length_str))| MapEntry {
                    dst_start,
                    src_start: src_start_str.parse().unwrap(),
                    length: length_str.parse().unwrap(),
                })
                .collect()
        })
        .collect();

    maps.iter_mut()
        .for_each(|map| map.sort_by_key(|entry| entry.src_start));

    let result = seeds
        .iter()
        .map(|&seed| {
            maps.iter().fold(seed, |src_num, map| {
                let index = match map.binary_search_by_key(&src_num, |entry| entry.src_start) {
                    Ok(i) => i,
                    Err(0) => return src_num,
                    Err(i) => i - 1,
                };
                let entry = &map[index];
                if src_num - entry.src_start >= entry.length {
                    src_num
                } else {
                    src_num - entry.src_start + entry.dst_start
                }
            })
        })
        .min()
        .unwrap();

    println!("{result}");

    let ranges: Vec<Range<u64>> = seeds
        .chunks(2)
        .map(|chunk| chunk[0]..(chunk[0] + chunk[1]))
        .collect();

    let result2 = ranges
        .iter()
        .map(|range| {
            maps.iter()
                .fold(vec![range.clone()], |acc, map| {
                    acc.iter()
                        .flat_map(|subrange| convert(subrange, map))
                        .collect()
                })
                .iter()
                .map(|range| range.start)
                .min()
                .unwrap()
        })
        .min()
        .unwrap();

    println!("{result2}");
}

fn convert(range: &Range<u64>, map: &[MapEntry]) -> Vec<Range<u64>> {
    let mut index = match map.binary_search_by_key(&range.start, |entry| entry.src_start) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) => i - 1,
    };

    let mut remaining_range = range.clone();
    let mut mapped_ranges = vec![];

    while !remaining_range.is_empty() {
        if map.len() <= index || map[index].src_start >= remaining_range.end {
            mapped_ranges.push(remaining_range.clone());
            break;
        }
        let entry = &map[index];

        if remaining_range.start > (entry.src_start + entry.length) {
            index += 1;
            continue;
        }

        if remaining_range.start < entry.src_start {
            mapped_ranges.push(remaining_range.start..entry.src_start);
        }

        let intersection_start = entry.src_start.max(remaining_range.start);
        let intersection_end = (entry.src_start + entry.length).min(remaining_range.end);
        let mapped_start = intersection_start - entry.src_start + entry.dst_start;
        let mapped_end = intersection_end - entry.src_start + entry.dst_start;

        mapped_ranges.push(mapped_start..mapped_end);

        remaining_range.start = intersection_end;
        index += 1;
    }

    mapped_ranges
}

fn day04() {
    let contents = fs::read_to_string("aoc04.txt").unwrap();
    let matches: Vec<u32> = contents
        .lines()
        .map(|line| {
            line.trim()
                .split_once(": ")
                .unwrap()
                .1
                .split_once(" | ")
                .unwrap()
        })
        .map(|(numbers_str, winning_numbers_str)| {
            (collect_numbers(numbers_str) & collect_numbers(winning_numbers_str)).count_ones()
        })
        .collect();

    let result: u64 = matches
        .iter()
        .map(|&game_matches| {
            if game_matches == 0 {
                0
            } else {
                1 << (game_matches - 1)
            }
        })
        .sum();

    println!("{result}");

    let mut total_won: Vec<u64> = Vec::new();
    total_won.resize(matches.len(), 1);

    for (i, &game_matches) in matches.iter().enumerate().rev() {
        for j in 1..=(game_matches as usize) {
            total_won[i] += total_won[i + j];
        }
    }

    let result2: u64 = total_won.iter().sum();

    println!("{result2}");
}

fn collect_numbers(s: &str) -> u128 {
    s.split_whitespace()
        .map(|num_str| num_str.parse::<u8>().unwrap())
        .fold(0, |acc, num| acc | (1 << num))
}

fn day03() {
    let contents = fs::read_to_string("aoc03.txt").unwrap();
    let grid: Vec<Vec<char>> = contents
        .lines()
        .map(|line| line.chars().collect())
        .collect();
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
                for i in (x - num_length)..x {
                    number_positions.insert((i, y), num);
                }
                num = 0;
                num_length = 0;
            }
        }
        for i in (row.len() - num_length)..row.len() {
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
                        .map(|(count_str, cube)| (cube.to_owned(), count_str.parse().unwrap()))
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
                cube_map.iter().all(|(color, &count)| match color.as_str() {
                    "red" => count <= 12,
                    "green" => count <= 13,
                    "blue" => count <= 14,
                    _ => false,
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
                    acc.entry(color.clone())
                        .and_modify(|e| *e = count.max(*e))
                        .or_insert(count);

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
