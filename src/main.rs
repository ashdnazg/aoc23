#![allow(dead_code)]
use std::{
    collections::{HashMap, HashSet},
    fs,
    iter::zip,
    ops::Range,
};

use itertools::Itertools;

fn main() {
    // day01();
    // day02();
    // day03();
    // day04();
    // day05();
    // day06();
    // day07();
    // day08();
    // day09();
    // day10();
    // day11();
    // day12();
    // day13();
    day14();
}

fn day14() {
    let contents = fs::read_to_string("aoc14.txt").unwrap();
    let grid = contents
        .lines()
        .map(|l| l.chars().collect_vec())
        .collect_vec();
    let result: usize = rotccw(&grid)
        .iter()
        .map(|row| {
            let mut sum = 0;
            let mut last_blocked = 0;
            for (i, c) in row.iter().enumerate() {
                match c {
                    'O' => {
                        last_blocked += 1;
                        sum += row.len() - last_blocked + 1
                    }
                    '.' => {}
                    '#' => last_blocked = i + 1,
                    _ => unreachable!(),
                }
            }
            sum
        })
        .sum();

    println!("{result}");

    let mut cache: HashMap<Vec<Vec<char>>, u64> = HashMap::new();

    let mut steps = 0;
    let mut current_grid = rotccw(&grid);
    while !cache.contains_key(&current_grid) {
        cache.insert(current_grid.clone(), steps);
        for _ in 0..4 {
            roll_blocks(&mut current_grid);
            current_grid = rotcw(&current_grid);
        }
        steps += 1;
    }

    let period_start = cache[&current_grid];
    let period_length = steps - period_start;

    let phase = (1000000000 - period_start) % period_length;

    let phase_grid = cache
        .iter()
        .find(|(_, &s)| s == phase + period_start)
        .unwrap()
        .0;

    let load: usize = phase_grid
        .iter()
        .map(|l| {
            l.iter()
                .enumerate()
                .filter(|(_, &c)| c == 'O')
                .map(|(i, _)| l.len() - i)
                .sum::<usize>()
        })
        .sum();

    println!("{load}")
}

fn rotcw<T: Copy>(grid: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    (0..grid[0].len())
        .map(|x| (0..grid.len()).rev().map(|y| grid[y][x]).collect_vec())
        .collect_vec()
}

fn rotccw<T: Copy>(grid: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    (0..grid[0].len())
        .rev()
        .map(|x| (0..grid.len()).map(|y| grid[y][x]).collect_vec())
        .collect_vec()
}

fn roll_blocks(grid: &mut [Vec<char>]) {
    for row in grid.iter_mut() {
        let mut first_free = 0;
        for i in 0..row.len() {
            match row[i] {
                'O' => {
                    row[i] = '.';
                    row[first_free] = 'O';
                    first_free += 1;
                }
                '.' => {}
                '#' => first_free = i + 1,
                _ => unreachable!(),
            }
        }
    }
}

fn day13() {
    let contents = fs::read_to_string("aoc13.txt").unwrap();
    let grids = contents
        .split("\n\n")
        .map(|grid_str| {
            grid_str
                .lines()
                .map(|l| l.trim().chars().map(|c| c == '#').collect_vec())
                .collect_vec()
        })
        .collect_vec();

    let row_sum: usize = grids.iter().filter_map(find_reflection_row).sum::<usize>() * 100;
    let col_sum: usize = grids
        .iter()
        .map(transpose)
        .filter_map(|g| find_reflection_row(&g))
        .sum();

    println!("{}", row_sum + col_sum);

    let row_sum2: usize = grids
        .iter()
        .filter_map(find_almost_reflection)
        .sum::<usize>()
        * 100;
    let col_sum2: usize = grids
        .iter()
        .map(transpose)
        .filter_map(|g| find_almost_reflection(&g))
        .sum();

    println!("{}", row_sum2 + col_sum2);
}

fn find_reflection_row(grid: &Vec<Vec<bool>>) -> Option<usize> {
    (1..grid.len()).find(|i| {
        let count = usize::min(*i, grid.len() - i);
        (0..count).all(|j| grid[i - j - 1] == grid[i + j])
    })
}

fn find_almost_reflection(grid: &Vec<Vec<bool>>) -> Option<usize> {
    (1..grid.len()).find(|i| {
        let count = usize::min(*i, grid.len() - i);
        (0..count)
            .map(|j| {
                grid[i - j - 1]
                    .iter()
                    .zip(grid[i + j].iter())
                    .filter(|(a, b)| a != b)
                    .count()
            })
            .sum::<usize>()
            == 1
    })
}

fn day12() {
    let contents = fs::read_to_string("aoc12.txt").unwrap();
    let records = contents
        .lines()
        .map(|l| {
            let (conditions_str, groups_str) = l.trim().split_once(' ').unwrap();
            let groups = groups_str.split(',').map(|n| n.parse().unwrap()).collect();
            let (conditions, mask) = conditions_str.chars().rev().fold(
                (0, 0),
                |(conditions_acc, mask_acc), c| match c {
                    '#' => (conditions_acc << 1 | 1, mask_acc << 1 | 1),
                    '.' => (conditions_acc << 1, mask_acc << 1 | 1),
                    '?' => (conditions_acc << 1, mask_acc << 1),
                    _ => unreachable!(),
                },
            );
            Record {
                conditions,
                mask,
                groups,
                length: conditions_str.len() as u8,
            }
        })
        .collect_vec();

    let result: usize = records.iter().map(Record::possible_arrangements).sum();
    println!("{result}");

    let records2 = contents
        .lines()
        .map(|l| {
            let (conditions_str, groups_str) = l.trim().split_once(' ').unwrap();
            let groups = std::iter::repeat(groups_str.split(','))
                .take(5)
                .flatten()
                .map(|n| n.parse().unwrap())
                .collect();
            let (conditions, mask) = std::iter::repeat(conditions_str)
                .take(5)
                .join("?")
                .chars()
                .rev()
                .fold((0, 0), |(conditions_acc, mask_acc), c| match c {
                    '#' => (conditions_acc << 1 | 1, mask_acc << 1 | 1),
                    '.' => (conditions_acc << 1, mask_acc << 1 | 1),
                    '?' => (conditions_acc << 1, mask_acc << 1),
                    _ => unreachable!(),
                });
            Record {
                conditions,
                mask,
                groups,
                length: (conditions_str.len() * 5 + 4) as u8,
            }
        })
        .collect_vec();

    let result2: usize = records2.iter().map(Record::possible_arrangements).sum();
    println!("{result2}");
}

#[derive(Debug)]
struct Record {
    conditions: u128,
    mask: u128,
    groups: Vec<u8>,
    length: u8,
}

impl Record {
    fn possible_arrangements(&self) -> usize {
        let mut cache: HashMap<(u8, usize), usize> = HashMap::new();
        self.recursive_possible_arrangements_memoized(0, 0, &mut cache)
    }

    fn recursive_possible_arrangements_memoized(
        &self,
        min_shift: u8,
        index: usize,
        cache: &mut HashMap<(u8, usize), usize>,
    ) -> usize {
        if let Some(count) = cache.get(&(min_shift, index)) {
            return *count;
        }
        let count = self.recursive_possible_arrangements(min_shift, index, cache);

        cache.insert((min_shift, index), count);

        count
    }

    fn recursive_possible_arrangements(
        &self,
        min_shift: u8,
        index: usize,
        cache: &mut HashMap<(u8, usize), usize>,
    ) -> usize {
        let remaining_bits = self.groups.iter().skip(index).sum::<u8>() as u32;
        let conditions_bits = (self.conditions >> min_shift).count_ones();
        if remaining_bits < conditions_bits {
            return 0;
        }

        if index == self.groups.len() {
            return 1;
        }
        let max_shift: u8 = self.length
            - (self.groups.iter().skip(index).sum::<u8>() + self.groups.len() as u8
                - index as u8
                - 1);
        let group_size = self.groups[index];
        let group = (1u128 << group_size) - 1;
        (min_shift..=max_shift)
            .map(|shift| {
                let shifted_group = group << shift;
                let group_mask = ((1u128 << (group_size + shift - min_shift + 1)) - 1) << min_shift;
                if shifted_group & self.mask == group_mask & self.conditions {
                    self.recursive_possible_arrangements_memoized(
                        shift + group_size + 1,
                        index + 1,
                        cache,
                    )
                } else {
                    0
                }
            })
            .sum()
    }
}

fn day11() {
    let contents = fs::read_to_string("aoc11.txt").unwrap();
    let grid: Vec<Vec<bool>> = contents
        .lines()
        .map(|l| l.chars().map(|c| c == '#').collect())
        .collect();

    let expanded_transposed_grid = transpose(&grid)
        .into_iter()
        .flat_map(|r| {
            if r.iter().all(|b| !b) {
                vec![r.clone(), r]
            } else {
                vec![r]
            }
        })
        .collect_vec();

    let expanded_grid = transpose(&expanded_transposed_grid)
        .into_iter()
        .flat_map(|r| {
            if r.iter().all(|b| !b) {
                vec![r.clone(), r]
            } else {
                vec![r]
            }
        })
        .collect_vec();

    let galaxy_coords: HashSet<_> = (0..expanded_grid.len())
        .cartesian_product(0..expanded_grid[0].len())
        .filter(|&(y, x)| expanded_grid[y][x])
        .collect();

    let result = galaxy_coords
        .iter()
        .cartesian_product(galaxy_coords.iter())
        .map(|(&(x1, y1), &(x2, y2))| x1.abs_diff(x2) + y1.abs_diff(y2))
        .sum::<usize>()
        / 2;

    println!("{result}");

    let transposed_grid = transpose(&grid);

    let expanded_ys = (0..grid.len())
        .scan(0usize, |y, j| {
            let old_y = *y;
            if grid[j].iter().all(|b| !b) {
                *y += 1000000;
            } else {
                *y += 1;
            }

            Some(old_y)
        })
        .collect_vec();

    let expanded_xs = (0..transposed_grid.len())
        .scan(0usize, |x, i| {
            let old_x = *x;
            if transposed_grid[i].iter().all(|b| !b) {
                *x += 1000000;
            } else {
                *x += 1;
            }

            Some(old_x)
        })
        .collect_vec();

    let galaxy_coords2: HashSet<_> = (0..grid.len())
        .cartesian_product(0..grid[0].len())
        .filter(|&(y, x)| grid[y][x])
        .collect();

    let result2 = galaxy_coords2
        .iter()
        .cartesian_product(galaxy_coords2.iter())
        .map(|(&(y1, x1), &(y2, x2))| {
            expanded_xs[x1].abs_diff(expanded_xs[x2]) + expanded_ys[y1].abs_diff(expanded_ys[y2])
        })
        .sum::<usize>()
        / 2;

    println!("{result2}");
}

fn transpose<T: Copy>(v: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    (0..v[0].len())
        .map(|i| (0..v.len()).map(|j| v[j][i]).collect())
        .collect()
}

fn day10() {
    let contents = fs::read_to_string("aoc10.txt").unwrap();
    let mut grid: Vec<Vec<char>> = contents.lines().map(|l| l.chars().collect()).collect();

    let mut start_x: i64 = 0;
    let mut start_y: i64 = 0;
    for (i, j) in (0..grid.len()).cartesian_product(0..grid[0].len()) {
        if grid[i][j] == 'S' {
            start_x = j as i64;
            start_y = i as i64;
            break;
        }
    }

    let mut first_x: i64 = 0;
    let mut first_y: i64 = 0;

    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let connections = get_connections(grid[(start_y + dy) as usize][(start_x + dx) as usize]);
        let reverse_dir = (-dx, -dy);
        if !connections.contains(&reverse_dir) {
            continue;
        }
        first_x = start_x + dx;
        first_y = start_y + dy;

        break;
    }

    let mut steps = 1;
    let mut x = first_x;
    let mut y = first_y;
    let mut last_dx = first_x - start_x;
    let mut last_dy = first_y - start_y;

    let mut loop_coords: HashSet<(i64, i64)> = [(start_x, start_y)].into_iter().collect();

    while !loop_coords.contains(&(x, y)) {
        loop_coords.insert((x, y));
        let connections = get_connections(grid[y as usize][x as usize]);
        let reverse_dir = (-last_dx, -last_dy);
        (last_dx, last_dy) = connections.into_iter().find(|&d| d != reverse_dir).unwrap();
        x += last_dx;
        y += last_dy;

        steps += 1;
    }

    grid[start_y as usize][start_x as usize] =
        get_letter(&[(-last_dx, -last_dy), (first_x - start_x, first_y - start_y)]);

    let result = steps / 2;

    println!("{result}");

    let mut contained_count = 0;

    for coord_sum in 0..(grid.len() + grid[0].len() - 2) {
        let mut flips_encountered = 0;
        for i in 0..=coord_sum {
            let j = coord_sum - i;
            let is_loop = loop_coords.contains(&(i as i64, j as i64));

            if !is_loop && flips_encountered % 2 == 1 {
                contained_count += 1;
            }

            if is_loop && grid[j][i] != 'F' && grid[j][i] != 'J' {
                flips_encountered += 1;
            }
        }
    }

    println!("{contained_count}");
}

fn get_connections(c: char) -> [(i64, i64); 2] {
    match c {
        '|' => [(0, 1), (0, -1)],
        '-' => [(1, 0), (-1, 0)],
        'L' => [(1, 0), (0, -1)],
        'J' => [(-1, 0), (0, -1)],
        '7' => [(-1, 0), (0, 1)],
        'F' => [(1, 0), (0, 1)],
        '.' => [(0, 0), (0, 0)],
        'S' => [(0, 0), (0, 0)],
        _ => unreachable!(),
    }
}

fn get_letter(connections: &[(i64, i64); 2]) -> char {
    let mut copy = *connections;
    copy.sort();
    match copy {
        [(0, -1), (0, 1)] => '|',
        [(-1, 0), (1, 0)] => '-',
        [(0, -1), (1, 0)] => 'L',
        [(0, -1), (-1, 0)] => 'J',
        [(-1, 0), (0, 1)] => '7',
        [(0, 1), (1, 0)] => 'F',
        _ => unreachable!(),
    }
}

fn day09() {
    let contents = fs::read_to_string("aoc09.txt").unwrap();
    let values: Vec<Vec<i64>> = contents
        .lines()
        .map(|l| l.split_whitespace().map(|n| n.parse().unwrap()).collect())
        .collect();

    let result: i64 = values.iter().map(|v| extrapolate(v)).sum();

    println!("{result}");

    let result2: i64 = values
        .iter()
        .map(|v| extrapolate(&v.iter().rev().cloned().collect::<Vec<_>>()))
        .sum();

    println!("{result2}");
}

fn extrapolate(values: &[i64]) -> i64 {
    if values.iter().all(|&v| v == 0) {
        return 0;
    }

    let diffs: Vec<i64> = values.windows(2).map(|pair| pair[1] - pair[0]).collect();

    values.last().unwrap() + extrapolate(&diffs)
}

fn day08() {
    let contents = fs::read_to_string("aoc08.txt").unwrap();
    let (instructions_str, connections_str) = contents.split_once("\n\n").unwrap();
    let connections: HashMap<&str, (&str, &str)> = connections_str
        .lines()
        .map(|line| line.trim_end_matches(')').split_once(" = (").unwrap())
        .map(|(parent, children_str)| (parent, children_str.split_once(", ").unwrap()))
        .collect();

    let instructions: Vec<char> = instructions_str.chars().collect();

    {
        let mut current_node = "AAA";
        let mut steps = 0;
        while current_node != "ZZZ" {
            current_node = match instructions[steps % instructions.len()] {
                'L' => connections[current_node].0,
                'R' => connections[current_node].1,
                _ => unreachable!(),
            };
            steps += 1;
        }
        println!("{steps}");
    }

    {
        let result = connections
            .keys()
            .filter(|&node| node.ends_with('A'))
            .map(|&start| {
                let mut current_node = start;
                let mut steps = 0;
                while !current_node.ends_with('Z') {
                    current_node = match instructions[steps % instructions.len()] {
                        'L' => connections[current_node].0,
                        'R' => connections[current_node].1,
                        _ => unreachable!(),
                    };
                    steps += 1;
                }

                steps as u64
            })
            .reduce(num::integer::lcm)
            .unwrap();

        println!("{result}");
    }
}

fn day07() {
    let contents = fs::read_to_string("aoc07.txt").unwrap();
    let mut cards_vec: Vec<(u32, u64)> = contents
        .lines()
        .map(|line| {
            let (cards_str, bid_str) = line.trim().split_once(' ').unwrap();
            let mut card_counts = [0; 15];
            let mut set_counts = [0; 6];
            let mut cards = 0;
            for c in cards_str.chars() {
                let value = match c {
                    'T' => 10,
                    'J' => 11,
                    'Q' => 12,
                    'K' => 13,
                    'A' => 14,
                    _ => c.to_digit(10).unwrap(),
                };

                cards = (cards << 4) + value;
                if card_counts[value as usize] > 0 {
                    set_counts[card_counts[value as usize]] -= 1;
                }
                card_counts[value as usize] += 1;
                set_counts[card_counts[value as usize]] += 1;
            }

            for (i, count) in set_counts.iter().skip(2).enumerate() {
                cards += count << (20 + i * 2)
            }

            (cards, bid_str.parse().unwrap())
        })
        .collect();

    cards_vec.sort_by_key(|&(cards, _)| cards);

    let result: u64 = cards_vec
        .iter()
        .enumerate()
        .map(|(i, (_, bid))| (i + 1) as u64 * bid)
        .sum();

    println!("{result}");

    let mut cards_vec2: Vec<(u32, u64)> = contents
        .lines()
        .map(|line| {
            let (cards_str, bid_str) = line.trim().split_once(' ').unwrap();
            let mut card_counts = [0; 15];
            let mut set_counts = [5, 0, 0, 0, 0, 0];
            let mut joker_count = 0;
            let mut cards = 0;
            for c in cards_str.chars() {
                let value = match c {
                    'T' => 10,
                    'J' => 1,
                    'Q' => 12,
                    'K' => 13,
                    'A' => 14,
                    _ => c.to_digit(10).unwrap(),
                };

                cards = (cards << 4) + value;

                if c == 'J' {
                    joker_count += 1;
                    continue;
                }

                set_counts[card_counts[value as usize]] -= 1;
                card_counts[value as usize] += 1;
                set_counts[card_counts[value as usize]] += 1;
            }
            let top_count = set_counts
                .iter()
                .enumerate()
                .rfind(|&(_, &count)| count > 0)
                .unwrap()
                .0;
            set_counts[top_count] -= 1;
            set_counts[top_count + joker_count] += 1;

            for (i, count) in set_counts.iter().skip(2).enumerate() {
                cards += count << (20 + i * 2)
            }

            (cards, bid_str.parse().unwrap())
        })
        .collect();

    cards_vec2.sort_by_key(|&(cards, _)| cards);

    let result2: u64 = cards_vec2
        .iter()
        .enumerate()
        .map(|(i, (_, bid))| (i + 1) as u64 * bid)
        .sum();

    println!("{result2}");
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
        .split(' ')
        .map(|num_str| num_str.parse().unwrap())
        .collect();

    let mut maps: Vec<Vec<MapEntry>> = maps_str
        .split("\n\n")
        .map(|map_str| {
            map_str
                .lines()
                .skip(1)
                .map(|entry_str| entry_str.trim().split_once(' ').unwrap())
                .map(|(dst_start_str, rest_str)| {
                    (
                        dst_start_str.parse().unwrap(),
                        rest_str.split_once(' ').unwrap(),
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
            if c.is_ascii_digit() || c == '.' {
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
                        .map(|count_and_cube| count_and_cube.split_once(' ').unwrap())
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
                .find(|c| c.is_ascii_digit())
                .unwrap()
                .to_digit(10)
                .unwrap() as u64;
            let last_digit = line
                .chars()
                .rfind(|c| c.is_ascii_digit())
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
                .find(|c| c.is_ascii_digit())
                .unwrap()
                .to_digit(10)
                .unwrap() as u64;
            let last_digit = line
                .chars()
                .rfind(|c| c.is_ascii_digit())
                .unwrap()
                .to_digit(10)
                .unwrap() as u64;

            first_digit * 10 + last_digit
        })
        .sum();

    println!("{}", calibration_sum2);
}
