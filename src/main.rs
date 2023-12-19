#![allow(dead_code)]
use std::{
    array,
    collections::{HashMap, HashSet},
    fs,
    iter::zip,
    mem::swap,
    ops::{Range, RangeInclusive},
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
    // day14();
    // day15();
    // day16();
    // day17();
    // day18();
    day19();
}

fn day19() {
    let contents = fs::read_to_string("aoc19.txt").unwrap();
    let (workflows_str, parts_str) = contents.split_once("\n\n").unwrap();
    let workflows: HashMap<&str, Vec<Rule>> = workflows_str
        .lines()
        .map(|l| l.trim().split_once('{').unwrap())
        .map(|(name, rest_str)| {
            (
                name,
                rest_str
                    .trim_end_matches('}')
                    .split(',')
                    .map(Rule::parse)
                    .collect_vec(),
            )
        })
        .collect();

    let parts: Vec<HashMap<char, u64>> = parts_str
        .lines()
        .map(|l| {
            l.trim()
                .trim_matches('}')
                .trim_matches('{')
                .split(',')
                .map(|var_str| var_str.split_once('=').unwrap())
                .map(|(var, value)| (var.chars().next().unwrap(), value.parse().unwrap()))
                .collect()
        })
        .collect();

    let result: u64 = parts
        .iter()
        .filter(|part| {
            let mut target = "in";
            loop {
                match target {
                    "A" => return true,
                    "R" => return false,
                    _ => {}
                }
                target = workflows[target]
                    .iter()
                    .find(|rule| rule.accepts(part))
                    .unwrap()
                    .target
                    .as_str();
            }
        })
        .flat_map(|part| part.values())
        .sum();
    println!("{result}");

    {
        let mut accepted_count: usize = 0;
        let mut part_ranges: Vec<(String, PartRange)> = vec![(
            "in".to_owned(),
            [
                ('x', 1..=4000),
                ('m', 1..=4000),
                ('a', 1..=4000),
                ('s', 1..=4000),
            ]
            .into_iter()
            .collect(),
        )];
        while let Some((target, part_range)) = part_ranges.pop() {
            let mut remaining_range = part_range;
            for rule in &workflows[target.as_str()] {
                let (accepted, rejected) = rule.split(&remaining_range);
                if accepted.values().all(|r| !r.is_empty()) {
                    match rule.target.as_str() {
                        "A" => {
                            accepted_count += accepted
                                .values()
                                .map(|r| r.clone().count())
                                .product::<usize>()
                        }
                        "R" => {}
                        _ => part_ranges.push((rule.target.clone(), accepted)),
                    }
                }
                if rejected.values().any(|r| r.is_empty()) {
                    break;
                }
                remaining_range = rejected;
            }
        }

        println!("{accepted_count}");
    }
}

struct Rule {
    threshold: u64,
    on: char,
    is_greater: bool,
    target: String,
}

type PartRange = HashMap<char, RangeInclusive<u64>>;

impl Rule {
    fn accepts(&self, part: &HashMap<char, u64>) -> bool {
        if self.is_greater {
            part[&self.on] > self.threshold
        } else {
            part[&self.on] < self.threshold
        }
    }

    fn split(&self, part_range: &PartRange) -> (PartRange, PartRange) {
        let mut accepted_result = part_range.clone();
        let mut rejected_result = part_range.clone();

        if self.is_greater {
            accepted_result
                .entry(self.on)
                .and_modify(|r| *r = *r.start().max(&(self.threshold + 1))..=*r.end());
            rejected_result
                .entry(self.on)
                .and_modify(|r| *r = *r.start()..=*r.end().min(&self.threshold));
        } else {
            accepted_result
                .entry(self.on)
                .and_modify(|r| *r = *r.start()..=*r.end().min(&(self.threshold - 1)));
            rejected_result
                .entry(self.on)
                .and_modify(|r| *r = *r.start().max(&self.threshold)..=*r.end());
        }

        (accepted_result, rejected_result)
    }

    fn parse(s: &str) -> Self {
        let Some((condition_str, target)) = s.split_once(':') else {
            return Rule {
                threshold: u64::MAX,
                on: 'x',
                is_greater: false,
                target: s.to_owned(),
            };
        };

        let on = condition_str.chars().next().unwrap();
        let is_greater = condition_str.chars().nth(1).unwrap() == '>';
        let threshold = condition_str.chars().skip(2).join("").parse().unwrap();

        Rule {
            threshold,
            on,
            is_greater,
            target: target.to_owned(),
        }
    }
}

fn day18() {
    let contents = fs::read_to_string("aoc18.txt").unwrap();
    let plan = contents
        .lines()
        .map(|l| l.split_once(' ').unwrap())
        .map(|(dir_str, rest_str)| {
            (
                dir_str.chars().next().unwrap(),
                rest_str.split_once(' ').unwrap(),
            )
        })
        .map(|(dir, (count_str, color_str))| {
            (
                dir,
                count_str.parse::<i64>().unwrap(),
                color_str
                    .trim_matches('(')
                    .trim_matches(')')
                    .trim_matches('#'),
            )
        })
        .map(|(dir, count, color_str)| {
            (
                dir,
                count,
                ['R', 'D', 'L', 'U']
                    [color_str.chars().nth(5).unwrap().to_digit(16).unwrap() as usize],
                i64::from_str_radix(color_str.chars().take(5).join("").as_str(), 16).unwrap(),
            )
        })
        .collect_vec();

    {
        let mut trench: HashSet<(i64, i64)> = HashSet::new();
        let mut x: i64 = 0;
        let mut y: i64 = 0;
        for (dir, count, _, _) in plan.iter() {
            let old_x = x;
            let old_y = y;
            match dir {
                'U' => {
                    y -= count;
                }
                'L' => {
                    x -= count;
                }
                'D' => {
                    y += count;
                }
                'R' => {
                    x += count;
                }
                _ => unreachable!(),
            }
            for i in x.min(old_x)..=x.max(old_x) {
                for j in y.min(old_y)..=y.max(old_y) {
                    trench.insert((i, j));
                }
            }
        }
        let min_x = trench.iter().map(|(x, _)| *x).min().unwrap();
        let max_x = trench.iter().map(|(x, _)| *x).max().unwrap();
        let min_y = trench.iter().map(|(_, y)| *y).min().unwrap();
        let max_y = trench.iter().map(|(_, y)| *y).max().unwrap();

        let mut empty_spaces: HashSet<(i64, i64)> = HashSet::new();
        let x_border_iter = (min_x..=max_x).flat_map(|x| [(x, min_y), (x, max_y)]);
        let y_border_iter = (min_y..=max_y).flat_map(|y| [(min_x, y), (max_x, y)]);
        let mut frontier: Vec<(i64, i64)> = x_border_iter
            .chain(y_border_iter)
            .filter(|&(x, y)| !trench.contains(&(x, y)))
            .collect();

        while !frontier.is_empty() {
            empty_spaces.extend(frontier.iter().cloned());
            frontier = frontier
                .into_iter()
                .flat_map(|(x, y)| [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
                .filter(|&(x, y)| {
                    (min_x..=max_x).contains(&x)
                        && (min_y..=max_y).contains(&y)
                        && !trench.contains(&(x, y))
                        && !empty_spaces.contains(&(x, y))
                })
                .unique()
                .collect_vec()
        }

        let result = (min_x..=max_x).count() * (min_y..=max_y).count() - empty_spaces.len();
        println!("{result}");
    }

    {
        let mut xs: HashSet<i64> = [0].into_iter().collect();
        let mut ys: HashSet<i64> = [0].into_iter().collect();
        {
            let mut x = 0;
            let mut y = 0;
            for (_, _, dir, count) in plan.iter() {
                match dir {
                    'U' => {
                        y -= count;
                    }
                    'L' => {
                        x -= count;
                    }
                    'D' => {
                        y += count;
                    }
                    'R' => {
                        x += count;
                    }
                    _ => unreachable!(),
                }
                xs.insert(x);
                ys.insert(y);
            }
        }

        let x_ranges = xs
            .iter()
            .cloned()
            .sorted()
            .tuple_windows()
            .flat_map(|(x1, x2)| [x1..=x1, (x1 + 1)..=(x2 - 1), x2..=x2])
            .filter(|r| !r.is_empty())
            .unique()
            .collect_vec();
        let y_ranges = ys
            .iter()
            .cloned()
            .sorted()
            .tuple_windows()
            .flat_map(|(y1, y2)| [y1..=y1, (y1 + 1)..=(y2 - 1), y2..=y2])
            .filter(|r| !r.is_empty())
            .unique()
            .collect_vec();

        let big_x_to_x: HashMap<i64, i64> = x_ranges
            .iter()
            .enumerate()
            .filter(|(_, r)| r.start() == r.end())
            .map(|(i, r)| (*r.start(), i as i64))
            .collect();
        let big_y_to_y: HashMap<i64, i64> = y_ranges
            .iter()
            .enumerate()
            .filter(|(_, r)| r.start() == r.end())
            .map(|(i, r)| (*r.start(), i as i64))
            .collect();

        let mut big_x = 0;
        let mut big_y = 0;
        let mut trench: HashSet<(i64, i64)> = HashSet::new();
        for (_, _, dir, count) in plan.iter() {
            let old_big_x = big_x;
            let old_big_y = big_y;
            match dir {
                'U' => {
                    big_y -= count;
                }
                'L' => {
                    big_x -= count;
                }
                'D' => {
                    big_y += count;
                }
                'R' => {
                    big_x += count;
                }
                _ => unreachable!(),
            }
            for i in big_x_to_x[&big_x].min(big_x_to_x[&old_big_x])
                ..=big_x_to_x[&big_x].max(big_x_to_x[&old_big_x])
            {
                for j in big_y_to_y[&big_y].min(big_y_to_y[&old_big_y])
                    ..=big_y_to_y[&big_y].max(big_y_to_y[&old_big_y])
                {
                    trench.insert((i, j));
                }
            }
        }

        let max_x = trench.iter().map(|(x, _)| *x).max().unwrap();
        let max_y = trench.iter().map(|(_, y)| *y).max().unwrap();

        let mut empty_spaces: HashSet<(i64, i64)> = HashSet::new();
        let x_border_iter = (0..=max_x).flat_map(|x| [(x, 0), (x, max_y)]);
        let y_border_iter = (0..=max_y).flat_map(|y| [(0, y), (max_x, y)]);
        let mut frontier: Vec<(i64, i64)> = x_border_iter
            .chain(y_border_iter)
            .filter(|&(x, y)| !trench.contains(&(x, y)))
            .collect();

        while !frontier.is_empty() {
            empty_spaces.extend(frontier.iter().cloned());
            frontier = frontier
                .into_iter()
                .flat_map(|(x, y)| [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
                .filter(|&(x, y)| {
                    (0..=max_x).contains(&x)
                        && (0..=max_y).contains(&y)
                        && !trench.contains(&(x, y))
                        && !empty_spaces.contains(&(x, y))
                })
                .unique()
                .collect_vec()
        }

        let result2: usize = (0..=max_x)
            .cartesian_product(0..=max_y)
            .filter(|&(x, y)| !empty_spaces.contains(&(x, y)))
            .map(|(x, y)| {
                x_ranges[x as usize].clone().count() * y_ranges[y as usize].clone().count()
            })
            .sum();

        println!("{result2}");
    }
}

fn day17() {
    let contents = fs::read_to_string("aoc17.txt").unwrap();
    let grid = contents
        .lines()
        .map(|l| l.chars().map(|c| c.to_digit(10).unwrap()).collect_vec())
        .collect_vec();

    let max_x = grid[0].len() as i64;
    let max_y = grid.len() as i64;
    {
        let starting_state = State {
            x: 0,
            y: 0,
            dx: 1,
            dy: 0,
            steps: 0,
        };
        let mut frontier: Vec<(State, u32)> = vec![(starting_state.clone(), 0)];
        let mut next_frontier: Vec<(State, u32)> = vec![];
        let mut best_visited: HashMap<State, u32> = HashMap::new();
        best_visited.insert(starting_state.clone(), 0);
        while !frontier.is_empty() {
            frontier.drain(0..frontier.len()).for_each(|(state, cost)| {
                expand(&state, max_x, max_y).for_each(|new_state| {
                    let new_cost = cost + grid[new_state.y as usize][new_state.x as usize];
                    if let Some(&best_cost) = best_visited.get(&new_state) {
                        if best_cost <= new_cost {
                            return;
                        }
                    }
                    best_visited.insert(new_state.clone(), new_cost);
                    next_frontier.push((new_state.clone(), new_cost));
                })
            });
            swap(&mut frontier, &mut next_frontier);
        }

        let result = best_visited
            .iter()
            .filter(|(s, _)| s.x == max_x - 1 && s.y == max_y - 1)
            .map(|(_, cost)| cost)
            .min()
            .unwrap();
        println!("{result}");
    }

    {
        let starting_state1 = State {
            x: 0,
            y: 0,
            dx: 1,
            dy: 0,
            steps: 0,
        };
        let starting_state2 = State {
            x: 0,
            y: 0,
            dx: 0,
            dy: 1,
            steps: 0,
        };
        let mut frontier: Vec<(State, u32)> =
            vec![(starting_state1.clone(), 0), (starting_state2.clone(), 0)];
        let mut next_frontier: Vec<(State, u32)> = vec![];
        let mut best_visited: HashMap<State, u32> = HashMap::new();
        best_visited.insert(starting_state1.clone(), 0);
        best_visited.insert(starting_state2.clone(), 0);
        while !frontier.is_empty() {
            frontier.drain(0..frontier.len()).for_each(|(state, cost)| {
                ultra_expand(&state, max_x, max_y).for_each(|new_state| {
                    let new_cost = cost + grid[new_state.y as usize][new_state.x as usize];
                    if let Some(&best_cost) = best_visited.get(&new_state) {
                        if best_cost <= new_cost {
                            return;
                        }
                    }
                    best_visited.insert(new_state.clone(), new_cost);
                    next_frontier.push((new_state.clone(), new_cost));
                })
            });
            swap(&mut frontier, &mut next_frontier);
        }

        let result2 = best_visited
            .iter()
            .filter(|(s, _)| s.x == max_x - 1 && s.y == max_y - 1)
            .map(|(_, cost)| cost)
            .min()
            .unwrap();
        println!("{result2}");
    }
}

fn expand(state: &State, max_x: i64, max_y: i64) -> impl Iterator<Item = State> {
    [
        State {
            x: state.x + state.dx,
            y: state.y + state.dy,
            dx: state.dx,
            dy: state.dy,
            steps: state.steps + 1,
        },
        State {
            x: state.x + state.dy,
            y: state.y + state.dx,
            dx: state.dy,
            dy: state.dx,
            steps: 1,
        },
        State {
            x: state.x - state.dy,
            y: state.y - state.dx,
            dx: -state.dy,
            dy: -state.dx,
            steps: 1,
        },
    ]
    .into_iter()
    .filter(move |new_state| {
        new_state.steps <= 3
            && new_state.x >= 0
            && new_state.y >= 0
            && new_state.y < max_y
            && new_state.x < max_x
    })
}

fn ultra_expand(state: &State, max_x: i64, max_y: i64) -> impl Iterator<Item = State> {
    let state_steps = state.steps;
    let state_dx = state.dx;
    [
        State {
            x: state.x + state.dx,
            y: state.y + state.dy,
            dx: state.dx,
            dy: state.dy,
            steps: state.steps + 1,
        },
        State {
            x: state.x + state.dy,
            y: state.y + state.dx,
            dx: state.dy,
            dy: state.dx,
            steps: 1,
        },
        State {
            x: state.x - state.dy,
            y: state.y - state.dx,
            dx: -state.dy,
            dy: -state.dx,
            steps: 1,
        },
    ]
    .into_iter()
    .filter(move |new_state| {
        (state_dx == new_state.dx || state_steps >= 4)
            && new_state.steps <= 10
            && new_state.x >= 0
            && new_state.y >= 0
            && new_state.y < max_y
            && new_state.x < max_x
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct State {
    x: i64,
    y: i64,
    dx: i64,
    dy: i64,
    steps: u64,
}

fn day16() {
    let contents = fs::read_to_string("aoc16.txt").unwrap();
    let grid = contents
        .lines()
        .map(|l| l.chars().collect_vec())
        .collect_vec();
    {
        let result = energize(
            &Beam {
                x: 0,
                y: 0,
                dx: 1,
                dy: 0,
            },
            &grid,
        );
        println!("{result}");
    }

    {
        let y_beams_iter = (0..grid.len()).flat_map(|y| {
            vec![
                Beam {
                    x: 0,
                    y: y as i64,
                    dx: 1,
                    dy: 0,
                },
                Beam {
                    x: grid[0].len() as i64 - 1,
                    y: y as i64,
                    dx: -1,
                    dy: 0,
                },
            ]
        });
        let x_beams_iter = (0..grid[0].len()).flat_map(|x| {
            vec![
                Beam {
                    x: x as i64,
                    y: 0,
                    dx: 0,
                    dy: 1,
                },
                Beam {
                    x: x as i64,
                    y: grid.len() as i64 - 1,
                    dx: 0,
                    dy: -1,
                },
            ]
        });
        let result2 = x_beams_iter
            .chain(y_beams_iter)
            .map(|beam| energize(&beam, &grid))
            .max()
            .unwrap();

        println!("{result2}");
    }
}

fn energize(starting_beam: &Beam, grid: &[Vec<char>]) -> usize {
    let mut inactive_beams: HashSet<Beam> = HashSet::new();
    let mut beams: Vec<Beam> = vec![starting_beam.clone()];
    while !beams.is_empty() {
        inactive_beams.extend(beams.iter().cloned());
        beams = beams
            .into_iter()
            .flat_map(|beam| advance(&beam, grid))
            .filter(|beam| {
                beam.x >= 0
                    && beam.y >= 0
                    && beam.x < grid[0].len() as i64
                    && beam.y < grid.len() as i64
            })
            .filter(|beam| !inactive_beams.contains(beam))
            .collect_vec();
    }

    inactive_beams
        .iter()
        .map(|beam| (beam.x, beam.y))
        .unique()
        .count()
}

fn advance(beam: &Beam, grid: &[Vec<char>]) -> Vec<Beam> {
    match grid[beam.y as usize][beam.x as usize] {
        '.' => vec![Beam {
            x: beam.x + beam.dx,
            y: beam.y + beam.dy,
            dx: beam.dx,
            dy: beam.dy,
        }],
        '/' => vec![Beam {
            x: beam.x - beam.dy,
            y: beam.y - beam.dx,
            dx: -beam.dy,
            dy: -beam.dx,
        }],
        '\\' => vec![Beam {
            x: beam.x + beam.dy,
            y: beam.y + beam.dx,
            dx: beam.dy,
            dy: beam.dx,
        }],
        '|' => {
            if beam.dx == 0 {
                vec![Beam {
                    x: beam.x + beam.dx,
                    y: beam.y + beam.dy,
                    dx: beam.dx,
                    dy: beam.dy,
                }]
            } else {
                vec![
                    Beam {
                        x: beam.x,
                        y: beam.y + 1,
                        dx: 0,
                        dy: 1,
                    },
                    Beam {
                        x: beam.x,
                        y: beam.y - 1,
                        dx: 0,
                        dy: -1,
                    },
                ]
            }
        }
        '-' => {
            if beam.dy == 0 {
                vec![Beam {
                    x: beam.x + beam.dx,
                    y: beam.y + beam.dy,
                    dx: beam.dx,
                    dy: beam.dy,
                }]
            } else {
                vec![
                    Beam {
                        x: beam.x + 1,
                        y: beam.y,
                        dx: 1,
                        dy: 0,
                    },
                    Beam {
                        x: beam.x - 1,
                        y: beam.y,
                        dx: -1,
                        dy: 0,
                    },
                ]
            }
        }
        _ => unreachable!(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct Beam {
    x: i64,
    y: i64,
    dx: i64,
    dy: i64,
}

fn day15() {
    let contents = fs::read_to_string("aoc15.txt").unwrap();
    let steps = contents.trim().split(',').collect_vec();

    let result: u32 = steps
        .iter()
        .map(|step| {
            step.chars()
                .fold(0, |acc, c| ((acc + (c as u32)) * 17) % 256)
        })
        .sum();

    println!("{result}");

    let mut boxes: [Vec<(&str, usize)>; 256] = array::from_fn(|_| vec![]);

    for step in steps.iter() {
        if let Some((label, num_str)) = step.split_once('=') {
            let new_entry = (label, num_str.parse().unwrap());
            let box_index = label
                .chars()
                .fold(0, |acc, c| ((acc + (c as usize)) * 17) % 256);
            if let Some(existing_entry) = boxes[box_index]
                .iter_mut()
                .find(|(existing_label, _)| *existing_label == label)
            {
                *existing_entry = new_entry;
            } else {
                boxes[box_index].push(new_entry)
            }
        } else {
            let label = step.trim_end_matches('-');
            let box_index = label
                .chars()
                .fold(0, |acc, c| ((acc + (c as usize)) * 17) % 256);
            if let Some((position, _)) = boxes[box_index]
                .iter()
                .find_position(|(existing_label, _)| *existing_label == label)
            {
                boxes[box_index].remove(position);
            }
        }
    }

    let result2: usize = boxes
        .iter()
        .enumerate()
        .map(|(box_num, lenses)| {
            lenses
                .iter()
                .enumerate()
                .map(|(slot, (_, focal_length))| (1 + box_num) * (slot + 1) * focal_length)
                .sum::<usize>()
        })
        .sum();
    println!("{result2}");
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
