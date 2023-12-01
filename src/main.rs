use std::fs;

fn main() {
    day01();
}

fn day01() {
    let contents = fs::read_to_string("aoc01.txt").unwrap();
    let calibration_sum: u64 = contents.lines()
    .map(|line| {
        let first_digit = line.chars().find(|c| c.is_digit(10)).unwrap().to_digit(10).unwrap() as u64;
        let last_digit = line.chars().rfind(|c| c.is_digit(10)).unwrap().to_digit(10).unwrap() as u64;

        first_digit * 10 + last_digit
    }).sum();

    println!("{}", calibration_sum);

    let calibration_sum2: u64 = contents.lines()
    .map(|line| line
        .replace("one", "one1one")
        .replace("two", "two2two")
        .replace("three", "three3three")
        .replace("four", "four4four")
        .replace("five", "five5five")
        .replace("six", "six6six")
        .replace("seven", "seven7seven")
        .replace("eight", "eight8eight")
        .replace("nine", "nine9nine")
    )
    .map(|line| {
        let first_digit = line.chars().find(|c| c.is_digit(10)).unwrap().to_digit(10).unwrap() as u64;
        let last_digit = line.chars().rfind(|c| c.is_digit(10)).unwrap().to_digit(10).unwrap() as u64;

        first_digit * 10 + last_digit
    }).sum();

    println!("{}", calibration_sum2);
}
