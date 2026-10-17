# blazing-radix

*The fastest single-threaded sorter in the galaxy.*

Sorts slices of integers and `f32` with an adaptive hybrid radix sort. Safe Rust API, no dependencies, and an optional reusable scratch buffer so repeated sorts don't allocate.

## Usage

```toml
[dependencies]
blazing-radix = "0.1"
```

```rust
use blazing_radix::{radix_sort, radix_sort_with_scratch};

let mut values = [7_i32, -3, 7, 0];
radix_sort(&mut values);
assert_eq!(values, [-3, 0, 7, 7]);

// Reuse one buffer across sorts instead of allocating each time.
let mut scratch = Vec::<i32>::with_capacity(values.len());
radix_sort_with_scratch(&mut values, scratch.spare_capacity_mut());
```

The scratch buffer can be uninitialized and needs at least `values.len()` elements.

Supported types are `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64` and `f32`. Floats sort by `f32::total_cmp`: −0.0 comes before +0.0, NaNs with the sign bit set come first and the rest last, and every bit pattern, including NaN payloads, is preserved.

Run `cargo test --release --locked` to check correctness: 28 engine tests, 32 integration tests and one doctest are included in the crate. CI runs release tests on Linux, macOS and Windows, checks Rust 1.88 compatibility, and enforces formatting and Clippy.

## How it works

It picks among five techniques based on input size, key width, value range and existing order:

| Technique | Used for |
|---|---|
| Radix sorting | General inputs, with adaptive digit widths and skipped constant bytes |
| Counting sort | Byte values, compact ranges and few distinct values |
| Sorting networks | Small arrays and small groups of keys |
| Merge sorting | Existing sorted runs and the output of small sorting networks |
| Insertion-based sorting and repair | Nearly sorted inputs, displaced elements and short tails |

Already-sorted, reversed and all-equal inputs are detected up front and finish early.

## Single-threaded comparison timings

Every library, this one included, sorts on a single thread; the parallel modes of rdst, voracious_radix_sort, turbosort and i_key_sort are turned off. blazing-radix itself is single-threaded today.

Apple M4 Pro, Rust 1.98.1, measured 2026-10-17. Lower is better; the fastest unrounded median in each row is bold. *Default release* builds for Rust's standard ARM64 target; *native release* adds `-C target-cpu=native` for every library.

The complete suite covers **1,374 workloads per build** across all nine supported types. A loss is any positive median difference against the fastest of the five other libraries, with no percentage cutoff. Each workload counts once, even if several libraries beat it.

| Build | Cases | Wins against all five | Exact ties | Losses |
|---|---:|---:|---:|---:|
| Default release | 1,374 | 1,374 | 0 | 0 |
| Native release | 1,374 | 1,374 | 0 | 0 |

### Random inputs, 1,048,576 elements

Times in milliseconds.

| Type | blazing-radix | radsort | rdst | voracious | turbosort | i_key_sort |
|---|---:|---:|---:|---:|---:|---:|
| **Default release** |  |  |  |  |  |  |
| u8 | **0.214** | 0.606 | 0.731 | 0.250 | 0.224 | 0.637 |
| i8 | **0.217** | 0.607 | 0.765 | 0.248 | 0.228 | 0.675 |
| u16 | **0.563** | 1.22 | 1.49 | 1.02 | 1.17 | 1.69 |
| i16 | **0.565** | 1.29 | 1.64 | 0.972 | 1.22 | 1.70 |
| u32 | **2.03** | 3.02 | 2.89 | 4.03 | 2.68 | 7.21 |
| i32 | **2.10** | 3.08 | 3.00 | 4.02 | 2.87 | 7.30 |
| u64 | **2.69** | 6.23 | 5.19 | 3.90 | 4.61 | 7.62 |
| i64 | **2.74** | 6.19 | 5.20 | 4.20 | 4.72 | 7.68 |
| f32 | **2.34** | 3.99 | 3.94 | 5.13 | 3.74 | 11.1 |
| **Native release** |  |  |  |  |  |  |
| u8 | **0.214** | 0.604 | 0.731 | 0.248 | 0.223 | 0.637 |
| i8 | **0.218** | 0.618 | 0.733 | 0.247 | 0.228 | 0.672 |
| u16 | **0.562** | 1.24 | 1.49 | 0.979 | 1.18 | 1.68 |
| i16 | **0.561** | 1.30 | 1.62 | 1.01 | 1.23 | 1.70 |
| u32 | **2.03** | 2.96 | 2.88 | 4.13 | 2.71 | 7.25 |
| i32 | **2.09** | 3.04 | 2.95 | 4.12 | 2.88 | 7.19 |
| u64 | **2.73** | 6.27 | 5.20 | 3.94 | 4.63 | 7.69 |
| i64 | **2.78** | 6.30 | 5.28 | 4.24 | 4.74 | 7.77 |
| f32 | **2.35** | 3.95 | 3.96 | 5.10 | 3.76 | 11.1 |

### Short random arrays

Times in nanoseconds.

| Type | Elements | blazing-radix | radsort | rdst | voracious | turbosort | i_key_sort |
|---|---|---:|---:|---:|---:|---:|---:|
| **Default release** |  |  |  |  |  |  |  |
| u32 | 3 | **1.26** | 452 | 5.09 | 4.25 | 6.87 | 3.50 |
| u32 | 4 | **1.58** | 456 | 7.71 | 5.69 | 7.18 | 5.06 |
| u32 | 8 | **5.29** | 467 | 23.5 | 29.8 | 16.1 | 26.5 |
| u32 | 16 | **15.0** | 481 | 96.6 | 86.4 | 24.7 | 86.1 |
| u32 | 32 | **49.7** | 520 | 175 | 101 | 301 | 95.7 |
| i32 | 3 | **1.26** | 455 | 5.51 | 4.12 | 7.19 | 3.48 |
| i32 | 4 | **1.67** | 459 | 8.31 | 5.54 | 8.60 | 4.74 |
| i32 | 8 | **5.48** | 471 | 36.1 | 23.4 | 17.4 | 26.0 |
| i32 | 16 | **14.7** | 485 | 107 | 82.6 | 24.5 | 85.8 |
| i32 | 32 | **49.8** | 527 | 189 | 95.4 | 310 | 97.5 |
| f32 | 3 | **1.68** | 451 | 5.82 | 8.19 | 8.17 | 5.51 |
| f32 | 4 | **1.81** | 458 | 8.80 | 7.22 | 8.17 | 6.11 |
| f32 | 8 | **5.97** | 475 | 52.2 | 40.2 | 17.4 | 26.7 |
| f32 | 16 | **17.5** | 498 | 176 | 114 | 24.4 | 93.4 |
| f32 | 32 | **53.4** | 575 | 498 | 165 | 388 | 184 |
| **Native release** |  |  |  |  |  |  |  |
| u32 | 3 | **1.23** | 452 | 5.02 | 4.37 | 6.47 | 3.50 |
| u32 | 4 | **1.58** | 453 | 7.49 | 5.84 | 7.87 | 5.14 |
| u32 | 8 | **5.31** | 460 | 30.1 | 32.8 | 16.8 | 32.3 |
| u32 | 16 | **14.8** | 481 | 106 | 91.2 | 23.9 | 92.8 |
| u32 | 32 | **50.2** | 519 | 178 | 98.5 | 303 | 94.6 |
| i32 | 3 | **1.24** | 456 | 5.77 | 4.40 | 6.45 | 3.43 |
| i32 | 4 | **1.76** | 460 | 9.68 | 6.80 | 7.27 | 5.14 |
| i32 | 8 | **5.99** | 470 | 37.9 | 34.2 | 17.9 | 28.4 |
| i32 | 16 | **14.9** | 486 | 111 | 94.7 | 24.7 | 92.4 |
| i32 | 32 | **48.9** | 522 | 188 | 96.3 | 305 | 96.3 |
| f32 | 3 | **1.73** | 453 | 6.34 | 9.57 | 7.43 | 5.86 |
| f32 | 4 | **1.83** | 460 | 8.70 | 7.94 | 9.26 | 6.36 |
| f32 | 8 | **6.09** | 477 | 49.9 | 40.2 | 18.1 | 26.6 |
| f32 | 16 | **17.9** | 513 | 179 | 121 | 25.1 | 103 |
| f32 | 32 | **52.5** | 573 | 498 | 164 | 389 | 181 |

### Structured inputs

Times in microseconds. The inputs are:

- `equal`: every value is 42.
- `shifted`: random values from −100 to 155.
- `prefix`: the top byte cycles through 0–7 by position; the low 32 bits are random.
- `few`: four distinct values.
- `merge4`: four sorted runs, one after another.

| Type | Elements | Input | blazing-radix | radsort | rdst | voracious | turbosort | i_key_sort |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Default release** |  |  |  |  |  |  |  |  |
| u8 | 1,048,576 | equal | **9.50** | 1,420 | 780 | 1,070 | 256 | 9.71 |
| i8 | 1,048,576 | equal | **9.54** | 1,330 | 309 | 1,060 | 257 | 9.79 |
| i16 | 256 | shifted | **0.308** | 1.16 | 1.41 | 2.99 | 4.26 | 0.380 |
| i64 | 256 | shifted | **0.315** | 2.68 | 5.27 | 1.21 | 4.37 | 0.409 |
| u64 | 1,048,576 | prefix | **3,420** | 4,840 | 8,460 | 10,800 | 3,720 | 9,350 |
| i64 | 1,048,576 | prefix | **3,510** | 4,720 | 6,590 | 14,100 | 3,700 | 9,190 |
| f32 | 1,048,576 | few | **614** | 6,940 | 3,250 | 9,130 | 7,730 | 4,850 |
| u32 | 1,048,576 | merge4 | **2,010** | 3,990 | 4,660 | 6,100 | 2,820 | 10,100 |
| **Native release** |  |  |  |  |  |  |  |  |
| u8 | 1,048,576 | equal | **9.27** | 1,430 | 773 | 1,050 | 255 | 9.75 |
| i8 | 1,048,576 | equal | **9.38** | 1,250 | 312 | 1,050 | 258 | 9.79 |
| i16 | 256 | shifted | **0.309** | 1.18 | 1.41 | 3.03 | 4.34 | 0.382 |
| i64 | 256 | shifted | **0.313** | 2.63 | 5.28 | 1.21 | 4.37 | 0.411 |
| u64 | 1,048,576 | prefix | **3,410** | 4,820 | 7,450 | 10,800 | 3,710 | 9,300 |
| i64 | 1,048,576 | prefix | **3,480** | 4,780 | 6,390 | 14,200 | 3,650 | 9,110 |
| f32 | 1,048,576 | few | **606** | 6,880 | 3,210 | 9,100 | 7,580 | 4,790 |
| u32 | 1,048,576 | merge4 | **2,030** | 3,980 | 4,690 | 6,220 | 2,850 | 10,200 |

### How it was measured

The full suite spans 2–16,777,216 elements and includes random, ordered, reversed, equal, low-cardinality, compact-range, run-based and boundary cases.

Libraries: radsort 0.1.1, rdst 0.20.14, voracious_radix_sort 1.2.0, turbosort 0.2.1 and i_key_sort 0.12.0. All parallel features are disabled and every call uses a single-threaded API. Every library, this one included, is called through its public allocating API. i_key_sort sorts floats by a `u32` key that matches `f32::total_cmp` order. Timed float inputs contain no NaNs or infinities.

Each time is the median of three separate process medians; each process measures 14 rounds per library, in rotating order. Small arrays are batched, and every algorithm is warmed before timing. Generating and resetting inputs and checking results are not timed; allocations inside a sort are. Every timed output is checked bit for bit. Runs with detected heavy competing work are retained separately and repeated. Results are for this CPU, and small differences can change between runs.
