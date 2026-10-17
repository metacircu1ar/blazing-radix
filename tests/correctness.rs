use blazing_radix::{RadixKey, radix_sort, radix_sort_with_scratch};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::mem::MaybeUninit;

fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn check<T: RadixKey + Debug>(input: &[T], compare: fn(&T, &T) -> Ordering) {
    let mut expected = input.to_vec();
    expected.sort_unstable_by(compare);
    let mut owned = input.to_vec();
    radix_sort(&mut owned);
    assert!(
        owned
            .iter()
            .zip(&expected)
            .all(|(a, b)| compare(a, b).is_eq())
    );

    // Scratch starts genuinely uninitialized, including on repeated calls.
    let mut scratch = Vec::<T>::with_capacity(input.len());
    for _ in 0..2 {
        let mut actual = input.to_vec();
        radix_sort_with_scratch(&mut actual, scratch.spare_capacity_mut());
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| compare(a, b).is_eq())
        );
    }
}

macro_rules! integer {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            let mut state = 123;
            for len in (0..=257).chain([
                511, 512, 513, 1023, 1024, 1025, 1026, 1027, 4095, 4096, 4097, 4098, 4099, 65_537,
            ]) {
                let mut data: Vec<$ty> = (0..len).map(|_| random(&mut state) as $ty).collect();
                check(&data, <$ty>::cmp);
                data.sort_unstable();
                check(&data, <$ty>::cmp);
                data.reverse();
                check(&data, <$ty>::cmp);
                data.fill(<$ty>::MIN);
                check(&data, <$ty>::cmp);
                for (i, value) in data.iter_mut().enumerate() {
                    *value = [<$ty>::MIN, 0, 1, <$ty>::MAX][i % 4];
                }
                check(&data, <$ty>::cmp);
            }
        }
    };
}

integer!(u8_cases, u8);
integer!(i8_cases, i8);
integer!(u16_cases, u16);
integer!(i16_cases, i16);
integer!(u32_cases, u32);
integer!(i32_cases, i32);
integer!(u64_cases, u64);
integer!(i64_cases, i64);

#[test]
fn wide_digit_dispatch_and_partial_prefixes_preserve_outliers() {
    let mut state = 0xabba;
    for len in [16383, 16384, 16385, 32767, 32768, 32769] {
        for pattern in 0..3 {
            let mut input: Vec<u32> = (0..len).map(|_| random(&mut state) as u32).collect();
            if pattern == 1 {
                input[16..].fill(0xdead_beef);
                input[len - 2] = 0;
            } else if pattern == 2 {
                input[..16].fill(17);
            }
            check(&input, u32::cmp);
            check(
                &input.iter().map(|&v| v as i32).collect::<Vec<_>>(),
                i32::cmp,
            );
            check(
                &input.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
                f32::total_cmp,
            );
        }
    }
    for shift in [32, 40, 48, 56, 61] {
        for len in [256, 257, 4096, 16384] {
            let input: Vec<u64> = (0..len)
                .map(|_| {
                    let bits = random(&mut state);
                    ((bits & 7) << shift) | ((bits >> 16) & 0xffff)
                })
                .collect();
            check(&input, u64::cmp);
            check(
                &input.iter().map(|&v| v as i64).collect::<Vec<_>>(),
                i64::cmp,
            );
        }
    }
}

#[test]
fn short_and_skewed_float_runs_preserve_all_payloads() {
    for short in [7, 8, 9, 31, 32, 33] {
        for long in [65, 127, 128, 129, 1025] {
            for reversed in 0..4 {
                for swapped in [false, true] {
                    let mut a: Vec<_> = (0..short)
                        .map(|i| f32::from_bits(0x7f80_0001 + i * 97))
                        .collect();
                    let mut b: Vec<_> = (0..long)
                        .map(|i| f32::from_bits(0x7f80_0002 + i * 3))
                        .collect();
                    if reversed & 1 != 0 {
                        a.reverse();
                    }
                    if reversed & 2 != 0 {
                        b.reverse();
                    }
                    let input = if swapped {
                        [b, a].concat()
                    } else {
                        [a, b].concat()
                    };
                    check(&input, f32::total_cmp);
                }
            }
        }
    }
}

#[test]
fn every_sixteen_bit_value() {
    let unsigned: Vec<u16> = (u16::MIN..=u16::MAX).rev().collect();
    check(&unsigned, u16::cmp);
    let signed: Vec<i16> = unsigned.into_iter().map(|v| v as i16).collect();
    check(&signed, i16::cmp);
}

#[test]
fn float_total_order_and_payloads() {
    let special = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f7fffff,
        0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0xff800001, 0x7fc00000, 0xffc00000,
        0x7fffffff, 0xffffffff,
    ];
    for bits in special {
        let data = vec![f32::from_bits(bits); 4099];
        check(&data, f32::total_cmp);
    }
    let mut state = 456;
    for len in (0..=257).chain([
        1023, 1024, 1025, 1026, 1027, 4095, 4096, 4097, 4098, 4099, 100_003,
    ]) {
        let mut data: Vec<_> = (0..len)
            .map(|_| f32::from_bits(random(&mut state) as u32))
            .collect();
        check(&data, f32::total_cmp);
        for (i, value) in data.iter_mut().enumerate() {
            *value = f32::from_bits(special[i % special.len()]);
        }
        check(&data, f32::total_cmp);
        data.sort_unstable_by(f32::total_cmp);
        check(&data, f32::total_cmp);
        data.reverse();
        check(&data, f32::total_cmp);
    }
}

#[test]
fn scratch_and_input_subslice_guards() {
    let mut input = [111_i32, 7, -2, i32::MIN, i32::MAX, 0, -1, 222];
    let mut scratch = [MaybeUninit::new(333); 10];
    radix_sort_with_scratch(&mut input[1..7], &mut scratch[1..9]);
    assert_eq!(input, [111, i32::MIN, -2, -1, 0, 7, i32::MAX, 222]);
    for i in [0, 7, 8, 9] {
        // SAFETY: These guard slots were initialized above and must not be used
        // by sorting the six-element slice.
        assert_eq!(unsafe { scratch[i].assume_init() }, 333);
    }
    for len in [33, 1027, 4099] {
        let mut input: Vec<i32> = (0..len as i32 + 2).rev().collect();
        let first = input[0];
        let last = input[len + 1];
        let mut expected = input[1..len + 1].to_vec();
        expected.sort_unstable();
        let mut scratch = vec![MaybeUninit::new(333); len + 5];
        radix_sort_with_scratch(&mut input[1..len + 1], &mut scratch[1..len + 4]);
        assert_eq!(input[0], first);
        assert_eq!(input[len + 1], last);
        assert_eq!(input[1..len + 1], expected);
        for i in [0, len + 1, len + 2, len + 3, len + 4] {
            // SAFETY: Initialized guard slots lie outside the used scratch range.
            assert_eq!(unsafe { scratch[i].assume_init() }, 333);
        }
    }
}

#[test]
fn short_scratch_panics_before_mutating() {
    for len in [3, 33, 4096] {
        let mut input: Vec<u64> = (0..len as u64).rev().collect();
        let expected = input.clone();
        let mut scratch = vec![MaybeUninit::new(77); len - 1];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            radix_sort_with_scratch(&mut input, &mut scratch);
        }));
        assert!(result.is_err());
        assert_eq!(input, expected);
        for slot in scratch {
            // SAFETY: Initialized above; rejection occurs before any writes.
            assert_eq!(unsafe { slot.assume_init() }, 77);
        }
    }
}

#[test]
fn byte_sorts_stay_inside_subslices() {
    // Byte sorts fill short buckets with overlapping stores; the guard bytes
    // around a sorted subslice must stay untouched for every length and for
    // dense, constant, and two-value inputs.
    let mut state = 99;
    for len in (33..=70).chain([255, 256, 257, 1023, 1024, 1027, 4095, 4096, 4099]) {
        for (lo, hi) in [(0_u8, 255_u8), (9, 9), (200, 201), (0, 15)] {
            let span = u64::from(hi - lo) + 1;
            let mut unsigned = vec![0xEE_u8; len + 2];
            for value in &mut unsigned[1..len + 1] {
                *value = lo + (random(&mut state) % span) as u8;
            }
            let mut signed: Vec<i8> = unsigned.iter().map(|&v| v as i8).collect();
            let mut expected = unsigned[1..len + 1].to_vec();
            expected.sort_unstable();
            radix_sort(&mut unsigned[1..len + 1]);
            assert_eq!(unsigned[0], 0xEE);
            assert_eq!(unsigned[len + 1], 0xEE);
            assert_eq!(unsigned[1..len + 1], expected);
            let mut expected: Vec<i8> = signed[1..len + 1].to_vec();
            expected.sort_unstable();
            radix_sort(&mut signed[1..len + 1]);
            assert_eq!(signed[0], 0xEE_u8 as i8);
            assert_eq!(signed[len + 1], 0xEE_u8 as i8);
            assert_eq!(signed[1..len + 1], expected);
        }
    }
}

#[test]
fn no_scratch_needed_for_bytes_or_trivial_slices() {
    let mut bytes = [2_u8, 0, 255, 1];
    radix_sort_with_scratch(&mut bytes, &mut []);
    assert_eq!(bytes, [0, 1, 2, 255]);
    let mut signed = [127_i8, 0, -128, -1];
    radix_sort_with_scratch(&mut signed, &mut []);
    assert_eq!(signed, [-128, -1, 0, 127]);
    radix_sort_with_scratch::<u64>(&mut [], &mut []);
    let mut singleton = [42_u64];
    radix_sort_with_scratch(&mut singleton, &mut []);
    assert_eq!(singleton, [42]);
}

#[test]
fn ordered_prefix_does_not_hide_a_late_inversion() {
    for len in [33, 64, 65, 255, 256, 257, 4097, 8193] {
        let ascending: Vec<_> = (0..len as u64).collect();
        let descending: Vec<_> = ascending.iter().copied().rev().collect();
        for original in [ascending, descending, vec![42_u64; len]] {
            for position in [8, len / 2, len - 2] {
                let mut input = original.clone();
                input[position] = u64::MAX;
                input[position + 1] = 0;
                check(&input, u64::cmp);
                let floats: Vec<_> = input.iter().map(|&v| f32::from_bits(v as u32)).collect();
                check(&floats, f32::total_cmp);
            }
        }
    }
}

#[test]
fn adaptive_prefix_collisions_and_scratch_guards() {
    let mut state = 731;
    for len in [256, 257, 1024, 1025, 4096, 4097, 8191, 8192, 8193, 65_537] {
        for pattern in 0..4 {
            let mut input = vec![0xfeed_u64; len + 2];
            for (i, value) in input[1..len + 1].iter_mut().enumerate() {
                let bits = random(&mut state);
                *value = match pattern {
                    0 => bits,
                    1 => bits & 255,
                    2 => ((i % 8) as u64) << 56 | (bits & 0xffffffff),
                    _ => {
                        if i == 0 {
                            u64::MAX
                        } else {
                            bits & 0xffff
                        }
                    }
                };
            }
            let mut expected = input[1..len + 1].to_vec();
            expected.sort_unstable();
            let signed: Vec<_> = input[1..len + 1].iter().map(|&v| v as i64).collect();
            check(&signed, i64::cmp);
            let mut scratch = vec![MaybeUninit::new(0xbeef); len + 4];
            radix_sort_with_scratch(&mut input[1..len + 1], &mut scratch[1..len + 3]);
            assert_eq!(input[1..len + 1], expected);
            assert_eq!((input[0], input[len + 1]), (0xfeed, 0xfeed));
            for index in [0, len + 1, len + 2, len + 3] {
                // SAFETY: Initialized guards lie outside the used scratch range.
                assert_eq!(unsafe { scratch[index].assume_init() }, 0xbeef);
            }
        }
    }
}

#[test]
fn sampled_keys_are_verified_before_filling() {
    let keys = [0xffc00001_u32, 0xff800001, 0x80000000, 0];
    let mut state = 577;
    for len in [33, 256, 4097] {
        for distinct in 1..=4 {
            let mut bits: Vec<_> = (0..len)
                .map(|_| keys[random(&mut state) as usize % distinct])
                .collect();
            for late_new_key in [false, true] {
                if late_new_key {
                    bits[len - 1] = 0x7fc00002;
                }
                check(&bits, u32::cmp);
                let floats: Vec<_> = bits.iter().map(|&b| f32::from_bits(b)).collect();
                check(&floats, f32::total_cmp);
            }
        }
    }
}

#[test]
fn partial_order_repairs_preserve_values_when_the_budget_expires() {
    for len in [65, 256, 1024, 4097] {
        for descending in [false, true] {
            for pattern in 0..6 {
                let mut input: Vec<_> = (0..len as i64).collect();
                match pattern {
                    0 => input.swap(8, 9),
                    1 => input.rotate_left(1),
                    2 => input.rotate_left(len / 2),
                    3 => {
                        for i in 0..9 {
                            input.swap(8 + i * 2, 9 + i * 2);
                        }
                    }
                    4 => {
                        // A second long insertion exhausts the shift budget.
                        for i in 0..8 {
                            input[len - 8 + i] = i as i64;
                        }
                    }
                    _ => input[..32].fill(0),
                }
                if descending {
                    input.reverse();
                }
                check(&input, i64::cmp);
            }
        }
    }
    // Plateau prefixes must not commit the repair to the wrong direction.
    let mut plateau = vec![1000_i32; 64];
    plateau.extend((0..1000).rev());
    check(&plateau, i32::cmp);

    let mut state = 599;
    let mut floats: Vec<_> = (0..4097)
        .map(|_| f32::from_bits(random(&mut state) as u32))
        .collect();
    floats.extend([
        f32::from_bits(0xff800001),
        -0.0,
        0.0,
        f32::from_bits(0x7fc00042),
    ]);
    floats.sort_unstable_by(f32::total_cmp);
    for descending in [false, true] {
        let mut input = floats.clone();
        if descending {
            input.reverse();
        }
        input.swap(8, 9);
        check(&input, f32::total_cmp);
        input.rotate_left(1);
        check(&input, f32::total_cmp);
    }
}

#[test]
fn compact_ranges_cross_bytes_and_reject_unsampled_outliers() {
    let mut state = 7311;
    for base in [0, 127, 1000, u32::MAX as u64 - 127, u64::MAX - 255] {
        for outlier in [false, true] {
            let mut input: Vec<_> = (0..4097).map(|_| base + random(&mut state) % 256).collect();
            if outlier {
                input[3000] = base.wrapping_add(4096);
            }
            check(&input, u64::cmp);
            let signed: Vec<_> = input.iter().map(|&v| v as i64).collect();
            check(&signed, i64::cmp);
            let words: Vec<_> = input.iter().map(|&v| v as u32).collect();
            check(&words, u32::cmp);
            let signed_words: Vec<_> = input.iter().map(|&v| v as i32).collect();
            check(&signed_words, i32::cmp);
        }
    }
    for base in [0x007fff80_u32, 0x7fffff80, 0xff800080, 0xffffff00] {
        let floats: Vec<_> = (0..4097)
            .map(|_| {
                let key = base + (random(&mut state) % 256) as u32;
                let bits = key ^ (!((key as i32 >> 31) as u32) | (1 << 31));
                f32::from_bits(bits)
            })
            .collect();
        check(&floats, f32::total_cmp);
    }
}

#[test]
fn full_sixteen_bit_histogram_handles_scratch_alignment_and_guards() {
    let mut state = 9377;
    for len in [262143, 262144, 262145] {
        for offset in [0, 1] {
            let input: Vec<_> = (0..len).map(|_| random(&mut state) as u16).collect();
            let mut expected = input.clone();
            expected.sort_unstable();
            let mut actual = vec![0xface_u16];
            actual.extend(input);
            actual.push(0xcafe);
            let mut scratch = vec![MaybeUninit::new(0xbeef_u16); len + 4];
            radix_sort_with_scratch(
                &mut actual[1..len + 1],
                &mut scratch[offset..offset + len + 2],
            );
            assert_eq!(actual[1..len + 1], expected);
            assert_eq!((actual[0], actual[len + 1]), (0xface, 0xcafe));
            for i in (0..offset).chain(offset + len..len + 4) {
                // SAFETY: Initialized guards are outside the permitted range.
                assert_eq!(unsafe { scratch[i].assume_init() }, 0xbeef);
            }
            let signed: Vec<_> = actual[1..len + 1].iter().map(|&v| v as i16).collect();
            check(&signed, i16::cmp);
        }
    }
}

#[test]
fn single_varying_byte_sorts_wide_keys_at_every_position() {
    for len in [256, 1023, 1024, 1025] {
        for byte in 0..8 {
            let mask = 255_u64 << (byte * 8);
            let input: Vec<_> = (0..len)
                .map(|i| (((i as u64 * 157) & 255) << (byte * 8)) | (0xa5a5a5a5a5a5a5a5 & !mask))
                .collect();
            check(&input, u64::cmp);
            let signed: Vec<_> = input.iter().map(|&v| v as i64).collect();
            check(&signed, i64::cmp);
        }
    }
}

fn check_run_pairs<T: RadixKey + Debug>(input: &[T], compare: fn(&T, &T) -> Ordering) {
    let len = input.len();
    for middle in [
        1,
        31,
        32,
        33,
        len / 2,
        len - 33,
        len - 32,
        len - 31,
        len - 1,
    ] {
        let mut runs = input.to_vec();
        runs[..middle].sort_unstable_by(compare);
        runs[middle..].sort_unstable_by(compare);
        for directions in 0..4 {
            let mut data = runs.clone();
            if directions & 1 != 0 {
                data[..middle].reverse();
            }
            if directions & 2 != 0 {
                data[middle..].reverse();
            }
            check(&data, compare);
            let guard = input[0];
            let mut guarded = vec![guard];
            guarded.extend_from_slice(&data);
            guarded.push(guard);
            let mut scratch = vec![MaybeUninit::new(guard); len + 5];
            radix_sort_with_scratch(&mut guarded[1..len + 1], &mut scratch[1..len + 4]);
            data.sort_unstable_by(compare);
            assert!(
                guarded[1..len + 1]
                    .iter()
                    .zip(&data)
                    .all(|(a, b)| compare(a, b).is_eq())
            );
            assert!(compare(&guarded[0], &guard).is_eq());
            assert!(compare(&guarded[len + 1], &guard).is_eq());
            for index in [0, len + 1, len + 2, len + 3, len + 4] {
                // SAFETY: Initialized guards lie outside the first len scratch slots.
                assert!(compare(&unsafe { scratch[index].assume_init() }, &guard).is_eq());
            }
        }
    }
}

#[test]
fn monotone_run_pairs_preserve_values_and_slice_guards() {
    let mut state = 823;
    for len in [64, 65, 66, 127, 128, 129, 256, 1024, 4097] {
        let input: Vec<_> = (0..len).map(|_| random(&mut state)).collect();
        check_run_pairs(&input, u64::cmp);
        macro_rules! converted {
            ($ty:ty) => {
                check_run_pairs(
                    &input.iter().map(|&v| v as $ty).collect::<Vec<_>>(),
                    <$ty>::cmp,
                )
            };
        }
        converted!(i64);
        converted!(u32);
        converted!(i32);
        converted!(u16);
        converted!(i16);
        let mut floats: Vec<_> = input.iter().map(|&v| f32::from_bits(v as u32)).collect();
        floats[..6].copy_from_slice(&[
            -0.0,
            0.0,
            f32::from_bits(0x7f800001),
            f32::from_bits(0xffc00042),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]);
        check_run_pairs(&floats, f32::total_cmp);
    }
}

#[test]
fn extra_runs_and_late_inversions_are_fully_verified() {
    for len in [128_usize, 257, 4097] {
        for runs in [2, 3, 4, 9] {
            let mut input: Vec<_> = (0..len as i64).rev().collect();
            for chunk in input.chunks_mut(len.div_ceil(runs)) {
                chunk.sort_unstable();
            }
            for reverse in [false, true] {
                let mut data = input.clone();
                if reverse {
                    data.reverse();
                }
                check(&data, i64::cmp);
                data.swap(len - 2, len - 3);
                check(&data, i64::cmp);
                data[len - 1] = i64::MIN;
                check(&data, i64::cmp);
            }
        }
    }
}

#[test]
fn wrapped_compact_ranges_preserve_order_and_reject_outliers() {
    for len in [1023, 1024, 1025, 4095, 4096, 4097] {
        for offset in [1_u64, 100, 255, 256] {
            let input: Vec<u64> = (0..len)
                .map(|i| ((i * 157) as u64 & 255).wrapping_sub(offset))
                .collect();
            check(&input, u64::cmp);
            check(
                &input.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                u32::cmp,
            );
            let floats: Vec<_> = input
                .iter()
                .map(|&v| {
                    let key = v as u32;
                    f32::from_bits(if key & (1 << 31) == 0 {
                        !key
                    } else {
                        key ^ (1 << 31)
                    })
                })
                .collect();
            check(&floats, f32::total_cmp);
            for outlier in [256, 1024, 1 << 32, 1 << 63] {
                let mut data = input.clone();
                data[len / 2] = outlier;
                check(&data, u64::cmp);
                check(
                    &data.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                    u32::cmp,
                );
            }
        }
    }
}

#[test]
fn sparse_float_alphabets_preserve_payloads_and_late_unknown_keys() {
    let keys = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f7fffff,
        0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0xff800001, 0x7fc00000, 0xffc00000,
        0x7fffffff,
    ];
    for distinct in 5..=17 {
        for len in [4095, 4096, 4097, 8193] {
            let input: Vec<f32> = (0..len)
                .map(|i| f32::from_bits(keys[(i * 19) % distinct]))
                .collect();
            check(&input, f32::total_cmp);
            for position in [64, 1023, 1024, len - 2] {
                let mut data = input.clone();
                data[position] = f32::from_bits(0x7fc01234);
                check(&data, f32::total_cmp);
            }
        }
    }
}

#[test]
fn alternating_float_runs_preserve_total_order_bits() {
    let mut state = 0x87a4_1367;
    for len in [255, 256, 257, 271, 272, 273, 4097] {
        let mut sorted: Vec<f32> = (0..len)
            .map(|_| f32::from_bits(random(&mut state) as u32))
            .collect();
        sorted[..6].copy_from_slice(&[
            -0.0,
            0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7f800001),
            f32::from_bits(0xffc00042),
        ]);
        sorted.sort_unstable_by(f32::total_cmp);
        let input: Vec<_> = sorted
            .iter()
            .step_by(2)
            .chain(sorted.iter().skip(1).step_by(2))
            .copied()
            .collect();
        check(&input, f32::total_cmp);
        let mut reversed = input;
        reversed.reverse();
        check(&reversed, f32::total_cmp);
    }
}

#[test]
fn nearly_sorted_repairs_cover_budgets_displacements_and_duplicates() {
    let mut state = 0x5eed;
    for len in [64_usize, 256, 1024, 4097] {
        for swaps in [7, 8, 9, 16, 32, 64, 65, 128] {
            for duplicates in [false, true] {
                let mut input: Vec<i64> = (0..len)
                    .map(|_| {
                        let bits = random(&mut state);
                        if duplicates {
                            (bits % 16) as i64
                        } else {
                            bits as i64
                        }
                    })
                    .collect();
                input.sort_unstable();
                for _ in 0..swaps {
                    let i = 8 + random(&mut state) as usize % (len - 9);
                    input.swap(i, i + 1);
                }
                check(&input, i64::cmp);
                let reversed: Vec<_> = input.iter().rev().copied().collect();
                check(&reversed, i64::cmp);
                let floats: Vec<_> = input.iter().map(|&v| v as f32).collect();
                check(&floats, f32::total_cmp);
                let halves: Vec<_> = input.iter().map(|&v| v as u32).collect();
                check(&halves, u32::cmp);
            }
        }
        // Small elements carried later (one long insertion each) and large
        // elements pulled earlier (a tail of one-step insertions each).
        for distance in [1, 2, 8, 64, len / 2] {
            for pulled in [false, true] {
                let mut input: Vec<i64> = (0..len as i64).collect();
                for _ in 0..8 {
                    let d = distance.min(len - 10);
                    let p = 8 + d + random(&mut state) as usize % (len - 9 - d);
                    if pulled {
                        input[p - d..=p].rotate_right(1);
                    } else {
                        input[p - d..=p].rotate_left(1);
                    }
                }
                check(&input, i64::cmp);
                input.reverse();
                check(&input, i64::cmp);
            }
        }
        // An ordered head followed by random data must fail cheaply and exactly.
        let mut input: Vec<u64> = (0..len).map(|_| random(&mut state)).collect();
        input[..len / 4].sort_unstable();
        check(&input, u64::cmp);
        input[..len / 4].reverse();
        check(&input, u64::cmp);
    }
}

/// Check both public APIs on a subslice, including the unused scratch slots.
fn check_guarded<T: RadixKey + Debug>(input: &[T], compare: fn(&T, &T) -> Ordering, guard: T) {
    let mut expected = input.to_vec();
    expected.sort_unstable_by(compare);
    let end = 4 + input.len();
    for with_scratch in [false, true] {
        let mut values = vec![guard; input.len() + 8];
        values[4..end].copy_from_slice(input);
        let mut scratch = vec![MaybeUninit::new(guard); input.len() + 8];
        scratch[4..end].fill(MaybeUninit::uninit());
        if with_scratch {
            radix_sort_with_scratch(&mut values[4..end], &mut scratch[4..end]);
        } else {
            radix_sort(&mut values[4..end]);
        }
        assert!(
            values[4..end]
                .iter()
                .zip(&expected)
                .all(|(a, b)| compare(a, b).is_eq()),
            "output mismatch: type={} len={} with_scratch={with_scratch}",
            std::any::type_name::<T>(),
            input.len(),
        );
        assert!(
            values[..4]
                .iter()
                .chain(&values[end..])
                .all(|v| compare(v, &guard).is_eq()),
            "input guard changed: type={} len={} with_scratch={with_scratch}",
            std::any::type_name::<T>(),
            input.len(),
        );
        for slot in scratch[..4].iter().chain(&scratch[end..]) {
            // SAFETY: These guards were initialized above and are outside the
            // scratch subslice passed to the sorter.
            let stored = unsafe { slot.assume_init_ref() };
            assert!(
                compare(stored, &guard).is_eq(),
                "scratch guard changed: type={} len={}",
                std::any::type_name::<T>(),
                input.len(),
            );
        }
    }
}

#[test]
fn tiny_disordered_slices_preserve_guards_and_exact_bits() {
    let mut state = 0xa974_2b39_1f83_5ce0;
    for len in (2..=17).chain([31, 32, 33]) {
        for _ in 0..64 {
            let random_bits: Vec<u32> = (0..len).map(|_| random(&mut state) as u32).collect();
            let few_bits: Vec<u32> = random_bits
                .iter()
                .map(|&v| [0, 1, 0x8000_0000, u32::MAX][v as usize & 3])
                .collect();
            for bits in [&random_bits, &few_bits] {
                check_guarded(bits, u32::cmp, 0xa5a5_1357);
                check_guarded(
                    &bits.iter().map(|&v| v as i32).collect::<Vec<_>>(),
                    i32::cmp,
                    -123_456_789,
                );
                check_guarded(
                    &bits.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
                    f32::total_cmp,
                    f32::from_bits(0xffc1_2345),
                );
            }
            let special: Vec<f32> = random_bits
                .iter()
                .map(|&v| {
                    f32::from_bits(
                        [
                            0,
                            0x8000_0000,
                            1,
                            0x8000_0001,
                            0x7f80_0000,
                            0xff80_0000,
                            0x7fc1_2345,
                            0xffc5_4321,
                        ][v as usize & 7],
                    )
                })
                .collect();
            check_guarded(&special, f32::total_cmp, f32::from_bits(0x7fff_4321));
        }
    }
}

#[test]
fn equal_byte_arrays_and_late_outliers_preserve_guards() {
    for len in [16, 32, 33, 256, 1024, 4096, 65_536, 1_048_576] {
        for value in [0_u8, 42, u8::MAX] {
            let equal = vec![value; len];
            check_guarded(&equal, u8::cmp, 0xa5);
            check_guarded(
                &equal.iter().map(|&v| v as i8).collect::<Vec<_>>(),
                i8::cmp,
                -91,
            );
        }
        // A long equal prefix must not cause an unchecked early return.
        for outlier in [0_u8, u8::MAX] {
            let mut almost_equal = vec![42_u8; len];
            almost_equal[len - 1] = outlier;
            check_guarded(&almost_equal, u8::cmp, 0xa5);
            check_guarded(
                &almost_equal.iter().map(|&v| v as i8).collect::<Vec<_>>(),
                i8::cmp,
                -91,
            );
        }
    }
}

#[test]
fn equal_float_arrays_preserve_nan_payloads_and_detect_late_outliers() {
    for len in [16, 32, 33, 256, 65_536, 1_048_576] {
        for bits in [
            0_u32,
            0x8000_0000,
            42.0_f32.to_bits(),
            0x7fc1_2345,
            0xffc5_4321,
        ] {
            let mut values = vec![f32::from_bits(bits); len];
            check_guarded(&values, f32::total_cmp, f32::from_bits(0x7fff_1234));
            values[len - 1] = f32::from_bits(bits ^ 0x8000_0000);
            check_guarded(&values, f32::total_cmp, f32::from_bits(0x7fff_1234));
        }
    }
}
