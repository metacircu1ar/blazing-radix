use super::*;

#[test]
fn numeric_float_ranges_verify_bits_before_writing() {
    for base in [
        -2_147_483_648_f32,
        -128.0,
        0.0,
        16_777_216.0,
        2_147_483_520.0,
    ] {
        let input: Vec<_> = (0..2051).map(|i| base + ((i * 157) % 129) as f32).collect();
        let mut expected = input.clone();
        expected.sort_unstable_by(f32::total_cmp);
        let mut actual = input.clone();
        assert!(float_range(&mut actual));
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    for bad in [
        -0.0,
        0.5,
        256.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0123),
        f32::from_bits(0xff80_0001),
        4_294_967_296.0,
    ] {
        for index in [0, 15, 16, 63, 64, 1023, 1024, 1025, 2050] {
            let mut input: Vec<_> = (0..2051).map(|i| ((i * 157) % 256) as f32).collect();
            input[index] = bad;
            let original: Vec<_> = input.iter().map(|v| v.to_bits()).collect();
            // A sampled outlier selects the grid proposal, which rejects
            // this input; an integer outside the span widens it past 256
            // keys. Any other single outlier is merged as an exception.
            let sampled = matches!(index, 0 | 15 | 1025 | 2050);
            if sampled || bad == 256.0 {
                assert!(!float_range(&mut input));
                assert_eq!(
                    input.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    original
                );
            } else {
                let mut expected: Vec<_> = original.iter().map(|&b| f32::from_bits(b)).collect();
                expected.sort_unstable_by(f32::total_cmp);
                assert!(float_range(&mut input));
                assert_eq!(
                    input.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn numeric_float_ranges_merge_bounded_exceptions_by_total_order() {
    let specials = [
        0.5,
        -0.0,
        100.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0123),
        f32::from_bits(0xff80_0001),
        4_294_967_296.0,
        -2_147_483_904.0,
        1e-40,
        255.25,
    ];
    let check = |input: &[f32], accepted: bool| {
        let mut actual = input.to_vec();
        let original: Vec<_> = input.iter().map(|v| v.to_bits()).collect();
        assert_eq!(float_range(&mut actual), accepted);
        let mut expected = input.to_vec();
        if accepted {
            expected.sort_unstable_by(f32::total_cmp);
        }
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            if accepted {
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            } else {
                original
            }
        );
    };
    for len in [129, 2051, 4097] {
        // The verified span excludes exceptions: 10..=265 spans 255 keys
        // even though the truncated exception keys fall outside it.
        let base: Vec<f32> = (0..len).map(|i| 10.0 + ((i * 157) % 256) as f32).collect();
        check(&base, true);
        for &special in &specials {
            for position in [16, 17, 63, 64, 65, 1023, 1024, len / 2 + 1, len - 2] {
                if position >= len || position == len / 2 {
                    continue;
                }
                let mut input = base.clone();
                input[position] = special;
                check(&input, true);
            }
        }
        // Exactly the limit is merged; one more rejects before writing.
        let mut input = base.clone();
        for (slot, &special) in input[20..28].iter_mut().zip(&specials) {
            *slot = special;
        }
        check(&input, true);
        let mut spread = base.clone();
        for (i, &special) in specials[..8].iter().enumerate() {
            spread[16 + i * (len - 20) / 8] = special;
        }
        check(&spread, true);
        input[len - 2] = specials[8];
        check(&input, false);
        spread[len - 2] = specials[8];
        check(&spread, false);
        // Exceptions in the sample positions use the grid proposal instead.
        for position in [0, 15, len - 1, len / 2] {
            let mut input = base.clone();
            input[position] = 0.5;
            check(&input, false);
        }
    }
}

#[test]
fn repair_budgets_count_cheap_and_long_repairs_separately() {
    // Adjacent swaps after the probe, then groups of small elements each
    // moved `distance` places later: every group is placed from the end
    // backwards so that each move is one insertion of `distance` shifts.
    fn attempt(len: usize, swaps: usize, moves: &[(usize, usize)]) -> (bool, Vec<u64>) {
        let mut values: Vec<u64> = (0..len as u64).collect();
        for k in 0..swaps {
            let i = 16 + k * 3;
            values.swap(i, i + 1);
        }
        let mut p = len - 8;
        for &(count, distance) in moves {
            for _ in 0..count {
                values[p - distance..=p].rotate_left(1);
                p -= distance + 2;
            }
        }
        let original = values.clone();
        let start = ordered_scan::<u64, false>(&values);
        assert!(start < len);
        let repaired = ordered_repair::<u64, false>(&mut values, start);
        let mut sorted = original.clone();
        sorted.sort_unstable();
        if repaired {
            assert_eq!(values, sorted);
        } else {
            let mut remaining = values.clone();
            remaining.sort_unstable();
            assert_eq!(remaining, sorted, "a failed repair must keep the multiset");
        }
        (repaired, values)
    }
    // Adjacent swaps settle a window at a time, so even many of them
    // stay within the cheap budget.
    for swaps in [1, 7, 8, 64, 65, 128, 512] {
        assert!(attempt(4096, swaps, &[]).0, "{swaps} adjacent swaps");
    }
    // Elements moved a few places are inserted one by one and count as
    // cheap repairs: 4096 elements allow 256 of them.
    for moves in [1, 8, 255, 256] {
        assert!(attempt(4096, 0, &[(moves, 5)]).0, "{moves} short moves");
    }
    assert!(
        !attempt(4096, 0, &[(257, 5)]).0,
        "257 short moves exceed the cheap budget"
    );
    for moves in [1, 7, 8] {
        assert!(attempt(4096, 0, &[(moves, 100)]).0, "{moves} long moves");
    }
    assert!(
        !attempt(4096, 0, &[(9, 100)]).0,
        "nine long moves exceed the budget"
    );
    // Mixed: both budgets are independent; one more of either fails.
    assert!(attempt(8192, 0, &[(8, 100), (512, 5)]).0);
    assert!(!attempt(8192, 0, &[(8, 100), (513, 5)]).0);
    assert!(!attempt(8192, 0, &[(9, 100), (512, 5)]).0);
    // A large element pulled forward is carried back in one move at a
    // quarter unit per element; nine such moves exceed the long budget.
    for (distance, pulled, expected) in [
        (1, 1, true),
        (2, 1, true),
        (64, 1, true),
        (65, 1, true),
        (300, 1, true),
        (100, 8, true),
        (100, 9, false),
        (600, 8, true),
        (800, 8, true),
    ] {
        let mut values: Vec<u64> = (0..8192).collect();
        for k in 0..pulled {
            let p = 8000 - k * (distance + 2);
            values[p - distance..=p].rotate_right(1);
        }
        let original = values.clone();
        let start = ordered_scan::<u64, false>(&values);
        assert_eq!(
            ordered_repair::<u64, false>(&mut values, start),
            expected,
            "{pulled} pulled by {distance}"
        );
        let mut sorted = original;
        sorted.sort_unstable();
        if !expected {
            values.sort_unstable();
        }
        assert_eq!(values, sorted);
    }
    // Units stay bounded by len regardless of the move counts: elements
    // inserted about 1020 places cost 255 units each, so four fit in
    // 1024 elements and a fifth does not.
    for (tail, expected) in [(1, true), (4, true), (5, false)] {
        let mut values: Vec<u64> = (0..1024).collect();
        for (slot, small) in values[1024 - tail..].iter_mut().zip(0..) {
            *slot = small;
        }
        let original = values.clone();
        assert_eq!(
            ordered_repair::<u64, false>(&mut values, 1024 - tail),
            expected
        );
        let mut sorted = original;
        sorted.sort_unstable();
        values.sort_unstable();
        assert_eq!(values, sorted);
    }
}

#[test]
fn float_scans_agree_with_key_order_on_mixed_signs_and_offsets() {
    fn reference<const DESCENDING: bool>(values: &[f32]) -> usize {
        for i in 1..values.len() {
            let (a, b) = (values[i - 1].key(), values[i].key());
            if if DESCENDING { a < b } else { a > b } {
                return i;
            }
        }
        values.len()
    }
    let mut state = 0x5eed_u64;
    let mut random = move || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    let specials = [
        0.0_f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_0123),
        f32::from_bits(0xff80_0001),
        1e-40,
        -1e-40,
    ];
    for trial in 0..4000 {
        let len = 1 + (random() % 200) as usize;
        // Sorted mixed-sign data, sometimes with specials, then a few
        // random disturbances so that inversions land in every kind of
        // block, including those starting negative and ending positive.
        let mut values: Vec<f32> = (0..len)
            .map(|_| {
                if trial % 3 == 0 && random() % 8 == 0 {
                    specials[random() as usize % specials.len()]
                } else {
                    (random() % 2001) as f32 - 1000.0
                }
            })
            .collect();
        values.sort_by(f32::total_cmp);
        if trial % 2 == 1 {
            values.reverse();
        }
        for _ in 0..(random() % 3) {
            let i = random() as usize % len;
            let j = random() as usize % len;
            values.swap(i, j);
        }
        for offset in [0, 1, 7, 17, 31, 33] {
            if offset >= len {
                break;
            }
            let tail = &values[offset..];
            assert_eq!(
                ordered_scan::<f32, false>(tail),
                reference::<false>(tail),
                "ascending trial {trial} offset {offset}: {tail:?}"
            );
            assert_eq!(
                ordered_scan::<f32, true>(tail),
                reference::<true>(tail),
                "descending trial {trial} offset {offset}: {tail:?}"
            );
        }
    }
    // The specific shape that hides an inversion without the exact
    // check: a block starting negative with a reversed positive pair.
    let mut values: Vec<f32> = (0..64).map(|i| i as f32 - 4.0).collect();
    values.swap(40, 41);
    assert_eq!(ordered_scan::<f32, false>(&values), 41);
    let mut values: Vec<f32> = (0..64).map(|i| 59.0 - i as f32).collect();
    values.swap(40, 41);
    assert_eq!(ordered_scan::<f32, true>(&values), 41);
}

#[test]
fn settled_windows_recheck_the_prefix_boundary() {
    fn check<const DESCENDING: bool>() {
        // A local swap at 20/21 selects window settling; the window also
        // holds 5, which belongs near the front. Settling carries 5 to
        // the window start, where only a rescan from the verified
        // element before it can expose the remaining inversion.
        // An earlier adjacent swap makes the next repair dense enough to
        // settle a window instead of inserting one element.
        let mut input: Vec<u64> = (0..128).map(|i| i * 10).collect();
        input.swap(12, 13);
        input.swap(20, 21);
        input[23] = 5;
        input[127] = 230;
        if DESCENDING {
            input.iter_mut().for_each(|value| *value = !*value);
        }
        let mut expected = input.clone();
        expected.sort_unstable();
        if DESCENDING {
            expected.reverse();
        }
        let start = ordered_scan::<u64, DESCENDING>(&input);
        assert_eq!(start, 13);
        let mut values = input.clone();
        assert!(ordered_repair::<u64, DESCENDING>(&mut values, start));
        assert_eq!(values, expected);
        crate::radix_sort(&mut input);
        if DESCENDING {
            expected.reverse();
        }
        assert_eq!(input, expected);
    }
    check::<false>();
    check::<true>();
}

#[test]
fn forward_carry_respects_the_last_two_shifts() {
    fn check<const DESCENDING: bool>() {
        for remaining in 0..=2 {
            for carry_start in [700, 1021] {
                // Eight long insertions cost 127 units each and leave
                // eight. A short insertion consumes all but `remaining`,
                // followed by a carry of exactly two elements costing one
                // unit, including the case that reaches the slice end.
                let mut input: Vec<u64> = (8..=515).chain(0..8).chain(516..1024).collect();
                input[600 - (8 - remaining)..=600].rotate_left(1);
                input[carry_start..carry_start + 3].rotate_right(1);
                if DESCENDING {
                    input.iter_mut().for_each(|value| *value = !*value);
                }
                let mut expected = input.clone();
                expected.sort_unstable();
                let mut guarded = vec![42];
                guarded.extend_from_slice(&input);
                guarded.push(42);
                let values = &mut guarded[1..1025];
                let start = ordered_scan::<u64, DESCENDING>(values);
                let repaired = ordered_repair::<u64, DESCENDING>(values, start);
                assert_eq!(repaired, remaining >= 1, "{remaining} {carry_start}");
                if repaired {
                    assert_eq!(ordered_scan::<u64, DESCENDING>(values), values.len());
                }
                values.sort_unstable();
                assert_eq!(values, expected);
                assert_eq!((guarded[0], guarded[1025]), (42, 42));

                // Public sorting must complete correctly after rejecting
                // repair, with both allocating and reusable-scratch APIs.
                let mut actual = input.clone();
                crate::radix_sort(&mut actual);
                assert_eq!(actual, expected);
                let mut scratch = vec![MaybeUninit::uninit(); input.len()];
                crate::radix_sort_with_scratch(&mut input, &mut scratch);
                assert_eq!(input, expected);
            }
        }
    }
    check::<false>();
    check::<true>();
}

fn check_group<T: Sealed>(input: Vec<T>) {
    let len = input.len();
    let mut expected = input.clone();
    expected.sort_unstable_by_key(|v| v.key());
    let guard = T::from_key(0xbeef);
    let mut values = vec![guard];
    values.extend(input);
    values.push(guard);
    let mut scratch = vec![MaybeUninit::new(guard); len + 2];
    finish_group::<T, u32>(&mut values[1..len + 1], &mut scratch[1..len + 1]);
    assert!(
        values[1..len + 1]
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.key() == b.key())
    );
    assert_eq!(
        (values[0].key(), values[len + 1].key()),
        (guard.key(), guard.key())
    );
    // SAFETY: Initialized guards lie outside the group's scratch range.
    unsafe {
        assert_eq!(scratch[0].assume_init().key(), guard.key());
        assert_eq!(scratch[len + 1].assume_init().key(), guard.key());
    }
}

#[test]
fn group_networks_sort_every_length_including_padding_collisions() {
    let mut state = 0x6e74_u64;
    let mut random = move || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    for len in 0..=140 {
        for pattern in 0..7 {
            let keys: Vec<u64> = (0..len)
                .map(|i| match pattern {
                    0 => random(),
                    1 => u64::MAX - (random() % 3),
                    2 => [0, u64::MAX, 1 << 63, (1 << 63) - 1][random() as usize % 4],
                    3 => i as u64,
                    4 => (len - i) as u64,
                    5 => 42,
                    _ => random() % 4,
                })
                .collect();
            check_group::<u64>(keys.clone());
            check_group::<i64>(keys.iter().map(|&k| k as i64).collect());
            check_group::<u16>(keys.iter().map(|&k| k as u16).collect());
            check_group::<FloatBits>(keys.iter().map(|&k| FloatBits(k as u32)).collect());
        }
    }
    // The 0-1 principle on every binary input proves the small networks.
    for input in 0_u32..(1 << 16) {
        let mut keys: [u64; 16] = std::array::from_fn(|i| u64::from((input >> i) & 1));
        network16(&mut keys);
        assert!(keys.windows(2).all(|w| w[0] <= w[1]), "{input:#x}");
        if input < 256 {
            let mut keys: [u64; 8] = std::array::from_fn(|i| u64::from((input >> i) & 1));
            network8(&mut keys);
            assert!(keys.windows(2).all(|w| w[0] <= w[1]));
        }
    }
}

#[test]
fn float_grids_extend_below_the_sampled_minimum() {
    for len in [129, 256, 513, 4097] {
        for step in [0.25_f32, 1.0, 0.03125] {
            // The sample spans most of the grid but not its smallest
            // points, which only appear at unsampled positions.
            let input: Vec<f32> = (0..len)
                .map(|i| {
                    let index: i32 = if i < 16 {
                        12 * i
                    } else if i == len - 1 || i == len / 2 {
                        128
                    } else {
                        ((i * 157) % 256) - 60
                    };
                    index as f32 * step - 3.5
                })
                .collect();
            let mut expected = input.clone();
            expected.sort_unstable_by(f32::total_cmp);
            let mut grid = input.clone();
            assert!(float_grid(&mut grid), "{len} {step}");
            assert_eq!(
                grid.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let mut sorted = input;
            crate::radix_sort(&mut sorted);
            assert_eq!(
                sorted.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn float_grids_verify_exact_reconstruction_and_late_outliers() {
    for exponent in [-120, -10, -2, 0, 30, 100] {
        let step = f32::from_bits(((exponent + 127) as u32) << 23);
        let input: Vec<_> = (0..4099)
            .map(|i| (((i * 157) % 256) as f32 - 128.5) * step)
            .collect();
        let mut expected = input.clone();
        expected.sort_unstable_by(f32::total_cmp);
        let guard = f32::from_bits(0x7fc0_beef);
        let mut guarded = vec![guard];
        guarded.extend_from_slice(&input);
        guarded.push(guard);
        assert!(float_grid(&mut guarded[1..input.len() + 1]));
        assert!(
            guarded[1..input.len() + 1]
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(guarded[0].to_bits(), guard.to_bits());
        assert_eq!(guarded[input.len() + 1].to_bits(), guard.to_bits());
        for bad in [0.25 * step, -0.0, f32::INFINITY, f32::NEG_INFINITY, guard] {
            let mut actual = input.clone();
            actual[4097] = bad;
            let original: Vec<_> = actual.iter().map(|v| v.to_bits()).collect();
            assert!(!float_grid(&mut actual));
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                original
            );
        }
    }
}

fn check_compact_guards<T: Sealed>(keys: Vec<u64>, shift: u32) {
    let mut expected = keys.clone();
    expected.sort_unstable();
    let guard = T::from_key(0xbeef);
    for mode in 0..3 {
        let mut values = vec![guard];
        values.extend(keys.iter().map(|&key| T::from_key(key)));
        values.push(guard);
        let len = keys.len();
        let actual = &mut values[1..len + 1];
        let accepted = match mode {
            0 => count_byte(actual, keys[0], shift),
            1 => compact_range::<T, false>(actual),
            _ => compact_range::<T, true>(actual),
        };
        if accepted {
            assert_eq!(actual.iter().map(|v| v.key()).collect::<Vec<_>>(), expected);
        } else {
            assert_eq!(actual.iter().map(|v| v.key()).collect::<Vec<_>>(), keys);
        }
        assert_eq!(values[0].key(), guard.key());
        assert_eq!(values[len + 1].key(), guard.key());
    }
}

#[test]
fn compact_fills_cover_overlapping_stores_tails_and_sign_rotation() {
    for tail in 0..=17 {
        let mut keys = Vec::new();
        for key in 0..32 {
            keys.extend(std::iter::repeat_n(key * 7, (key % 18) as usize));
        }
        keys.extend(std::iter::repeat_n(255, tail));
        keys.reverse();
        for shift in [0, 8] {
            let keys: Vec<_> = keys.iter().map(|&v| v << shift).collect();
            check_compact_guards::<u16>(keys.clone(), shift);
            check_compact_guards::<i16>(keys.clone(), shift);
            check_compact_guards::<u32>(keys.clone(), shift);
            check_compact_guards::<u64>(keys.clone(), shift);
            check_compact_guards::<f32>(keys, shift);
        }
        let wrapped: Vec<_> = keys.iter().map(|&v| (v + 65500) & 65535).collect();
        check_compact_guards::<u16>(wrapped.clone(), 0);
        check_compact_guards::<i16>(wrapped, 0);
    }
    for len in [127, 128, 129, 255, 256, 257, 511, 512, 513] {
        for shift in [0, 8] {
            let keys: Vec<_> = (0..len).map(|i| ((i * 157) & 255) << shift).collect();
            check_compact_guards::<u16>(keys.clone(), shift);
            check_compact_guards::<i16>(keys.clone(), shift);
            check_compact_guards::<u32>(keys.clone(), shift);
            check_compact_guards::<u64>(keys.clone(), shift);
            check_compact_guards::<f32>(keys, shift);
        }
    }
}

#[test]
fn wide_digits_preserve_full_keys_and_both_buffer_guards() {
    fn check<T: Sealed>(keys: &[u32]) {
        let guard = T::from_key(0xbeef);
        let mut values = vec![guard];
        values.extend(keys.iter().map(|&v| T::from_key(u64::from(v))));
        values.push(guard);
        let len = keys.len();
        let mut scratch = vec![MaybeUninit::new(guard); len + 2];
        prefix64::<T, u32, 3, 2048>(
            &mut values[1..len + 1],
            &mut scratch[1..len + 1],
            [0, 11, 22],
            None,
        );
        let mut expected = keys.to_vec();
        expected.sort_unstable();
        assert_eq!(
            values[1..len + 1]
                .iter()
                .map(|v| v.key() as u32)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            (values[0].key(), values[len + 1].key()),
            (guard.key(), guard.key())
        );
        // SAFETY: The initialized guards are outside the scratch subslice.
        unsafe {
            assert_eq!(scratch[0].assume_init().key(), guard.key());
            assert_eq!(scratch[len + 1].assume_init().key(), guard.key());
        }
    }
    for len in [1, 17, 2047, 2048, 2049, 16383, 16384, 16385] {
        let keys: Vec<_> = (0..len)
            .map(|i| (i as u32).wrapping_mul(0x9e37_79b9))
            .collect();
        check::<u32>(&keys);
        check::<i32>(&keys);
        check::<f32>(&keys);
        check::<crate::FloatBits>(&keys);
    }
}

#[test]
fn merging_covers_short_runs_batch_tails_and_guards() {
    for left in (1..=33).chain([63, 64, 65, 127, 128, 129, 255, 256, 257]) {
        for right in (1..=33).chain([63, 64, 65, 127, 128, 129, 255, 256, 257]) {
            for pattern in 0..4 {
                let mut input: Vec<u64> = (0..left)
                    .map(|i| match pattern {
                        0 => i as u64 * 2,
                        1 => i as u64 / 8,
                        2 => i as u64 + 100,
                        _ => i as u64,
                    })
                    .collect();
                input.extend((0..right).map(|i| match pattern {
                    0 => i as u64 * 2 + 1,
                    1 => i as u64 / 8,
                    2 => i as u64,
                    _ => i as u64 + 100,
                }));
                let mut expected = input.clone();
                expected.sort_unstable();
                let len = input.len();
                input.insert(0, u64::MAX);
                input.push(u64::MAX);
                let mut scratch = vec![MaybeUninit::new(u64::MAX); len + 4];
                merge_runs(&mut input[1..len + 1], &mut scratch[1..len + 3], left);
                assert_eq!(input[1..len + 1], expected);
                assert_eq!((input[0], input[len + 1]), (u64::MAX, u64::MAX));
                for i in [0, len + 1, len + 2, len + 3] {
                    // SAFETY: Initialized guard slots are outside the used scratch.
                    assert_eq!(unsafe { scratch[i].assume_init() }, u64::MAX);
                }
            }
        }
    }
}

#[test]
fn alternating_blocks_rejoin_the_general_merge() {
    for length in [127_usize, 128, 129, 135, 136, 137, 255, 256, 257] {
        for stop in [0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129] {
            for swapped in [false, true] {
                let left: Vec<u64> = (0..length).map(|i| i as u64 * 4).collect();
                let right: Vec<u64> = (0..length)
                    .map(|i| (i * 4 + 2 + i.saturating_sub(stop) * 7) as u64)
                    .collect();
                let mut input = if swapped {
                    [right, left]
                } else {
                    [left, right]
                }
                .concat();
                let mut expected = input.clone();
                expected.sort_unstable();
                let len = input.len();
                input.insert(0, u64::MAX);
                input.push(u64::MAX);
                let mut scratch = vec![MaybeUninit::new(u64::MAX); len + 2];
                merge_runs(&mut input[1..len + 1], &mut scratch[1..len + 1], length);
                assert_eq!(&input[1..len + 1], expected);
                assert_eq!((input[0], input[len + 1]), (u64::MAX, u64::MAX));
                // SAFETY: These initialized guards are outside the used scratch.
                unsafe {
                    assert_eq!(scratch[0].assume_init(), u64::MAX);
                    assert_eq!(scratch[len + 1].assume_init(), u64::MAX);
                }
            }
        }
    }
}

use crate::FloatBits;

fn check_counters<T: Sealed, const N: usize>(input: Vec<T>) {
    let mut narrow = input.clone();
    let mut wide = input.clone();
    let mut expected = input;
    expected.sort_unstable_by_key(|v| v.key());
    let mut scratch = Vec::<T>::with_capacity(expected.len());
    kernel::<T, u32, N>(&mut narrow, scratch.spare_capacity_mut());
    kernel::<T, usize, N>(&mut wide, scratch.spare_capacity_mut());
    for ((narrow, wide), expected) in narrow.into_iter().zip(wide).zip(expected) {
        assert_eq!(narrow.key(), expected.key());
        assert_eq!(wide.key(), expected.key());
    }
}

#[test]
fn both_counter_widths_preserve_all_key_types() {
    // Explicitly exercise the >u32::MAX fallback kernel on manageable
    // allocations, plus both sides of the replicated-histogram threshold.
    for len in [0, 1, 2, 3, 33, 34, 35, 4095, 4096, 4097, 4098, 4099] {
        let bits: Vec<_> = (0..len)
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15).rotate_left(23))
            .collect();
        check_counters::<u16, 2>(bits.iter().map(|&v| v as u16).collect());
        check_counters::<i16, 2>(bits.iter().map(|&v| v as i16).collect());
        check_counters::<u32, 4>(bits.iter().map(|&v| v as u32).collect());
        check_counters::<i32, 4>(bits.iter().map(|&v| v as i32).collect());
        check_counters::<u64, 8>(bits.clone());
        check_counters::<i64, 8>(bits.iter().map(|&v| v as i64).collect());
        check_counters::<FloatBits, 4>(bits.iter().map(|&v| FloatBits(v as u32)).collect());
    }
}

#[test]
fn counter_capacity_includes_final_increment() {
    assert!(narrow_counts(u32::MAX as usize));
    if usize::BITS > 32 {
        assert!(!narrow_counts((u32::MAX as usize).wrapping_add(1)));
    }
    let mut table = [0_u32; 256];
    table[..4].copy_from_slice(&[1, 1, 1, 0]);
    let mut offset = u32::MAX - 3;
    prefix_four(&mut table, &mut offset, 0);
    assert_eq!(offset, u32::MAX);
    assert_eq!(table[2] + <u32 as Counter>::ONE, u32::MAX);
}

#[test]
fn skipped_bytes_preserve_original_bits_and_pass_parity() {
    for active in 0_u16..256 {
        let mask = (0..8).fold(0, |mask, byte| {
            mask | if active & (1 << byte) != 0 {
                255_u64 << (byte * 8)
            } else {
                0
            }
        });
        let bits: Vec<_> = (0..257_u64)
            .map(|i| (i.wrapping_mul(0x9e3779b97f4a7c15) & mask) | (0xa5a5a5a5a5a5a5a5 & !mask))
            .collect();
        check_counters::<u64, 8>(bits.clone());
        check_counters::<i64, 8>(bits.iter().map(|&v| v as i64).collect());
        if active < 16 {
            check_counters::<u32, 4>(bits.iter().map(|&v| v as u32).collect());
            check_counters::<i32, 4>(bits.iter().map(|&v| v as i32).collect());
            check_counters::<FloatBits, 4>(bits.iter().map(|&v| FloatBits(v as u32)).collect());
        }
    }
}

#[test]
fn singleton_byte_reconstruction_preserves_signed_and_float_bits() {
    for len in [1023, 1024, 1025] {
        for byte in 0..8 {
            let mask = 255_u64 << (byte * 8);
            let bits: Vec<_> = (0..len)
                .map(|i| {
                    (((i as u64).wrapping_mul(157) & 255) << (byte * 8))
                        | (0xa5a5a5a5a5a5a5a5 & !mask)
                })
                .collect();
            check_counters::<u64, 8>(bits.clone());
            check_counters::<i64, 8>(bits.iter().map(|&v| v as i64).collect());
            if byte < 4 {
                check_counters::<u32, 4>(bits.iter().map(|&v| v as u32).collect());
                check_counters::<i32, 4>(bits.iter().map(|&v| v as i32).collect());
                check_counters::<FloatBits, 4>(bits.iter().map(|&v| FloatBits(v as u32)).collect());
                check_counters::<FloatBits, 4>(
                    bits.iter()
                        .map(|&v| FloatBits::from_key(v as u32 as u64))
                        .collect(),
                );
            }
            if byte < 2 {
                check_counters::<u16, 2>(bits.iter().map(|&v| v as u16).collect());
                check_counters::<i16, 2>(bits.iter().map(|&v| v as i16).collect());
            }
        }
    }
}

fn check_full_histogram<T: Sealed>(input: Vec<T>) {
    let len = input.len();
    let mut expected = input.clone();
    expected.sort_unstable_by_key(|v| v.key());
    let guard = T::from_key(0xbeef);
    let mut actual = vec![guard];
    actual.extend(input);
    actual.push(guard);
    let mut scratch = vec![MaybeUninit::new(guard); len + 4];
    assert!(counting16(
        &mut actual[1..len + 1],
        &mut scratch[1..len + 3]
    ));
    assert!(
        actual[1..len + 1]
            .iter()
            .zip(expected)
            .all(|(a, b)| a.key() == b.key())
    );
    assert_eq!(actual[0].key(), guard.key());
    assert_eq!(actual[len + 1].key(), guard.key());
    for i in [0, len + 1, len + 2, len + 3] {
        // SAFETY: Initialized guards lie outside the permitted scratch.
        assert_eq!(unsafe { scratch[i].assume_init() }.key(), guard.key());
    }
}

#[test]
fn full_histogram_fixed_stores_cover_exact_and_overlapping_tails() {
    for tail in [31, 32, 33] {
        let mut input = vec![0_u16; 262144];
        for (key, count) in [
            (0x0101, 31),
            (0x0303, 32),
            (0x0505, 33),
            (0x0707, 1),
            (0x0909, tail),
        ] {
            input.extend(std::iter::repeat_n(key, count));
        }
        let last = input.len() - 1;
        input.swap(0, last);
        check_full_histogram::<i16>(input.iter().map(|&v| v as i16).collect());
        check_full_histogram(input);
    }
}

fn check_prefix<const P: usize, const R: usize>(digit: usize) {
    for shift in [0, 64 - P * digit] {
        let original: Vec<_> = (0..513_u64)
            .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let mask = (1_u64 << (P * digit)) - 1;
        let mut expected = original.clone();
        // A stable reference checks that skipped low bits remain intact.
        expected.sort_by_key(|v| (v >> shift) & mask);
        let mut actual = original;
        let mut scratch = vec![MaybeUninit::new(0xfeed_u64); actual.len() + 2];
        let shifts: [usize; P] = std::array::from_fn(|pass| shift + pass * digit);
        prefix64::<u64, u32, P, R>(&mut actual, &mut scratch[1..514], shifts, None);
        assert_eq!(actual, expected);
        // SAFETY: Initialized guard elements are outside the scratch subslice.
        assert_eq!(unsafe { scratch[0].assume_init() }, 0xfeed);
        // SAFETY: Initialized guard elements are outside the scratch subslice.
        assert_eq!(unsafe { scratch[514].assume_init() }, 0xfeed);
    }
}

#[test]
fn prefix_passes_cover_both_widths_and_buffer_parities() {
    check_prefix::<1, 256>(8);
    check_prefix::<2, 256>(8);
    check_prefix::<3, 256>(8);
    check_prefix::<4, 256>(8);
    check_prefix::<1, 512>(9);
    check_prefix::<2, 512>(9);
    check_prefix::<3, 512>(9);
    check_prefix::<4, 512>(9);
}

#[test]
fn byte_counters_fit_at_the_small_capacity_boundary() {
    for len in [33, 64, 127, 128, 254, 255, 256] {
        for input in [vec![42_u8; len], (0..len).map(|i| (i % 4) as u8).collect()] {
            let mut expected = input.clone();
            expected.sort_unstable();
            let mut actual = input;
            byte_scatter::<u8, 2>(&mut actual);
            assert_eq!(actual, expected);
        }
    }
}

fn check_byte_counters<T: Sealed, const SIGN: u8>(input: Vec<T>) {
    let mut expected = input.clone();
    expected.sort_unstable_by_key(|value| value.key());
    for sort in [
        byte_kernel::<T, u32, SIGN, 1> as fn(&mut [T]),
        byte_kernel::<T, u16, SIGN, 4>,
        byte_kernel::<T, u32, SIGN, 4>,
        byte_kernel::<T, usize, SIGN, 4>,
    ] {
        let mut actual = input.clone();
        sort(&mut actual);
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.key() == b.key())
        );
    }
}

#[test]
fn byte_counter_widths_and_unrolling_tails() {
    // Lengths up to 70 cover every mix of overlapping and exact fills near
    // the end of the slice; the longer ones cover the unrolling tails.
    for len in (0..=70).chain([1023, 1024, 1025, 1026, 1027, 4095, 4096, 4097, 4098, 4099]) {
        check_byte_counters::<u8, 0>((0..len).map(|i| (i * 157) as u8).collect());
        check_byte_counters::<i8, 128>((0..len).map(|i| (i * 157) as i8).collect());
        // Runs longer than one fill, mixed with single elements and a tail.
        check_byte_counters::<u8, 0>((0..len).map(|i| (i / 20) as u8).collect());
        check_byte_counters::<i8, 128>((0..len).map(|i| (i / 20) as i8).collect());
        check_byte_counters::<u8, 0>(
            (0..len)
                .map(|i| if i % 50 == 0 { 7 } else { 200 })
                .collect(),
        );
        check_byte_counters::<i8, 128>(
            (0..len)
                .map(|i| if i % 50 == 0 { -7 } else { 100 })
                .collect(),
        );
    }
}

#[test]
fn small_networks_sort_every_binary_input() {
    // The 0-1 principle: a comparator network that sorts every binary
    // input sorts every input.
    fn check<const N: usize>(network: fn(&mut [u64; N])) {
        for input in 0_u32..(1 << N) {
            let mut keys: [u64; N] = std::array::from_fn(|i| u64::from((input >> i) & 1));
            network(&mut keys);
            assert!(
                keys.windows(2).all(|w| w[0] <= w[1]),
                "{N} keys, input {input:#b}"
            );
        }
    }
    check::<3>(network3);
    check::<4>(network4);
    check::<5>(network5);
    check::<6>(network6);
    check::<7>(network7);
    check::<8>(network8);
    check::<9>(network9);
    check::<10>(network10);
    check::<11>(network11);
    check::<12>(network12);
    check::<13>(network13);
    check::<14>(network14);
    check::<15>(network15);
    check::<16>(network16);
}

#[test]
fn tiny_and_short_sorts_cover_every_length_and_order() {
    let mut state = 0x7e57_u64;
    let mut random = move || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    for len in 0..=48 {
        for pattern in 0..6 {
            let mut keys: Vec<u64> = (0..len)
                .map(|i| match pattern {
                    0 => random(),
                    1 => [0, 1, u64::MAX, 42][random() as usize % 4],
                    2 => i as u64,
                    3 => (len - i) as u64,
                    4 => 7,
                    _ => random() % 3,
                })
                .collect();
            if pattern == 5 && len > 2 {
                // Sorted with one late inversion.
                keys.sort_unstable();
                keys.swap(len - 2, len - 1);
            }
            for guard in [u64::MAX, 0] {
                let mut expected = keys.clone();
                expected.sort_unstable();
                let mut values = vec![guard];
                values.extend(keys.iter().copied());
                values.push(guard);
                small_sort(&mut values[1..len + 1]);
                assert_eq!(&values[1..len + 1], &expected[..], "{len} {pattern}");
                assert_eq!((values[0], values[len + 1]), (guard, guard));
                let mut signed: Vec<i32> = keys.iter().map(|&k| k as i32).collect();
                let mut expected_signed = signed.clone();
                expected_signed.sort_unstable();
                crate::radix_sort(&mut signed);
                assert_eq!(signed, expected_signed);
                let mut floats: Vec<f32> = keys.iter().map(|&k| f32::from_bits(k as u32)).collect();
                let mut expected_floats = floats.clone();
                expected_floats.sort_by(f32::total_cmp);
                crate::radix_sort(&mut floats);
                assert!(
                    floats
                        .iter()
                        .zip(&expected_floats)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{len} {pattern}"
                );
            }
        }
    }
}

#[test]
fn compact_scatter_orders_wrapped_ranges_and_rejects_outliers() {
    fn check<T: Sealed>(keys: &[u64], wrapped: bool) -> bool {
        let mask = u64::MAX >> (64 - 8 * std::mem::size_of::<T>());
        let rotate = if wrapped { 256 } else { 0 };
        let guard = T::from_key(0xbeef & mask);
        let mut values = vec![guard];
        values.extend(keys.iter().map(|&key| T::from_key(key)));
        values.push(guard);
        let len = keys.len();
        let accepted = if wrapped {
            compact_scatter::<T, true>(&mut values[1..len + 1], mask, rotate)
        } else {
            compact_scatter::<T, false>(&mut values[1..len + 1], mask, rotate)
        };
        let mut expected = keys.to_vec();
        expected.sort_unstable();
        let actual: Vec<u64> = values[1..len + 1].iter().map(|v| v.key()).collect();
        assert_eq!(actual, if accepted { expected } else { keys.to_vec() });
        assert_eq!(
            (values[0].key(), values[len + 1].key()),
            (guard.key(), guard.key())
        );
        accepted
    }
    for len in [1, 2, 3, 255, 256, 257, 511, 512] {
        // A range crossing a byte boundary, as shifted signed inputs do.
        let keys: Vec<u64> = (0..len)
            .map(|i| 0x7f9c + ((i * 157) % 256) as u64)
            .collect();
        assert!(check::<u16>(&keys, false));
        assert!(check::<u32>(&keys, false));
        assert!(check::<u64>(&keys, false));
        // A range wrapping the key width: the small keys come first.
        let wrapped: Vec<u64> = keys.iter().map(|&k| (k + 65500) & 0xffff).collect();
        assert!(check::<u16>(&wrapped, true));
        // A range wrapping without crossing 256 after rotation.
        let high: Vec<u64> = (0..len)
            .map(|i| 0xff00 + ((i * 157) % 200) as u64)
            .collect();
        assert!(check::<u16>(&high, true));
        // One value beyond the span is rejected before any write; a
        // single element is its own span.
        let mut outlier = keys.clone();
        outlier[len / 2] = 0x7f9c + 256;
        assert_eq!(check::<u32>(&outlier, false), len == 1);
    }
}

#[test]
fn short_byte_repairs_stay_within_budget_and_fall_back_to_counting() {
    // Byte slices above the general repair bound: sparse insertions
    // repair in place, a dense first window is declared dense, dense
    // disorder after a sparse start trips the density rule, one long
    // move fits the budget and a third long move does not. Every path
    // still sorts through the public entry point.
    for len in [REPAIRED_BYTES + 1, 128, 200, 256] {
        let mut values: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        values.sort_unstable();
        let expected = values.clone();
        let sorted_by_entry = |mut input: Vec<u8>| {
            crate::radix_sort(&mut input);
            assert_eq!(input, expected);
        };
        let mut input = values.clone();
        for i in (8..len - 1).step_by(24) {
            input.swap(i, i + 1);
        }
        let start = ordered_scan::<u8, false>(&input);
        assert!(!dense_bytes::<u8, false>(&input, start), "{len}");
        assert!(ordered_repair::<u8, false>(&mut input, start), "{len}");
        assert_eq!(input, expected);

        let mut input = values.clone();
        for i in (8..len - 1).step_by(2) {
            input.swap(i, i + 1);
        }
        let start = ordered_scan::<u8, false>(&input);
        assert!(dense_bytes::<u8, false>(&input, start), "{len}");
        sorted_by_entry(input);

        let mut input = values.clone();
        input.swap(10, 11);
        for i in (48..len - 1).step_by(2) {
            input.swap(i, i + 1);
        }
        let start = ordered_scan::<u8, false>(&input);
        assert!(!dense_bytes::<u8, false>(&input, start), "{len}");
        let original = input.clone();
        assert!(!ordered_repair::<u8, false>(&mut input, start), "{len}");
        let mut multiset = input.clone();
        multiset.sort_unstable();
        assert_eq!(multiset, expected, "a failed repair keeps the multiset");
        sorted_by_entry(original);

        let mut input = values.clone();
        input[8..].rotate_right(1);
        let start = ordered_scan::<u8, false>(&input);
        assert!(ordered_repair::<u8, false>(&mut input, start), "{len}");
        assert_eq!(input, expected);

        if len >= 128 {
            for (moves, repaired) in [(2, true), (3, false)] {
                let mut input = values.clone();
                for k in 0..moves {
                    let p = 10 + k * 40;
                    input[p..p + 36].rotate_right(1);
                }
                let start = ordered_scan::<u8, false>(&input);
                let original = input.clone();
                assert_eq!(
                    ordered_repair::<u8, false>(&mut input, start),
                    repaired,
                    "{len} {moves}"
                );
                if repaired {
                    assert_eq!(input, expected);
                }
                sorted_by_entry(original);
            }
        }

        let mut descending = values.clone();
        descending.reverse();
        for i in (8..len - 1).step_by(24) {
            descending.swap(i, i + 1);
        }
        let start = ordered_scan::<u8, true>(&descending);
        assert!(!dense_bytes::<u8, true>(&descending, start));
        assert!(ordered_repair::<u8, true>(&mut descending, start));
        assert!(descending.windows(2).all(|w| w[0] >= w[1]));
        descending.reverse();
        assert_eq!(descending, expected);
    }
    // The dense window needs five inversions below 128 elements, four
    // below 192 and three beyond.
    for (len, limit) in [(97, 5), (127, 5), (128, 4), (191, 4), (192, 3), (256, 3)] {
        for inversions in [limit - 1, limit] {
            let mut input: Vec<u8> = (0..len).map(|i| (i / 2) as u8).collect();
            for k in 0..inversions {
                let i = 9 + 2 * k;
                input.swap(i, i + 1);
            }
            let start = ordered_scan::<u8, false>(&input);
            assert_eq!(start, 10);
            assert_eq!(
                dense_bytes::<u8, false>(&input, start),
                inversions == limit,
                "{len} {inversions}"
            );
        }
    }
    // Byte slices up to the stack merge bound merge two runs in place;
    // longer ones leave the decision to counting.
    for len in [40, 97, STACK_MERGE, STACK_MERGE + 1, 256] {
        let mut input: Vec<u8> = (0..len).map(|i| (i * 7 % 256) as u8).collect();
        input.sort_unstable();
        let expected = input.clone();
        input.rotate_left(len / 2);
        let start = ordered_scan::<u8, false>(&input);
        let result = two_runs(&mut input, start, false);
        if len <= STACK_MERGE {
            assert!(matches!(result, Some(Preparation::Done)), "{len}");
            assert_eq!(input, expected);
        } else {
            assert!(result.is_none(), "{len}");
            crate::radix_sort(&mut input);
            assert_eq!(input, expected);
        }
    }
}

#[test]
fn byte_scatter_lanes_cover_remainders_fills_and_signs() {
    for len in [17, 33, 97, 98, 99, 100, 255, 256] {
        for distinct in [3, 16, 17, 200] {
            let input: Vec<u8> = (0..len).map(|i| ((i * 7919) % distinct) as u8).collect();
            let mut expected = input.clone();
            expected.sort_unstable();
            let mut two = input.clone();
            byte_scatter::<u8, 2>(&mut two);
            assert_eq!(two, expected, "u8 {len} {distinct}");
            let mut four = input.clone();
            byte_scatter::<u8, 4>(&mut four);
            assert_eq!(four, expected, "u8 {len} {distinct}");
            let signed: Vec<i8> = input.iter().map(|&b| (b as i8).wrapping_sub(100)).collect();
            let mut expected = signed.clone();
            expected.sort_unstable();
            let mut two = signed.clone();
            byte_scatter::<i8, 2>(&mut two);
            assert_eq!(two, expected, "i8 {len} {distinct}");
            let mut four = signed.clone();
            byte_scatter::<i8, 4>(&mut four);
            assert_eq!(four, expected, "i8 {len} {distinct}");
        }
    }
}

#[test]
fn network_tail_merges_every_length_with_duplicates_and_extremes() {
    // The merge network is exact for every pair of sorted 0-1 runs.
    for zeros_head in 0..=32 {
        for zeros_tail in 0..=8 {
            let mut keys = [u64::MAX; NETWORK_TAIL];
            keys[..zeros_head].fill(0);
            keys[32..32 + zeros_tail].fill(0);
            merge32_8(&mut keys);
            assert!(keys.windows(2).all(|w| w[0] <= w[1]));
            assert_eq!(
                keys.iter().filter(|&&k| k == 0).count(),
                zeros_head + zeros_tail
            );
        }
    }
    let mut state = 0x7a11_u64;
    let mut random = move || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    for len in 33..=NETWORK_TAIL {
        for round in 0..8 {
            // Largest keys, duplicates and negative values exercise the
            // padding and both runs' sentinels.
            let wide: Vec<u64> = (0..len)
                .map(|i| match (round + i) % 4 {
                    0 => u64::MAX,
                    1 => random() % 3,
                    _ => random(),
                })
                .collect();
            let mut expected = wide.clone();
            expected.sort_unstable();
            let mut actual = wide.clone();
            network_sort_with_tail(&mut actual);
            assert_eq!(actual, expected, "u64 {len} {round}");
            let signed: Vec<i64> = wide.iter().map(|&w| w as i64).collect();
            let mut expected = signed.clone();
            expected.sort_unstable();
            let mut actual = signed.clone();
            network_sort_with_tail(&mut actual);
            assert_eq!(actual, expected, "i64 {len} {round}");
            let bytes: Vec<i8> = wide.iter().map(|&w| w as i8).collect();
            let mut expected = bytes.clone();
            expected.sort_unstable();
            let mut actual = bytes.clone();
            network_sort_with_tail(&mut actual);
            assert_eq!(actual, expected, "i8 {len} {round}");
            let floats: Vec<f32> = wide
                .iter()
                .map(|&w| match w % 5 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f32::NEG_INFINITY,
                    _ => (w as i32) as f32 / 7.0,
                })
                .collect();
            let mut expected = floats.clone();
            expected.sort_unstable_by(f32::total_cmp);
            let mut actual = floats.clone();
            network_sort_with_tail(&mut actual);
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "f32 {len} {round}"
            );
        }
    }
    // Inputs through sixteen keys pad with the largest key as well.
    for len in 2..=16 {
        let input: Vec<u64> = (0..len)
            .map(|i| if i % 3 == 0 { u64::MAX } else { random() % 4 })
            .collect();
        let mut expected = input.clone();
        expected.sort_unstable();
        let mut actual = input.clone();
        crate::radix_sort(&mut actual);
        assert_eq!(actual, expected, "{len}");
    }
}

#[test]
fn constant_scans_see_a_late_difference_at_every_width() {
    for len in [2, 8, 63, 64, 65, 1023, 1024, 1025, 4095, 4096, 4097, 8192] {
        for position in [0, len / 2, len - 1] {
            let mut bytes = vec![42_u8; len];
            assert!(constant(&bytes));
            bytes[position] = 43;
            assert!(!constant(&bytes), "u8 {len} {position}");
            let mut signed: Vec<i8> = vec![-42; len];
            assert!(constant(&signed));
            signed[position] = -41;
            assert!(!constant(&signed), "i8 {len} {position}");
            let mut shorts: Vec<i16> = vec![-1000; len];
            assert!(constant(&shorts));
            shorts[position] = -1001;
            assert!(!constant(&shorts), "i16 {len} {position}");
            let mut words: Vec<i32> = vec![-7; len];
            assert!(constant(&words));
            words[position] = 7;
            assert!(!constant(&words), "i32 {len} {position}");
            let mut floats = vec![-0.0_f32; len];
            assert!(constant(&floats));
            floats[position] = 0.0;
            assert!(!constant(&floats), "f32 {len} {position}");
            let mut wide: Vec<u64> = vec![u64::MAX; len];
            assert!(constant(&wide));
            wide[position] = u64::MAX - 1;
            assert!(!constant(&wide), "u64 {len} {position}");
        }
    }
}
