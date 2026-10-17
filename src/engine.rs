use crate::sealed::Sealed;
use std::mem::MaybeUninit;

#[derive(Clone, Copy)]
pub(crate) enum Preparation {
    Done,
    Radix,
    Merge(usize),
}

impl Preparation {
    #[inline]
    pub(crate) fn is_done(self) -> bool {
        matches!(self, Self::Done)
    }
}

/// Finish tiny, ordered, or few-key inputs before allocating scratch. All
/// comparisons use the same integer and float total ordering as radix passes.
#[inline]
pub(crate) fn prepare<T: Sealed>(values: &mut [T]) -> Preparation {
    let width = std::mem::size_of::<T>();
    if values.len() > 32 {
        match ordered(values) {
            Preparation::Radix => {}
            result => return result,
        }
        if width > 1 && values.len() >= 64 && few_values(values) {
            return Preparation::Done;
        }
        if T::SMALL_SORT_LIMIT >= values.len() {
            T::small_sort(values);
            return Preparation::Done;
        }
        if width > 1
            && values.len() >= 129
            && narrow_counts(values.len())
            && (compact_range::<T, false>(values)
                || compact_range::<T, true>(values)
                || single_byte(values))
        {
            return Preparation::Done;
        }
        if values.len() >= 129 && narrow_counts(values.len()) && T::numeric_range(values) {
            return Preparation::Done;
        }
        return if width > 1 && values.len() >= 4096 && sparse_values(values) {
            Preparation::Done
        } else {
            Preparation::Radix
        };
    }
    T::small_sort(values);
    Preparation::Done
}

/// Count exactly reconstructible integer-valued floats in a compact numeric
/// range, or propose a power-of-two grid when the sample is unsuitable.
/// Full bitwise verification precedes every output write.
#[inline(never)]
pub(crate) fn float_range(values: &mut [f32]) -> bool {
    let key_of = |value: f32| (value as i32 as u32) ^ (1 << 31);
    let first = key_of(values[0]);
    let mut minimum = first;
    let mut maximum = first;
    for &value in values[..16]
        .iter()
        .chain(values.last())
        .chain(values.get(values.len() / 2))
    {
        let integer = value as i32;
        if (integer as f32).to_bits() != value.to_bits() {
            return float_grid(values);
        }
        let key = (integer as u32) ^ (1 << 31);
        minimum = minimum.min(key);
        maximum = maximum.max(key);
        if maximum - minimum > 255 {
            return float_grid(values);
        }
    }
    let mut counts = [[0_u32; 256]; 4];
    let mut minima = [minimum; 4];
    let mut maxima = [maximum; 4];
    // A few values that do not reconstruct from an integer keep their exact
    // bits and are merged into the counted output by total order. The common
    // all-integer scan is unchanged; a block with exceptions is rescanned so
    // that counts and extrema exclude them.
    let mut exceptions = [0.0_f32; FLOAT_EXCEPTIONS];
    let mut exceptional = 0;
    for block in validation_blocks(values) {
        let saved = (minima, maxima);
        let mut chunks = block.chunks_exact(4);
        let mut exact = true;
        for chunk in &mut chunks {
            for lane in 0..4 {
                let value = chunk[lane];
                let integer = value as i32;
                exact &= (integer as f32).to_bits() == value.to_bits();
                let key = (integer as u32) ^ (1 << 31);
                counts[lane][key as usize & 255] += 1;
                minima[lane] = minima[lane].min(key);
                maxima[lane] = maxima[lane].max(key);
            }
        }
        for &value in chunks.remainder() {
            let integer = value as i32;
            exact &= (integer as f32).to_bits() == value.to_bits();
            let key = (integer as u32) ^ (1 << 31);
            counts[0][key as usize & 255] += 1;
            minima[0] = minima[0].min(key);
            maxima[0] = maxima[0].max(key);
        }
        if !exact {
            (minima, maxima) = saved;
            for &value in block {
                let integer = value as i32;
                let key = (integer as u32) ^ (1 << 31);
                if (integer as f32).to_bits() == value.to_bits() {
                    minima[0] = minima[0].min(key);
                    maxima[0] = maxima[0].max(key);
                } else {
                    if exceptional == FLOAT_EXCEPTIONS {
                        return false;
                    }
                    exceptions[exceptional] = value;
                    exceptional += 1;
                }
            }
        }
        minimum = *minima.iter().min().unwrap();
        maximum = *maxima.iter().max().unwrap();
        if maximum - minimum > 255 {
            return false;
        }
    }
    let [first, second, third, fourth] = &mut counts;
    for (bucket, count) in first.iter_mut().enumerate() {
        *count += second[bucket] + third[bucket] + fourth[bucket];
    }
    // Exceptions were counted under their truncated integer key; remove them
    // from the merged histogram before filling. Keys in the verified span map
    // to distinct low bytes, so each in-span bucket then counts one value.
    let exceptions = &mut exceptions[..exceptional];
    for &exception in exceptions.iter() {
        counts[0][key_of(exception) as usize & 255] -= 1;
    }
    if exceptions.is_empty() {
        fill_key_range(
            values,
            minimum.into()..=maximum.into(),
            0,
            |key| counts[0][key as usize & 255] as usize,
            |key| ((key as u32 ^ (1 << 31)) as i32) as f32,
        );
        return true;
    }
    exceptions.sort_unstable_by(f32::total_cmp);
    let mut next = 0;
    let mut start = 0;
    for key in minimum..=maximum {
        let value = ((key ^ (1 << 31)) as i32) as f32;
        // An exception never shares a total-order key with a reconstructed
        // value: equal keys would mean equal bits, and it did not reconstruct.
        while next < exceptions.len() && exceptions[next].key() < value.key() {
            values[start] = exceptions[next];
            start += 1;
            next += 1;
        }
        let end = start + counts[0][key as usize & 255] as usize;
        fill_bucket(values, start, end, value);
        start = end;
    }
    for &exception in &exceptions[next..] {
        values[start] = exception;
        start += 1;
    }
    debug_assert_eq!(start, values.len());
    true
}

/// Maximum number of non-integer values merged into a numeric float count.
const FLOAT_EXCEPTIONS: usize = 8;

/// Count an exactly reconstructible power-of-two grid of finite floats.
/// Sampling only proposes the grid; every value and the complete span are
/// verified before filling, including the sign bit of zero.
#[inline(never)]
fn float_grid(values: &mut [f32]) -> bool {
    let sample = || {
        values[..16]
            .iter()
            .chain(values.last())
            .chain(values.get(values.len() / 2))
    };
    let (mut low, mut high) = (values[0], values[0]);
    for &value in sample() {
        if !value.is_finite() {
            return false;
        }
        low = low.min(value);
        high = high.max(value);
    }
    let span = high - low;
    if !span.is_finite() || span <= 0.0 {
        return false;
    }
    let exponent = ((span.to_bits() >> 23) as i32) - 127 - 7;
    if !(-126..=126).contains(&exponent) {
        return false;
    }
    let step = f32::from_bits(((exponent + 127) as u32) << 23);
    let scale = f32::from_bits(((127 - exponent) as u32) << 23);
    let index = |value: f32| ((value - low) * scale) as i32;
    let reconstruct = |integer: i32| low + integer as f32 * step;
    if sample().any(|&v| reconstruct(index(v)).to_bits() != v.to_bits()) {
        return false;
    }
    let mut counts = [[0_u32; 256]; 4];
    let (mut minima, mut maxima) = ([i32::MAX; 4], [i32::MIN; 4]);
    for block in validation_blocks(values) {
        let mut chunks = block.chunks_exact(4);
        let mut exact = true;
        for chunk in &mut chunks {
            for lane in 0..4 {
                let value = chunk[lane];
                let key = index(value);
                exact &= reconstruct(key).to_bits() == value.to_bits();
                counts[lane][key as usize & 255] += 1;
                minima[lane] = minima[lane].min(key);
                maxima[lane] = maxima[lane].max(key);
            }
        }
        for &value in chunks.remainder() {
            let key = index(value);
            exact &= reconstruct(key).to_bits() == value.to_bits();
            counts[0][key as usize & 255] += 1;
            minima[0] = minima[0].min(key);
            maxima[0] = maxima[0].max(key);
        }
        let min = *minima.iter().min().unwrap();
        let max = *maxima.iter().max().unwrap();
        if !exact || i64::from(max) - i64::from(min) > 255 {
            return false;
        }
    }
    let [first, second, third, fourth] = &mut counts;
    for (bucket, count) in first.iter_mut().enumerate() {
        *count += second[bucket] + third[bucket] + fourth[bucket];
    }
    let first = *minima.iter().min().unwrap();
    let last = *maxima.iter().max().unwrap();
    // Grid indexes are signed: values below the sampled minimum have negative
    // indexes. Walk the verified span (at most 256 keys) as offsets from the
    // smallest index; low bytes of the signed indexes stay distinct.
    fill_key_range(
        values,
        0..=(i64::from(last) - i64::from(first)) as u64,
        0,
        |offset| counts[0][first.wrapping_add(offset as i32) as usize & 255] as usize,
        |offset| reconstruct(first.wrapping_add(offset as i32)),
    );
    true
}

/// A sample proposes at most four keys. The full scan must confirm membership
/// before any write; a late unseen key leaves the original input untouched.
#[inline]
fn few_values<T: Sealed>(values: &mut [T]) -> bool {
    let mut representatives = [values[0]; 4];
    let mut distinct = 1;
    for &value in values
        .last()
        .into_iter()
        .chain(values.get(values.len() / 2))
        .chain(&values[1..15])
    {
        if !representatives[..distinct]
            .iter()
            .any(|v| v.key() == value.key())
        {
            if distinct == 4 {
                return false;
            }
            representatives[distinct] = value;
            distinct += 1;
        }
    }
    representatives[..distinct].sort_unstable_by_key(|v| v.key());
    match distinct {
        1 => finish_few::<T, 1>(values, &representatives),
        2 => finish_few::<T, 2>(values, &representatives),
        3 => finish_few::<T, 3>(values, &representatives),
        _ => finish_few::<T, 4>(values, &representatives),
    }
}

/// A short sample proposes a compact canonical key range. Complete min/max
/// verification makes the low byte an injective bucket index before any fill.
/// Four lanes avoid repeated-key dependencies; no scratch allocation is needed.
#[inline(never)]
fn compact_range<T: Sealed, const WRAPPED: bool>(values: &mut [T]) -> bool {
    let mask = u64::MAX >> (64 - 8 * std::mem::size_of::<T>());
    let rotate = if WRAPPED { 256 } else { 0 };
    let key_of = |value: T| value.key().wrapping_add(rotate) & mask;
    let first = key_of(values[0]);
    let mut minimum = first;
    let mut maximum = first;
    for &value in values
        .last()
        .into_iter()
        .chain(values.get(values.len() / 2))
        .chain(&values[1..15])
    {
        let key = key_of(value);
        minimum = minimum.min(key);
        maximum = maximum.max(key);
        if maximum - minimum > 255 {
            return false;
        }
    }
    if !WRAPPED && minimum >> 8 == maximum >> 8 {
        return count_byte(values, values[0].key(), 0);
    }
    if values.len() <= COMPACT_SCATTER {
        return compact_scatter::<T, WRAPPED>(values, mask, rotate);
    }
    // A separate narrow reduction vectorizes independently of histogram
    // updates. It is cheap enough for 16-bit ranges that cross a byte edge.
    let narrow = std::mem::size_of::<T>() == 2;
    if narrow {
        let (lo, hi) = values.iter().fold((u16::MAX, 0_u16), |(lo, hi), &v| {
            let key = key_of(v) as u16;
            (lo.min(key), hi.max(key))
        });
        minimum = u64::from(lo);
        maximum = u64::from(hi);
        if maximum - minimum > 255 {
            return false;
        }
    }
    let mut counts = [[0_u32; 256]; 4];
    let mut minima = [minimum; 4];
    let mut maxima = [maximum; 4];
    for block in validation_blocks(values) {
        let mut chunks = block.chunks_exact(4);
        for chunk in &mut chunks {
            for lane in 0..4 {
                let key = key_of(chunk[lane]);
                counts[lane][key as usize & 255] += 1;
                if !narrow {
                    minima[lane] = minima[lane].min(key);
                    maxima[lane] = maxima[lane].max(key);
                }
            }
        }
        for &value in chunks.remainder() {
            let key = key_of(value);
            counts[0][key as usize & 255] += 1;
            if !narrow {
                minima[0] = minima[0].min(key);
                maxima[0] = maxima[0].max(key);
            }
        }
        if !narrow {
            minimum = *minima.iter().min().unwrap();
            maximum = *maxima.iter().max().unwrap();
        }
        if maximum - minimum > 255 {
            return false;
        }
    }
    let [first, second, third, fourth] = &mut counts;
    for (bucket, count) in first.iter_mut().enumerate() {
        *count += second[bucket] + third[bucket] + fourth[bucket];
    }
    let mut start = 0;
    let mut fill = |keys: std::ops::RangeInclusive<u64>| {
        start = fill_key_range(
            values,
            keys,
            start,
            |key| counts[0][key as usize & 255] as usize,
            |key| T::from_key(key.wrapping_sub(rotate) & mask),
        );
    };
    if WRAPPED {
        // Rotated keys at or above 256 are the smallest original keys.
        // Emit those before keys that wrapped around the width boundary.
        fill(minimum.max(256)..=maximum);
        fill(minimum..=maximum.min(255));
    } else {
        fill(minimum..=maximum);
    }
    true
}

/// Longest slice whose compact range is scattered through a stack buffer.
const COMPACT_SCATTER: usize = 512;

/// A short compact range crossing a byte boundary: verify the exact span,
/// count the offset within it in two lanes, and scatter once into a stack
/// buffer in canonical order. Fills would walk 256 buckets for a few values
/// each, and a wider histogram would cost more than the input itself.
#[inline(never)]
fn compact_scatter<T: Sealed, const WRAPPED: bool>(
    values: &mut [T],
    mask: u64,
    rotate: u64,
) -> bool {
    assert!(values.len() <= COMPACT_SCATTER);
    let key_of = |value: T| value.key().wrapping_add(rotate) & mask;
    let first = key_of(values[0]);
    let (minimum, maximum) = values.iter().fold((first, first), |(low, high), &value| {
        let key = key_of(value);
        (low.min(key), high.max(key))
    });
    if maximum - minimum > 255 {
        return false;
    }
    // Rotated keys at or above 256 are the smallest original keys; in a
    // range crossing that point they come first in canonical order.
    let crossing = WRAPPED && minimum < 256 && maximum >= 256;
    let digit_of = |key: u64| {
        if crossing && key >= 256 {
            key - 256
        } else if crossing {
            key + (maximum - 255) - minimum
        } else {
            key - minimum
        }
    };
    let mut lanes = [[0_u16; 256]; 2];
    let mut chunks = values.chunks_exact(2);
    for chunk in &mut chunks {
        lanes[0][digit_of(key_of(chunk[0])) as usize] += 1;
        lanes[1][digit_of(key_of(chunk[1])) as usize] += 1;
    }
    for &value in chunks.remainder() {
        lanes[0][digit_of(key_of(value)) as usize] += 1;
    }
    let [first_lane, second_lane] = &mut lanes;
    let mut counts = *first_lane;
    for (count, &other) in counts.iter_mut().zip(second_lane.iter()) {
        *count += other;
    }
    exclusive_prefix(&mut counts);
    // Each lane scatters through its own offsets, so adjacent elements
    // never update the same counter.
    for ((offset, &start), &count) in second_lane
        .iter_mut()
        .zip(counts.iter())
        .zip(first_lane.iter())
    {
        *offset = start + count;
    }
    let mut scratch = [MaybeUninit::<T>::uninit(); COMPACT_SCATTER];
    let mut chunks = values.chunks_exact(2);
    for chunk in &mut chunks {
        let offset = &mut counts[digit_of(key_of(chunk[0])) as usize];
        scratch[usize::from(*offset)].write(chunk[0]);
        *offset += 1;
        let offset = &mut second_lane[digit_of(key_of(chunk[1])) as usize];
        scratch[usize::from(*offset)].write(chunk[1]);
        *offset += 1;
    }
    for &value in chunks.remainder() {
        let offset = &mut counts[digit_of(key_of(value)) as usize];
        scratch[usize::from(*offset)].write(value);
        *offset += 1;
    }
    // SAFETY: The verified span makes the digit an injective canonical
    // index. The per-lane exclusive prefixes initialize exactly len
    // disjoint scratch slots. Both buffers are aligned with room for len.
    unsafe {
        std::ptr::copy_nonoverlapping(
            scratch.as_ptr().cast::<T>(),
            values.as_mut_ptr(),
            values.len(),
        );
    }
    true
}

fn single_byte<T: Sealed>(values: &mut [T]) -> bool {
    let first = values[0].key();
    let variation = values[..16]
        .iter()
        .chain(values.last())
        .chain(values.get(values.len() / 2))
        .fold(0, |bits, v| bits | (v.key() ^ first));
    if variation == 0 {
        return false;
    }
    let shift = variation.trailing_zeros() / 8 * 8;
    variation >> shift <= 255 && count_byte(values, first, shift)
}

/// A sampled byte is only a proposal. Verify every other bit before filling
/// exact inverse keys, using independent counters and validity reductions.
#[inline(never)]
fn count_byte<T: Sealed>(values: &mut [T], first: u64, shift: u32) -> bool {
    if values.len() <= 512 {
        return scatter_small_byte(values, first, shift);
    }
    let mask = 255_u64 << shift;
    let base = first & !mask;
    let raw_first = first ^ T::SIGN_MASK;
    let mut counts = [[0_u32; 256]; 4];
    let mut variation = [0_u64; 4];
    for block in validation_blocks(values) {
        let mut chunks = block.chunks_exact(4);
        for chunk in &mut chunks {
            for lane in 0..4 {
                let key = chunk[lane].radix_key();
                variation[lane] |= key ^ raw_first;
                counts[lane][((key >> shift) & 255) as usize] += 1;
            }
        }
        for &value in chunks.remainder() {
            let key = value.radix_key();
            variation[0] |= key ^ raw_first;
            counts[0][((key >> shift) & 255) as usize] += 1;
        }
        if (variation[0] | variation[1] | variation[2] | variation[3]) & !mask != 0 {
            return false;
        }
    }
    let [first, second, third, fourth] = &mut counts;
    for (bucket, count) in first.iter_mut().enumerate() {
        *count += second[bucket] + third[bucket] + fourth[bucket];
    }
    let rotation = ((T::SIGN_MASK >> shift) & 255) as usize;
    fill_key_range(
        values,
        0..=255,
        0,
        |bucket| counts[0][bucket as usize ^ rotation] as usize,
        |bucket| T::from_key(base | (bucket << shift)),
    );
    true
}

/// Dense small histograms are cheaper to scatter than to fill bucket by bucket.
/// The stack buffer is bounded; all unchanged bits are verified before writing.
#[inline(never)]
fn scatter_small_byte<T: Sealed>(values: &mut [T], first: u64, shift: u32) -> bool {
    assert!(values.len() <= 512);
    let mask = 255_u64 << shift;
    let mut counts = [[0_u16; 256]; 2];
    let mut variation = [0_u64; 2];
    let mut chunks = values.chunks_exact(2);
    for chunk in &mut chunks {
        for lane in 0..2 {
            let key = chunk[lane].key();
            variation[lane] |= key ^ first;
            counts[lane][((key >> shift) & 255) as usize] += 1;
        }
    }
    for &value in chunks.remainder() {
        let key = value.key();
        variation[0] |= key ^ first;
        counts[0][((key >> shift) & 255) as usize] += 1;
    }
    if (variation[0] | variation[1]) & !mask != 0 {
        return false;
    }
    let [counts, second] = &mut counts;
    let mut occupied = [false; 16];
    let mut distinct = 0;
    for ((block, other), flag) in counts
        .chunks_exact_mut(16)
        .zip(second.chunks_exact(16))
        .zip(&mut occupied)
    {
        let mut any = 0;
        for (count, other) in block.iter_mut().zip(other) {
            *count += *other;
            any |= *count;
            distinct += usize::from(*count != 0);
        }
        *flag = any != 0;
    }
    if distinct <= 16 {
        let mut start = 0;
        for block in (0..16).filter(|&block| occupied[block]) {
            for (bucket, &count) in counts.iter().enumerate().skip(block * 16).take(16) {
                let end = start + usize::from(count);
                fill_bucket(
                    values,
                    start,
                    end,
                    T::from_key((first & !mask) | ((bucket as u64) << shift)),
                );
                start = end;
            }
        }
    } else {
        exclusive_prefix(counts);
        let mut scratch = [MaybeUninit::<T>::uninit(); 512];
        for &value in values.iter() {
            let offset = &mut counts[((value.key() >> shift) & 255) as usize];
            scratch[usize::from(*offset)].write(value);
            *offset += 1;
        }
        // SAFETY: Verified unchanged bits make byte order equal to full key
        // order. Exclusive histogram prefixes initialize exactly len disjoint
        // scratch slots. Both buffers are aligned and have room for len.
        unsafe {
            std::ptr::copy_nonoverlapping(
                scratch.as_ptr().cast::<T>(),
                values.as_mut_ptr(),
                values.len(),
            );
        }
    }
    true
}

#[inline]
fn fill_bucket<T: Copy>(values: &mut [T], start: usize, end: usize, value: T) {
    if end == start {
        return;
    }
    if !fixed_fill::<T, 16>(values, start, end, value) {
        values[start..end].fill(value);
    }
}

/// Write one bucket with a single fixed-width store when it fits in the
/// slice, returning false when it does not. Elements of at most four bytes
/// use thirty-two-element stores, wider ones sixteen, so a store is at most
/// 128 bytes.
#[inline(always)]
fn fixed_fill<T: Copy, const NARROW: usize>(
    values: &mut [T],
    start: usize,
    end: usize,
    value: T,
) -> bool {
    if std::mem::size_of::<T>() <= 4 {
        fixed_store::<T, 32>(values, start, end, value)
    } else {
        fixed_store::<T, NARROW>(values, start, end, value)
    }
}

#[inline(always)]
fn fixed_store<T: Copy, const W: usize>(
    values: &mut [T],
    start: usize,
    end: usize,
    value: T,
) -> bool {
    if end - start <= W && values.len() - start >= W {
        // SAFETY: The W-element store stays inside the input slice. A
        // complete verified histogram partitions the slice; any overlap with
        // later buckets is overwritten by their subsequent fills, including
        // a whole store for an empty bucket.
        unsafe {
            values
                .as_mut_ptr()
                .add(start)
                .cast::<[T; W]>()
                .write([value; W])
        };
        true
    } else {
        false
    }
}

/// Fill consecutive key buckets of a verified histogram. Dense histograms
/// (at least two values per key) branch on each count. Sparser ones proceed
/// in blocks of sixteen keys: an empty block costs one predictable branch,
/// and the buckets of an occupied block are written without branching on
/// their individual counts: short buckets, even empty ones, use the fixed
/// sixteen-element store when it fits, and later buckets overwrite the
/// excess. Moderately sparse histograms thus avoid a mispredicted branch per
/// bucket, while very sparse ones skip most blocks. Returns the end of the
/// last bucket.
#[inline]
fn fill_key_range<T: Copy>(
    values: &mut [T],
    keys: std::ops::RangeInclusive<u64>,
    mut start: usize,
    count_of: impl Fn(u64) -> usize,
    value_of: impl Fn(u64) -> T,
) -> usize {
    let (low, high) = keys.into_inner();
    if high < low {
        return start;
    }
    // Spans are at most a few hundred keys; exclusive ranges over offsets
    // keep the bucket loops free of inclusive-range bookkeeping.
    let span = high - low + 1;
    if values.len() as u64 >= 2 * span {
        // Dense histograms have few empty buckets; branching on each count
        // is predictable and avoids writing stores for empty buckets.
        for offset in 0..span {
            let key = low + offset;
            let end = start + count_of(key);
            fill_bucket(values, start, end, value_of(key));
            start = end;
        }
        return start;
    }
    let mut block = 0;
    while block < span {
        let block_span = (span - block).min(16);
        let occupied = (0..block_span).fold(0, |any, i| any | count_of(low + block + i)) != 0;
        if occupied {
            for i in 0..block_span {
                let key = low + block + i;
                let end = start + count_of(key);
                if !fixed_fill::<T, 16>(values, start, end, value_of(key)) && end > start {
                    values[start..end].fill(value_of(key));
                }
                start = end;
            }
        }
        block += block_span;
    }
    start
}

/// Reject misleading samples after a short block, then amortize validation
/// across larger blocks when the proposed distribution continues to hold.
fn validation_blocks<T>(values: &[T]) -> impl Iterator<Item = &[T]> {
    let (head, tail) = values.split_at(values.len().min(64));
    std::iter::once(head).chain(tail.chunks(1024))
}

#[inline(never)]
fn finish_few<T: Sealed, const K: usize>(values: &mut [T], representatives: &[T; 4]) -> bool {
    let keys: [u64; K] = std::array::from_fn(|i| representatives[i].key());
    let mut counts = [0_usize; K];
    let mut all_known = true;
    // Independent counts and one validity reduction let the compiler vectorize
    // repeated-value inputs without indexed histogram dependencies.
    for block in validation_blocks(values) {
        for value in block {
            let key = value.key();
            let mut known = false;
            for i in 0..K {
                let matches = key == keys[i];
                counts[i] += usize::from(matches);
                known |= matches;
            }
            all_known &= known;
        }
        if !all_known {
            return false;
        }
    }
    if !all_known {
        return false;
    }
    let mut start = 0;
    for i in 0..K {
        let end = start + counts[i];
        values[start..end].fill(representatives[i]);
        start = end;
    }
    true
}

/// Count small alphabets that are not confined to a compact integer range.
/// Sampling chooses representatives; every value is checked before any fill.
fn sparse_values<T: Sealed>(values: &mut [T]) -> bool {
    let mut representatives = [values[0]; 16];
    let mut distinct = 1;
    for &value in values[..64].iter().chain(values.last()) {
        if !representatives[..distinct]
            .iter()
            .any(|v| v.key() == value.key())
        {
            if distinct == 16 {
                return false;
            }
            representatives[distinct] = value;
            distinct += 1;
        }
    }
    representatives[..distinct].sort_unstable_by_key(|v| v.key());
    // Unused lanes duplicate the first key and are excluded from output.
    representatives[distinct..].fill(values[0]);
    if distinct <= 8 {
        finish_sparse::<T, 8>(values, &representatives, distinct)
    } else {
        finish_sparse::<T, 16>(values, &representatives, distinct)
    }
}

#[inline(never)]
fn finish_sparse<T: Sealed, const K: usize>(
    values: &mut [T],
    representatives: &[T; 16],
    distinct: usize,
) -> bool {
    let keys: [u64; K] = std::array::from_fn(|i| representatives[i].key());
    let mut counts = [0_usize; K];
    for block in validation_blocks(values) {
        let mut all_known = true;
        for value in block {
            let key = value.key();
            let mut known = false;
            for i in 0..K {
                let matches = key == keys[i];
                counts[i] += usize::from(matches);
                known |= matches;
            }
            all_known &= known;
        }
        if !all_known {
            return false;
        }
    }
    let mut start = 0;
    for i in 0..distinct {
        let end = start + counts[i];
        values[start..end].fill(representatives[i]);
        start = end;
    }
    true
}

/// Reject random inputs after a short prefix; accept only after checking every
/// adjacent pair. Comparing sortable keys preserves float total ordering.
#[inline]
fn ordered<T: Sealed>(values: &mut [T]) -> Preparation {
    // One reduction settles a constant short slice before the probe, whose
    // scalar comparisons would cost as much; matching end values keep the
    // reduction off every other input. The bound per width is measured:
    // beyond it the probe first is cheaper.
    let guard_first = match std::mem::size_of::<T>() {
        1 | 2 => values.len() <= 1024,
        4 => values.len() <= 128,
        _ => values.len() <= 64,
    };
    if guard_first
        && values[0].raw_bits() == values[values.len() - 1].raw_bits()
        && constant(values)
    {
        return Preparation::Done;
    }
    let mut ascending = true;
    let mut descending = true;
    for pair in values[..8].windows(2) {
        let a = pair[0].key();
        let b = pair[1].key();
        ascending &= a <= b;
        descending &= a >= b;
        if !ascending && !descending {
            // Disordered input continues with the few-key probe and the
            // length-based choice between comparison and radix sorting.
            return Preparation::Radix;
        }
    }
    if ascending && descending && constant(values) {
        return Preparation::Done;
    }
    // Short byte slices beyond the general repair's reach count faster than
    // any settled window or merge could finish.
    let direct = std::mem::size_of::<T>() == 1
        && values.len() > REPAIRED_BYTES
        && values.len() <= DIRECT_BYTES;
    if ascending {
        let start = ordered_scan::<T, false>(values);
        if start == values.len() {
            return Preparation::Done;
        }
        if direct {
            if values[start].key() < values[start - 8].key()
                && let Some(result) = two_runs(values, start, false)
            {
                return result;
            }
            if dense_bytes::<T, false>(values, start) || !ordered_repair::<T, false>(values, start)
            {
                byte_scatter::<T, 4>(values);
            }
            return Preparation::Done;
        }
        // If the verified ascending prefix is a plateau, its first decrease
        // establishes descending order. Choose that direction before editing.
        if !descending || values[start - 1].key() != values[0].key() {
            // A short insertion belongs in bounded repair. Avoid scanning a
            // long suffix for a second run when only a few neighbors moved.
            if values[start].key() < values[start.saturating_sub(8)].key()
                || (values.len() - start >= 32 && values[start + 7].key() < values[start].key())
            {
                if let Some(result) = two_runs(values, start, false) {
                    return result;
                }
                if futile_repair::<T, false>(values, start) {
                    return ordered_fallback(values);
                }
            }
            return if ordered_repair::<T, false>(values, start) {
                Preparation::Done
            } else {
                ordered_fallback(values)
            };
        }
    }
    if descending {
        let start = ordered_scan::<T, true>(values);
        if start == values.len() {
            values.reverse();
            return Preparation::Done;
        }
        if direct {
            if values[start].key() > values[start - 8].key()
                && let Some(result) = two_runs(values, start, true)
            {
                return result;
            }
            if !dense_bytes::<T, true>(values, start) && ordered_repair::<T, true>(values, start) {
                values.reverse();
            } else {
                byte_scatter::<T, 4>(values);
            }
            return Preparation::Done;
        }
        if values[start].key() > values[start.saturating_sub(8)].key()
            || (values.len() - start >= 32 && values[start + 7].key() > values[start].key())
        {
            if let Some(result) = two_runs(values, start, true) {
                return result;
            }
            if futile_repair::<T, true>(values, start) {
                return ordered_fallback(values);
            }
        }
        if ordered_repair::<T, true>(values, start) {
            values.reverse();
            return Preparation::Done;
        }
    }
    ordered_fallback(values)
}

/// Longest byte slice whose disorder goes straight to counting after at
/// most a few short insertions.
const DIRECT_BYTES: usize = 256;
/// Longest byte slice repaired under the general policy: its count has a
/// fixed cost that a few repairs or settled windows do not reach.
const REPAIRED_BYTES: usize = 96;

/// Whether the window after the verified prefix of a short byte slice holds
/// more inversions than a repair could settle for less than counting.
/// Every repair of a byte slice mispredicts at least once, so a few per
/// window already cost more than the count, whose fixed part weighs more
/// on the shorter slices. This is only a dispatch shortcut.
#[inline]
fn dense_bytes<T: Sealed, const DESCENDING: bool>(values: &[T], start: usize) -> bool {
    let window = &values[start - 1..values.len().min(start + 31)];
    let inversions = window
        .windows(2)
        .filter(|pair| {
            let (a, b) = (pair[0].key(), pair[1].key());
            if DESCENDING { a < b } else { a > b }
        })
        .count();
    inversions
        >= if values.len() < 128 {
            5
        } else if values.len() < 192 {
            4
        } else {
            3
        }
}

/// Whether every key equals the first. One reduction per block costs half
/// of an order scan, and a block that differs leaves early. Each type
/// reduces at its own width on its stored bits, so every vector holds as
/// many lanes as the type allows and no key transform is needed.
#[inline]
fn constant<T: Sealed>(values: &[T]) -> bool {
    // A short first block rejects most varying input early; the rest
    // reduces in large blocks. Up to 1,024 elements reduce in one block,
    // where a split would cost a second reduction for nothing.
    let (head, tail) = values.split_at(if values.len() <= 1024 {
        values.len()
    } else {
        128
    });
    match std::mem::size_of::<T>() {
        1 => {
            let first = values[0].raw_bits() as u8;
            let block = |block: &[T]| {
                block.iter().fold(0, |different, value| {
                    different | (value.raw_bits() as u8 ^ first)
                }) == 0
            };
            block(head) && tail.chunks(16384).all(block)
        }
        2 => {
            let first = values[0].raw_bits() as u16;
            let block = |block: &[T]| {
                block.iter().fold(0, |different, value| {
                    different | (value.raw_bits() as u16 ^ first)
                }) == 0
            };
            block(head) && tail.chunks(8192).all(block)
        }
        4 => {
            let first = values[0].raw_bits();
            let block = |block: &[T]| {
                block
                    .iter()
                    .fold(0, |different, value| different | (value.raw_bits() ^ first))
                    == 0
            };
            if values.len() <= 1024 {
                // Fixed 64-element blocks unroll completely; one variable
                // block costs a loop and a longer tail here.
                return values.chunks(64).all(block);
            }
            block(head) && tail.chunks(4096).all(block)
        }
        _ => {
            let first = values[0].key();
            let block = |block: &[T]| {
                block
                    .iter()
                    .fold(0, |different, value| different | (value.key() ^ first))
                    == 0
            };
            block(head) && tail.chunks(2048).all(block)
        }
    }
}

/// After a failed second-run check, a dense block of elements beyond the
/// verified prefix's boundary value indicates several substantial runs rather
/// than a local inversion. Bounded repair would insert most of them one by one
/// and still give up, so it is skipped. This is only a dispatch shortcut: the
/// general path sorts either way.
#[inline]
fn futile_repair<T: Sealed, const DESCENDING: bool>(values: &[T], start: usize) -> bool {
    let Some(window) = values.get(start..start + 16) else {
        return false;
    };
    let boundary = values[start - 1].key();
    let beyond = window
        .iter()
        .filter(|v| {
            if DESCENDING {
                v.key() > boundary
            } else {
                v.key() < boundary
            }
        })
        .count();
    beyond >= 12
}

fn ordered_fallback<T: Sealed>(values: &mut [T]) -> Preparation {
    if T::SMALL_PARTIAL && values.len() <= 256 {
        T::small_sort(values);
        Preparation::Done
    } else {
        Preparation::Radix
    }
}

/// The prefix is already verified. A second long monotone run is cheaper to
/// rotate or merge than to repair one element at a time. Verify it completely,
/// optionally repairing its last few elements, before reversing either run.
#[cold]
#[inline(never)]
fn two_runs<T: Sealed>(values: &mut [T], middle: usize, descending: bool) -> Option<Preparation> {
    // Longer byte slices count faster than they merge through scratch.
    if (std::mem::size_of::<T>() == 1 && values.len() > STACK_MERGE)
        || middle < 8
        || values.len() - middle < 8
    {
        return None;
    }
    let tail = &mut values[middle..];
    let mut up = true;
    let mut down = true;
    for pair in tail[..8].windows(2) {
        up &= pair[0].key() <= pair[1].key();
        down &= pair[0].key() >= pair[1].key();
    }
    let tail_descending = if up && verify_run::<T, false>(tail) {
        false
    } else if down && verify_run::<T, true>(tail) {
        true
    } else {
        return None;
    };
    if descending {
        values[..middle].reverse();
    }
    if tail_descending {
        values[middle..].reverse();
    }
    if values[middle - 1].key() <= values[middle].key() {
        return Some(Preparation::Done);
    }
    if values[values.len() - 1].key() <= values[0].key() {
        values.rotate_left(middle);
        return Some(Preparation::Done);
    }
    if values.len() <= STACK_MERGE {
        // Short overlapping runs merge through a stack buffer, so the
        // allocating API does not reserve scratch for them.
        let mut scratch = [MaybeUninit::<T>::uninit(); STACK_MERGE];
        merge_runs(values, &mut scratch[..values.len()], middle);
        return Some(Preparation::Done);
    }
    Some(Preparation::Merge(middle))
}

/// Longest run pair merged through a stack buffer instead of scratch.
const STACK_MERGE: usize = 128;

fn verify_run<T: Sealed, const DESCENDING: bool>(values: &mut [T]) -> bool {
    let start = ordered_scan::<T, DESCENDING>(values);
    start == values.len()
        || (values.len() - start <= 8 && ordered_repair::<T, DESCENDING>(values, start))
}

/// Complete a fully verified pair of ascending runs, or enter the radix path.
#[inline]
pub(crate) fn finish<T: Sealed, const N: usize>(
    values: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    preparation: Preparation,
) {
    match preparation {
        Preparation::Done => {}
        Preparation::Merge(middle) => merge_runs(values, scratch, middle),
        Preparation::Radix => sort::<T, N>(values, scratch),
    }
}

/// Merge the overlapping region from both ends into scratch. Independent
/// front/back cursors expose two comparison chains; batches amortize bounds.
#[inline(never)]
fn merge_runs<T: Sealed>(values: &mut [T], scratch: &mut [MaybeUninit<T>], middle: usize) {
    let first_right = values[middle].key();
    let last_left = values[middle - 1].key();
    let skip = values[..middle].partition_point(|v| v.key() <= first_right);
    let right = values[middle..].partition_point(|v| v.key() < last_left);
    let values = &mut values[skip..middle + right];
    let middle = middle - skip;
    let len = values.len();
    if middle < len / 4 || len - middle < len / 4 {
        merge_skewed(values, scratch, middle);
    } else {
        merge_core(values, scratch, middle);
    }
}

fn merge_skewed<T: Sealed>(values: &mut [T], scratch: &mut [MaybeUninit<T>], middle: usize) {
    let (left, right) = values.split_at(middle);
    let (short, long) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let temp = scratch.as_mut_ptr().cast::<T>();
    let (mut cursor, mut out) = (0, 0);
    // SAFETY: Each verified input run is initialized and sorted. A copied
    // block is bounded by the long run and its last key is no greater than
    // the next short-run key. Every output slot is written once, and the
    // entire disjoint scratch result is initialized before copying it back.
    unsafe {
        for &value in short {
            let key = value.key();
            while long.len() - cursor >= 8 && long[cursor + 7].key() <= key {
                std::ptr::copy_nonoverlapping(long.as_ptr().add(cursor), temp.add(out), 8);
                cursor += 8;
                out += 8;
            }
            while cursor < long.len() && long[cursor].key() <= key {
                temp.add(out).write(long[cursor]);
                cursor += 1;
                out += 1;
            }
            temp.add(out).write(value);
            out += 1;
        }
        std::ptr::copy_nonoverlapping(
            long.as_ptr().add(cursor),
            temp.add(out),
            long.len() - cursor,
        );
        std::ptr::copy_nonoverlapping(temp, values.as_mut_ptr(), values.len());
    }
}

fn merge_core<T: Sealed>(values: &mut [T], scratch: &mut [MaybeUninit<T>], middle: usize) {
    let len = values.len();
    let (mut a, mut b, mut a_end, mut b_end) = (0, middle, middle, len);
    let (mut out, mut out_end) = (0, len);
    if len >= 128 {
        let pairs = if values[0].key() <= values[middle].key() {
            merge_alternating::<T, true>(&values[..middle], &values[middle..], scratch)
        } else {
            merge_alternating::<T, false>(&values[..middle], &values[middle..], scratch)
        };
        a += pairs;
        b += pairs;
        out += 2 * pairs;
    }
    let data = values.as_mut_ptr();
    let temp = scratch.as_mut_ptr().cast::<T>();
    // SAFETY: Input and scratch are disjoint aligned buffers with room for len
    // elements. Both input runs are sorted and initialized. A batch starts
    // with at least eight elements in each run and consumes exactly eight in
    // total, so its front/back reads cannot cross. Front/back outputs occupy
    // disjoint positions. The scalar tail checks each cursor before reading;
    // the final copies initialize all remaining scratch slots. Only then is
    // the complete initialized result copied back to the input slice.
    unsafe {
        while a_end - a >= 8 && b_end - b >= 8 {
            // An entire ordered block can bypass the dependent merge cursors.
            // Fixed counts let the compiler use inline vector loads/stores.
            if data.add(a + 7).read().key() <= data.add(b).read().key() {
                std::ptr::copy_nonoverlapping(data.add(a), temp.add(out), 8);
                a += 8;
                out += 8;
                continue;
            }
            if data.add(b + 7).read().key() < data.add(a).read().key() {
                std::ptr::copy_nonoverlapping(data.add(b), temp.add(out), 8);
                b += 8;
                out += 8;
                continue;
            }
            for _ in 0..4 {
                let front_a = data.add(a).read();
                let front_b = data.add(b).read();
                let back_a = data.add(a_end - 1).read();
                let back_b = data.add(b_end - 1).read();
                let first_a = front_a.key() <= front_b.key();
                let last_a = back_a.key() > back_b.key();
                temp.add(out).write(if first_a { front_a } else { front_b });
                out_end -= 1;
                temp.add(out_end)
                    .write(if last_a { back_a } else { back_b });
                a += usize::from(first_a);
                b += usize::from(!first_a);
                a_end -= usize::from(last_a);
                b_end -= usize::from(!last_a);
                out += 1;
            }
        }
        while a < a_end && b < b_end {
            let left = data.add(a).read();
            let right = data.add(b).read();
            let take_left = left.key() <= right.key();
            temp.add(out).write(if take_left { left } else { right });
            a += usize::from(take_left);
            b += usize::from(!take_left);
            out += 1;
        }
        std::ptr::copy_nonoverlapping(data.add(a), temp.add(out), a_end - a);
        out += a_end - a;
        std::ptr::copy_nonoverlapping(data.add(b), temp.add(out), b_end - b);
        std::ptr::copy_nonoverlapping(temp, data, len);
    }
}

/// Emit verified alternating blocks without dependent merge cursors. Include
/// the next pair's first value in each check so the remaining runs cannot
/// contain an element smaller than the last value written. A failed block
/// leaves both input cursors at its beginning for the general merge.
fn merge_alternating<T: Sealed, const LEFT_FIRST: bool>(
    left: &[T],
    right: &[T],
    scratch: &mut [MaybeUninit<T>],
) -> usize {
    let (first, second) = if LEFT_FIRST {
        (left, right)
    } else {
        (right, left)
    };
    let limit = first.len().min(second.len()).saturating_sub(1) / 8 * 8;
    let mut start = 0;
    while start < limit {
        let mut alternating = true;
        for lane in 0..8 {
            let i = start + lane;
            alternating &= first[i].key() <= second[i].key();
            alternating &= second[i].key() <= first[i + 1].key();
        }
        if !alternating {
            break;
        }
        for lane in 0..8 {
            let i = start + lane;
            scratch[2 * i].write(first[i]);
            scratch[2 * i + 1].write(second[i]);
        }
        start += 8;
    }
    start
}

#[inline]
pub(crate) fn ordered_scan<T: Sealed, const DESCENDING: bool>(values: &[T]) -> usize {
    if T::FLOAT_BITS {
        return float_scan::<T, DESCENDING>(values);
    }
    let block = if std::mem::size_of::<T>() != 4 {
        16
    } else {
        32
    };
    let compare = |a, b| if DESCENDING { a >= b } else { a <= b };
    let mut previous = values[0].key();
    let mut chunks = values.chunks_exact(block);
    for (index, chunk) in chunks.by_ref().enumerate() {
        let mut in_order = compare(previous, chunk[0].key());
        for pair in chunk.windows(2) {
            in_order &= compare(pair[0].key(), pair[1].key());
        }
        if !in_order {
            // The block reduction keeps the common ordered scan independent.
            // Only a failing block needs the exact first inversion position.
            return first_inversion::<T, DESCENDING>(
                values,
                (index * block).max(1),
                (index + 1) * block,
            );
        }
        previous = chunk[block - 1].key();
    }
    let mut start = values.len() - chunks.remainder().len();
    let mut tails = chunks.remainder().chunks_exact(8);
    for chunk in &mut tails {
        let mut in_order = compare(previous, chunk[0].key());
        for pair in chunk.windows(2) {
            in_order &= compare(pair[0].key(), pair[1].key());
        }
        if !in_order {
            return first_inversion::<T, DESCENDING>(values, start.max(1), start + 8);
        }
        previous = chunk[7].key();
        start += 8;
    }
    for (index, value) in tails.remainder().iter().enumerate() {
        let key = value.key();
        if !compare(previous, key) {
            return start + index;
        }
        previous = key;
    }
    values.len()
}

/// Float total order within one block compares raw bit patterns as signed
/// integers after flipping the magnitude bits of a block that starts
/// negative. Within one sign that is exact, and a pair whose first element
/// is negative and second non-negative is accepted correctly in either
/// space while the opposite pair is flagged. Non-negative pairs inside a
/// block that starts negative compare in reverse (and negative pairs inside
/// a block that starts non-negative, for descending order), so such a block
/// always takes the exact check: in sorted input it is the one block holding
/// the sign boundary. This replaces the per-element key transform with one
/// exclusive-or per element.
#[inline]
fn float_scan<T: Sealed, const DESCENDING: bool>(values: &[T]) -> usize {
    const BLOCK: usize = 32;
    let compare = |a: u32, b: u32, flip: u32| {
        let (a, b) = ((a ^ flip) as i32, (b ^ flip) as i32);
        if DESCENDING { a >= b } else { a <= b }
    };
    let mut previous = values[0].raw_bits();
    let mut chunks = values.chunks_exact(BLOCK);
    let mut index = 0;
    for chunk in chunks.by_ref() {
        let first_negative = (chunk[0].raw_bits() as i32) < 0;
        let last_negative = (chunk[BLOCK - 1].raw_bits() as i32) < 0;
        let flip = if first_negative { u32::MAX >> 1 } else { 0 };
        let mut in_order = compare(previous, chunk[0].raw_bits(), flip);
        for pair in chunk.windows(2) {
            in_order &= compare(pair[0].raw_bits(), pair[1].raw_bits(), flip);
        }
        let mixed = if DESCENDING {
            !first_negative && last_negative
        } else {
            first_negative && !last_negative
        };
        if !in_order || mixed {
            // Only the block holding the sign boundary is checked while
            // ordered; one vectorized key comparison settles it before the
            // scalar search for the exact position.
            let previous_key = values[(index * BLOCK).max(1) - 1].key();
            let first = chunk[0].key();
            let mut exact = if DESCENDING {
                previous_key >= first
            } else {
                previous_key <= first
            };
            for pair in chunk.windows(2) {
                let (a, b) = (pair[0].key(), pair[1].key());
                exact &= if DESCENDING { a >= b } else { a <= b };
            }
            if !exact {
                return first_inversion::<T, DESCENDING>(
                    values,
                    (index * BLOCK).max(1),
                    (index + 1) * BLOCK,
                );
            }
        }
        previous = chunk[BLOCK - 1].raw_bits();
        index += 1;
    }
    let start = index * BLOCK;
    let mut previous = values[start.max(1) - 1].key();
    for (offset, value) in chunks.remainder().iter().enumerate() {
        let key = value.key();
        if if DESCENDING {
            previous < key
        } else {
            previous > key
        } {
            return start + offset;
        }
        previous = key;
    }
    values.len()
}

#[cold]
#[inline(never)]
fn first_inversion<T: Sealed, const DESCENDING: bool>(
    values: &[T],
    start: usize,
    end: usize,
) -> usize {
    for i in start..end {
        let a = values[i - 1].key();
        let b = values[i].key();
        if if DESCENDING { a < b } else { a > b } {
            return i;
        }
    }
    end
}

/// Resume at the verified prefix. Budgets bound the work even when a monotone
/// sample precedes disordered data. Scalar insertions count one unit per
/// shifted element and contiguous moves a quarter per element; the units
/// never exceed len. Cheap repairs (at most eight units) may repeat up to
/// sixty-four times, or len/16 times for long inputs, and longer moves at
/// most eight times. A failed attempt therefore costs at most about one pass
/// and leaves a permutation of the input. A single element that is too large
/// for its place is carried forward in one move; local disorder such as
/// adjacent swaps is settled a window at a time without data-dependent
/// branches.
#[inline(never)]
fn ordered_repair<T: Sealed, const DESCENDING: bool>(values: &mut [T], mut index: usize) -> bool {
    let out_of_order = |a, b| if DESCENDING { a < b } else { a > b };
    let len = values.len();
    // A short byte slice counts faster than a settled window could finish,
    // and dense disorder there goes to counting after a few repairs or two
    // long moves, within a quarter of the slice's length of moves. The
    // shortest slices keep the general policy, whose few repairs cost less
    // than their comparison sort.
    let short_bytes = std::mem::size_of::<T>() == 1 && len > REPAIRED_BYTES && len <= DIRECT_BYTES;
    let cheap_limit = if short_bytes {
        len
    } else {
        (len / 4).clamp(8, CHEAP_REPAIRS).max(len / 16)
    };
    let long_limit = if short_bytes { 2 } else { LONG_REPAIRS };
    let start = index;
    let mut cheap = 0;
    let mut long = 0;
    let mut remaining = if short_bytes { len / 4 } else { len };
    let mut settled_at = usize::MAX;
    let mut previous_repair = usize::MAX;
    while index < len {
        let value = values[index];
        let key = value.key();
        let cost;
        if index + 1 < len
            && out_of_order(values[index - 1].key(), values[index + 1].key())
            && index >= 2
            && !out_of_order(values[index - 2].key(), key)
        {
            // Only the previous element is out of place and at least two
            // elements belong before it: carry it forward to the first
            // element that is not smaller, then resume behind it. The suffix
            // is unverified, so the binary search only proposes a place; the
            // scan after the move verifies it and any mistake is one more
            // repair. The search is bounded by the remaining budget.
            let outlier = values[index - 1];
            let outlier_key = outlier.key();
            let limit = len.min(index + 2 + remaining.saturating_mul(4));
            let end = index
                + 2
                + values[index + 2..limit].partition_point(|v| out_of_order(outlier_key, v.key()));
            cost = (end - index).div_ceil(4);
            // The scan may not advance past its initial two elements, so
            // check that minimum cost too before moving anything.
            if cost > remaining {
                return false;
            }
            values.copy_within(index..end, index - 1);
            values[end - 1] = outlier;
            index -= 1;
        } else if index >= 64 && out_of_order(values[index - 64].key(), key) {
            // The prefix is sorted. Find a long insertion in logarithmic
            // comparisons, then move its contiguous range in one operation.
            let position = values[..index].partition_point(|v| !out_of_order(v.key(), key));
            cost = (index - position).div_ceil(4);
            if cost > remaining {
                return false;
            }
            values.copy_within(position..index, position + 1);
            values[position] = value;
        } else if !short_bytes
            && index >= 2
            && !out_of_order(values[index - 2].key(), key)
            && index != settled_at
            && previous_repair != usize::MAX
            && index - previous_repair < REPAIR_WINDOW / 2
        {
            // The element fits right behind the verified prefix and the
            // previous repair was within half a window: dense local
            // disorder. Settle a
            // whole window, then rescan from the last verified element,
            // because a smaller element from inside the window may now sit
            // at the prefix boundary. If the same inversion survives, the
            // next visit inserts it instead. An isolated swap takes the
            // cheaper scalar insertion below.
            let end = len.min(index - 1 + REPAIR_WINDOW);
            settle_window::<T, DESCENDING>(&mut values[index - 1..end]);
            settled_at = index;
            cost = CHEAP_SHIFTS;
            if cost > remaining {
                return false;
            }
            index -= 2;
        } else {
            let mut position = index;
            while position > 0 && out_of_order(values[position - 1].key(), key) {
                if remaining == index - position {
                    // Restore the held value before falling back: partial
                    // repairs must preserve the exact multiset.
                    values[position] = value;
                    return false;
                }
                values[position] = values[position - 1];
                position -= 1;
            }
            values[position] = value;
            cost = index - position;
        }
        remaining -= cost;
        if cost <= CHEAP_SHIFTS {
            cheap += 1;
        } else {
            long += 1;
        }
        if cheap > cheap_limit || long > long_limit {
            return false;
        }
        if short_bytes && cheap > 8 && cheap * 4 > index - start {
            return false;
        }
        previous_repair = index;
        index += ordered_scan::<T, DESCENDING>(&values[index..]);
    }
    true
}

/// Exchange out-of-order neighbours at even, then odd, offsets for a few
/// rounds. An element displaced by at most two positions settles, and the
/// work has no data-dependent branches, so a window of adjacent swaps costs
/// about as much as one scalar insertion.
#[inline]
fn settle_window<T: Sealed, const DESCENDING: bool>(window: &mut [T]) {
    for round in 0..REPAIR_ROUNDS {
        for pair in window[round & 1..].chunks_exact_mut(2) {
            let (a, b) = (pair[0], pair[1]);
            let (ka, kb) = (a.key(), b.key());
            let swap = if DESCENDING { ka < kb } else { ka > kb };
            pair[0] = if swap { b } else { a };
            pair[1] = if swap { a } else { b };
        }
    }
}

/// A repair costing at most this many units is cheap.
const CHEAP_SHIFTS: usize = 8;
/// Minimum cheap repairs per attempt; long inputs allow len/16.
const CHEAP_REPAIRS: usize = 64;
/// Maximum longer moves per attempt.
const LONG_REPAIRS: usize = 8;
/// Elements settled together after a local inversion.
const REPAIR_WINDOW: usize = 32;
/// Exchange rounds per settled window.
const REPAIR_ROUNDS: usize = 4;

trait Counter: Copy + std::ops::Add<Output = Self> + std::ops::AddAssign {
    const ZERO: Self;
    const ONE: Self;
    const MAX: usize;
    fn index(self) -> usize;
}

macro_rules! counter {
    ($ty:ty) => {
        impl Counter for $ty {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MAX: usize = <$ty>::MAX as usize;
            #[inline(always)]
            fn index(self) -> usize {
                self as usize
            }
        }
    };
}
counter!(u16);
counter!(u32);
counter!(usize);

#[inline(always)]
fn count<C: Counter, const N: usize>(frequencies: &mut [[C; 256]; N], key: u64) {
    for (byte, table) in frequencies.iter_mut().enumerate() {
        table[((key >> (byte * 8)) & 255) as usize] += C::ONE;
    }
}

#[inline]
pub(crate) fn sort<T: Sealed, const N: usize>(values: &mut [T], scratch: &mut [MaybeUninit<T>]) {
    // Counts and exclusive offsets never exceed len, including the final
    // increment of a bucket cursor. Thus len == u32::MAX still fits exactly.
    // Larger slices retain full-width counters without truncating any offset.
    if narrow_counts(values.len()) {
        if N == 2 && values.len() >= 262144 && counting16(values, scratch) {
            return;
        }
        if N == 4 && values.len() >= WIDE_DIGITS {
            // Sixteen spread positions: consecutive elements of a sorted run
            // would share their high bits and hide the key variation.
            let first = values[0].key();
            let variation = values
                .iter()
                .step_by(values.len() / 16)
                .take(16)
                .fold(0, |bits, value| bits | (value.key() ^ first));
            if variation.count_ones() >= 20 {
                prefix64::<T, u32, 3, 2048>(values, scratch, [0, 11, 22], None);
                return;
            }
        }
        if N == 8 && values.len() >= 129 {
            adaptive64::<T, u32>(values, scratch);
        } else {
            kernel::<T, u32, N>(values, scratch);
        }
    } else {
        kernel::<T, usize, N>(values, scratch);
    }
}

/// Length from which 32-bit keys take three 11-bit passes instead of four
/// byte passes.
const WIDE_DIGITS: usize = 8192;

/// Reuse a sufficiently large scratch region for a complete 16-bit histogram.
/// Alignment is derived from the slice, never assumed from the key type.
fn counting16<T: Sealed>(values: &mut [T], scratch: &mut [MaybeUninit<T>]) -> bool {
    assert!(std::mem::size_of::<T>() == 2 && narrow_counts(values.len()));
    let first = values[0].key();
    let variation = values
        .iter()
        .step_by(values.len() / 16)
        .take(16)
        .chain(values.last())
        .chain(values.get(values.len() / 2))
        .fold(0, |bits, value| bits | (value.key() ^ first));
    // A sample confined to one byte favors the small histogram kernel. This
    // is only a dispatch hint: that kernel still verifies the complete input.
    if variation & 255 == 0 || variation >> 8 == 0 {
        return false;
    }
    let scratch = &mut scratch[..values.len()];
    // SAFETY: MaybeUninit permits every bit pattern, including uninitialized
    // bytes, and align_to_mut returns disjoint slices with u32 alignment.
    let (_, aligned, _) = unsafe { scratch.align_to_mut::<MaybeUninit<u32>>() };
    let Some(storage) = aligned.get_mut(..65536) else {
        return false;
    };
    for count in storage.iter_mut() {
        count.write(0);
    }
    // SAFETY: All 65536 aligned u32 slots were initialized immediately above;
    // this exclusive view does not outlive the scratch reborrow.
    let counts =
        unsafe { std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u32>(), 65536) };
    // Four increments per iteration keep more table accesses in flight; the
    // table exceeds the first-level cache, so independent misses matter more
    // than the rare repeated key.
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        counts[chunk[0].key() as usize] += 1;
        counts[chunk[1].key() as usize] += 1;
        counts[chunk[2].key() as usize] += 1;
        counts[chunk[3].key() as usize] += 1;
    }
    for value in chunks.remainder() {
        counts[value.key() as usize] += 1;
    }
    let mut start = 0;
    let len = values.len();
    for (key, &count) in counts.iter().enumerate() {
        if count != 0 {
            let end = start + count as usize;
            let value = T::from_key(key as u64);
            let mut position = start;
            while position < end && len - position >= 32 {
                // SAFETY: This fixed-width store stays within the input slice.
                // Each store covers part of the bucket and may overlap later
                // buckets, which are subsequently filled with their own
                // values over their whole range. The complete histogram
                // partitions the whole slice; T is a Copy primitive.
                unsafe {
                    values
                        .as_mut_ptr()
                        .add(position)
                        .cast::<[T; 32]>()
                        .write([value; 32])
                };
                position += 32;
            }
            if position < end {
                values[position..end].fill(value);
            }
            start = end;
        }
    }
    true
}

/// Sort enough high bits to separate most keys, then finish only collisions.
/// Large collision groups use the full radix kernel, keeping worst-case work
/// linear rather than depending on comparison sorting for unbounded groups.
#[inline(never)]
fn adaptive64<T: Sealed, C: Counter>(values: &mut [T], scratch: &mut [MaybeUninit<T>]) {
    let first = values[0].key();
    let variation = values
        .iter()
        .fold(0, |bits, value| bits | (value.key() ^ first));
    if variation == 0 {
        return;
    }
    let byte_shift = (variation.trailing_zeros() as usize / 8) * 8;
    if variation >> byte_shift <= 255 {
        // Every other byte is constant, so one pass sorts the complete keys.
        prefix64::<T, C, 1, 256>(values, scratch, [byte_shift], None);
        return;
    }
    // Sorted digits with too few varying bits leave large collision groups.
    // Avoid their passes when the full histogram kernel can skip those gaps.
    let required = values
        .len()
        .div_ceil(128)
        .next_power_of_two()
        .trailing_zeros();
    let handled = if values.len() >= 8192 {
        dispatch_prefix64::<T, C, 512>(values, scratch, 2, variation, required)
    } else {
        dispatch_prefix64::<T, C, 256>(values, scratch, 4, variation, required)
    };
    if !handled {
        kernel::<T, C, 8>(values, scratch);
    }
}

/// Finish every group of keys that share the sorted digits under `mask`.
/// A single pass reports its bucket ends directly; several passes leave the
/// combined digit groups to one scan of the sorted array.
fn finish_scan<T: Sealed, C: Counter>(values: &mut [T], scratch: &mut [MaybeUninit<T>], mask: u64) {
    const BLOCK: usize = 16;
    let len = values.len();
    let mut start = 0;
    let mut end = 1;
    while end < len {
        let stop = if end + BLOCK <= len {
            // A block whose neighbours all differ, including the pair at its
            // entry, holds only singletons: close the open group and skip
            // to its last element, which may start a group with the next
            // block. The reduction vectorizes; only a block with an equal
            // neighbour pair takes the scalar group walk below, once.
            let block = &values[end - 1..end + BLOCK];
            let mut any_equal = false;
            for pair in block.windows(2) {
                any_equal |= pair[0].key() & mask == pair[1].key() & mask;
            }
            if !any_equal {
                finish_group::<T, C>(&mut values[start..end], &mut scratch[start..end]);
                start = end + BLOCK - 1;
                end += BLOCK;
                continue;
            }
            end + BLOCK
        } else {
            len
        };
        while end < stop {
            if values[end].key() & mask != values[end - 1].key() & mask {
                finish_group::<T, C>(&mut values[start..end], &mut scratch[start..end]);
                start = end;
            }
            end += 1;
        }
    }
    finish_group::<T, C>(&mut values[start..len], &mut scratch[start..len]);
}

fn finish_buckets<T: Sealed, C: Counter, const R: usize>(
    values: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    ends: &[C; R],
) {
    let mut start = 0;
    for end in ends.iter().map(|end| end.index()) {
        if end - start > 1 {
            finish_group::<T, C>(&mut values[start..end], &mut scratch[start..end]);
        }
        start = end;
    }
}

/// Sort one collision group. Singletons and pairs finish inline; small
/// groups use comparator networks on their keys, medium groups comparison
/// sorting, and large groups the full kernel, keeping the worst case linear.
#[inline]
fn finish_group<T: Sealed, C: Counter>(group: &mut [T], scratch: &mut [MaybeUninit<T>]) {
    match group.len() {
        0 | 1 => {}
        2 => {
            if group[0].key() > group[1].key() {
                group.swap(0, 1);
            }
        }
        _ => sort_group::<T, C>(group, scratch),
    }
}

/// One branch-free compare-exchange of two values by key.
#[inline(always)]
fn exchange_values<T: Sealed>(group: &mut [T], i: usize, j: usize) {
    let (a, b) = (group[i], group[j]);
    let swap = a.key() > b.key();
    group[i] = if swap { b } else { a };
    group[j] = if swap { a } else { b };
}

#[inline(never)]
fn sort_group<T: Sealed, C: Counter>(group: &mut [T], scratch: &mut [MaybeUninit<T>]) {
    match group.len() {
        // The common tiny groups exchange values directly, without a key
        // array or an indirect call.
        3 => {
            exchange_values(group, 0, 1);
            exchange_values(group, 1, 2);
            exchange_values(group, 0, 1);
        }
        4 => {
            exchange_values(group, 0, 1);
            exchange_values(group, 2, 3);
            exchange_values(group, 0, 2);
            exchange_values(group, 1, 3);
            exchange_values(group, 1, 2);
        }
        0..=2 => network_sort::<T, 4>(group, network4),
        5..=8 => network_sort::<T, 8>(group, network8),
        9..=16 => network_sort::<T, 16>(group, network16),
        17..=32 => network_sort::<T, 32>(group, network32),
        33..=128 => T::small_sort(group),
        // Larger groups choose their own digits again; each level sorts
        // some varying bits, so the recursion is bounded by the key width,
        // and a group's passes stay within its own cache footprint.
        _ => adaptive64::<T, C>(group, scratch),
    }
}

macro_rules! exchange {
    ($keys:ident; $(($i:literal, $j:literal))*) => {
        $(
            let (a, b) = ($keys[$i], $keys[$j]);
            $keys[$i] = a.min(b);
            $keys[$j] = a.max(b);
        )*
    };
}

#[inline(always)]
fn network9(keys: &mut [u64; 9]) {
    exchange!(keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (0, 2) (1, 3) (4, 6) (5, 7) (1, 2) (5, 6) (0, 4) (3, 7) (2, 6)
        (1, 5) (0, 8) (2, 4) (3, 5) (1, 2) (3, 4) (5, 6) (4, 8) (3, 5) (2, 4) (6, 8) (1, 2) (3, 4)
        (5, 6) (7, 8)
    );
}

#[inline(always)]
fn network10(keys: &mut [u64; 10]) {
    exchange!(keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (8, 9) (0, 2) (1, 3) (4, 6) (5, 7) (1, 2) (5, 6) (0, 4) (3, 7)
        (2, 6) (1, 5) (0, 8) (2, 4) (3, 5) (1, 2) (3, 4) (5, 6) (4, 8) (1, 9) (2, 4) (6, 8) (5, 9)
        (3, 5) (7, 9) (1, 2) (3, 4) (5, 6) (7, 8)
    );
}

#[inline(always)]
fn network11(keys: &mut [u64; 11]) {
    exchange!(keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (8, 9) (0, 2) (1, 3) (4, 6) (5, 7) (8, 10) (1, 2) (5, 6) (0, 4)
        (3, 7) (9, 10) (2, 6) (1, 5) (9, 10) (0, 8) (2, 4) (3, 5) (1, 2) (3, 4) (5, 6) (4, 8) (2,
        10) (1, 9) (6, 10) (2, 4) (5, 9) (6, 8) (3, 5) (7, 9) (1, 2) (3, 4) (5, 6) (7, 8) (9, 10)
    );
}

#[inline(always)]
fn network12(keys: &mut [u64; 12]) {
    exchange!(keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (8, 9) (10, 11) (0, 2) (1, 3) (4, 6) (5, 7) (8, 10) (9, 11) (1,
        2) (5, 6) (0, 4) (3, 7) (9, 10) (2, 6) (1, 5) (9, 10) (0, 8) (2, 4) (3, 5) (1, 2) (3, 4) (5,
        6) (4, 8) (2, 10) (1, 9) (3, 11) (6, 10) (2, 4) (5, 9) (7, 11) (6, 8) (3, 5) (7, 9) (1, 2)
        (3, 4) (5, 6) (7, 8) (9, 10)
    );
}

#[inline(always)]
fn network13(keys: &mut [u64; 13]) {
    exchange!(keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (8, 9) (10, 11) (0, 2) (1, 3) (4, 6) (5, 7) (8, 10) (9, 11) (1,
        2) (5, 6) (0, 4) (3, 7) (9, 10) (8, 12) (2, 6) (1, 5) (10, 12) (0, 8) (2, 4) (3, 5) (9, 10)
        (11, 12) (1, 2) (3, 4) (5, 6) (4, 12) (2, 10) (1, 9) (3, 11) (4, 8) (6, 10) (5, 9) (7, 11)
        (2, 4) (6, 8) (10, 12) (3, 5) (7, 9) (1, 2) (3, 4) (5, 6) (7, 8) (9, 10) (11, 12)
    );
}

#[inline(always)]
fn network14(keys: &mut [u64; 14]) {
    exchange!(keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (8, 9) (10, 11) (12, 13) (0, 2) (1, 3) (4, 6) (5, 7) (8, 10) (9,
        11) (1, 2) (5, 6) (0, 4) (3, 7) (9, 10) (8, 12) (2, 6) (1, 5) (10, 12) (9, 13) (0, 8) (2, 4)
        (3, 5) (11, 13) (9, 10) (1, 2) (3, 4) (5, 6) (11, 12) (4, 12) (2, 10) (1, 9) (5, 13) (3, 11)
        (4, 8) (6, 10) (5, 9) (7, 11) (2, 4) (6, 8) (10, 12) (3, 5) (7, 9) (11, 13) (1, 2) (3, 4)
        (5, 6) (7, 8) (9, 10) (11, 12)
    );
}

#[inline(always)]
fn network15(keys: &mut [u64; 15]) {
    exchange!(keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (8, 9) (10, 11) (12, 13) (0, 2) (1, 3) (4, 6) (5, 7) (8, 10) (9,
        11) (12, 14) (1, 2) (5, 6) (0, 4) (3, 7) (9, 10) (13, 14) (8, 12) (2, 6) (1, 5) (10, 14) (9,
        13) (0, 8) (2, 4) (3, 5) (10, 12) (11, 13) (1, 2) (3, 4) (5, 6) (9, 10) (11, 12) (13, 14)
        (4, 12) (2, 10) (6, 14) (1, 9) (5, 13) (3, 11) (4, 8) (6, 10) (5, 9) (7, 11) (2, 4) (6, 8)
        (10, 12) (3, 5) (7, 9) (11, 13) (1, 2) (3, 4) (5, 6) (7, 8) (9, 10) (11, 12) (13, 14)
    );
}

macro_rules! network16_exchanges {
    ($apply:ident; $keys:ident) => {
        $apply!($keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (8, 9) (10, 11) (12, 13) (14, 15) (0, 2) (1, 3) (4, 6) (5, 7)
        (8, 10) (9, 11) (12, 14) (13, 15) (1, 2) (5, 6) (9, 10) (13, 14) (0, 4) (1, 5) (2, 6) (3, 7)
        (8, 12) (9, 13) (10, 14) (11, 15) (2, 4) (3, 5) (10, 12) (11, 13) (1, 2) (3, 4) (5, 6) (9,
        10) (11, 12) (13, 14) (0, 8) (1, 9) (2, 10) (3, 11) (4, 12) (5, 13) (6, 14) (7, 15) (4, 8)
        (5, 9) (6, 10) (7, 11) (2, 4) (3, 5) (6, 8) (7, 9) (10, 12) (11, 13) (1, 2) (3, 4) (5, 6)
        (7, 8) (9, 10) (11, 12) (13, 14)
        );
    };
}

#[inline(always)]
fn network16(keys: &mut [u64; 16]) {
    network16_exchanges!(exchange; keys);
}

/// Comparison sorting for short slices: an ordered probe, then branch-free
/// sorting whose work depends only on the length. Up to 32 keys use one
/// comparator network. Longer inputs sort columns of sixteen keys through a
/// vertical network over whole rows, which the compiler maps to lane-wise
/// vector minimum and maximum, and merge the sorted runs pairwise. Values are
/// rebuilt from their total-order keys with every original bit intact.
#[inline(never)]
pub(crate) fn small_sort<T: Sealed>(values: &mut [T]) {
    let len = values.len();
    if len <= TINY {
        tiny(values);
        return;
    }
    match len {
        // Lengths through 16 probe and sort through the exact network, as
        // the tiny path does; longer short inputs probe through
        // `short_ordered` before their network. Inputs beyond 32 elements
        // arrive from `prepare` with their order already settled.
        9 => tiny_keys::<T, 9, true, true>(values, network9),
        10 => tiny_keys::<T, 10, true, true>(values, network10),
        11 => tiny_keys::<T, 11, true, true>(values, network11),
        12 => tiny_keys::<T, 12, true, true>(values, network12),
        13 => tiny_keys::<T, 13, true, true>(values, network13),
        14 => tiny_keys::<T, 14, true, true>(values, network14),
        15 => tiny_keys::<T, 15, true, true>(values, network15),
        16 => tiny_keys::<T, 16, true, true>(values, network16),
        17..=32 => {
            if !short_ordered(values) {
                network_run_sort(values);
            }
        }
        33..=NETWORK_TAIL => network_sort_with_tail(values),
        41..=64 => column_sort::<T, 64>(values),
        65..=INSERTION_TAIL => column_sort_with_tail(values),
        73..=128 => column_sort::<T, 128>(values),
        _ => column_sort::<T, 256>(values),
    }
}

/// Longest input sorted entirely by the inlined tiny path.
pub(crate) const TINY: usize = 8;
/// Longest input that the entry points send straight to the comparison
/// sort, before any preparation.
pub(crate) const SHORT: usize = 32;
/// Longest input that sorts 32 keys through the network and merges the
/// rest instead of entering the column sort.
pub(crate) const NETWORK_TAIL: usize = 40;
/// Longest input that sorts 64 keys and inserts the rest one at a time. Two
/// more merge levels for a third column block cost more up to this length.
const INSERTION_TAIL: usize = 72;

/// Sort at most eight values with the exact network for their count. The
/// keys are loaded once; an order probe without data-dependent branches
/// returns ordered input (reversing descending input), and the network runs
/// on the keys otherwise. Inlined into the public entry points so the
/// shortest inputs pay no call.
#[inline(always)]
pub(crate) fn tiny<T: Sealed>(values: &mut [T]) {
    match values.len() {
        0 | 1 => {}
        // One exchange costs less than a probe whose outcome is random.
        2 => exchange_values(values, 0, 1),
        // Three and four keys sort faster through their networks than any
        // probe of their order could decide.
        3 => tiny_keys::<T, 3, false, false>(values, network3),
        4 => tiny_keys::<T, 4, false, false>(values, network4),
        5 => tiny_keys::<T, 5, true, true>(values, network5),
        6 => tiny_keys::<T, 6, true, true>(values, network6),
        7 => tiny_keys::<T, 7, true, true>(values, network7),
        _ => tiny_keys::<T, 8, true, true>(values, network8),
    }
}

/// The optional probe checks ascending order, and descending order too when
/// the network is long enough for a reversal to pay for the extra
/// comparisons. The slice becomes an array of its exact length first, so
/// that the loads, the probe and the stores unroll without length checks.
#[inline(always)]
fn tiny_keys<T: Sealed, const N: usize, const PROBE: bool, const REVERSE: bool>(
    values: &mut [T],
    network: impl FnOnce(&mut [u64; N]),
) {
    let Ok(values) = <&mut [T; N]>::try_from(values) else {
        unreachable!("tiny network length");
    };
    let mut keys = [u64::MAX; N];
    for (slot, value) in keys.iter_mut().zip(values.iter()) {
        *slot = value.key();
    }
    // Reduce the order violations to one word before branching on it. The
    // compiler would otherwise rewrite the reduction into one data-dependent
    // branch per pair, which mispredicts on random input; the barrier keeps
    // the selects and leaves one predictable branch. It changes no result.
    if PROBE {
        let mut unordered = 0;
        let mut not_reversed = 0;
        for pair in keys.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            unordered |= u64::from(a > b);
            if REVERSE {
                not_reversed |= u64::from(a < b);
            }
        }
        if std::hint::black_box(unordered) == 0 {
            return;
        }
        if REVERSE && std::hint::black_box(not_reversed) == 0 {
            values.reverse();
            return;
        }
    }
    network(&mut keys);
    for (value, &key) in values.iter_mut().zip(keys.iter()) {
        *value = T::from_key(key);
    }
}

/// Thirty-two keys through the network and the rest, padded with the
/// largest key, through the eight-key network, then the odd-even merge
/// network over both runs. Every step is a compare-exchange: a binary
/// insertion per tail key would mispredict at every step, and a merge loop
/// would wait on its own cursor for every key.
#[inline(never)]
fn network_sort_with_tail<T: Sealed>(values: &mut [T]) {
    let len = values.len();
    assert!((33..=NETWORK_TAIL).contains(&len));
    let mut keys = [u64::MAX; NETWORK_TAIL];
    for (slot, value) in keys.iter_mut().zip(values.iter()) {
        *slot = value.key();
    }
    let (head, tail) = keys.split_at_mut(32);
    network32(head.try_into().unwrap());
    network8(tail.try_into().unwrap());
    merge32_8(&mut keys);
    for (value, &key) in values.iter_mut().zip(keys.iter()) {
        *value = T::from_key(key);
    }
}

/// Batcher's odd-even merge of a sorted run of 32 keys with a sorted run of
/// 8 keys: 85 comparators in 6 layers.
#[inline(always)]
fn merge32_8(keys: &mut [u64; NETWORK_TAIL]) {
    exchange!(keys;
        (0, 32) (16, 32) (8, 16) (24, 32) (4, 36) (20, 36) (12, 20) (28, 36) (4, 8) (12, 16) (20, 24)
        (28, 32) (2, 34) (18, 34) (10, 18) (26, 34) (6, 38) (22, 38) (14, 22) (30, 38) (6, 10) (14, 18)
        (22, 26) (30, 34) (2, 4) (6, 8) (10, 12) (14, 16) (18, 20) (22, 24) (26, 28) (30, 32) (34, 36)
        (1, 33) (17, 33) (9, 17) (25, 33) (5, 37) (21, 37) (13, 21) (29, 37) (5, 9) (13, 17) (21, 25)
        (29, 33) (3, 35) (19, 35) (11, 19) (27, 35) (7, 39) (23, 39) (15, 23) (31, 39) (7, 11) (15, 19)
        (23, 27) (31, 35) (3, 5) (7, 9) (11, 13) (15, 17) (19, 21) (23, 25) (27, 29) (31, 33) (35, 37)
        (1, 2) (3, 4) (5, 6) (7, 8) (9, 10) (11, 12) (13, 14) (15, 16) (17, 18) (19, 20) (21, 22)
        (23, 24) (25, 26) (27, 28) (29, 30) (31, 32) (33, 34) (35, 36) (37, 38)
    );
}

/// Check ascending and descending order in one pass of eight-element
/// chunks. Ordered input returns without further work; descending input is
/// reversed. The chunked form keeps the comparisons predictable on ordered
/// input, where this probe decides the whole sort; a word reduction with a
/// barrier, as in the tiny probe, cost twice as much here.
#[inline(always)]
fn short_ordered<T: Sealed>(values: &mut [T]) -> bool {
    let first = values[0].key();
    if values[values.len() - 1].key() == first
        && values
            .iter()
            .fold(0, |different, value| different | (value.key() ^ first))
            == 0
    {
        // Constant input: one reduction instead of two comparisons per pair.
        return true;
    }
    let mut ascending = true;
    let mut descending = true;
    let mut previous = first;
    let mut chunks = values.chunks_exact(8);
    for chunk in &mut chunks {
        let first = chunk[0].key();
        let mut up = previous <= first;
        let mut down = previous >= first;
        for pair in chunk.windows(2) {
            let (a, b) = (pair[0].key(), pair[1].key());
            up &= a <= b;
            down &= a >= b;
        }
        ascending &= up;
        descending &= down;
        previous = chunk[7].key();
    }
    for value in chunks.remainder() {
        let key = value.key();
        ascending &= previous <= key;
        descending &= previous >= key;
        previous = key;
    }
    if ascending {
        return true;
    }
    if descending {
        values.reverse();
        return true;
    }
    false
}

/// At most 32 values through the 32-key network. Padding keys are the
/// largest key, which never precedes a real key; equal keys are identical
/// values, so the first len lanes are exactly the sorted input even when a
/// real key equals the padding.
#[inline(never)]
fn network_run_sort<T: Sealed>(values: &mut [T]) {
    debug_assert!(values.len() <= 32);
    let mut keys = [u64::MAX; 32];
    for (slot, value) in keys.iter_mut().zip(values.iter()) {
        *slot = value.key();
    }
    network32(&mut keys);
    for (value, &key) in values.iter_mut().zip(keys.iter()) {
        *value = T::from_key(key);
    }
}

/// A row of key lanes: four 32-bit keys or two 64-bit keys in sixteen
/// aligned bytes. The vertical networks exchange whole rows, so every lane
/// is sorted independently by the same comparator sequence.
trait Row: Copy {
    const LANES: usize;
    /// Every lane holds the largest key.
    const PADDING: Self;
    fn lane_min(a: Self, b: Self) -> Self;
    fn lane_max(a: Self, b: Self) -> Self;
    fn lane(self, index: usize) -> u64;
    fn set_lane(&mut self, index: usize, key: u64);
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Row4([u32; 4]);

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Row2([u64; 2]);

impl Row for Row4 {
    const LANES: usize = 4;
    const PADDING: Self = Self([u32::MAX; 4]);
    #[inline(always)]
    fn lane_min(a: Self, b: Self) -> Self {
        Self(std::array::from_fn(|lane| a.0[lane].min(b.0[lane])))
    }
    #[inline(always)]
    fn lane_max(a: Self, b: Self) -> Self {
        Self(std::array::from_fn(|lane| a.0[lane].max(b.0[lane])))
    }
    #[inline(always)]
    fn lane(self, index: usize) -> u64 {
        u64::from(self.0[index])
    }
    #[inline(always)]
    fn set_lane(&mut self, index: usize, key: u64) {
        debug_assert!(key <= u64::from(u32::MAX));
        self.0[index] = key as u32;
    }
}

impl Row for Row2 {
    const LANES: usize = 2;
    const PADDING: Self = Self([u64::MAX; 2]);
    #[inline(always)]
    fn lane_min(a: Self, b: Self) -> Self {
        Self([a.0[0].min(b.0[0]), a.0[1].min(b.0[1])])
    }
    #[inline(always)]
    fn lane_max(a: Self, b: Self) -> Self {
        Self([a.0[0].max(b.0[0]), a.0[1].max(b.0[1])])
    }
    #[inline(always)]
    fn lane(self, index: usize) -> u64 {
        self.0[index]
    }
    #[inline(always)]
    fn set_lane(&mut self, index: usize, key: u64) {
        self.0[index] = key;
    }
}

macro_rules! exchange_rows {
    ($rows:ident; $(($i:literal, $j:literal))*) => {
        $(
            let (a, b) = ($rows[$i], $rows[$j]);
            $rows[$i] = Row::lane_min(a, b);
            $rows[$j] = Row::lane_max(a, b);
        )*
    };
}

/// Sort every lane of sixteen rows with the 16-key network applied to whole
/// rows. Kept out of line: one vectorized copy per row type.
#[inline(never)]
fn column_sort16<R: Row>(rows: &mut [R; 16]) {
    network16_exchanges!(exchange_rows; rows);
}

/// Sorted keys for up to SLOTS values: column blocks of sixteen rows, then a
/// merge tree over runs of sixteen. Padding keys are the largest key and
/// sort to the end of every run and merge; whole padding blocks and merges
/// with all-padding runs are skipped.
#[inline(never)]
fn column_keys<T: Sealed, R: Row, const BLOCKS: usize, const SLOTS: usize>(
    values: &[T],
    keys: &mut [u64; SLOTS],
) {
    let len = values.len();
    assert!(SLOTS == BLOCKS * 16 * R::LANES && len <= SLOTS && SLOTS >= 64);
    let mut blocks = [[R::PADDING; 16]; BLOCKS];
    for (index, value) in values.iter().enumerate() {
        let run = index / 16;
        blocks[run / R::LANES][index % 16].set_lane(run % R::LANES, value.key());
    }
    for (number, block) in blocks.iter_mut().enumerate() {
        if number * 16 * R::LANES < len {
            column_sort16(block);
        }
    }
    for (number, block) in blocks.iter().enumerate() {
        for (row, lanes) in block.iter().enumerate() {
            for lane in 0..R::LANES {
                keys[(number * R::LANES + lane) * 16 + row] = lanes.lane(lane);
            }
        }
    }
    merge_tree(keys, len);
}

/// Merge runs of sixteen sorted keys pairwise until one run of SLOTS keys
/// remains in `keys`. Two independent merges proceed together so that four
/// dependency chains overlap; the last level of 64 or 128 keys uses a merge
/// network instead, and 256 keys a single two-ended merge.
#[inline]
fn merge_tree<const SLOTS: usize>(keys: &mut [u64; SLOTS], len: usize) {
    let mut other = [u64::MAX; SLOTS];
    let mut in_other = false;
    let (mut source, mut dest) = (&mut *keys, &mut other);
    let mut width = 16;
    while 2 * width < SLOTS {
        let span = 2 * width;
        let mut base = 0;
        while base < SLOTS {
            let next = base + span;
            if base + width >= len {
                // The second run (and everything after it) is all padding.
                dest[base..].copy_from_slice(&source[base..]);
                break;
            }
            if next + width < len && next + span <= SLOTS {
                let (left_in, right_in) = source[base..next + span].split_at(span);
                let (left_out, right_out) = dest[base..next + span].split_at_mut(span);
                merge_two_pairs(left_in, right_in, left_out, right_out);
                base = next + span;
            } else {
                merge_keys(&source[base..next], &mut dest[base..next]);
                base = next;
            }
        }
        std::mem::swap(&mut source, &mut dest);
        in_other = !in_other;
        width *= 2;
    }
    match SLOTS {
        64 => merge_network64(source.as_mut_slice().try_into().unwrap()),
        128 => merge_network128(source.as_mut_slice().try_into().unwrap()),
        _ => {
            merge_keys(source, dest);
            in_other = !in_other;
        }
    }
    if in_other {
        keys.copy_from_slice(&other);
    }
}

/// Column-sort through SLOTS slots and write the values back.
#[inline(never)]
fn column_sort<T: Sealed, const SLOTS: usize>(values: &mut [T]) {
    let mut keys = [u64::MAX; SLOTS];
    column_keys_for(values, &mut keys);
    for (value, &key) in values.iter_mut().zip(keys.iter()) {
        *value = T::from_key(key);
    }
}

/// Dispatch on the key width: 64-bit keys use two lanes per row, narrower
/// keys four. The unused branch is a compile-time constant for each type.
#[inline(always)]
fn column_keys_for<T: Sealed, const SLOTS: usize>(values: &[T], keys: &mut [u64; SLOTS]) {
    let wide = std::mem::size_of::<T>() == 8;
    match (SLOTS, wide) {
        (64, true) => column_keys::<T, Row2, 2, SLOTS>(values, keys),
        (64, false) => column_keys::<T, Row4, 1, SLOTS>(values, keys),
        (128, true) => column_keys::<T, Row2, 4, SLOTS>(values, keys),
        (128, false) => column_keys::<T, Row4, 2, SLOTS>(values, keys),
        (256, true) => column_keys::<T, Row2, 8, SLOTS>(values, keys),
        (256, false) => column_keys::<T, Row4, 4, SLOTS>(values, keys),
        _ => unreachable!("unsupported column sort size"),
    }
}

/// Sixty-four keys through the column sort, then the remaining keys are
/// inserted one at a time with a binary search and one contiguous move.
#[inline(never)]
fn column_sort_with_tail<T: Sealed>(values: &mut [T]) {
    let len = values.len();
    assert!((64..=INSERTION_TAIL).contains(&len));
    let mut keys = [u64::MAX; 64];
    column_keys_for(&values[..64], &mut keys);
    let mut sorted = [u64::MAX; INSERTION_TAIL];
    sorted[..64].copy_from_slice(&keys);
    for (count, value) in (64..).zip(&values[64..]) {
        let key = value.key();
        let position = sorted[..count].partition_point(|&k| k <= key);
        sorted.copy_within(position..count, position + 1);
        sorted[position] = key;
    }
    for (value, &key) in values.iter_mut().zip(sorted.iter()) {
        *value = T::from_key(key);
    }
}

/// Merge the two sorted halves of `input` into `output` from both ends at
/// once: the front chain emits the smallest half of the keys and the back
/// chain the largest half. Each chain can consume a whole run only on its
/// final step, so neither reads past a run, and the two halves of the key
/// multiset partition it exactly.
#[inline]
fn merge_keys(input: &[u64], output: &mut [u64]) {
    let half = input.len() / 2;
    debug_assert!(input.len() == 2 * half && output.len() == input.len() && half > 0);
    let (a, b) = input.split_at(half);
    let (mut i, mut j) = (0, 0);
    let (mut ri, mut rj) = (half - 1, half - 1);
    let mut back = output.len();
    for front in 0..half {
        let (x, y) = (a[i], b[j]);
        let take_a = x <= y;
        output[front] = if take_a { x } else { y };
        i += usize::from(take_a);
        j += usize::from(!take_a);
        let (x, y) = (a[ri], b[rj]);
        let take_a = x > y;
        back -= 1;
        output[back] = if take_a { x } else { y };
        ri = ri.wrapping_sub(usize::from(take_a));
        rj = rj.wrapping_sub(usize::from(!take_a));
    }
}

/// Two independent merges of equal shape in one loop, as `merge_keys`, so
/// that their four dependency chains overlap.
#[inline]
fn merge_two_pairs(left: &[u64], right: &[u64], left_out: &mut [u64], right_out: &mut [u64]) {
    let half = left.len() / 2;
    debug_assert!(
        left.len() == 2 * half
            && right.len() == left.len()
            && left_out.len() == left.len()
            && right_out.len() == left.len()
            && half > 0
    );
    let (a0, b0) = left.split_at(half);
    let (a1, b1) = right.split_at(half);
    let (mut i0, mut j0, mut i1, mut j1) = (0, 0, 0, 0);
    let (mut ri0, mut rj0, mut ri1, mut rj1) = (half - 1, half - 1, half - 1, half - 1);
    let mut back = 2 * half;
    for front in 0..half {
        let (x, y) = (a0[i0], b0[j0]);
        let take_a = x <= y;
        left_out[front] = if take_a { x } else { y };
        i0 += usize::from(take_a);
        j0 += usize::from(!take_a);
        let (x, y) = (a1[i1], b1[j1]);
        let take_a = x <= y;
        right_out[front] = if take_a { x } else { y };
        i1 += usize::from(take_a);
        j1 += usize::from(!take_a);
        back -= 1;
        let (x, y) = (a0[ri0], b0[rj0]);
        let take_a = x > y;
        left_out[back] = if take_a { x } else { y };
        ri0 = ri0.wrapping_sub(usize::from(take_a));
        rj0 = rj0.wrapping_sub(usize::from(!take_a));
        let (x, y) = (a1[ri1], b1[rj1]);
        let take_a = x > y;
        right_out[back] = if take_a { x } else { y };
        ri1 = ri1.wrapping_sub(usize::from(take_a));
        rj1 = rj1.wrapping_sub(usize::from(!take_a));
    }
}

// Batcher odd-even merge networks for two sorted halves: 161 and 385
// exchanges, applied in place. They replace the dependent merge cursors on
// the last level, where only one merge remains.
#[inline(never)]
fn merge_network64(keys: &mut [u64; 64]) {
    exchange!(keys;
        (0, 32) (16, 48) (8, 40) (24, 56) (4, 36) (20, 52) (12, 44) (28, 60) (2, 34) (18, 50) (10,
        42) (26, 58) (6, 38) (22, 54) (14, 46) (30, 62) (1, 33) (17, 49) (9, 41) (25, 57) (5, 37)
        (21, 53) (13, 45) (29, 61) (3, 35) (19, 51) (11, 43) (27, 59) (7, 39) (23, 55) (15, 47) (31,
        63) (16, 32) (24, 40) (20, 36) (28, 44) (18, 34) (26, 42) (22, 38) (30, 46) (17, 33) (25,
        41) (21, 37) (29, 45) (19, 35) (27, 43) (23, 39) (31, 47) (8, 16) (24, 32) (40, 48) (12, 20)
        (28, 36) (44, 52) (10, 18) (26, 34) (42, 50) (14, 22) (30, 38) (46, 54) (9, 17) (25, 33)
        (41, 49) (13, 21) (29, 37) (45, 53) (11, 19) (27, 35) (43, 51) (15, 23) (31, 39) (47, 55)
        (4, 8) (12, 16) (20, 24) (28, 32) (36, 40) (44, 48) (52, 56) (6, 10) (14, 18) (22, 26) (30,
        34) (38, 42) (46, 50) (54, 58) (5, 9) (13, 17) (21, 25) (29, 33) (37, 41) (45, 49) (53, 57)
        (7, 11) (15, 19) (23, 27) (31, 35) (39, 43) (47, 51) (55, 59) (2, 4) (6, 8) (10, 12) (14,
        16) (18, 20) (22, 24) (26, 28) (30, 32) (34, 36) (38, 40) (42, 44) (46, 48) (50, 52) (54,
        56) (58, 60) (3, 5) (7, 9) (11, 13) (15, 17) (19, 21) (23, 25) (27, 29) (31, 33) (35, 37)
        (39, 41) (43, 45) (47, 49) (51, 53) (55, 57) (59, 61) (1, 2) (3, 4) (5, 6) (7, 8) (9, 10)
        (11, 12) (13, 14) (15, 16) (17, 18) (19, 20) (21, 22) (23, 24) (25, 26) (27, 28) (29, 30)
        (31, 32) (33, 34) (35, 36) (37, 38) (39, 40) (41, 42) (43, 44) (45, 46) (47, 48) (49, 50)
        (51, 52) (53, 54) (55, 56) (57, 58) (59, 60) (61, 62)
    );
}

#[inline(never)]
fn merge_network128(keys: &mut [u64; 128]) {
    exchange!(keys;
        (0, 64) (32, 96) (16, 80) (48, 112) (8, 72) (40, 104) (24, 88) (56, 120) (4, 68) (36, 100)
        (20, 84) (52, 116) (12, 76) (44, 108) (28, 92) (60, 124) (2, 66) (34, 98) (18, 82) (50, 114)
        (10, 74) (42, 106) (26, 90) (58, 122) (6, 70) (38, 102) (22, 86) (54, 118) (14, 78) (46,
        110) (30, 94) (62, 126) (1, 65) (33, 97) (17, 81) (49, 113) (9, 73) (41, 105) (25, 89) (57,
        121) (5, 69) (37, 101) (21, 85) (53, 117) (13, 77) (45, 109) (29, 93) (61, 125) (3, 67) (35,
        99) (19, 83) (51, 115) (11, 75) (43, 107) (27, 91) (59, 123) (7, 71) (39, 103) (23, 87) (55,
        119) (15, 79) (47, 111) (31, 95) (63, 127) (32, 64) (48, 80) (40, 72) (56, 88) (36, 68) (52,
        84) (44, 76) (60, 92) (34, 66) (50, 82) (42, 74) (58, 90) (38, 70) (54, 86) (46, 78) (62,
        94) (33, 65) (49, 81) (41, 73) (57, 89) (37, 69) (53, 85) (45, 77) (61, 93) (35, 67) (51,
        83) (43, 75) (59, 91) (39, 71) (55, 87) (47, 79) (63, 95) (16, 32) (48, 64) (80, 96) (24,
        40) (56, 72) (88, 104) (20, 36) (52, 68) (84, 100) (28, 44) (60, 76) (92, 108) (18, 34) (50,
        66) (82, 98) (26, 42) (58, 74) (90, 106) (22, 38) (54, 70) (86, 102) (30, 46) (62, 78) (94,
        110) (17, 33) (49, 65) (81, 97) (25, 41) (57, 73) (89, 105) (21, 37) (53, 69) (85, 101) (29,
        45) (61, 77) (93, 109) (19, 35) (51, 67) (83, 99) (27, 43) (59, 75) (91, 107) (23, 39) (55,
        71) (87, 103) (31, 47) (63, 79) (95, 111) (8, 16) (24, 32) (40, 48) (56, 64) (72, 80) (88,
        96) (104, 112) (12, 20) (28, 36) (44, 52) (60, 68) (76, 84) (92, 100) (108, 116) (10, 18)
        (26, 34) (42, 50) (58, 66) (74, 82) (90, 98) (106, 114) (14, 22) (30, 38) (46, 54) (62, 70)
        (78, 86) (94, 102) (110, 118) (9, 17) (25, 33) (41, 49) (57, 65) (73, 81) (89, 97) (105,
        113) (13, 21) (29, 37) (45, 53) (61, 69) (77, 85) (93, 101) (109, 117) (11, 19) (27, 35)
        (43, 51) (59, 67) (75, 83) (91, 99) (107, 115) (15, 23) (31, 39) (47, 55) (63, 71) (79, 87)
        (95, 103) (111, 119) (4, 8) (12, 16) (20, 24) (28, 32) (36, 40) (44, 48) (52, 56) (60, 64)
        (68, 72) (76, 80) (84, 88) (92, 96) (100, 104) (108, 112) (116, 120) (6, 10) (14, 18) (22,
        26) (30, 34) (38, 42) (46, 50) (54, 58) (62, 66) (70, 74) (78, 82) (86, 90) (94, 98) (102,
        106) (110, 114) (118, 122) (5, 9) (13, 17) (21, 25) (29, 33) (37, 41) (45, 49) (53, 57) (61,
        65) (69, 73) (77, 81) (85, 89) (93, 97) (101, 105) (109, 113) (117, 121) (7, 11) (15, 19)
        (23, 27) (31, 35) (39, 43) (47, 51) (55, 59) (63, 67) (71, 75) (79, 83) (87, 91) (95, 99)
        (103, 107) (111, 115) (119, 123) (2, 4) (6, 8) (10, 12) (14, 16) (18, 20) (22, 24) (26, 28)
        (30, 32) (34, 36) (38, 40) (42, 44) (46, 48) (50, 52) (54, 56) (58, 60) (62, 64) (66, 68)
        (70, 72) (74, 76) (78, 80) (82, 84) (86, 88) (90, 92) (94, 96) (98, 100) (102, 104) (106,
        108) (110, 112) (114, 116) (118, 120) (122, 124) (3, 5) (7, 9) (11, 13) (15, 17) (19, 21)
        (23, 25) (27, 29) (31, 33) (35, 37) (39, 41) (43, 45) (47, 49) (51, 53) (55, 57) (59, 61)
        (63, 65) (67, 69) (71, 73) (75, 77) (79, 81) (83, 85) (87, 89) (91, 93) (95, 97) (99, 101)
        (103, 105) (107, 109) (111, 113) (115, 117) (119, 121) (123, 125) (1, 2) (3, 4) (5, 6) (7,
        8) (9, 10) (11, 12) (13, 14) (15, 16) (17, 18) (19, 20) (21, 22) (23, 24) (25, 26) (27, 28)
        (29, 30) (31, 32) (33, 34) (35, 36) (37, 38) (39, 40) (41, 42) (43, 44) (45, 46) (47, 48)
        (49, 50) (51, 52) (53, 54) (55, 56) (57, 58) (59, 60) (61, 62) (63, 64) (65, 66) (67, 68)
        (69, 70) (71, 72) (73, 74) (75, 76) (77, 78) (79, 80) (81, 82) (83, 84) (85, 86) (87, 88)
        (89, 90) (91, 92) (93, 94) (95, 96) (97, 98) (99, 100) (101, 102) (103, 104) (105, 106)
        (107, 108) (109, 110) (111, 112) (113, 114) (115, 116) (117, 118) (119, 120) (121, 122)
        (123, 124) (125, 126)
    );
}

/// Sort at most N values through a fixed comparator network on their keys.
/// Unused lanes hold the largest key, which never precedes a real key; equal
/// keys are identical values, so the first len lanes are exactly the sorted
/// input even when a real key equals the padding.
#[inline]
fn network_sort<T: Sealed, const N: usize>(group: &mut [T], network: fn(&mut [u64; N])) {
    debug_assert!(group.len() <= N);
    let mut keys = [u64::MAX; N];
    for (slot, value) in keys.iter_mut().zip(group.iter()) {
        *slot = value.key();
    }
    network(&mut keys);
    for (value, &key) in group.iter_mut().zip(keys.iter()) {
        *value = T::from_key(key);
    }
}

// Branch-free comparator networks. The 3, 5, 6 and 7-key networks are the
// known minimal ones; the others are Batcher odd-even merges, with the
// comparators touching padded lanes removed for counts between powers of
// two: 3, 5, 9, 12, 16, 19, 28, 32, 38, 42, 48, 53, 59, 63 and 191
// exchanges. Two keys exchange directly.
#[inline(always)]
fn network3(keys: &mut [u64; 3]) {
    exchange!(keys;
        (0, 2) (0, 1) (1, 2)
    );
}

#[inline(always)]
fn network4(keys: &mut [u64; 4]) {
    exchange!(keys; (0, 1) (2, 3) (0, 2) (1, 3) (1, 2));
}

#[inline(always)]
fn network5(keys: &mut [u64; 5]) {
    exchange!(keys;
        (0, 3) (1, 4) (0, 2) (1, 3) (0, 1) (2, 4) (1, 2) (3, 4) (2, 3)
    );
}

#[inline(always)]
fn network6(keys: &mut [u64; 6]) {
    exchange!(keys;
        (0, 5) (1, 3) (2, 4) (1, 2) (3, 4) (0, 3) (2, 5) (0, 1) (2, 3) (4, 5) (1, 2) (3, 4)
    );
}

#[inline(always)]
fn network7(keys: &mut [u64; 7]) {
    exchange!(keys;
        (0, 6) (2, 3) (4, 5) (0, 2) (1, 4) (3, 6) (0, 1) (2, 5) (3, 4) (1, 2) (4, 6) (2, 3) (4, 5)
        (1, 2) (3, 4) (5, 6)
    );
}

#[inline(always)]
fn network8(keys: &mut [u64; 8]) {
    exchange!(keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (0, 2) (1, 3) (4, 6) (5, 7) (1, 2) (5, 6) (0, 4) (1, 5) (2, 6)
        (3, 7) (2, 4) (3, 5) (1, 2) (3, 4) (5, 6)
    );
}

#[inline(always)]
fn network32(keys: &mut [u64; 32]) {
    exchange!(keys;
        (0, 1) (2, 3) (4, 5) (6, 7) (8, 9) (10, 11) (12, 13) (14, 15) (16, 17) (18, 19) (20, 21)
        (22, 23) (24, 25) (26, 27) (28, 29) (30, 31) (0, 2) (1, 3) (4, 6) (5, 7) (8, 10) (9, 11)
        (12, 14) (13, 15) (16, 18) (17, 19) (20, 22) (21, 23) (24, 26) (25, 27) (28, 30) (29, 31)
        (1, 2) (5, 6) (9, 10) (13, 14) (17, 18) (21, 22) (25, 26) (29, 30) (0, 4) (1, 5) (2, 6) (3,
        7) (8, 12) (9, 13) (10, 14) (11, 15) (16, 20) (17, 21) (18, 22) (19, 23) (24, 28) (25, 29)
        (26, 30) (27, 31) (2, 4) (3, 5) (10, 12) (11, 13) (18, 20) (19, 21) (26, 28) (27, 29) (1, 2)
        (3, 4) (5, 6) (9, 10) (11, 12) (13, 14) (17, 18) (19, 20) (21, 22) (25, 26) (27, 28) (29,
        30) (0, 8) (1, 9) (2, 10) (3, 11) (4, 12) (5, 13) (6, 14) (7, 15) (16, 24) (17, 25) (18, 26)
        (19, 27) (20, 28) (21, 29) (22, 30) (23, 31) (4, 8) (5, 9) (6, 10) (7, 11) (20, 24) (21, 25)
        (22, 26) (23, 27) (2, 4) (3, 5) (6, 8) (7, 9) (10, 12) (11, 13) (18, 20) (19, 21) (22, 24)
        (23, 25) (26, 28) (27, 29) (1, 2) (3, 4) (5, 6) (7, 8) (9, 10) (11, 12) (13, 14) (17, 18)
        (19, 20) (21, 22) (23, 24) (25, 26) (27, 28) (29, 30) (0, 16) (1, 17) (2, 18) (3, 19) (4,
        20) (5, 21) (6, 22) (7, 23) (8, 24) (9, 25) (10, 26) (11, 27) (12, 28) (13, 29) (14, 30)
        (15, 31) (8, 16) (9, 17) (10, 18) (11, 19) (12, 20) (13, 21) (14, 22) (15, 23) (4, 8) (5, 9)
        (6, 10) (7, 11) (12, 16) (13, 17) (14, 18) (15, 19) (20, 24) (21, 25) (22, 26) (23, 27) (2,
        4) (3, 5) (6, 8) (7, 9) (10, 12) (11, 13) (14, 16) (15, 17) (18, 20) (19, 21) (22, 24) (23,
        25) (26, 28) (27, 29) (1, 2) (3, 4) (5, 6) (7, 8) (9, 10) (11, 12) (13, 14) (15, 16) (17,
        18) (19, 20) (21, 22) (23, 24) (25, 26) (27, 28) (29, 30)
    );
}

/// Choose the digits to sort: each window of `digit` bits starts at the
/// highest bit that still varies below the previous window, so a gap of
/// constant bits between a short high prefix and varying low bits costs no
/// pass and leaves no large collision groups. Returns false when the chosen
/// digits vary in fewer than `required` bits and the kernel should run.
fn dispatch_prefix64<T: Sealed, C: Counter, const R: usize>(
    values: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    extra: usize,
    variation: u64,
    required: u32,
) -> bool {
    let digit = R.trailing_zeros() as usize;
    let bits = (64 - variation.leading_zeros()) as usize;
    let length_bits = (usize::BITS - (values.len() - 1).leading_zeros()) as usize;
    let desired = length_bits + extra;
    let wanted = bits.div_ceil(digit).min(desired.div_ceil(digit)).min(4);
    let mask = (R - 1) as u64;
    // Most significant window first; the passes below run least significant
    // first, so the shifts are reversed before use. Windows with few varying
    // bits, such as a short high prefix, are followed by more windows until
    // the sorted bits can separate the keys into small groups.
    let mut selected = [0usize; 4];
    let mut passes = 0;
    let mut uncovered = variation;
    // A short high prefix over large groups sorts that prefix alone: the
    // groups then choose their own digits with passes that stay within
    // their cache footprint, while sorting enough low windows here would
    // stream the whole input several times.
    let top = 63 - variation.leading_zeros() as usize;
    let first = (variation >> (top + 1).saturating_sub(digit)) & mask;
    let span = (u64::BITS - (first >> first.trailing_zeros()).leading_zeros()) as usize;
    let large_groups = span <= 4 && first != variation && values.len() >> span >= LARGE_GROUP;
    while passes < 4
        && uncovered != 0
        && !(large_groups && passes == 1)
        && (passes < wanted
            || ((variation & !uncovered).count_ones() as usize) < length_bits + DENSE_GROUPS)
    {
        let top = 63 - uncovered.leading_zeros() as usize;
        let shift = (top + 1).saturating_sub(digit);
        selected[passes] = shift;
        passes += 1;
        uncovered &= !(mask << shift);
    }
    let sorted_mask = variation & !uncovered;
    if uncovered != 0 && !large_groups && sorted_mask.count_ones() < required {
        return false;
    }
    // Varying bits outside the sorted windows leave groups to finish.
    let unsorted = uncovered != 0;
    if passes == 1 {
        // Constant low bits of the remaining digit are skipped too, so a
        // short varying span can use sixteen buckets and their exact ends.
        let shift = selected[0];
        let varying = (variation >> shift) & mask;
        let low = varying.trailing_zeros() as usize;
        let span = (u64::BITS - (varying >> low).leading_zeros()) as usize;
        if span <= 4 {
            let mut ends = [C::ZERO; 16];
            prefix64::<T, C, 1, 16>(values, scratch, [shift + low], Some(&mut ends));
            if unsorted {
                finish_buckets::<T, C, 16>(values, scratch, &ends);
            }
        } else {
            let mut ends = [C::ZERO; R];
            prefix64::<T, C, 1, R>(values, scratch, [shift], Some(&mut ends));
            if unsorted {
                finish_buckets::<T, C, R>(values, scratch, &ends);
            }
        }
        return true;
    }
    match passes {
        2 => prefix64::<T, C, 2, R>(values, scratch, [selected[1], selected[0]], None),
        3 => prefix64::<T, C, 3, R>(
            values,
            scratch,
            [selected[2], selected[1], selected[0]],
            None,
        ),
        _ => prefix64::<T, C, 4, R>(
            values,
            scratch,
            [selected[3], selected[2], selected[1], selected[0]],
            None,
        ),
    }
    if unsorted {
        finish_scan::<T, C>(values, scratch, sorted_mask);
    }
    true
}

/// Group size from which the kernel's per-group passes beat extra windows.
const LARGE_GROUP: usize = 8192;
/// Another window is sorted while the sorted bits leave more than one key
/// per eight buckets on average: finishing that many small groups costs more
/// than one pass.
const DENSE_GROUPS: usize = 3;

/// Stable passes over the digits at `shifts`, least significant first.
/// `ends` receives the last pass's bucket ends, which are exact group
/// boundaries after a single pass.
fn prefix64<T: Sealed, C: Counter, const P: usize, const R: usize>(
    values: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    shifts: [usize; P],
    ends: Option<&mut [C; R]>,
) {
    assert!((1..=4).contains(&P) && R.is_power_of_two());
    let digit = R.trailing_zeros() as usize;
    assert!(shifts.iter().all(|&shift| shift + digit <= 64));
    assert!(scratch.len() >= values.len() && values.len() <= C::MAX);
    let mut counts = [[C::ZERO; R]; P];
    let digit_of = |key: u64, shift: usize| ((key >> shift) & (R - 1) as u64) as usize;
    // Two counting chains alternate elements, and a block of eight keys
    // sharing a digit updates each chain's counter once: sorted runs share
    // their high digits for long stretches and would otherwise wait on the
    // counter just stored for every element. Eight-key blocks keep the
    // batching branch predictable when runs share a digit only briefly.
    let mut second = [[C::ZERO; R]; P];
    let mut chunks = values.chunks_exact(8);
    for chunk in &mut chunks {
        let keys: [u64; 8] = std::array::from_fn(|lane| chunk[lane].key());
        for ((table, other), &shift) in counts.iter_mut().zip(second.iter_mut()).zip(shifts.iter())
        {
            let digits = keys.map(|key| digit_of(key, shift));
            if digits[0] == digits[7] && digits[1..7].iter().all(|&digit| digit == digits[0]) {
                let four = C::ONE + C::ONE + C::ONE + C::ONE;
                table[digits[0]] += four;
                other[digits[0]] += four;
            } else {
                for pair in digits.chunks_exact(2) {
                    table[pair[0]] += C::ONE;
                    other[pair[1]] += C::ONE;
                }
            }
        }
    }
    for value in chunks.remainder() {
        let key = value.key();
        for (table, &shift) in counts.iter_mut().zip(shifts.iter()) {
            table[digit_of(key, shift)] += C::ONE;
        }
    }
    for (table, other) in counts.iter_mut().zip(second.iter()) {
        for (count, &extra) in table.iter_mut().zip(other.iter()) {
            *count += extra;
        }
    }
    for table in &mut counts {
        let mut sum = C::ZERO;
        for count in table {
            let next = sum + *count;
            *count = sum;
            sum = next;
        }
    }
    let data = values.as_mut_ptr();
    let mut source = data;
    let mut dest = scratch.as_mut_ptr().cast::<T>();
    let len = values.len();
    for (table, &shift) in counts.iter_mut().zip(shifts.iter()) {
        // SAFETY: Histograms count exactly these canonical key digits.
        // Exclusive prefixes partition the destination and each pass
        // initializes every slot before it is read. The buffers are
        // disjoint, aligned, and sized for the whole input. Values retain
        // their original representation. Reading a block of four before
        // its scattered stores lets the loads overlap the bucket updates.
        unsafe {
            let mut i = 0;
            while i + 4 <= len {
                let a = source.add(i).read();
                let b = source.add(i + 1).read();
                let c = source.add(i + 2).read();
                let d = source.add(i + 3).read();
                let (ka, kb) = (digit_of(a.key(), shift), digit_of(b.key(), shift));
                let (kc, kd) = (digit_of(c.key(), shift), digit_of(d.key(), shift));
                if ka == kd && ka == kb && kc == kd {
                    // Sorted runs keep one digit for long stretches: one
                    // counter update per block avoids waiting four times
                    // for the counter just stored. The branch is taken
                    // throughout a run and almost never on random digits.
                    let position = table[ka];
                    table[ka] = position + C::ONE + C::ONE + C::ONE + C::ONE;
                    let first = dest.add(position.index());
                    first.write(a);
                    first.add(1).write(b);
                    first.add(2).write(c);
                    first.add(3).write(d);
                    i += 4;
                    continue;
                }
                let pa = table[ka];
                table[ka] = pa + C::ONE;
                let pb = table[kb];
                table[kb] = pb + C::ONE;
                let pc = table[kc];
                table[kc] = pc + C::ONE;
                let pd = table[kd];
                table[kd] = pd + C::ONE;
                dest.add(pa.index()).write(a);
                dest.add(pb.index()).write(b);
                dest.add(pc.index()).write(c);
                dest.add(pd.index()).write(d);
                i += 4;
            }
            while i < len {
                let value = source.add(i).read();
                let bucket = digit_of(value.key(), shift);
                let position = table[bucket];
                table[bucket] += C::ONE;
                dest.add(position.index()).write(value);
                i += 1;
            }
        }
        std::mem::swap(&mut source, &mut dest);
    }
    if source != data {
        // SAFETY: Odd pass counts leave the complete initialized result in
        // scratch, which is disjoint from the input and has room for len.
        unsafe { std::ptr::copy_nonoverlapping(source, data, values.len()) };
    }
    if let Some(ends) = ends {
        *ends = counts[P - 1];
    }
}

#[inline]
fn narrow_counts(len: usize) -> bool {
    len <= u32::MAX as usize
}

#[inline(never)]
fn kernel<T: Sealed, C: Counter, const N: usize>(values: &mut [T], scratch: &mut [MaybeUninit<T>]) {
    // These checks also protect the unsafe kernel if another internal caller is
    // added. They are compile-time constants for the sealed implementations.
    assert!(N == 2 || N == 4 || N == 8);
    assert!(scratch.len() >= values.len());
    assert!(values.len() <= C::MAX);
    let mut frequencies = [[C::ZERO; 256]; N];
    // Two counting chains alternate elements, and a block of eight keys
    // sharing a byte updates each chain's counter once: sorted runs share
    // their high bytes for long stretches and would otherwise wait on the
    // counter just stored for every element. Short inputs keep one chain.
    if values.len() >= 4096 {
        let mut second = [[C::ZERO; 256]; N];
        let mut chunks = values.chunks_exact(8);
        for chunk in &mut chunks {
            let keys: [u64; 8] = std::array::from_fn(|lane| chunk[lane].radix_key());
            for (byte, (table, other)) in frequencies.iter_mut().zip(second.iter_mut()).enumerate()
            {
                let digits = keys.map(|key| ((key >> (byte * 8)) & 255) as usize);
                if digits[0] == digits[7] && digits[1..7].iter().all(|&digit| digit == digits[0]) {
                    let four = C::ONE + C::ONE + C::ONE + C::ONE;
                    table[digits[0]] += four;
                    other[digits[0]] += four;
                } else {
                    for pair in digits.chunks_exact(2) {
                        table[pair[0]] += C::ONE;
                        other[pair[1]] += C::ONE;
                    }
                }
            }
        }
        for &value in chunks.remainder() {
            count(&mut frequencies, value.radix_key());
        }
        for (a, b) in frequencies
            .as_flattened_mut()
            .iter_mut()
            .zip(second.as_flattened())
        {
            *a += *b;
        }
    } else {
        let mut chunks = values.chunks_exact(4);
        for chunk in &mut chunks {
            count(&mut frequencies, chunk[0].radix_key());
            count(&mut frequencies, chunk[1].radix_key());
            count(&mut frequencies, chunk[2].radix_key());
            count(&mut frequencies, chunk[3].radix_key());
        }
        for &value in chunks.remainder() {
            count(&mut frequencies, value.radix_key());
        }
    }

    // If every byte of every key matches the first key, sorting cannot change
    // the result. Each sealed radix-key mapping is injective, so this also
    // proves that all original bit patterns (including NaN payloads) match.
    // Checking complete histograms makes this valid for any constant input.
    let mut active = 0_u8;
    if let Some(&first) = values.first() {
        let key = first.radix_key();
        for (byte, table) in frequencies.iter().enumerate() {
            if table[((key >> (byte * 8)) & 255) as usize].index() != values.len() {
                active |= 1 << byte;
            }
        }
        if active == 0 {
            return;
        }
        if active.count_ones() == 1 && values.len() >= 1024 {
            let byte = active.trailing_zeros() as usize;
            let shift = byte * 8;
            let base = first.key() & !(255 << shift);
            let rotation = ((T::SIGN_MASK >> shift) & 255) as usize;
            let mut start = 0;
            // With every other byte constant, each bucket identifies one
            // exact original value. Canonical order rotates signed high bytes.
            for bucket in 0..256 {
                let end = start + frequencies[byte][bucket ^ rotation].index();
                if end != start {
                    values[start..end].fill(T::from_key(base | ((bucket as u64) << shift)));
                }
                start = end;
            }
            return;
        }
    }

    let mut offsets = [C::ZERO; N];
    for i in (0..256).step_by(4) {
        // Keep independent prefix chains interleaved. For signed integers,
        // visit the high-byte buckets in 128..255, 0..127 order. This pays
        // for sign adjustment once per bucket instead of once per element.
        prefix_four(&mut frequencies[0], &mut offsets[0], i);
        prefix_four(
            &mut frequencies[1],
            &mut offsets[1],
            i ^ (((T::SIGN_MASK >> 8) & 255) as usize),
        );
        if N >= 4 {
            prefix_four(&mut frequencies[2], &mut offsets[2], i);
            prefix_four(
                &mut frequencies[3],
                &mut offsets[3],
                i ^ (((T::SIGN_MASK >> 24) & 255) as usize),
            );
        }
        if N == 8 {
            prefix_four(&mut frequencies[4], &mut offsets[4], i);
            prefix_four(&mut frequencies[5], &mut offsets[5], i);
            prefix_four(&mut frequencies[6], &mut offsets[6], i);
            prefix_four(
                &mut frequencies[7],
                &mut offsets[7],
                i ^ (((T::SIGN_MASK >> 56) & 255) as usize),
            );
        }
    }

    let data = values.as_mut_ptr();
    let temp = scratch.as_mut_ptr().cast::<T>();
    let len = values.len();
    if u32::from(active) != (1 << N) - 1 {
        // SAFETY: Histograms and rotated prefixes describe the original keys.
        // Each selected stable pass initializes its whole destination; odd
        // pass counts are copied back. This path never encodes stored values.
        unsafe { scatter_active(values, scratch, &mut frequencies, active) };
        return;
    }
    // SAFETY: Both buffers have room for len aligned T values and are disjoint
    // because they come from exclusive slices. The first source is initialized.
    // Prefix tables count the input's radix keys; signed high-byte buckets
    // are rotated to signed order. FloatBits is encoded while writing the
    // first pass, uses those encoded keys for intermediate passes, then is
    // decoded on the final write. The sealed implementations guarantee
    // encode(x).intermediate_key() == x.radix_key() and decode(encode(x)) == x,
    // so every pass uses the counted byte frequencies and restores exact bits.
    // Each scatter initializes
    // every destination slot once before it can become the next source.
    // Sealed types are Copy primitives or a transparent integer wrapper.
    // The even number of passes restores original representations in values.
    unsafe {
        scatter::<T, C, 0, N>(data, temp, len, &mut frequencies[0]);
        scatter::<T, C, 8, N>(temp, data, len, &mut frequencies[1]);
        if N >= 4 {
            scatter::<T, C, 16, N>(data, temp, len, &mut frequencies[2]);
            scatter::<T, C, 24, N>(temp, data, len, &mut frequencies[3]);
        }
        if N == 8 {
            scatter::<T, C, 32, N>(data, temp, len, &mut frequencies[4]);
            scatter::<T, C, 40, N>(temp, data, len, &mut frequencies[5]);
            scatter::<T, C, 48, N>(data, temp, len, &mut frequencies[6]);
            scatter::<T, C, 56, N>(temp, data, len, &mut frequencies[7]);
        }
    }
}

/// Both slices are disjoint and large enough, and every table is the exclusive
/// prefix for its original radix-key byte. Only bytes marked active are used.
unsafe fn scatter_active<T: Sealed, C: Counter, const N: usize>(
    values: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    offsets: &mut [[C; 256]; N],
    active: u8,
) {
    let data = values.as_mut_ptr();
    let mut source = data;
    let mut dest = scratch.as_mut_ptr().cast::<T>();
    for (byte, table) in offsets.iter_mut().enumerate() {
        if active & (1 << byte) == 0 {
            continue;
        }
        // SAFETY: The first source is initialized input; subsequent sources
        // were fully initialized by the preceding scatter. Counts partition
        // 0..len and all bytes use the same original, unmodified representation.
        unsafe {
            for i in 0..values.len() {
                let value = source.add(i).read();
                let bucket = ((value.radix_key() >> (byte * 8)) & 255) as usize;
                let position = table[bucket];
                table[bucket] += C::ONE;
                dest.add(position.index()).write(value);
            }
        }
        std::mem::swap(&mut source, &mut dest);
    }
    if source != data {
        // SAFETY: An odd number of complete passes leaves every value in the
        // disjoint scratch buffer, which has at least values.len() elements.
        unsafe { std::ptr::copy_nonoverlapping(source, data, values.len()) };
    }
}

#[inline(always)]
fn prefix_four<C: Counter>(table: &mut [C; 256], offset: &mut C, i: usize) {
    for count in &mut table[i..i + 4] {
        let next = *offset + *count;
        *count = *offset;
        *offset = next;
    }
}

/// `source` and `dest` must be disjoint, aligned buffers of `len` T values;
/// Source must be initialized. For SHIFT == 0, offsets count `radix_key()`;
/// later passes count `intermediate_key()`. Signed high-byte prefixes use
/// rotated bucket order. Both key mappings must yield the same byte counts
/// after first-pass encoding. Last-pass decoding must restore original bits.
/// N is 2, 4 or 8; SHIFT is a byte boundary below N * 8; len fits C.
#[inline(always)]
unsafe fn scatter<T: Sealed, C: Counter, const SHIFT: u32, const N: usize>(
    source: *const T,
    dest: *mut T,
    len: usize,
    offsets: &mut [C; 256],
) {
    let mut source = source;
    let mut blocks = len / 4;
    // SAFETY: source advances in groups of four while four elements remain; the
    // tail reads only the remaining len % 4 elements. For each bucket, its
    // prefix plus the number already seen is an unused index below len.
    // Buckets partition 0..len.
    // write() initializes scratch without reading its previous contents.
    unsafe {
        while blocks != 0 {
            // Read the whole block before its first scattered store. The input
            // and output are disjoint; exposing independent loads lets the
            // compiler overlap key extraction with bucket-index loads.
            let a = source.read();
            let b = source.add(1).read();
            let c = source.add(2).read();
            let d = source.add(3).read();
            let ka = ((if SHIFT == 0 {
                a.radix_key()
            } else {
                a.intermediate_key()
            } >> SHIFT)
                & 255) as usize;
            let kb = ((if SHIFT == 0 {
                b.radix_key()
            } else {
                b.intermediate_key()
            } >> SHIFT)
                & 255) as usize;
            let kc = ((if SHIFT == 0 {
                c.radix_key()
            } else {
                c.intermediate_key()
            } >> SHIFT)
                & 255) as usize;
            let kd = ((if SHIFT == 0 {
                d.radix_key()
            } else {
                d.intermediate_key()
            } >> SHIFT)
                & 255) as usize;
            let (ia, ib, ic, id);
            if ka == kd && ka == kb && kc == kd {
                // One counter update for a block sharing its digit, as in
                // the digit passes: sorted runs would otherwise wait on the
                // counter just stored for every element.
                ia = offsets[ka];
                ib = ia + C::ONE;
                ic = ib + C::ONE;
                id = ic + C::ONE;
                offsets[ka] = id + C::ONE;
            } else {
                ia = offsets[ka];
                offsets[ka] = ia + C::ONE;
                ib = offsets[kb];
                offsets[kb] = ib + C::ONE;
                ic = offsets[kc];
                offsets[kc] = ic + C::ONE;
                id = offsets[kd];
                offsets[kd] = id + C::ONE;
            }
            dest.add(ia.index()).write(if SHIFT == 0 {
                a.encode()
            } else if SHIFT as usize == (N - 1) * 8 {
                a.decode()
            } else {
                a
            });
            dest.add(ib.index()).write(if SHIFT == 0 {
                b.encode()
            } else if SHIFT as usize == (N - 1) * 8 {
                b.decode()
            } else {
                b
            });
            dest.add(ic.index()).write(if SHIFT == 0 {
                c.encode()
            } else if SHIFT as usize == (N - 1) * 8 {
                c.decode()
            } else {
                c
            });
            dest.add(id.index()).write(if SHIFT == 0 {
                d.encode()
            } else if SHIFT as usize == (N - 1) * 8 {
                d.decode()
            } else {
                d
            });
            source = source.add(4);
            blocks -= 1;
        }
        for _ in 0..len % 4 {
            let value = source.read();
            let key = ((if SHIFT == 0 {
                value.radix_key()
            } else {
                value.intermediate_key()
            } >> SHIFT)
                & 255) as usize;
            dest.add(offsets[key].index()).write(if SHIFT == 0 {
                value.encode()
            } else if SHIFT as usize == (N - 1) * 8 {
                value.decode()
            } else {
                value
            });
            offsets[key] += C::ONE;
            source = source.add(1);
        }
    }
}

/// Byte-only dispatcher. Byte types require no scratch or heap allocation.
#[inline]
pub(crate) fn sort_bytes<T: Sealed, const SIGN: u8>(values: &mut [T]) {
    if values.len() <= 256 {
        byte_scatter::<T, 2>(values);
    } else if narrow_counts(values.len()) {
        byte_kernel::<T, u32, SIGN, 4>(values);
    } else {
        byte_kernel::<T, usize, SIGN, 4>(values);
    }
}

/// Small byte slices fit in a stack buffer. Scattering avoids walking the
/// output buckets with many short fills. Compact counting lanes handle
/// repeated keys, and each lane scatters through its own offsets within
/// every bucket, so adjacent elements never update the same counter: a
/// nearly sorted input would otherwise wait on the counter just stored, and
/// it takes four lanes where random input needs only two. Low-cardinality
/// histograms fill only their occupied blocks.
#[inline(never)]
fn byte_scatter<T: Sealed, const LANES: usize>(values: &mut [T]) {
    assert!(std::mem::size_of::<T>() == 1 && values.len() <= 256);
    assert!(LANES == 2 || LANES == 4);
    // Each lane receives at most 128 values, so byte counters cannot overflow.
    let mut lanes = [[0_u8; 256]; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            lane[value.key() as usize] += 1;
        }
    }
    for (lane, value) in lanes.iter_mut().zip(chunks.remainder()) {
        lane[value.key() as usize] += 1;
    }
    let mut counts = [0_u16; 256];
    let mut occupied = [false; 16];
    let mut distinct = 0;
    for (block, (chunk, flag)) in counts.chunks_exact_mut(16).zip(&mut occupied).enumerate() {
        let mut any = 0;
        for (i, count) in chunk.iter_mut().enumerate() {
            *count = lanes
                .iter()
                .map(|lane| u16::from(lane[block * 16 + i]))
                .sum();
            any |= *count;
            distinct += usize::from(*count != 0);
        }
        *flag = any != 0;
    }
    if distinct <= 16 {
        let data = values.as_mut_ptr().cast::<u8>();
        let len = values.len();
        let mut offset = 0;
        for block in (0..16).filter(|&block| occupied[block]) {
            for (key, &count) in counts.iter().enumerate().skip(block * 16).take(16) {
                let count = usize::from(count);
                let byte = T::from_key(key as u64).raw_bits() as u8;
                // SAFETY: Buckets partition 0..len in key order and the
                // helper's overlapping stores stay inside the slice.
                unsafe { fill_byte_bucket(data, len, offset, count, byte) };
                offset += count;
            }
        }
        return;
    }
    exclusive_prefix(&mut counts);
    // Within a bucket, each lane's elements follow those of the lanes before.
    let mut offsets = [[0_u16; 256]; LANES];
    offsets[0] = counts;
    for lane in 1..LANES {
        for key in 0..256 {
            offsets[lane][key] = offsets[lane - 1][key] + u16::from(lanes[lane - 1][key]);
        }
    }
    let mut temp = [MaybeUninit::<T>::uninit(); 256];
    let mut chunks = values.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (offsets, &value) in offsets.iter_mut().zip(chunk) {
            let offset = &mut offsets[value.key() as usize];
            temp[usize::from(*offset)].write(value);
            *offset += 1;
        }
    }
    for (offsets, &value) in offsets.iter_mut().zip(chunks.remainder()) {
        let offset = &mut offsets[value.key() as usize];
        temp[usize::from(*offset)].write(value);
        *offset += 1;
    }
    // SAFETY: The per-lane exclusive prefixes and the matching histogram
    // write exactly 0..values.len() once each. T is Copy and the stack
    // buffer is disjoint.
    unsafe {
        std::ptr::copy_nonoverlapping(temp.as_ptr().cast::<T>(), values.as_mut_ptr(), values.len());
    }
}

/// Exclusive prefix sums over 256 counters in sixteen independent blocks, so
/// the dependent additions overlap instead of forming one 256-step chain.
#[inline]
fn exclusive_prefix(counts: &mut [u16; 256]) {
    let mut totals = [0_u16; 16];
    for (total, block) in totals.iter_mut().zip(counts.chunks_exact(16)) {
        *total = block.iter().sum();
    }
    let mut base = 0;
    for total in &mut totals {
        let next = base + *total;
        *total = base;
        base = next;
    }
    for (block, &start) in counts.chunks_exact_mut(16).zip(&totals) {
        let mut sum = start;
        for count in block {
            let next = sum + *count;
            *count = sum;
            sum = next;
        }
    }
}

#[inline(never)]
fn byte_kernel<T: Sealed, C: Counter, const SIGN: u8, const B: usize>(values: &mut [T]) {
    // The only sealed one-byte types are u8/i8, both valid for every bit pattern.
    assert!(std::mem::size_of::<T>() == 1);
    assert!(B == 1 || B == 4 || B == 8);
    assert!(values.len() <= C::MAX);
    // With one table, equal or repeated keys turn the increments into one
    // store-to-load forwarding chain, and the core then serializes even the
    // increments of other buckets. Several tables keep as many chains in
    // flight; they are merged once below.
    let mut frequencies = [[C::ZERO; 256]; B];
    let mut chunks = values.chunks_exact(8);
    for chunk in &mut chunks {
        for (lane, value) in chunk.iter().enumerate() {
            frequencies[lane % B][value.key() as usize] += C::ONE;
        }
    }
    for (lane, value) in chunks.remainder().iter().enumerate() {
        frequencies[lane % B][value.key() as usize] += C::ONE;
    }
    // Merge the replicated tables once, in a vectorizable loop, rather than
    // summing them per bucket between the fills below.
    let (first, rest) = frequencies.split_first_mut().unwrap();
    for table in rest {
        for (a, b) in first.iter_mut().zip(table.iter()) {
            *a += *b;
        }
    }
    fill_bytes::<T, C, SIGN>(values, first);
}

/// Fill byte buckets block by block: an empty block of sixteen buckets is
/// skipped with one predictable branch, and the buckets of an occupied block
/// are written without branching on their individual counts. Sparse and
/// moderately dense histograms both avoid a mispredicted branch per bucket.
fn fill_bytes<T: Sealed, C: Counter, const SIGN: u8>(values: &mut [T], counts: &[C; 256]) {
    let data = values.as_mut_ptr().cast::<u8>();
    let len = values.len();
    let mut offset = 0;
    for (block, chunk) in counts.chunks_exact(16).enumerate() {
        if chunk.iter().fold(0, |any, count| any | count.index()) == 0 {
            continue;
        }
        for (key, count) in chunk.iter().enumerate() {
            let count = count.index();
            // SAFETY: Each input contributes to exactly one counter and the
            // bucket counts sum to len, so the buckets partition 0..len in key
            // order and offset + count <= len, as the helper requires.
            unsafe {
                fill_byte_bucket(data, len, offset, count, ((block * 16 + key) as u8) ^ SIGN)
            };
            offset += count;
        }
    }
}

/// Fill one byte bucket. A short bucket, even an empty one, is written with
/// one overlapping fixed-width store instead of a memset call or a branch on
/// its count; the higher buckets then overwrite the excess bytes.
///
/// # Safety
///
/// `data` must point to `len` writable bytes, `offset + count <= len`, and
/// every byte from `offset + count` to `len` must be written afterwards by
/// the fills of higher buckets.
#[inline(always)]
unsafe fn fill_byte_bucket(data: *mut u8, len: usize, offset: usize, count: usize, byte: u8) {
    // SAFETY: An exact fill writes only this bucket. An overlapping fill of N
    // bytes stays inside the slice (offset + N <= len); its excess bytes lie
    // in higher buckets, each of which is written afterwards over its whole
    // range, so every byte ends with the value of its own bucket.
    unsafe {
        if count <= 64 && offset + 64 <= len {
            data.add(offset)
                .cast::<[u8; 64]>()
                .write_unaligned([byte; 64]);
        } else if count == 0 {
        } else if count <= 32 && offset + 32 <= len {
            data.add(offset)
                .cast::<[u8; 32]>()
                .write_unaligned([byte; 32]);
        } else if count <= 16 && offset + 16 <= len {
            data.add(offset)
                .cast::<[u8; 16]>()
                .write_unaligned([byte; 16]);
        } else if count <= 256 && offset + 256 <= len {
            // Four fixed stores beat a call for the common bucket of a
            // large uniform input; the excess lies in higher buckets.
            for step in 0..4 {
                data.add(offset + step * 64)
                    .cast::<[u8; 64]>()
                    .write_unaligned([byte; 64]);
            }
        } else {
            std::ptr::write_bytes(data.add(offset), byte, count);
        }
    }
}

#[cfg(test)]
mod tests;
