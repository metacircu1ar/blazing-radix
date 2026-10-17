//! Radix sorting for `i8`, `u8`, `i16`, `u16`, `i32`, `u32`,
//! `i64`, `u64`, and `f32`.
//!
//! Floating-point values follow [`f32::total_cmp`], including signed zeros and
//! NaNs. Every bit pattern, including NaN payloads, is preserved.
//!
//! ```
//! use blazing_radix::{radix_sort, radix_sort_with_scratch};
//!
//! let mut values = [7_i32, -3, 7, 0];
//! radix_sort(&mut values);
//! assert_eq!(values, [-3, 0, 7, 7]);
//!
//! // Reserve without initializing the scratch elements. Reuse this allocation.
//! let mut scratch = Vec::<i32>::with_capacity(values.len());
//! radix_sort_with_scratch(&mut values, scratch.spare_capacity_mut());
//! ```

#![warn(missing_docs)]

use std::mem::MaybeUninit;

#[allow(
    clippy::chunks_exact_to_as_chunks,
    reason = "Preserve benchmarked chunk loops; changing their shape needs performance validation."
)]
mod engine;

mod sealed {
    use std::mem::MaybeUninit;

    pub trait Sealed: Copy {
        // Signed integers count raw bytes; their high-byte prefix order is
        // rotated by this mask. Floats instead encode their sortable key.
        const SIGN_MASK: u64 = 0;
        // Disordered inputs up to this length use the network-and-merge
        // comparison sort. Two radix passes on 16-bit keys overtake it near
        // 64 elements; wider keys keep it through 128.
        const SMALL_SORT_LIMIT: usize = match std::mem::size_of::<Self>() {
            8 | 4 => 128,
            2 => 64,
            _ => crate::engine::NETWORK_TAIL,
        };
        const SMALL_PARTIAL: bool = false;
        // Floats stored as IEEE bits can verify order on raw bits per block.
        const FLOAT_BITS: bool = false;
        // The stored representation of types of at most four bytes, used to
        // compare values for equality without the key transform.
        #[inline(always)]
        fn raw_bits(self) -> u32 {
            self.key() as u32
        }
        fn key(self) -> u64;
        // Exact inverse of key() for the supported type's key-width domain.
        fn from_key(key: u64) -> Self;
        #[inline]
        fn small_sort(values: &mut [Self]) {
            crate::engine::small_sort(values);
        }
        #[inline]
        fn numeric_range(_values: &mut [Self]) -> bool {
            false
        }
        #[inline(always)]
        fn radix_key(self) -> u64 {
            self.key() ^ Self::SIGN_MASK
        }
        // Engine invariant: encode(x).intermediate_key() == x.radix_key(),
        // and decode(encode(x)) == x bit for bit.
        #[inline(always)]
        fn intermediate_key(self) -> u64 {
            self.radix_key()
        }
        #[inline(always)]
        fn encode(self) -> Self {
            self
        }
        #[inline(always)]
        fn decode(self) -> Self {
            self
        }
        fn sort(values: &mut [Self]);
        fn sort_with_scratch(values: &mut [Self], scratch: &mut [MaybeUninit<Self>]);
    }
}

/// A supported arithmetic type. This trait is sealed; it cannot be implemented
/// outside this crate.
pub trait RadixKey: sealed::Sealed {}

/// Sorts integers in ascending order and floats in [`f32::total_cmp`] order.
///
/// Small slices use comparison sort without allocation. Ordered inputs and
/// selected few-key inputs also finish without scratch. Other byte slices use
/// stack-based counting or scattering. Wider slices allocate one uninitialized
/// buffer of `values.len()` elements. The radix kernels skip constant digits;
/// 64-bit keys can first sort a high-key prefix, then finish colliding groups.
#[inline]
pub fn radix_sort<T: RadixKey>(values: &mut [T]) {
    T::sort(values);
}

/// Sorts with caller-owned scratch memory, without allocating.
///
/// The scratch buffer need not be initialized. Its contents after sorting are
/// unspecified. Only its first `values.len()` elements are used. Eight-bit types
/// and slices shorter than two elements do not use scratch memory.
///
/// # Panics
///
/// Panics before modifying either buffer if a wider type needs more scratch
/// elements than provided.
#[inline]
pub fn radix_sort_with_scratch<T: RadixKey>(values: &mut [T], scratch: &mut [MaybeUninit<T>]) {
    T::sort_with_scratch(values, scratch);
}

macro_rules! wider {
    ($ty:ty, $bytes:literal, $key:expr, $from_key:expr $(, $extra:item)*) => {
        impl RadixKey for $ty {}

        impl sealed::Sealed for $ty {
            #[inline(always)]
            fn key(self) -> u64 {
                ($key)(self)
            }
            #[inline(always)]
            fn from_key(key: u64) -> Self {
                ($from_key)(key)
            }
            $($extra)*

            // The entry points keep only the tiny path inline: everything else
            // is one call away, so the shortest inputs and the ordered scans
            // do not pay for the registers and stack of the wider paths.
            #[inline]
            fn sort(values: &mut [Self]) {
                if values.len() <= engine::TINY {
                    engine::tiny(values);
                    return;
                }
                if values.len() <= engine::SHORT {
                    Self::small_sort(values);
                    return;
                }
                sort_slice::<Self, $bytes>(values);
            }

            #[inline]
            fn sort_with_scratch(values: &mut [Self], scratch: &mut [MaybeUninit<Self>]) {
                if values.len() < 2 {
                    return;
                }
                assert!(scratch.len() >= values.len(), "scratch buffer is too short");
                if values.len() <= engine::TINY {
                    engine::tiny(values);
                    return;
                }
                if values.len() <= engine::SHORT {
                    Self::small_sort(values);
                    return;
                }
                sort_slice_with_scratch::<Self, $bytes>(values, scratch);
            }
        }
    };
}

#[inline(never)]
fn sort_slice<T: sealed::Sealed, const N: usize>(values: &mut [T]) {
    let preparation = engine::prepare(values);
    if preparation.is_done() {
        return;
    }
    let mut scratch = Vec::<T>::with_capacity(values.len());
    engine::finish::<T, N>(values, scratch.spare_capacity_mut(), preparation);
}

#[inline(never)]
fn sort_slice_with_scratch<T: sealed::Sealed, const N: usize>(
    values: &mut [T],
    scratch: &mut [MaybeUninit<T>],
) {
    let preparation = engine::prepare(values);
    if preparation.is_done() {
        return;
    }
    engine::finish::<T, N>(values, scratch, preparation);
}

macro_rules! integer_wider {
    ($ty:ty, $bytes:literal, $key:expr, $from_key:expr $(, $extra:item)*) => {
        wider!($ty, $bytes, $key, $from_key $(, $extra)*);
    };
}

integer_wider!(u16, 2, |v: u16| u64::from(v), |k| k as u16);
integer_wider!(
    i16,
    2,
    |v: i16| u64::from((v as u16) ^ (1 << 15)),
    |k: u64| (k as u16 ^ (1 << 15)) as i16,
    const SIGN_MASK: u64 = 1 << 15;,
    #[inline(always)]
    fn raw_bits(self) -> u32 {
        u32::from(self as u16)
    }
);
integer_wider!(u32, 4, |v: u32| u64::from(v), |k| k as u32);
integer_wider!(
    i32,
    4,
    |v: i32| u64::from((v as u32) ^ (1 << 31)),
    |k: u64| (k as u32 ^ (1 << 31)) as i32,
    const SIGN_MASK: u64 = 1 << 31;,
    #[inline(always)]
    fn raw_bits(self) -> u32 {
        self as u32
    }
);
integer_wider!(u64, 8, |v: u64| v, |k| k);
integer_wider!(
    i64,
    8,
    |v: i64| (v as u64) ^ (1 << 63),
    |k: u64| (k ^ (1 << 63)) as i64,
    const SIGN_MASK: u64 = 1 << 63;
);
#[derive(Clone, Copy)]
#[repr(transparent)]
struct FloatBits(u32);

// FloatBits holds original IEEE bits at the API boundary. During radix passes
// it holds the monotone unsigned encoding, avoiding repeated sign-dependent
// transforms. encode/decode are exact inverses for every 32-bit pattern.

const _: () = assert!(std::mem::align_of::<FloatBits>() == std::mem::align_of::<f32>());
const _: () = assert!(std::mem::size_of::<FloatBits>() == std::mem::size_of::<f32>());

wider!(
    FloatBits,
    4,
    |v: FloatBits| {
        let bits = v.0;
        let mask = (bits >> 31).wrapping_neg() | (1 << 31);
        u64::from(bits ^ mask)
    },
    |key| FloatBits(key as u32).decode(),
    #[inline(always)]
    fn intermediate_key(self) -> u64 {
        u64::from(self.0)
    },
    #[inline(always)]
    fn encode(self) -> Self {
        Self(self.key() as u32)
    },
    #[inline(always)]
    fn decode(self) -> Self {
        Self(self.0 ^ (!((self.0 as i32 >> 31) as u32) | (1 << 31)))
    }
);

impl RadixKey for f32 {}

impl sealed::Sealed for f32 {
    const SMALL_SORT_LIMIT: usize = 128;
    const SMALL_PARTIAL: bool = true;
    const FLOAT_BITS: bool = true;
    #[inline(always)]
    fn raw_bits(self) -> u32 {
        self.to_bits()
    }
    #[inline(always)]
    fn key(self) -> u64 {
        FloatBits(self.to_bits()).key()
    }

    #[inline(always)]
    fn from_key(key: u64) -> Self {
        Self::from_bits(FloatBits::from_key(key).0)
    }

    fn numeric_range(values: &mut [Self]) -> bool {
        engine::float_range(values)
    }

    #[inline]
    fn sort(values: &mut [Self]) {
        if values.len() <= engine::TINY {
            engine::tiny(values);
            return;
        }
        if values.len() <= engine::SHORT {
            Self::small_sort(values);
            return;
        }
        sort_floats(values);
    }

    #[inline]
    fn sort_with_scratch(values: &mut [Self], scratch: &mut [MaybeUninit<Self>]) {
        if values.len() < 2 {
            return;
        }
        assert!(scratch.len() >= values.len(), "scratch buffer is too short");
        if values.len() <= engine::TINY {
            engine::tiny(values);
            return;
        }
        if values.len() <= engine::SHORT {
            Self::small_sort(values);
            return;
        }
        sort_floats_with_scratch(values, scratch);
    }
}

// Inlined: the float order scans are short enough that the call itself
// showed in reversed 64-element inputs.
#[inline]
fn sort_floats(values: &mut [f32]) {
    let preparation = engine::prepare(values);
    if preparation.is_done() {
        return;
    }
    let mut scratch = Vec::<f32>::with_capacity(values.len());
    sort_float_bits(values, scratch.spare_capacity_mut(), preparation);
}

#[inline(never)]
fn sort_floats_with_scratch(values: &mut [f32], scratch: &mut [MaybeUninit<f32>]) {
    let preparation = engine::prepare(values);
    if preparation.is_done() {
        return;
    }
    sort_float_bits(values, scratch, preparation);
}

fn sort_float_bits(
    values: &mut [f32],
    scratch: &mut [MaybeUninit<f32>],
    preparation: engine::Preparation,
) {
    // SAFETY: f32 and the transparent u32 wrapper have the same size and
    // alignment, and both accept every 32-bit pattern. These reborrows are
    // exclusive and do not outlive the original slices. MaybeUninit preserves
    // the scratch initialization state. Integer loads/stores also preserve
    // every NaN payload without moving values through float registers.
    let (bits, scratch) = unsafe {
        (
            std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<FloatBits>(), values.len()),
            std::slice::from_raw_parts_mut(
                scratch.as_mut_ptr().cast::<MaybeUninit<FloatBits>>(),
                scratch.len(),
            ),
        )
    };
    engine::finish::<FloatBits, 4>(bits, scratch, preparation);
}

macro_rules! byte {
    ($ty:ty, $sign:expr) => {
        impl RadixKey for $ty {}

        impl sealed::Sealed for $ty {
            #[inline(always)]
            fn key(self) -> u64 {
                u64::from((self as u8) ^ $sign)
            }

            #[inline(always)]
            fn raw_bits(self) -> u32 {
                u32::from(self as u8)
            }

            #[inline(always)]
            fn from_key(key: u64) -> Self {
                (key as u8 ^ $sign) as Self
            }

            #[inline]
            fn sort(values: &mut [Self]) {
                if values.len() <= engine::TINY {
                    engine::tiny(values);
                    return;
                }
                if values.len() <= engine::SHORT {
                    Self::small_sort(values);
                    return;
                }
                sort_byte_slice::<Self, $sign>(values);
            }

            #[inline]
            fn sort_with_scratch(values: &mut [Self], _scratch: &mut [MaybeUninit<Self>]) {
                Self::sort(values);
            }
        }
    };
}

#[inline(never)]
fn sort_byte_slice<T: sealed::Sealed, const SIGN: u8>(values: &mut [T]) {
    if engine::prepare(values).is_done() {
        return;
    }
    engine::sort_bytes::<T, SIGN>(values);
}

byte!(u8, 0);
byte!(i8, 128);
