//! Per-search pseudo-random number generation.
//!
//! Objective:
//! - Provide a fast, allocation-free generator owned by exactly one search.
//!
//! Technical approach:
//! - xoshiro256++ state seeded through SplitMix64, so any `u64` seed yields a
//!   well-mixed non-zero state.
//! - Floating-point draws use the high 53 bits; unbiased bounded integers use
//!   Lemire's widening-multiply rejection method.
//!
//! Flow (pseudocode):
//! 1. Expand the seed with SplitMix64 into four state words.
//! 2. Each draw advances the xoshiro state and returns the scrambled word.
//! 3. Derived draws (`f64`, ranges, coin flips) are pure functions of one word.
//!
//! Important criteria:
//! - No shared state: threads never contend on a generator.
//! - Deterministic for a given seed and draw sequence.
//!
//! Inputs/Outputs:
//! - Input: a `u64` seed.
//! - Output: `u64`, `f64` in `[0, 1)`, bounded integers and uniform intervals.

/// SplitMix64 step used for seeding and stream derivation.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derives an independent stream seed from a base seed and a stream index.
#[must_use]
pub fn derive_seed(base: u64, stream: u64) -> u64 {
    let mut state = base ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03);
    splitmix64(&mut state)
}

/// xoshiro256++ generator owned by a single search.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Creates a generator from a 64-bit seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        Self {
            s: [
                splitmix64(&mut state),
                splitmix64(&mut state),
                splitmix64(&mut state),
                splitmix64(&mut state),
            ],
        }
    }

    /// Returns the next raw 64-bit word.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Converts a raw word into a float in `[0, 1)` using its high 53 bits.
    #[inline]
    #[must_use]
    pub fn unit_from_word(word: u64) -> f64 {
        (word >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Returns a uniform float in `[0, 1)`.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        Self::unit_from_word(self.next_u64())
    }

    /// Returns a uniform float in `[low, high)`.
    #[inline]
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.f64()
    }

    /// Returns an unbiased integer in `[0, bound)`; `bound` must be positive.
    #[inline]
    pub fn below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        let bound = bound as u64;
        let mut product = u128::from(self.next_u64()) * u128::from(bound);
        let mut low = product as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                product = u128::from(self.next_u64()) * u128::from(bound);
                low = product as u64;
            }
        }
        (product >> 64) as usize
    }

    /// Returns an unbiased integer in `[low, high]` inclusive.
    #[inline]
    pub fn range_inclusive(&mut self, low: usize, high: usize) -> usize {
        low + self.below(high - low + 1)
    }

    /// Shuffles a slice in place with Fisher-Yates.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let other = self.below(index + 1);
            items.swap(index, other);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn floats_in_unit_interval_with_expected_mean() {
        let mut rng = Rng::new(1);
        let mut sum = 0.0;
        for _ in 0..200_000 {
            let value = rng.f64();
            assert!((0.0..1.0).contains(&value));
            sum += value;
        }
        assert!((sum / 200_000.0 - 0.5).abs() < 0.005);
    }

    #[test]
    fn below_is_uniform_and_in_range() {
        let mut rng = Rng::new(3);
        let mut counts = [0_u32; 7];
        for _ in 0..70_000 {
            counts[rng.below(7)] += 1;
        }
        for count in counts {
            assert!((9_300..10_700).contains(&count), "count {count}");
        }
    }

    #[test]
    fn derived_streams_differ() {
        assert_ne!(derive_seed(1, 0), derive_seed(1, 1));
    }

    #[test]
    fn shuffle_is_permutation() {
        let mut rng = Rng::new(5);
        let mut items: Vec<usize> = (0..50).collect();
        rng.shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
    }
}
