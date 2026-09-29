//! Tiny, fast, deterministic PRNG (xoshiro256++), dependency-free.
//!
//! Used for HNSW level assignment (reproducible graphs from a seed), synthetic benchmark data and
//! tests. **Not** cryptographically secure.

/// xoshiro256++ by Blackman & Vigna. 256-bit state, period 2^256 − 1.
#[derive(Clone, Debug)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    /// Seeds the full state from one `u64` via SplitMix64 (the reference seeding procedure).
    #[must_use]
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed;
        let mut next = || {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0, 1)` with 53 bits of precision.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform `f32` in `[0, 1)` with 24 bits of precision.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform integer in `[0, n)` using Lemire's multiply-shift (unbiased via rejection).
    ///
    /// # Panics
    /// Panics if `n == 0`.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below(0) is empty");
        let threshold = n.wrapping_neg() % n;
        loop {
            let m = u128::from(self.next_u64()) * u128::from(n);
            if (m as u64) >= threshold {
                return (m >> 64) as u64;
            }
        }
    }

    /// Standard normal sample (Box–Muller).
    #[inline]
    pub fn next_gaussian(&mut self) -> f32 {
        // 1 - U maps [0,1) to (0,1], keeping ln() finite.
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        ((-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()) as f32
    }

    /// Fills `out` with uniform samples in `[lo, hi)`.
    pub fn fill_uniform(&mut self, out: &mut [f32], lo: f32, hi: f32) {
        let span = hi - lo;
        for x in out {
            *x = lo + span * self.next_f32();
        }
    }

    /// Fills `out` with standard normal samples.
    pub fn fill_gaussian(&mut self, out: &mut [f32]) {
        for x in out {
            *x = self.next_gaussian();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Xoshiro256pp::seed_from_u64(42);
        let mut b = Xoshiro256pp::seed_from_u64(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
            let f = a.next_f32();
            let _ = b.next_f32();
            assert!((0.0..1.0).contains(&f));
            let k = a.below(7);
            let _ = b.below(7);
            assert!(k < 7);
        }
        assert_ne!(
            Xoshiro256pp::seed_from_u64(1).next_u64(),
            Xoshiro256pp::seed_from_u64(2).next_u64()
        );
    }

    #[test]
    fn gaussian_moments() {
        let mut r = Xoshiro256pp::seed_from_u64(7);
        let n = 200_000;
        let (mut sum, mut sq) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let x = f64::from(r.next_gaussian());
            sum += x;
            sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }
}
