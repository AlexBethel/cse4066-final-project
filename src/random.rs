use rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

static mut RNG: Option<StdRng> = None;

pub fn random<T>() -> T
where
    Standard: Distribution<T>,
{
    let rng = unsafe { &mut RNG }.get_or_insert_with(|| StdRng::from_seed([2; 32]));
    rng.gen()
}
