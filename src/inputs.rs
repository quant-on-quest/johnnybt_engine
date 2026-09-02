//! What one simulation is handed: the normalized arrays, and where each
//! packed row lives.

use ndarray::{ArrayView1, ArrayView2, ArrayView3, ArrayView4};

// Row layout of the packed rates matrix, mirroring johnnybt.market.spec.RATES.
pub const LOT: usize = 0;
pub const MIN_LOT: usize = 1;
pub const MIN_NOTIONAL: usize = 2;
pub const MULTIPLIER: usize = 3;
pub const BUY_FEE: usize = 4;
pub const SELL_FEE: usize = 5;
pub const FIXED_FEE: usize = 6;
pub const MIN_FEE: usize = 7;

// Flags rows, mirroring the flags matrix.
pub const SHORTABLE: usize = 0;
pub const SAME_BAR: usize = 1;

// Reported series rows.
pub const EQUITY: usize = 0;
pub const CASH: usize = 1;
pub const FEES: usize = 2;
pub const BOUGHT: usize = 3;
pub const SOLD: usize = 4;
pub const REPORTED: usize = 5;

/// One simulation's inputs, as views over the caller's arrays.
///
/// Shapes are the engine's contract (`johnnybt.engine.kernel.EngineInputs`)
/// and are validated before they get here: nothing in the walk checks a
/// bound it did not have to.
#[allow(clippy::too_many_arguments)]
pub struct Inputs<'a> {
    pub plan: ArrayView4<'a, f64>,
    pub at: ArrayView4<'a, i32>,
    pub standing: ArrayView4<'a, bool>,
    /// One `(T, N)` plane per price point — each a view over the caller's
    /// own buffer, so a plane the frame already holds contiguously is never
    /// copied to be stacked.
    pub prices: Vec<ArrayView2<'a, f64>>,
    pub mark: ArrayView2<'a, f64>,
    pub impound: Vec<ArrayView2<'a, bool>>,
    pub previous: ArrayView2<'a, f64>,
    pub has_previous: bool,
    pub new_day: ArrayView1<'a, bool>,
    pub buyable: Vec<ArrayView2<'a, bool>>,
    pub sellable: Vec<ArrayView2<'a, bool>>,
    pub instrument: ArrayView1<'a, i32>,
    pub epoch: ArrayView1<'a, i32>,
    pub rates: ArrayView3<'a, f64>,
    pub flags: ArrayView2<'a, bool>,
    pub capital: f64,
    pub buffer: f64,
    pub audit: bool,
    pub record_positions: bool,
}

impl<'a> Inputs<'a> {
    /// How many price points a bar carries.
    #[inline]
    pub fn points(&self) -> usize {
        self.prices.len()
    }

    /// How many bars.
    #[inline]
    pub fn steps(&self) -> usize {
        self.mark.shape()[0]
    }

    /// The price at one point of one bar for one instrument.
    #[inline]
    pub fn price(&self, phase: usize, t: usize, i: usize) -> f64 {
        self.prices[phase][(t, i)]
    }

    /// Whether a buy can fill there.
    #[inline]
    pub fn buyable(&self, phase: usize, t: usize, i: usize) -> bool {
        self.buyable[phase][(t, i)]
    }

    /// Whether a sell can fill there.
    #[inline]
    pub fn sellable(&self, phase: usize, t: usize, i: usize) -> bool {
        self.sellable[phase][(t, i)]
    }

    /// Whether a holding is stuck at the limit there.
    #[inline]
    pub fn impound(&self, phase: usize, t: usize, i: usize) -> bool {
        self.impound[phase][(t, i)]
    }

    /// The instrument's class code.
    #[inline]
    pub fn class(&self, i: usize) -> usize {
        self.instrument[i] as usize
    }

    /// One rate for one instrument under one epoch.
    #[inline]
    pub fn rate(&self, row: usize, e: usize, c: usize) -> f64 {
        self.rates[(row, e, c)]
    }

    /// A fee on one fill: proportional plus fixed, floored at the minimum.
    #[inline]
    pub fn fee(&self, turnover: f64, rate: f64, e: usize, c: usize) -> f64 {
        let mut fee = turnover * rate + self.rates[(FIXED_FEE, e, c)];
        if fee < self.rates[(MIN_FEE, e, c)] {
            fee = self.rates[(MIN_FEE, e, c)];
        }
        fee
    }
}

/// The sizing rule: a weight becomes a whole-lot quantity, rounded down.
#[inline]
pub fn decide(
    investable: f64,
    weight: f64,
    price: f64,
    lot: f64,
    min_lot: f64,
    min_notional: f64,
    multiplier: f64,
) -> f64 {
    if weight == 0.0 || price.is_nan() || price <= 0.0 {
        return 0.0;
    }
    let value = investable * weight;
    if value.abs() < min_notional {
        return 0.0;
    }
    let raw = value.abs() / (price * multiplier);
    let quantity = (raw / lot).floor() * lot;
    if quantity < min_lot {
        return 0.0;
    }
    if weight > 0.0 {
        quantity
    } else {
        -quantity
    }
}

/// Where the walk stands: one price point of one bar, under one rule epoch.
///
/// Every phase of the account and every policy hook reads the same three
/// coordinates; carrying them as one value keeps the signatures short and
/// makes it impossible to hand a bar's epoch to another bar's point.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub t: usize,
    pub phase: usize,
    pub e: usize,
}
