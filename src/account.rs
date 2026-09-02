//! The account: what every market's walk has in common.
//!
//! Cash, the per-tranche books, the liquidation pool, the last quote of
//! every name, and what each tranche's book has actually attained. The
//! phases here are the ones no bookkeeping policy disputes — marking,
//! impounding, the pool's sales, the equity snapshot, the close. How a
//! decision is funded, sized and filled is the policy's (`Bookkeeping`).

use crate::inputs::*;

/// One account's state, for one run.
pub struct Account {
    /// Which run this account is.
    pub r: usize,
    pub tranches: usize,
    pub n: usize,
    /// The account's cash. Policies move money in and out of it.
    pub cash: f64,
    /// `(K, N)` what each tranche holds, in units.
    pub qty: Vec<f64>,
    /// `(K, N)` what each tranche's standing decision asks it to hold.
    pub target: Vec<f64>,
    /// `(N,)` the liquidation pool: shares impounded out of their tranches.
    pub liq: Vec<f64>,
    /// `(N,)` the last quote seen for each name — what a holding is worth.
    pub last: Vec<f64>,
    /// `(K, N)` the weight each book actually reached. A standing firing is
    /// measured against this rather than the last target struck: the two
    /// part company exactly where a fill was refused, and that is where the
    /// account has to come back and try again.
    pub attained: Vec<f64>,
    /// `(K,)` which decision the book attained whole; -1 says the last
    /// firing left it somewhere in between.
    pub attained_row: Vec<i64>,
    /// `(K,)` which decision fires at the current point, -1 for none.
    pub fire: Vec<i64>,
    /// `(K, N)` which reductions the settlement refused when sized.
    pub blocked: Vec<bool>,
    /// The bar's fees, purchase turnover and sale turnover so far.
    pub fees: f64,
    pub bought: f64,
    pub sold: f64,
    /// The equity read at the point sizing happens.
    pub equity: f64,
}

impl Account {
    pub fn new(r: usize, tranches: usize, n: usize, capital: f64) -> Self {
        Account {
            r,
            tranches,
            n,
            cash: capital,
            qty: vec![0.0; tranches * n],
            target: vec![0.0; tranches * n],
            liq: vec![0.0; n],
            last: vec![0.0; n],
            attained: vec![f64::NAN; tranches * n],
            attained_row: vec![-1; tranches],
            fire: vec![-1; tranches],
            blocked: vec![false; tranches * n],
            fees: 0.0,
            bought: 0.0,
            sold: 0.0,
            equity: 0.0,
        }
    }

    /// The flat index of tranche `k`, name `i`.
    #[inline]
    pub fn at(&self, k: usize, i: usize) -> usize {
        k * self.n + i
    }

    /// The value of one name's `units` at its last quote.
    #[inline]
    pub fn worth(&self, inp: &Inputs, at: Point, i: usize, units: f64) -> f64 {
        units * self.last[i] * inp.rate(MULTIPLIER, at.e, inp.class(i))
    }

    /// Before the bar opens, a holding is worth the exchange's reference
    /// close; on an ex-dividend bar the shares scale by the ratio.
    ///
    /// Returns each scaled name with its ratio, for the policy to scale its
    /// own per-name state by.
    pub fn mark_previous(&mut self, inp: &Inputs, t: usize, scaled: &mut Vec<(usize, f64)>) {
        scaled.clear();
        if !inp.has_previous {
            return;
        }
        for i in 0..self.n {
            let q = inp.previous[(t, i)];
            if !q.is_nan() && q > 0.0 {
                let base = self.last[i];
                if base > 0.0 && base != q {
                    let ratio = base / q;
                    for k in 0..self.tranches {
                        let cell = self.at(k, i);
                        if self.qty[cell] != 0.0 {
                            self.qty[cell] *= ratio;
                            self.target[cell] *= ratio;
                        }
                    }
                    if self.liq[i] > 0.0 {
                        self.liq[i] *= ratio;
                    }
                    scaled.push((i, ratio));
                }
                self.last[i] = q;
            }
        }
    }

    /// Everything the account holds in one name, tranches and pool alike.
    pub fn held(&self, i: usize) -> f64 {
        let mut total = self.liq[i];
        for k in 0..self.tranches {
            total += self.qty[self.at(k, i)];
        }
        total
    }

    /// The bar's accumulators start empty.
    pub fn open_bar(&mut self) {
        self.fees = 0.0;
        self.bought = 0.0;
        self.sold = 0.0;
    }

    /// A price point quotes: every name with a price is worth that now.
    pub fn mark_point(&mut self, inp: &Inputs, at: Point) {
        for i in 0..self.n {
            let p = inp.prices[(at.phase, at.t, i)];
            if !p.is_nan() && p > 0.0 {
                self.last[i] = p;
            }
        }
    }

    /// Which tranches fire at this point, after impounding what is stuck.
    ///
    /// A position stuck at the limit leaves its tranche at every point that
    /// tranche has a firing at — before the pool sells here, so it gets to
    /// try at this point too, and **including a firing with nothing to
    /// say**: the transfer happens before the answer is compared against
    /// the book, so a holding that goes limit-down mid-morning is in the
    /// pool from the next point on rather than waiting for the answer to
    /// move. A standing firing whose answer the book already attained is
    /// answered without touching the instruments.
    ///
    /// Returns whether anybody decides here.
    pub fn resolve_firings(&mut self, inp: &Inputs, at: Point) -> bool {
        let (r, t, phase) = (self.r, at.t, at.phase);
        let mut deciding = false;
        for k in 0..self.tranches {
            let mut d = inp.at[(r, k, t, phase)];
            if d >= 0 {
                for i in 0..self.n {
                    let cell = self.at(k, i);
                    if inp.impound[(phase, t, i)] && self.qty[cell] > 0.0 {
                        self.liq[i] += self.qty[cell];
                        self.qty[cell] = 0.0;
                    }
                }
            }
            if d >= 0 && inp.standing[(r, k, t, phase)] {
                let mut settled = true;
                if d as i64 != self.attained_row[k] {
                    for i in 0..self.n {
                        let w = inp.plan[(r, k, d as usize, i)];
                        let a = self.attained[self.at(k, i)];
                        if w != a && (!w.is_nan() || !a.is_nan()) {
                            settled = false;
                            break;
                        }
                    }
                }
                if settled {
                    d = -1;
                }
            }
            self.fire[k] = d as i64;
            if d >= 0 {
                deciding = true;
            }
        }
        deciding
    }

    /// The liquidation pool tries to leave at every point.
    pub fn pool_sell(&mut self, inp: &Inputs, at: Point) {
        let (t, phase, e) = (at.t, at.phase, at.e);
        for i in 0..self.n {
            if self.liq[i] <= 0.0 {
                continue;
            }
            let p = inp.prices[(phase, t, i)];
            if p.is_nan() || p <= 0.0 || !inp.sellable[(phase, t, i)] {
                continue;
            }
            let c = inp.class(i);
            let give = self.liq[i];
            let turnover = give * p * inp.rate(MULTIPLIER, e, c);
            let fee = inp.fee(turnover, inp.rate(SELL_FEE, e, c), e, c);
            self.liq[i] = 0.0;
            self.cash += turnover - fee;
            self.fees += fee;
            self.sold += turnover;
        }
    }

    /// The account's equity, read where sizing happens: after the pool has
    /// sold into it, so its proceeds are cash and its fees already paid.
    pub fn snapshot_equity(&mut self, inp: &Inputs, at: Point) {
        let mut equity = self.cash;
        for i in 0..self.n {
            if self.liq[i] != 0.0 {
                equity += self.worth(inp, at, i, self.liq[i]);
            }
        }
        for k in 0..self.tranches {
            for i in 0..self.n {
                let units = self.qty[self.at(k, i)];
                if units != 0.0 {
                    equity += self.worth(inp, at, i, units);
                }
            }
        }
        self.equity = equity;
    }

    /// The market value of one tranche's book.
    pub fn holding(&self, inp: &Inputs, k: usize, at: Point) -> f64 {
        let mut holding = 0.0f64;
        for i in 0..self.n {
            let units = self.qty[self.at(k, i)];
            if units != 0.0 {
                holding += self.worth(inp, at, i, units);
            }
        }
        holding
    }

    /// The gross exposure a decision asks for, and the equity it may put on.
    ///
    /// Gross, not net: a long-short plan whose legs offset still puts on
    /// both books, and sizing by the net would liquidate them. For a
    /// long-only plan |w| is w bit for bit. `reachable` caps the target
    /// equity at what the tranche can actually reach — its own holdings
    /// plus whatever cash the policy says is free.
    pub fn investable(&self, inp: &Inputs, share: f64, reachable: f64) -> f64 {
        if share < 1e-12 {
            return 0.0;
        }
        let mut target_equity = self.equity * share;
        if reachable < target_equity {
            target_equity = reachable;
        }
        target_equity * inp.buffer / share
    }

    /// The target this decision strikes for one name, or the book itself
    /// when the plan says nothing about it.
    #[inline]
    pub fn strike(&self, inp: &Inputs, k: usize, i: usize, weight: f64, investable: f64, at: Point) -> f64 {
        if weight.is_nan() {
            return self.qty[self.at(k, i)];
        }
        let (t, phase, e) = (at.t, at.phase, at.e);
        let c = inp.class(i);
        let mut want = decide(
            investable,
            weight,
            inp.prices[(phase, t, i)],
            inp.rate(LOT, e, c),
            inp.rate(MIN_LOT, e, c),
            inp.rate(MIN_NOTIONAL, e, c),
            inp.rate(MULTIPLIER, e, c),
        );
        if want < 0.0 && !inp.flags[(SHORTABLE, c)] {
            want = 0.0;
        }
        want
    }

    /// Record what the tranche's book attained after trading decision `d`.
    pub fn remember_attained(&mut self, inp: &Inputs, k: usize, d: usize) {
        let r = self.r;
        let mut whole = true;
        for i in 0..self.n {
            let cell = self.at(k, i);
            if self.blocked[cell] {
                whole = false;
            } else {
                self.attained[cell] = inp.plan[(r, k, d, i)];
            }
        }
        self.attained_row[k] = if whole { d as i64 } else { -1 };
    }

    /// The bar closes: what is held is worth the closing price, and the
    /// bar's series are reported.
    pub fn close_bar(&mut self, inp: &Inputs, at: Point, reported: &mut [f64], positions: &mut [f64], steps: usize) {
        let t = at.t;
        for i in 0..self.n {
            let m = inp.mark[(t, i)];
            if !m.is_nan() && m > 0.0 {
                self.last[i] = m;
            }
        }
        let mut value = 0.0f64;
        for i in 0..self.n {
            let total = self.held(i);
            if total != 0.0 {
                value += self.worth(inp, at, i, total);
            }
            if inp.record_positions {
                positions[t * self.n + i] = total;
            }
        }
        reported[EQUITY * steps + t] = self.cash + value;
        reported[CASH * steps + t] = self.cash;
        reported[FEES * steps + t] = self.fees;
        reported[BOUGHT * steps + t] = self.bought;
        reported[SOLD * steps + t] = self.sold;
    }
}
