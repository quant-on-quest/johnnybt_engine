//! The account: what every market's walk has in common.
//!
//! Cash, the per-tranche books, the liquidation pool, the last quote of
//! every name, and what each tranche's book has actually attained. The
//! phases here are the ones no bookkeeping policy disputes — marking,
//! impounding, the pool's sales, the equity snapshot, the close. How a
//! decision is funded, sized and filled is the policy's (`Bookkeeping`).

use crate::inputs::*;

/// The tranche a fill wears when the liquidation pool made it.
pub const POOL: i32 = -1;

/// One fill, as it happened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    pub bar: u32,
    pub point: u32,
    /// The tranche whose book filled; `POOL` for the liquidation pool.
    pub tranche: i32,
    pub asset: u32,
    /// Units, positive bought and negative sold.
    pub units: f64,
    /// The price per unit it filled at.
    pub price: f64,
    /// What the fill cost in fees.
    pub fee: f64,
}

/// One account's state, for one run.
pub struct Account {
    /// Which run this account is.
    pub run: usize,
    pub tranches: usize,
    pub assets: usize,
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
    /// `(K,)` the names each tranche's book is doing anything with, ascending:
    /// a unit held, a target struck, a refusal recorded, or state the policy
    /// keeps. A market has thousands of names and a book touches a dozen;
    /// every per-point phase walks these and nothing else, in the same
    /// ascending order the dense scan took — so every sum adds the same
    /// non-zero terms in the same order, and the answer is the same bit.
    pub active: Vec<Vec<usize>>,
    /// The names the liquidation pool holds, ascending.
    pub pool: Vec<usize>,
    /// The bar's fees, purchase turnover and sale turnover so far.
    pub fees: f64,
    pub bought: f64,
    pub sold: f64,
    /// The equity read at the point sizing happens.
    pub equity: f64,
    /// Whether fills go on record. Off, `fill` is one branch and nothing
    /// else — the walk's cost does not move.
    pub record_fills: bool,
    /// Every fill so far, in walk order, when recorded.
    pub fills: Vec<Fill>,
}

impl Account {
    pub fn new(run: usize, tranches: usize, assets: usize, capital: f64) -> Self {
        Account {
            run,
            tranches,
            assets,
            cash: capital,
            qty: vec![0.0; tranches * assets],
            target: vec![0.0; tranches * assets],
            liq: vec![0.0; assets],
            last: vec![0.0; assets],
            attained: vec![f64::NAN; tranches * assets],
            attained_row: vec![-1; tranches],
            fire: vec![-1; tranches],
            blocked: vec![false; tranches * assets],
            active: vec![Vec::new(); tranches],
            pool: Vec::new(),
            fees: 0.0,
            bought: 0.0,
            sold: 0.0,
            equity: 0.0,
            record_fills: false,
            fills: Vec::new(),
        }
    }

    /// Put one fill on record, when fills are being recorded.
    #[inline]
    pub fn fill(&mut self, at: Point, tranche: i32, asset: usize, units: f64, price: f64, fee: f64) {
        if self.record_fills {
            self.fills.push(Fill {
                bar: at.bar as u32,
                point: at.phase as u32,
                tranche,
                asset: asset as u32,
                units,
                price,
                fee,
            });
        }
    }

    /// The flat index of tranche `k`, name `i`.
    #[inline]
    pub fn at(&self, tranche: usize, asset: usize) -> usize {
        tranche * self.assets + asset
    }

    /// Note that tranche `k` is doing something with name `i`.
    #[inline]
    pub fn touch(&mut self, tranche: usize, asset: usize) {
        insert_sorted(&mut self.active[tranche], asset);
    }

    /// The names tranche `k` is doing something with, ascending.
    #[inline]
    pub fn named(&self, tranche: usize) -> &[usize] {
        &self.active[tranche]
    }

    /// The names a decision row weights, ascending — a missing weight
    /// (NaN, "keep the book") names the instrument too.
    pub fn row_names(&self, inp: &Inputs, tranche: usize, decision: usize) -> Vec<usize> {
        let run = self.run;
        (0..self.assets)
            .filter(|&asset| inp.plan[(run, tranche, decision, asset)] != 0.0)
            .collect()
    }

    /// Every name any book or the pool holds, ascending.
    pub fn union_names(&self) -> Vec<usize> {
        let mut out = self.pool.clone();
        for tranche in 0..self.tranches {
            out = merged(&out, &self.active[tranche]);
        }
        out
    }

    /// Forget the names whose cells are back at rest: nothing held, no
    /// target, no refusal, and nothing the policy keeps for them.
    pub fn sweep(&mut self, at_rest: impl Fn(usize, usize) -> bool) {
        for tranche in 0..self.tranches {
            let assets = self.assets;
            let (qty, target, blocked) = (&self.qty, &self.target, &self.blocked);
            self.active[tranche].retain(|&asset| {
                let cell = tranche * assets + asset;
                !(qty[cell] == 0.0
                    && target[cell] == 0.0
                    && !blocked[cell]
                    && at_rest(tranche, asset))
            });
        }
    }

    /// The value of one name's `units` at its last quote.
    #[inline]
    pub fn worth(&self, inp: &Inputs, at: Point, asset: usize, units: f64) -> f64 {
        units * self.last[asset] * inp.rate(MULTIPLIER, at.epoch, inp.class(asset))
    }

    /// Before the bar opens, a holding is worth the exchange's reference
    /// close; on an ex-dividend bar the shares scale by the ratio.
    ///
    /// Returns each scaled name with its ratio, for the policy to scale its
    /// own per-name state by.
    pub fn mark_previous(&mut self, inp: &Inputs, bar: usize, scaled: &mut Vec<(usize, f64)>) {
        scaled.clear();
        if !inp.has_previous {
            return;
        }
        for asset in 0..self.assets {
            let quote = inp.previous[(bar, asset)];
            if !quote.is_nan() && quote > 0.0 {
                let base = self.last[asset];
                if base > 0.0 && base != quote {
                    let ratio = base / quote;
                    for tranche in 0..self.tranches {
                        let cell = self.at(tranche, asset);
                        if self.qty[cell] != 0.0 {
                            self.qty[cell] *= ratio;
                            self.target[cell] *= ratio;
                        }
                    }
                    if self.liq[asset] > 0.0 {
                        self.liq[asset] *= ratio;
                    }
                    scaled.push((asset, ratio));
                }
                self.last[asset] = quote;
            }
        }
    }

    /// Everything the account holds in one name, tranches and pool alike.
    #[inline]
    pub fn held(&self, asset: usize) -> f64 {
        let mut total = self.liq[asset];
        for tranche in 0..self.tranches {
            total += self.qty[self.at(tranche, asset)];
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
        for asset in 0..self.assets {
            let price = inp.price(at.phase, at.bar, asset);
            if !price.is_nan() && price > 0.0 {
                self.last[asset] = price;
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
        let (run, bar, phase) = (self.run, at.bar, at.phase);
        let mut deciding = false;
        for tranche in 0..self.tranches {
            let mut decision = inp.at[(run, tranche, bar, phase)];
            if decision >= 0 {
                for idx in 0..self.active[tranche].len() {
                    let asset = self.active[tranche][idx];
                    let cell = self.at(tranche, asset);
                    if inp.impound(phase, bar, asset) && self.qty[cell] > 0.0 {
                        self.liq[asset] += self.qty[cell];
                        self.qty[cell] = 0.0;
                        insert_sorted(&mut self.pool, asset);
                    }
                }
            }
            if decision >= 0 && inp.standing[(run, tranche, bar, phase)] {
                let mut settled = true;
                if decision as i64 != self.attained_row[tranche] {
                    for asset in 0..self.assets {
                        let weight = inp.plan[(run, tranche, decision as usize, asset)];
                        let attained = self.attained[self.at(tranche, asset)];
                        if weight != attained && (!weight.is_nan() || !attained.is_nan()) {
                            settled = false;
                            break;
                        }
                    }
                }
                if settled {
                    decision = -1;
                }
            }
            self.fire[tranche] = decision as i64;
            if decision >= 0 {
                deciding = true;
            }
        }
        deciding
    }

    /// The liquidation pool tries to leave at every point.
    pub fn pool_sell(&mut self, inp: &Inputs, at: Point) {
        let (bar, phase, epoch) = (at.bar, at.phase, at.epoch);
        let mut emptied = false;
        for idx in 0..self.pool.len() {
            let asset = self.pool[idx];
            if self.liq[asset] <= 0.0 {
                continue;
            }
            let price = inp.price(phase, bar, asset);
            if price.is_nan() || price <= 0.0 || !inp.sellable(phase, bar, asset) {
                continue;
            }
            let category = inp.class(asset);
            let give = self.liq[asset];
            let turnover = give * price * inp.rate(MULTIPLIER, epoch, category);
            let fee = inp.fee(
                turnover,
                inp.rate(SELL_FEE, epoch, category),
                epoch,
                category,
            );
            self.liq[asset] = 0.0;
            self.cash += turnover - fee;
            self.fees += fee;
            self.sold += turnover;
            self.fill(at, POOL, asset, -give, price, fee);
            emptied = true;
        }
        if emptied {
            let liq = &self.liq;
            self.pool.retain(|&asset| liq[asset] > 0.0);
        }
    }

    /// The account's equity, read where sizing happens: after the pool has
    /// sold into it, so its proceeds are cash and its fees already paid.
    pub fn snapshot_equity(&mut self, inp: &Inputs, at: Point) {
        let mut equity = self.cash;
        for &asset in self.pool.iter() {
            if self.liq[asset] != 0.0 {
                equity += self.worth(inp, at, asset, self.liq[asset]);
            }
        }
        for tranche in 0..self.tranches {
            for &asset in self.active[tranche].iter() {
                let units = self.qty[self.at(tranche, asset)];
                if units != 0.0 {
                    equity += self.worth(inp, at, asset, units);
                }
            }
        }
        self.equity = equity;
    }

    /// The market value of one tranche's book.
    pub fn holding(&self, inp: &Inputs, tranche: usize, at: Point) -> f64 {
        let mut holding = 0.0f64;
        for &asset in self.active[tranche].iter() {
            let units = self.qty[self.at(tranche, asset)];
            if units != 0.0 {
                holding += self.worth(inp, at, asset, units);
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
    pub fn strike(
        &self,
        inp: &Inputs,
        tranche: usize,
        asset: usize,
        weight: f64,
        investable: f64,
        at: Point,
    ) -> f64 {
        if weight.is_nan() {
            return self.qty[self.at(tranche, asset)];
        }
        let (bar, phase, epoch) = (at.bar, at.phase, at.epoch);
        let category = inp.class(asset);
        let mut want = decide(
            investable,
            weight,
            inp.price(phase, bar, asset),
            inp.rate(LOT, epoch, category),
            inp.rate(MIN_LOT, epoch, category),
            inp.rate(MIN_NOTIONAL, epoch, category),
            inp.rate(MULTIPLIER, epoch, category),
        );
        if want < 0.0 && !inp.flags[(SHORTABLE, category)] {
            want = 0.0;
        }
        want
    }

    /// Record what the tranche's book attained after trading decision `d`.
    pub fn remember_attained(&mut self, inp: &Inputs, tranche: usize, decision: usize) {
        let run = self.run;
        let mut whole = true;
        for asset in 0..self.assets {
            let cell = self.at(tranche, asset);
            if self.blocked[cell] {
                whole = false;
            } else {
                self.attained[cell] = inp.plan[(run, tranche, decision, asset)];
            }
        }
        self.attained_row[tranche] = if whole { decision as i64 } else { -1 };
    }

    /// The bar closes: what is held is worth the closing price, and the
    /// bar's series are reported.
    pub fn close_bar(
        &mut self,
        inp: &Inputs,
        at: Point,
        reported: &mut [f64],
        positions: &mut [f64],
        steps: usize,
    ) {
        let bar = at.bar;
        for asset in 0..self.assets {
            let quote = inp.mark[(bar, asset)];
            if !quote.is_nan() && quote > 0.0 {
                self.last[asset] = quote;
            }
        }
        let mut value = 0.0f64;
        if inp.record_positions {
            for asset in 0..self.assets {
                let total = self.held(asset);
                if total != 0.0 {
                    value += self.worth(inp, at, asset, total);
                }
                positions[bar * self.assets + asset] = total;
            }
        } else {
            for asset in self.union_names() {
                let total = self.held(asset);
                if total != 0.0 {
                    value += self.worth(inp, at, asset, total);
                }
            }
        }
        reported[EQUITY * steps + bar] = self.cash + value;
        reported[CASH * steps + bar] = self.cash;
        reported[FEES * steps + bar] = self.fees;
        reported[BOUGHT * steps + bar] = self.bought;
        reported[SOLD * steps + bar] = self.sold;
    }
}

/// Insert `i` into an ascending list, if it is not there.
#[inline]
pub fn insert_sorted(names: &mut Vec<usize>, asset: usize) {
    if let Err(at) = names.binary_search(&asset) {
        names.insert(at, asset);
    }
}

/// The union of two ascending lists, ascending.
pub fn merged(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut first, mut second) = (0, 0);
    while first < left.len() && second < right.len() {
        if left[first] < right[second] {
            out.push(left[first]);
            first += 1;
        } else if right[second] < left[first] {
            out.push(right[second]);
            second += 1;
        } else {
            out.push(left[first]);
            first += 1;
            second += 1;
        }
    }
    out.extend_from_slice(&left[first..]);
    out.extend_from_slice(&right[second..]);
    out
}
