//! How a decision is funded, sized and filled — the part of an account walk
//! that a market's conventions, or a reference machine's, get to decide.
//!
//! The account (`Account`) owns what every walk shares; a `Bookkeeping`
//! owns the rest and is handed the account at each phase. `Plain` is the
//! framework's own policy, market-neutral: a buy fills only from cash the
//! account has, a refused reduction is cut to what may be sold, nothing is
//! frozen and nothing is owed. A vendor's policy lives in its own crate.

use crate::account::{merged, Account};
use crate::inputs::*;

/// One bookkeeping policy, walked alongside the account.
pub trait Bookkeeping {
    /// Fresh state for one run.
    fn new(tranches: usize, assets: usize) -> Self;

    /// A corporate action scaled every book's units of name `i` by `ratio`;
    /// scale whatever per-name state the policy keeps.
    fn corporate_action(&mut self, acct: &Account, asset: usize, ratio: f64);

    /// A new trading day opens: settlement releases.
    fn new_day(&mut self, acct: &Account);

    /// A bar opens: bar-local state starts empty.
    fn open_bar(&mut self);

    /// Standing orders try again at this point, before anyone decides.
    fn backfill(&mut self, acct: &mut Account, inp: &Inputs, at: Point);

    /// Every tranche that fires at this point sizes and trades.
    fn decide_and_trade(&mut self, acct: &mut Account, inp: &Inputs, at: Point);

    /// The bar closes: unfilled orders expire.
    fn close_bar(&mut self, acct: &mut Account);

    /// Whether the policy keeps nothing for tranche `k`'s name `i` — the
    /// account forgets a name once its own cell and the policy's are at rest.
    fn at_rest(&self, tranche: usize, asset: usize) -> bool;
}

/// The framework's own policy.
///
/// Sells pay into cash and buys draw from it; a buy the cash cannot cover
/// is cut to the lots it can. A buy the market blocks becomes a standing
/// order: its cost — the fill's turnover plus fee, at the size struck — is
/// locked until it fills at a later point of the bar or the bar closes,
/// exactly as a venue holds the money behind an open order. Locked cash
/// is still the account's (it counts in equity); it is just not spendable.
/// Settlement is a quantity per tranche — what the book held when the day
/// opened — and only bites when a tranche decides more than once inside a
/// day (`audit`).
pub struct Plain {
    tranches: usize,
    assets: usize,
    /// `(K, N)` units each tranche may still sell today.
    budget: Vec<f64>,
    /// `(K, N)` cash locked behind each standing buy.
    locked: Vec<f64>,
    /// Everything locked, across the account.
    locked_total: f64,
}

impl Plain {
    #[inline]
    fn at(&self, tranche: usize, asset: usize) -> usize {
        tranche * self.assets + asset
    }

    /// Free one cell's locked cash.
    #[inline]
    fn release(&mut self, cell: usize) {
        if self.locked[cell] > 0.0 {
            self.locked_total -= self.locked[cell];
            self.locked[cell] = 0.0;
        }
    }

    /// What the account may spend: its cash less what its orders hold.
    #[inline]
    fn spendable(&self, acct: &Account) -> f64 {
        acct.cash - self.locked_total
    }

    /// Size one tranche's decision against what it can reach.
    fn size(
        &mut self,
        acct: &mut Account,
        inp: &Inputs,
        tranche: usize,
        decision: usize,
        at: Point,
    ) {
        let run = acct.run;
        let row = acct.row_names(inp, tranche, decision);
        let mut share = 0.0f64;
        for &asset in row.iter() {
            let weight = inp.plan[(run, tranche, decision, asset)];
            if !weight.is_nan() {
                share += weight.abs();
            }
        }
        let reachable = acct.holding(inp, tranche, at) + self.spendable(acct);
        let investable = acct.investable(inp, share, reachable);
        // The row's names and the book's: any other name strikes zero on a
        // zero weight and holds nothing, so its cell is already at rest.
        for asset in merged(&row, acct.named(tranche)) {
            let want = acct.strike(
                inp,
                tranche,
                asset,
                inp.plan[(run, tranche, decision, asset)],
                investable,
                at,
            );
            let cell = acct.at(tranche, asset);
            acct.target[cell] = want;
            // A reduction the day's budget cannot cover is refused in part:
            // it sells what it may, and the book stays between answers.
            let give = acct.qty[cell] - want;
            acct.blocked[cell] =
                inp.audit && acct.qty[cell] > 0.0 && give > self.budget[self.at(tranche, asset)];
            if want != 0.0 || acct.blocked[cell] {
                acct.touch(tranche, asset);
            }
        }
    }

    /// One tranche's sells at this point.
    fn sell(&mut self, acct: &mut Account, inp: &Inputs, tranche: usize, at: Point) {
        let (bar, phase, epoch) = (at.bar, at.phase, at.epoch);
        for idx in 0..acct.named(tranche).len() {
            let asset = acct.named(tranche)[idx];
            let price = inp.price(phase, bar, asset);
            if price.is_nan() || price <= 0.0 {
                continue;
            }
            let cell = acct.at(tranche, asset);
            let mut give = acct.qty[cell] - acct.target[cell];
            if give <= 0.0 || !inp.sellable(phase, bar, asset) {
                continue;
            }
            if acct.blocked[cell] && give > self.budget[cell] {
                give = self.budget[cell];
            }
            // No lot rounding on the way out: a corporate action leaves a
            // fractional count, and flooring every sell strands it forever.
            if give <= 0.0 {
                continue;
            }
            let category = inp.class(asset);
            let turnover = give * price * inp.rate(MULTIPLIER, epoch, category);
            let fee = inp.fee(
                turnover,
                inp.rate(SELL_FEE, epoch, category),
                epoch,
                category,
            );
            acct.qty[cell] -= give;
            if inp.audit && acct.qty[cell] + give > 0.0 {
                self.budget[cell] -= give;
            }
            acct.cash += turnover - fee;
            acct.fees += fee;
            acct.sold += turnover;
        }
    }

    /// What one name's buy costs at this point, cut to what cash covers.
    ///
    /// Returns `(units, turnover, fee)`, units zero when nothing fits.
    fn affordable(
        &self,
        acct: &Account,
        inp: &Inputs,
        tranche: usize,
        asset: usize,
        at: Point,
    ) -> (f64, f64, f64) {
        let (bar, phase, epoch) = (at.bar, at.phase, at.epoch);
        let price = inp.price(phase, bar, asset);
        let cell = acct.at(tranche, asset);
        let mut take = acct.target[cell] - acct.qty[cell];
        if take <= 0.0 || price.is_nan() || price <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let category = inp.class(asset);
        let unit = price * inp.rate(MULTIPLIER, epoch, category);
        let rate = inp.rate(BUY_FEE, epoch, category);
        let lot = inp.rate(LOT, epoch, category);
        // Whole lots the cash covers, fees included; the fee floor can push
        // a full-cash order over by a few yuan, so back off a lot until it
        // fits.
        let spendable = self.spendable(acct);
        let per_unit = unit * (1.0 + rate);
        let mut fitting =
            ((spendable - inp.rate(FIXED_FEE, epoch, category)) / per_unit / lot).floor() * lot;
        if fitting < take {
            take = fitting;
        }
        let mut turnover = take * unit;
        let mut fee = inp.fee(turnover, rate, epoch, category);
        // An infinite target would back off one lot at a time forever.
        while take > 0.0 && take.is_finite() && turnover + fee > spendable {
            fitting -= lot;
            take = fitting;
            turnover = take * unit;
            fee = inp.fee(turnover, rate, epoch, category);
        }
        if take <= 0.0 || acct.qty[cell] + take < inp.rate(MIN_LOT, epoch, category) {
            return (0.0, 0.0, 0.0);
        }
        (take, turnover, fee)
    }

    /// One buy for one name: fill what fits, or lock its cash behind a
    /// standing order when the market blocks it.
    fn buy(&mut self, acct: &mut Account, inp: &Inputs, tranche: usize, asset: usize, at: Point) {
        let (take, turnover, fee) = self.affordable(acct, inp, tranche, asset, at);
        if take <= 0.0 {
            return;
        }
        let cell = acct.at(tranche, asset);
        acct.touch(tranche, asset);
        if !inp.buyable(at.phase, at.bar, asset) {
            let cost = turnover + fee;
            self.locked[cell] = cost;
            self.locked_total += cost;
            return;
        }
        acct.qty[cell] += take;
        if inp.flags[(SAME_BAR, inp.class(asset))] {
            self.budget[cell] += take;
        }
        acct.cash -= turnover + fee;
        acct.fees += fee;
        acct.bought += turnover;
    }
}

impl Bookkeeping for Plain {
    fn new(tranches: usize, assets: usize) -> Self {
        Plain {
            tranches,
            assets,
            budget: vec![0.0; tranches * assets],
            locked: vec![0.0; tranches * assets],
            locked_total: 0.0,
        }
    }

    fn corporate_action(&mut self, _acct: &Account, asset: usize, ratio: f64) {
        for tranche in 0..self.tranches {
            let cell = self.at(tranche, asset);
            if self.budget[cell] != 0.0 {
                self.budget[cell] *= ratio;
            }
        }
    }

    fn new_day(&mut self, acct: &Account) {
        for tranche in 0..self.tranches {
            for &asset in acct.named(tranche) {
                let cell = self.at(tranche, asset);
                self.budget[cell] = acct.qty[cell];
            }
        }
    }

    fn open_bar(&mut self) {}

    fn backfill(&mut self, acct: &mut Account, inp: &Inputs, at: Point) {
        // A standing order tries again where the market lets it: its cash
        // comes back to the account and the fill is checked against the
        // point's own price.
        for tranche in 0..self.tranches {
            for idx in 0..acct.named(tranche).len() {
                let asset = acct.named(tranche)[idx];
                let cell = self.at(tranche, asset);
                if self.locked[cell] <= 0.0 || !inp.buyable(at.phase, at.bar, asset) {
                    continue;
                }
                self.release(cell);
                self.buy(acct, inp, tranche, asset, at);
            }
        }
    }

    fn decide_and_trade(&mut self, acct: &mut Account, inp: &Inputs, at: Point) {
        for tranche in 0..self.tranches {
            let decision = acct.fire[tranche];
            if decision < 0 {
                continue;
            }
            let decision = decision as usize;
            // A fresh decision replaces whatever stood: its orders are
            // withdrawn and their cash is reachable again.
            for idx in 0..acct.named(tranche).len() {
                let cell = self.at(tranche, acct.named(tranche)[idx]);
                self.release(cell);
            }
            self.size(acct, inp, tranche, decision, at);
            self.sell(acct, inp, tranche, at);
            for idx in 0..acct.named(tranche).len() {
                let asset = acct.named(tranche)[idx];
                self.buy(acct, inp, tranche, asset, at);
            }
            acct.remember_attained(inp, tranche, decision);
        }
    }

    fn close_bar(&mut self, acct: &mut Account) {
        // Standing orders expire with the bar; their cash is free again.
        for tranche in 0..self.tranches {
            for idx in 0..acct.named(tranche).len() {
                let cell = self.at(tranche, acct.named(tranche)[idx]);
                self.release(cell);
            }
        }
    }

    fn at_rest(&self, tranche: usize, asset: usize) -> bool {
        let cell = self.at(tranche, asset);
        self.budget[cell] == 0.0 && self.locked[cell] == 0.0
    }
}
