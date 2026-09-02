//! Walking the bars: one account, then every account in parallel.

use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

use crate::account::Account;
use crate::bookkeeping::Bookkeeping;
use crate::inputs::*;

/// Run one account: every tranche, every bar, under one policy.
pub fn run_account<B: Bookkeeping>(r: usize, inp: &Inputs, reported: &mut [f64], positions: &mut [f64]) {
    let tranches = inp.plan.shape()[1];
    let n = inp.plan.shape()[3];
    let phases = inp.prices.shape()[0];
    let steps = inp.prices.shape()[1];

    let mut acct = Account::new(r, tranches, n, inp.capital);
    let mut book = B::new(tranches, n);
    let mut scaled: Vec<(usize, f64)> = Vec::new();

    for t in 0..steps {
        let e = inp.epoch[t] as usize;

        acct.mark_previous(inp, t, &mut scaled);
        for &(i, ratio) in scaled.iter() {
            book.corporate_action(&acct, i, ratio);
        }
        // Settlement releases on the day boundary, not the bar boundary.
        if inp.new_day[t] {
            book.new_day(&acct);
        }
        acct.open_bar();
        book.open_bar();

        // The auction: each price point of the bar, in order.
        for phase in 0..phases {
            let at = Point { t, phase, e };
            acct.mark_point(inp, at);
            let deciding = acct.resolve_firings(inp, at);
            acct.pool_sell(inp, at);
            book.backfill(&mut acct, inp, at);
            if deciding {
                acct.snapshot_equity(inp, at);
            }
            book.decide_and_trade(&mut acct, inp, at);
        }

        book.close_bar(&mut acct);
        acct.close_bar(inp, Point { t, phase: phases - 1, e }, reported, positions, steps);
        acct.sweep(|k, i| book.at_rest(k, i));
    }
}

/// Run every account in parallel, one policy for all.
///
/// Returns `(5, R, T)` reported series and `(R, T, N)` positions (or a
/// `(1, 1, 1)` placeholder when positions are not recorded).
pub fn run_all<B: Bookkeeping>(inputs: &Inputs) -> (Array3<f64>, Array3<f64>) {
    let runs = inputs.plan.shape()[0];
    let steps = inputs.prices.shape()[1];
    let n = inputs.plan.shape()[3];
    let record_positions = inputs.record_positions;

    let mut reported = Array3::<f64>::zeros((runs, REPORTED, steps));
    let mut positions = if record_positions {
        Array3::<f64>::zeros((runs, steps, n))
    } else {
        Array3::<f64>::zeros((1, 1, 1))
    };

    let report_iter = reported.axis_iter_mut(Axis(0)).into_iter();
    if record_positions {
        report_iter
            .zip(positions.axis_iter_mut(Axis(0)))
            .enumerate()
            .par_bridge()
            .for_each(|(r, (mut rep, mut pos))| {
                run_account::<B>(
                    r,
                    inputs,
                    rep.as_slice_mut().expect("contiguous"),
                    pos.as_slice_mut().expect("contiguous"),
                );
            });
    } else {
        let mut dummies: Vec<Array2<f64>> = (0..runs).map(|_| Array2::zeros((1, 1))).collect();
        report_iter
            .zip(dummies.iter_mut())
            .enumerate()
            .par_bridge()
            .for_each(|(r, (mut rep, dummy))| {
                run_account::<B>(
                    r,
                    inputs,
                    rep.as_slice_mut().expect("contiguous"),
                    dummy.as_slice_mut().expect("contiguous"),
                );
            });
    }

    // Reported comes back as (5, runs, steps) for the caller's indexing.
    let mut packed = Array3::<f64>::zeros((REPORTED, runs, steps));
    for r in 0..runs {
        for s in 0..REPORTED {
            for t in 0..steps {
                packed[(s, r, t)] = reported[(r, s, t)];
            }
        }
    }
    (packed, positions)
}
