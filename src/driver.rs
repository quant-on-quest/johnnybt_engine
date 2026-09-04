//! Walking the bars: one account, then every account in parallel.

use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

use crate::account::{Account, Fill};
use crate::bookkeeping::Bookkeeping;
use crate::inputs::*;

/// Run one account: every tranche, every bar, under one policy.
pub fn run_account<B: Bookkeeping>(
    run: usize,
    inp: &Inputs,
    reported: &mut [f64],
    positions: &mut [f64],
) -> Vec<Fill> {
    let tranches = inp.plan.shape()[1];
    let assets = inp.plan.shape()[3];
    let phases = inp.points();
    let steps = inp.steps();

    let mut acct = Account::new(run, tranches, assets, inp.capital);
    acct.record_fills = inp.record_fills;
    let mut book = B::new(tranches, assets);
    let mut scaled: Vec<(usize, f64)> = Vec::new();

    for bar in 0..steps {
        let epoch = inp.epoch[bar] as usize;

        acct.mark_previous(inp, bar, &mut scaled);
        for &(asset, ratio) in scaled.iter() {
            book.corporate_action(&acct, asset, ratio);
        }
        // Settlement releases on the day boundary, not the bar boundary.
        if inp.new_day[bar] {
            book.new_day(&acct);
        }
        acct.open_bar();
        book.open_bar();

        // The auction: each price point of the bar, in order.
        for phase in 0..phases {
            let at = Point { bar, phase, epoch };
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
        acct.close_bar(
            inp,
            Point {
                bar,
                phase: phases - 1,
                epoch,
            },
            reported,
            positions,
            steps,
        );
        acct.sweep(|tranche, asset| book.at_rest(tranche, asset));
    }
    acct.fills
}

/// Run every account in parallel, one policy for all.
///
/// Returns `(5, R, T)` reported series, `(R, T, N)` positions (or a
/// `(1, 1, 1)` placeholder when positions are not recorded) and each run's
/// fills in walk order (empty unless recorded).
pub fn run_all<B: Bookkeeping>(inputs: &Inputs) -> (Array3<f64>, Array3<f64>, Vec<Vec<Fill>>) {
    let runs = inputs.plan.shape()[0];
    let steps = inputs.steps();
    let assets = inputs.plan.shape()[3];
    let record_positions = inputs.record_positions;
    let mut reported = Array3::<f64>::zeros((runs, REPORTED, steps));
    let mut positions = if record_positions {
        Array3::<f64>::zeros((runs, steps, assets))
    } else {
        Array3::<f64>::zeros((1, 1, 1))
    };
    let mut fills: Vec<Vec<Fill>> = (0..runs).map(|_| Vec::new()).collect();
    let report_iter = reported.axis_iter_mut(Axis(0)).into_iter();
    if record_positions {
        report_iter
            .zip(positions.axis_iter_mut(Axis(0)))
            .zip(fills.iter_mut())
            .enumerate()
            .par_bridge()
            .for_each(|(run, ((mut rep, mut pos), sink))| {
                *sink = run_account::<B>(
                    run,
                    inputs,
                    rep.as_slice_mut().expect("contiguous"),
                    pos.as_slice_mut().expect("contiguous"),
                );
            });
    } else {
        let mut dummies: Vec<Array2<f64>> = (0..runs).map(|_| Array2::zeros((1, 1))).collect();
        report_iter
            .zip(dummies.iter_mut())
            .zip(fills.iter_mut())
            .enumerate()
            .par_bridge()
            .for_each(|(run, ((mut rep, dummy), sink))| {
                *sink = run_account::<B>(
                    run,
                    inputs,
                    rep.as_slice_mut().expect("contiguous"),
                    dummy.as_slice_mut().expect("contiguous"),
                );
            });
    }
    let mut packed = Array3::<f64>::zeros((REPORTED, runs, steps));
    for run in 0..runs {
        for series in 0..REPORTED {
            for bar in 0..steps {
                packed[(series, run, bar)] = reported[(run, series, bar)];
            }
        }
    }
    (packed, positions, fills)
}
