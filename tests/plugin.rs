//! The frame-in, frame-out marshalling, on the smallest account there is.

#![cfg(feature = "polars")]

use johnnybt_engine::bookkeeping::Plain;
use johnnybt_engine::plugin::{simulate, Firing, SimulateKwargs};
use polars::prelude::*;

fn lists_f64(name: &str, rows: &[Vec<f64>]) -> Series {
    let inner: Vec<Series> = rows
        .iter()
        .map(|row| Series::new("".into(), row.as_slice()))
        .collect();
    Series::new(name.into(), inner)
}

fn lists_bool(name: &str, rows: &[Vec<bool>]) -> Series {
    let inner: Vec<Series> = rows
        .iter()
        .map(|row| Series::new("".into(), row.as_slice()))
        .collect();
    Series::new(name.into(), inner)
}

fn lists_i32(name: &str, rows: &[Vec<i32>]) -> Series {
    let inner: Vec<Series> = rows
        .iter()
        .map(|row| Series::new("".into(), row.as_slice()))
        .collect();
    Series::new(name.into(), inner)
}

#[test]
fn an_account_holding_nothing_keeps_its_capital() {
    let steps = 4;
    let price = vec![vec![10.0]; steps];
    let yes = vec![vec![true]; steps];
    let no = vec![vec![false]; steps];
    let inputs = vec![
        lists_f64("mark", &price),
        lists_f64("price_0", &price),
        lists_bool("buyable_0", &yes),
        lists_bool("sellable_0", &yes),
        lists_bool("impound_0", &no),
        lists_i32("instrument", &vec![vec![0]; steps]),
        Series::new("epoch".into(), vec![0i32; steps]),
        Series::new("new_day".into(), vec![true; steps]),
        lists_f64("plan_0_0", &vec![vec![0.0]; steps]),
    ];
    let mut rates = vec![vec![vec![0.0; 10]; 1]; 8];
    rates[0][0][0] = 100.0; // lot
    rates[3][0][0] = 1.0; // multiplier
    let kwargs = SimulateKwargs {
        points: 1,
        tranches: 1,
        mark: 0,
        previous: None,
        prices: vec![1],
        buyable: Some(vec![2]),
        sellable: Some(vec![3]),
        impound: Some(vec![4]),
        firings: vec![Firing {
            tranche: 0,
            point: 0,
            plan: 8,
            standing: None,
        }],
        instrument: 5,
        epoch: 6,
        new_day: 7,
        rates,
        flags: vec![vec![false; 10], vec![false; 10]],
        capital: 1_000_000.0,
        buffer: 0.97,
        audit: false,
        positions: true,
    };
    let out = simulate::<Plain>(&inputs, &kwargs).expect("walks");
    let fields = out.struct_().expect("a struct");
    let equity = fields.field_by_name("equity").expect("equity");
    assert_eq!(equity.f64().unwrap().get(steps - 1), Some(1_000_000.0));
    let held = fields.field_by_name("position").expect("positions");
    assert_eq!(held.len(), steps);
}
