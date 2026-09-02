//! Frame in, frame out: what a polars expression plugin hands the walk.
//!
//! The plugin is called under `group_by(account).agg(...)`: one group is one
//! account, one row is one bar, and a cell holding the market is a list —
//! every instrument the account's universe covers, in one fixed order. Which
//! column plays which part is said by the kwargs, the market's rules and the
//! account's terms ride along in them, and the answer is one struct per bar.
//! polars runs the groups on its own thread pool; the walk never sees a
//! thread.

use ndarray::{Array1, Array2, Array3, Array4};
use polars::prelude::*;
use serde::Deserialize;

use crate::bookkeeping::Bookkeeping;
use crate::driver::run_all;
use crate::inputs::{Inputs, BOUGHT, CASH, EQUITY, FEES, SOLD};

/// One firing: a tranche's decision at one price point of a bar, given as a
/// list column that is null on the bars where nothing fires.
#[derive(Deserialize, Clone, Debug)]
pub struct Firing {
    pub tranche: usize,
    pub point: usize,
    /// The list column of target weights, by input index.
    pub plan: usize,
    /// A boolean column: the firing acts only on a change; None for never.
    pub standing: Option<usize>,
}

/// What the kwargs say about the inputs and the account.
#[derive(Deserialize, Clone, Debug)]
pub struct SimulateKwargs {
    pub points: usize,
    pub tranches: usize,
    /// Input indices of the list columns, per part.
    pub mark: usize,
    pub previous: Option<usize>,
    pub prices: Vec<usize>,
    pub buyable: Option<Vec<usize>>,
    pub sellable: Option<Vec<usize>>,
    pub impound: Option<Vec<usize>>,
    pub firings: Vec<Firing>,
    /// A list column of class codes (any row; they are the same on every bar).
    pub instrument: usize,
    /// Scalar columns.
    pub epoch: usize,
    pub new_day: usize,
    /// `(rates, epochs, classes)` and `(flags, classes)`, mirroring the rulebook.
    pub rates: Vec<Vec<Vec<f64>>>,
    pub flags: Vec<Vec<bool>>,
    pub capital: f64,
    pub buffer: f64,
    pub audit: bool,
    /// Whether to report the `(N,)` quantities held on each bar as well.
    #[serde(default)]
    pub positions: bool,
}

/// The output field of the expression: one struct per bar.
pub fn output_field(name: &str, positions: bool) -> Field {
    let mut fields: Vec<Field> = ["equity", "cash", "fees", "bought", "sold"]
        .into_iter()
        .map(|f| Field::new(f.into(), DataType::Float64))
        .collect();
    if positions {
        fields.push(Field::new("position".into(), DataType::List(Box::new(DataType::Float64))));
    }
    Field::new(name.into(), DataType::Struct(fields))
}

fn width_of(column: &Series) -> PolarsResult<usize> {
    let list = column.list()?;
    for row in list.amortized_iter() {
        if let Some(inner) = row {
            return Ok(inner.as_ref().len());
        }
    }
    Ok(0)
}

/// Read a `(T, N)` float list column into one contiguous matrix.
///
/// A column with no null rows is one memcpy of its flattened values; a
/// column with nulls (a firing's targets) is read row by row.
fn floats(column: &Series, steps: usize, width: usize) -> PolarsResult<Array2<f64>> {
    let list = column.list()?;
    if list.null_count() == 0 {
        let flat = list.explode(ExplodeOptions { empty_as_null: false, keep_nulls: false })?;
        let values = flat.f64()?.rechunk();
        if let Ok(slice) = values.cont_slice() {
            if slice.len() == steps * width {
                return Array2::from_shape_vec((steps, width), slice.to_vec())
                    .map_err(|e| polars_err!(ComputeError: "list column does not reshape: {e}"));
            }
        }
    }
    let mut out = Array2::<f64>::from_elem((steps, width), f64::NAN);
    for (t, row) in list.amortized_iter().enumerate() {
        if let Some(inner) = row {
            let values = inner.as_ref().f64()?;
            for (i, v) in values.iter().enumerate() {
                out[(t, i)] = v.unwrap_or(f64::NAN);
            }
        }
    }
    Ok(out)
}

/// Read a `(T, N)` boolean list column into one contiguous matrix.
fn booleans(column: &Series, steps: usize, width: usize) -> PolarsResult<Array2<bool>> {
    let list = column.list()?;
    if list.null_count() == 0 {
        let flat = list.explode(ExplodeOptions { empty_as_null: false, keep_nulls: false })?;
        let values = flat.bool()?.rechunk();
        if values.len() == steps * width && values.null_count() == 0 {
            let collected: Vec<bool> = values.downcast_iter().flat_map(|arr| arr.values_iter()).collect();
            return Array2::from_shape_vec((steps, width), collected)
                .map_err(|e| polars_err!(ComputeError: "list column does not reshape: {e}"));
        }
    }
    let mut out = Array2::<bool>::from_elem((steps, width), false);
    for (t, row) in list.amortized_iter().enumerate() {
        if let Some(inner) = row {
            let values = inner.as_ref().bool()?;
            for (i, v) in values.iter().enumerate() {
                out[(t, i)] = v.unwrap_or(false);
            }
        }
    }
    Ok(out)
}

/// Stack `P` `(T, N)` matrices into `(P, T, N)`.
fn stacked_floats(inputs: &[Series], which: &[usize], steps: usize, width: usize) -> PolarsResult<Array3<f64>> {
    let mut out = Array3::<f64>::from_elem((which.len(), steps, width), f64::NAN);
    for (p, &index) in which.iter().enumerate() {
        let plane = floats(&inputs[index], steps, width)?;
        out.index_axis_mut(ndarray::Axis(0), p).assign(&plane);
    }
    Ok(out)
}

fn stacked_booleans(
    inputs: &[Series],
    which: Option<&Vec<usize>>,
    points: usize,
    steps: usize,
    width: usize,
    default: bool,
) -> PolarsResult<Array3<bool>> {
    let mut out = Array3::<bool>::from_elem((points, steps, width), default);
    if let Some(which) = which {
        for (p, &index) in which.iter().enumerate() {
            let plane = booleans(&inputs[index], steps, width)?;
            out.index_axis_mut(ndarray::Axis(0), p).assign(&plane);
        }
    }
    Ok(out)
}

/// Walk one account under policy `B` and answer one struct per bar.
pub fn simulate<B: Bookkeeping>(inputs: &[Series], kwargs: &SimulateKwargs) -> PolarsResult<Series> {
    let steps = inputs[kwargs.mark].len();
    let width = width_of(&inputs[kwargs.mark])?;
    let points = kwargs.points;
    let tranches = kwargs.tranches;
    if kwargs.prices.len() != points {
        polars_bail!(ComputeError: "the kwargs name {} price points and {} price columns", points, kwargs.prices.len());
    }

    let mark = floats(&inputs[kwargs.mark], steps, width)?;
    let has_previous = kwargs.previous.is_some();
    let previous = match kwargs.previous {
        Some(index) => floats(&inputs[index], steps, width)?,
        None => Array2::<f64>::zeros((steps, width)),
    };
    let prices = stacked_floats(inputs, &kwargs.prices, steps, width)?;
    let buyable = stacked_booleans(inputs, kwargs.buyable.as_ref(), points, steps, width, true)?;
    let sellable = stacked_booleans(inputs, kwargs.sellable.as_ref(), points, steps, width, true)?;
    let impound = stacked_booleans(inputs, kwargs.impound.as_ref(), points, steps, width, false)?;

    // Decisions: every firing's non-null rows become plan rows, numbered in
    // bar order per tranche; the firing table points each (bar, point) at
    // its row.
    let mut rows_per_tranche = vec![0usize; tranches];
    let mut firing_rows: Vec<Vec<(usize, usize)>> = Vec::with_capacity(kwargs.firings.len());
    for firing in &kwargs.firings {
        let list = inputs[firing.plan].list()?;
        let mut found = Vec::new();
        for (t, row) in list.amortized_iter().enumerate() {
            if row.is_some() {
                found.push((t, 0));
            }
        }
        firing_rows.push(found);
    }
    // Number the rows: firings of one tranche interleave by bar, then by point.
    let mut order: Vec<(usize, usize, usize, usize)> = Vec::new(); // (tranche, bar, point, firing index)
    for (f, firing) in kwargs.firings.iter().enumerate() {
        for &(t, _) in &firing_rows[f] {
            order.push((firing.tranche, t, firing.point, f));
        }
    }
    order.sort();
    let mut row_of: Vec<Vec<usize>> = firing_rows.iter().map(|rows| vec![0; rows.len()]).collect();
    let mut cursor: Vec<std::collections::HashMap<usize, usize>> = vec![Default::default(); kwargs.firings.len()];
    for (f, rows) in firing_rows.iter().enumerate() {
        for (j, &(t, _)) in rows.iter().enumerate() {
            cursor[f].insert(t, j);
        }
    }
    for &(k, t, _p, f) in &order {
        let j = cursor[f][&t];
        row_of[f][j] = rows_per_tranche[k];
        rows_per_tranche[k] += 1;
    }
    let deepest = rows_per_tranche.iter().copied().max().unwrap_or(0).max(1);
    let mut plan = Array4::<f64>::zeros((1, tranches, deepest, width));
    let mut at = Array4::<i32>::from_elem((1, tranches, steps, points), -1);
    let mut standing = Array4::<bool>::from_elem((1, tranches, steps, points), false);
    for (f, firing) in kwargs.firings.iter().enumerate() {
        let list = inputs[firing.plan].list()?;
        let flags = match firing.standing {
            Some(index) => Some(inputs[index].bool()?.clone()),
            None => None,
        };
        let mut j = 0usize;
        for (t, row) in list.amortized_iter().enumerate() {
            let Some(inner) = row else { continue };
            let d = row_of[f][j];
            j += 1;
            let values = inner.as_ref().f64()?;
            for (i, v) in values.iter().enumerate() {
                plan[(0, firing.tranche, d, i)] = v.unwrap_or(0.0);
            }
            at[(0, firing.tranche, t, firing.point)] = d as i32;
            if let Some(flags) = &flags {
                standing[(0, firing.tranche, t, firing.point)] = flags.get(t).unwrap_or(false);
            }
        }
    }

    // The market's rules, as the walk indexes them.
    let instrument: Array1<i32> = {
        let list = inputs[kwargs.instrument].list()?;
        let mut codes = Array1::<i32>::zeros(width);
        for row in list.amortized_iter() {
            if let Some(inner) = row {
                let values = inner.as_ref().i32()?;
                for (i, v) in values.iter().enumerate() {
                    codes[i] = v.unwrap_or(0);
                }
                break;
            }
        }
        codes
    };
    let epoch: Array1<i32> = inputs[kwargs.epoch].i32()?.iter().map(|v| v.unwrap_or(0)).collect();
    let new_day: Array1<bool> = inputs[kwargs.new_day].bool()?.iter().map(|v| v.unwrap_or(false)).collect();
    let epochs = kwargs.rates.first().map(|e| e.len()).unwrap_or(0);
    let classes = kwargs.flags.first().map(|c| c.len()).unwrap_or(0);
    let mut rates = Array3::<f64>::zeros((kwargs.rates.len(), epochs, classes));
    for (r, by_epoch) in kwargs.rates.iter().enumerate() {
        for (e, by_class) in by_epoch.iter().enumerate() {
            for (c, &v) in by_class.iter().enumerate() {
                rates[(r, e, c)] = v;
            }
        }
    }
    let mut flags = Array2::<bool>::from_elem((kwargs.flags.len(), classes), false);
    for (r, by_class) in kwargs.flags.iter().enumerate() {
        for (c, &v) in by_class.iter().enumerate() {
            flags[(r, c)] = v;
        }
    }

    let run = Inputs {
        plan: plan.view(),
        at: at.view(),
        standing: standing.view(),
        prices: prices.view(),
        mark: mark.view(),
        impound: impound.view(),
        previous: previous.view(),
        has_previous,
        new_day: new_day.view(),
        buyable: buyable.view(),
        sellable: sellable.view(),
        instrument: instrument.view(),
        epoch: epoch.view(),
        rates: rates.view(),
        flags: flags.view(),
        capital: kwargs.capital,
        buffer: kwargs.buffer,
        audit: kwargs.audit,
        record_positions: kwargs.positions,
    };
    let (reported, held) = run_all::<B>(&run);
    let series = |row: usize, name: &str| -> Series {
        let values: Vec<f64> = (0..steps).map(|t| reported[(row, 0, t)]).collect();
        Series::new(name.into(), values)
    };
    let mut fields = vec![
        series(EQUITY, "equity"),
        series(CASH, "cash"),
        series(FEES, "fees"),
        series(BOUGHT, "bought"),
        series(SOLD, "sold"),
    ];
    if kwargs.positions {
        let flat: Vec<f64> = held.iter().copied().collect();
        let quantities = Series::new("position".into(), flat).reshape_list(&[
            ReshapeDimension::Specified(Dimension::new(steps as u64)),
            ReshapeDimension::Specified(Dimension::new(width as u64)),
        ])?;
        fields.push(quantities);
    }
    let out = StructChunked::from_series("simulate".into(), steps, fields.iter())?;
    Ok(out.into_series())
}
