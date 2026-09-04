//! Frame in, frame out: what a polars expression plugin hands the walk.
//!
//! The plugin is called under `group_by(account).agg(...)`: one group is one
//! account, one row is one bar, and a cell holding the market is a list —
//! every instrument the account's universe covers, in one fixed order. Which
//! column plays which part is said by the kwargs, the market's rules and the
//! account's terms ride along in them, and the answer is one struct per bar.
//! polars runs the groups on its own thread pool; the walk never sees a
//! thread.

use ndarray::{Array1, Array2, Array3, Array4, ArrayView2};
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
    /// Whether every fill is answered, as per-bar lists.
    #[serde(default)]
    pub fills: bool,
}

/// The output field of the expression: one struct per bar.
/// The fill columns, one list per bar, the lists of a bar in step:
/// tranche, asset and point as `Int32`, units, price and fee as `Float64`.
pub const FILL_COLUMNS: [&str; 6] = [
    "fill_tranche",
    "fill_asset",
    "fill_point",
    "fill_units",
    "fill_price",
    "fill_fee",
];

pub fn output_field(name: &str, positions: bool, fills: bool) -> Field {
    let mut fields: Vec<Field> = ["equity", "cash", "fees", "bought", "sold"]
        .into_iter()
        .map(|firing| Field::new(firing.into(), DataType::Float64))
        .collect();
    if positions {
        fields.push(Field::new(
            "position".into(),
            DataType::List(Box::new(DataType::Float64)),
        ));
    }
    if fills {
        for (index, column) in FILL_COLUMNS.iter().enumerate() {
            let inner = if index < 3 {
                DataType::Int32
            } else {
                DataType::Float64
            };
            fields.push(Field::new((*column).into(), DataType::List(Box::new(inner))));
        }
    }
    Field::new(name.into(), DataType::Struct(fields))
}

/// Lay one run's fills out as the six per-bar list columns.
fn fill_columns(fills: &[crate::account::Fill], steps: usize) -> Vec<Series> {
    let mut tranche = ListPrimitiveChunkedBuilder::<Int32Type>::new(
        FILL_COLUMNS[0].into(),
        steps,
        fills.len(),
        DataType::Int32,
    );
    let mut asset = ListPrimitiveChunkedBuilder::<Int32Type>::new(
        FILL_COLUMNS[1].into(),
        steps,
        fills.len(),
        DataType::Int32,
    );
    let mut point = ListPrimitiveChunkedBuilder::<Int32Type>::new(
        FILL_COLUMNS[2].into(),
        steps,
        fills.len(),
        DataType::Int32,
    );
    let mut units = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        FILL_COLUMNS[3].into(),
        steps,
        fills.len(),
        DataType::Float64,
    );
    let mut price = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        FILL_COLUMNS[4].into(),
        steps,
        fills.len(),
        DataType::Float64,
    );
    let mut fee = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        FILL_COLUMNS[5].into(),
        steps,
        fills.len(),
        DataType::Float64,
    );
    // Fills come in walk order, so each bar's are one contiguous stretch.
    let mut cursor = 0usize;
    for bar in 0..steps {
        let start = cursor;
        while cursor < fills.len() && fills[cursor].bar as usize == bar {
            cursor += 1;
        }
        let slice = &fills[start..cursor];
        tranche.append_slice(&slice.iter().map(|fill| fill.tranche).collect::<Vec<i32>>());
        asset.append_slice(&slice.iter().map(|fill| fill.asset as i32).collect::<Vec<i32>>());
        point.append_slice(&slice.iter().map(|fill| fill.point as i32).collect::<Vec<i32>>());
        units.append_slice(&slice.iter().map(|fill| fill.units).collect::<Vec<f64>>());
        price.append_slice(&slice.iter().map(|fill| fill.price).collect::<Vec<f64>>());
        fee.append_slice(&slice.iter().map(|fill| fill.fee).collect::<Vec<f64>>());
    }
    vec![
        tranche.finish().into_series(),
        asset.finish().into_series(),
        point.finish().into_series(),
        units.finish().into_series(),
        price.finish().into_series(),
        fee.finish().into_series(),
    ]
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

/// A `(T, N)` float plane: the frame's own flattened values when the
/// column has no null row (no copy), or a matrix read row by row.
enum Plane {
    Flat(Float64Chunked),
    Owned(Array2<f64>),
}

impl Plane {
    fn view(&self, steps: usize, width: usize) -> PolarsResult<ArrayView2<'_, f64>> {
        match self {
            Plane::Flat(values) => {
                let slice = values
                    .cont_slice()
                    .map_err(|error| polars_err!(ComputeError: "not contiguous: {error}"))?;
                ArrayView2::from_shape((steps, width), slice).map_err(
                    |error| polars_err!(ComputeError: "list column does not reshape: {error}"),
                )
            }
            Plane::Owned(matrix) => Ok(matrix.view()),
        }
    }
}

/// Read a `(T, N)` float list column.
fn floats(column: &Series, steps: usize, width: usize) -> PolarsResult<Plane> {
    let list = column.list()?;
    if list.null_count() == 0 {
        let flat = list.explode(ExplodeOptions {
            empty_as_null: false,
            keep_nulls: false,
        })?;
        let values = flat.f64()?.rechunk().into_owned();
        if values.null_count() == 0 && values.len() == steps * width && values.cont_slice().is_ok()
        {
            return Ok(Plane::Flat(values));
        }
    }
    let mut out = Array2::<f64>::from_elem((steps, width), f64::NAN);
    for (bar, row) in list.amortized_iter().enumerate() {
        if let Some(inner) = row {
            let values = inner.as_ref().f64()?;
            for (asset, value) in values.iter().enumerate() {
                out[(bar, asset)] = value.unwrap_or(f64::NAN);
            }
        }
    }
    Ok(Plane::Owned(out))
}

/// Read a `(T, N)` boolean list column into one matrix, bits unpacked in one pass.
fn booleans(column: &Series, steps: usize, width: usize) -> PolarsResult<Array2<bool>> {
    let list = column.list()?;
    let mut out = Array2::<bool>::from_elem((steps, width), false);
    if list.null_count() == 0 {
        let flat = list.explode(ExplodeOptions {
            empty_as_null: false,
            keep_nulls: false,
        })?;
        let values = flat.bool()?.rechunk();
        if values.len() == steps * width && values.null_count() == 0 {
            let cells = out.as_slice_mut().expect("fresh matrix is contiguous");
            for (cell, bit) in cells
                .iter_mut()
                .zip(values.downcast_iter().flat_map(|arr| arr.values_iter()))
            {
                *cell = bit;
            }
            return Ok(out);
        }
    }
    for (bar, row) in list.amortized_iter().enumerate() {
        if let Some(inner) = row {
            let values = inner.as_ref().bool()?;
            for (asset, value) in values.iter().enumerate() {
                out[(bar, asset)] = value.unwrap_or(false);
            }
        }
    }
    Ok(out)
}

/// One boolean plane per point, or `default` everywhere when the kwargs name none.
fn boolean_planes(
    inputs: &[Series],
    which: Option<&Vec<usize>>,
    points: usize,
    steps: usize,
    width: usize,
    default: bool,
) -> PolarsResult<Vec<Array2<bool>>> {
    match which {
        Some(which) => which
            .iter()
            .map(|&index| booleans(&inputs[index], steps, width))
            .collect(),
        None => Ok((0..points)
            .map(|_| Array2::<bool>::from_elem((steps, width), default))
            .collect()),
    }
}

/// Walk one account under policy `B` and answer one struct per bar.
pub fn simulate<B: Bookkeeping>(
    inputs: &[Series],
    kwargs: &SimulateKwargs,
) -> PolarsResult<Series> {
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
        None => Plane::Owned(Array2::<f64>::zeros((steps, width))),
    };
    let prices: Vec<Plane> = kwargs
        .prices
        .iter()
        .map(|&index| floats(&inputs[index], steps, width))
        .collect::<PolarsResult<_>>()?;
    let buyable = boolean_planes(inputs, kwargs.buyable.as_ref(), points, steps, width, true)?;
    let sellable = boolean_planes(inputs, kwargs.sellable.as_ref(), points, steps, width, true)?;
    let impound = boolean_planes(inputs, kwargs.impound.as_ref(), points, steps, width, false)?;

    // Decisions: every firing's non-null rows become plan rows, numbered in
    // bar order per tranche; the firing table points each (bar, point) at
    // its row.
    let mut rows_per_tranche = vec![0usize; tranches];
    let mut firing_rows: Vec<Vec<(usize, usize)>> = Vec::with_capacity(kwargs.firings.len());
    for firing in &kwargs.firings {
        let list = inputs[firing.plan].list()?;
        let mut found = Vec::new();
        for (bar, row) in list.amortized_iter().enumerate() {
            if row.is_some() {
                found.push((bar, 0));
            }
        }
        firing_rows.push(found);
    }
    // Number the rows: firings of one tranche interleave by bar, then by point.
    let mut order: Vec<(usize, usize, usize, usize)> = Vec::new(); // (tranche, bar, point, firing index)
    for (index, firing) in kwargs.firings.iter().enumerate() {
        for &(bar, _) in &firing_rows[index] {
            order.push((firing.tranche, bar, firing.point, index));
        }
    }
    order.sort();
    let mut row_of: Vec<Vec<usize>> = firing_rows.iter().map(|rows| vec![0; rows.len()]).collect();
    let mut cursor: Vec<std::collections::HashMap<usize, usize>> =
        vec![Default::default(); kwargs.firings.len()];
    for (firing, rows) in firing_rows.iter().enumerate() {
        for (row, &(bar, _)) in rows.iter().enumerate() {
            cursor[firing].insert(bar, row);
        }
    }
    for &(tranche, bar, _p, firing) in &order {
        let row = cursor[firing][&bar];
        row_of[firing][row] = rows_per_tranche[tranche];
        rows_per_tranche[tranche] += 1;
    }
    let deepest = rows_per_tranche.iter().copied().max().unwrap_or(0).max(1);
    let mut plan = Array4::<f64>::zeros((1, tranches, deepest, width));
    let mut at = Array4::<i32>::from_elem((1, tranches, steps, points), -1);
    let mut standing = Array4::<bool>::from_elem((1, tranches, steps, points), false);
    for (index, firing) in kwargs.firings.iter().enumerate() {
        let list = inputs[firing.plan].list()?;
        let flags = match firing.standing {
            Some(index) => Some(inputs[index].bool()?.clone()),
            None => None,
        };
        let mut fired = 0usize;
        for (bar, row) in list.amortized_iter().enumerate() {
            let Some(inner) = row else { continue };
            let decision = row_of[index][fired];
            fired += 1;
            let values = inner.as_ref().f64()?;
            let mut target = plan.slice_mut(ndarray::s![0, firing.tranche, decision, ..]);
            match values.cont_slice() {
                Ok(slice) if slice.len() == width => {
                    target
                        .as_slice_mut()
                        .expect("plan row is contiguous")
                        .copy_from_slice(slice);
                }
                _ => {
                    for (asset, value) in values.iter().enumerate() {
                        target[asset] = value.unwrap_or(0.0);
                    }
                }
            }
            at[(0, firing.tranche, bar, firing.point)] = decision as i32;
            if let Some(flags) = &flags {
                standing[(0, firing.tranche, bar, firing.point)] = flags.get(bar).unwrap_or(false);
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
                for (asset, value) in values.iter().enumerate() {
                    codes[asset] = value.unwrap_or(0);
                }
                break;
            }
        }
        codes
    };
    let epoch: Array1<i32> = inputs[kwargs.epoch]
        .i32()?
        .iter()
        .map(|value| value.unwrap_or(0))
        .collect();
    let new_day: Array1<bool> = inputs[kwargs.new_day]
        .bool()?
        .iter()
        .map(|value| value.unwrap_or(false))
        .collect();
    let epochs = kwargs.rates.first().map(|error| error.len()).unwrap_or(0);
    let classes = kwargs
        .flags
        .first()
        .map(|category| category.len())
        .unwrap_or(0);
    let mut rates = Array3::<f64>::zeros((kwargs.rates.len(), epochs, classes));
    for (run, by_epoch) in kwargs.rates.iter().enumerate() {
        for (error, by_class) in by_epoch.iter().enumerate() {
            for (category, &value) in by_class.iter().enumerate() {
                rates[(run, error, category)] = value;
            }
        }
    }
    let mut flags = Array2::<bool>::from_elem((kwargs.flags.len(), classes), false);
    for (run, by_class) in kwargs.flags.iter().enumerate() {
        for (category, &value) in by_class.iter().enumerate() {
            flags[(run, category)] = value;
        }
    }

    let run = Inputs {
        plan: plan.view(),
        at: at.view(),
        standing: standing.view(),
        prices: prices
            .iter()
            .map(|plane| plane.view(steps, width))
            .collect::<PolarsResult<_>>()?,
        mark: mark.view(steps, width)?,
        impound: impound.iter().map(|plane| plane.view()).collect(),
        previous: previous.view(steps, width)?,
        has_previous,
        new_day: new_day.view(),
        buyable: buyable.iter().map(|plane| plane.view()).collect(),
        sellable: sellable.iter().map(|plane| plane.view()).collect(),
        instrument: instrument.view(),
        epoch: epoch.view(),
        rates: rates.view(),
        flags: flags.view(),
        capital: kwargs.capital,
        buffer: kwargs.buffer,
        audit: kwargs.audit,
        record_positions: kwargs.positions,
        record_fills: kwargs.fills,
    };
    let (reported, held, fills) = run_all::<B>(&run);
    let series = |row: usize, name: &str| -> Series {
        let values: Vec<f64> = (0..steps).map(|bar| reported[(row, 0, bar)]).collect();
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
    if kwargs.fills {
        fields.extend(fill_columns(&fills[0], steps));
    }
    let out = StructChunked::from_series("simulate".into(), steps, fields.iter())?;
    Ok(out.into_series())
}
