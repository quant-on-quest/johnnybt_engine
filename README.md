# johnnybt_engine

An account walk, as a Rust library. It takes what a portfolio should hold on
each bar and reports what it actually held, what it paid, and what it was
worth — the path-dependent half of a backtest, the half that cannot be
vectorised.

There is no Python here. Two crates put a face on it: a polars expression
plugin marshals frames in and out (`polars` feature), and a bookkeeping crate
supplies a market's fill conventions.

## What it is

```
decisions ──► Account (phases every market shares) ──► equity, positions, fills
                  └── Bookkeeping (how a decision is funded, sized, filled)
```

The kernel knows nothing about any particular market. `Account` owns the
phases every walk shares — mark, corporate actions, move refused names into
the disposal pool, retry the pool at each price point, snapshot equity, value
at the close. How a decision turns into fills is a `Bookkeeping`
implementation: `Plain`, included here, is market-neutral (a buy fills only
from cash on hand, a refused reduction is cut to what may be sold, nothing is
frozen and nothing is owed). A different market, or a different broker's
arithmetic, is a different implementation in its own crate.

**Decisions, not rebalances.** The execution price is a `(point, bar, asset)`
stack; what to do at each point is a small integer table (`at`), and what to
hold is `plan`. An order stands from its point until the next decision
replaces it. Refining the price grid — five intraday marks, minutes, ticks —
grows the integer table, not the plan, so the expensive axis (assets) is paid
for once per decision.

## Performance

Measured on a 3,035-bar × 5,459-asset window, five tranches, three price
points: the walk is linear in the width of the names actually in play, about
4 ns per cell. Nine tenths of a dense walk is scanning zeros, so each book
keeps an ascending set of live names (held, targeted, refused, or carrying
policy state) and every phase visits only those. Sums still run in ascending
name order and add only non-zero terms, which keeps the result bit-identical
to the dense walk.

`rayon` parallelises across runs; within a run the walk is serial, because it
is a recurrence — each bar's cash is the previous bar's answer.

## Building

```bash
cargo test                      # the kernel
cargo test --features polars    # plus the frame marshalling
```

## Licence

MIT. See [LICENSE](LICENSE).
