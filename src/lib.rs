//! The account loop, in Rust, with no Python in it.
//!
//! One walk (`driver`), one account (`account`), and a pluggable policy for
//! the part a market gets to decide — how a decision is funded, sized and
//! filled (`bookkeeping::Bookkeeping`). `Plain` is the framework's own,
//! market-neutral policy. A vendor's policy lives in its own crate and links
//! this one.
//!
//! The walk sees ndarray views (`inputs::Inputs`); where those views come
//! from is a binding's business. The `polars` feature adds `plugin`: the
//! frame-in, frame-out marshalling a polars expression plugin needs, so a
//! plugin crate is one `#[polars_expr]` function naming its policy.
//!
//! Every policy has a pure-Python twin that a test holds bit-for-bit equal
//! (strict IEEE 754 on both sides, no fused multiply-add): `Plain`'s lives in
//! `johnnybt.engine.reference`.

pub mod account;
pub mod bookkeeping;
pub mod driver;
pub mod inputs;
#[cfg(feature = "polars")]
pub mod plugin;
