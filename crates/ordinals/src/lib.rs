//! Types for interoperating with ordinals, inscriptions, and runes.
#![allow(clippy::large_enum_variant)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

use {
	alloc::format,
	alloc::string::String,
	alloc::string::ToString,
	bitcoin::{
		consensus::{Decodable, Encodable},
		constants::{
			COIN_VALUE, DIFFCHANGE_INTERVAL, MAX_SCRIPT_ELEMENT_SIZE, SUBSIDY_HALVING_INTERVAL,
		},
		opcodes,
		script::{self, Instruction},
		Network, OutPoint, ScriptBuf, Transaction,
	},
	core2::io,
	derive_more::{Display, FromStr},
	serde::{Deserialize, Serialize},
	sp_std::{
		cmp,
		collections::{btree_map::BTreeMap as HashMap, vec_deque::VecDeque},
		fmt::{self, Display, Formatter},
		num::{ParseFloatError, ParseIntError},
		ops::{Add, AddAssign, Sub},
		str::FromStr,
		vec::Vec,
	},
	thiserror_no_std::Error,
};

pub use {
	artifact::Artifact, cenotaph::Cenotaph, charm::Charm, decimal_sat::DecimalSat, degree::Degree,
	edict::Edict, epoch::Epoch, etching::Etching, flaw::Flaw, height::Height, pile::Pile,
	rarity::Rarity, rune::Rune, rune_id::RuneId, runestone::Runestone, sat::Sat,
	sat_point::SatPoint, spaced_rune::SpacedRune, spaced_rune::Txid, terms::Terms,
};

pub const CYCLE_EPOCHS: u32 = 6;

fn default<T: Default>() -> T {
	Default::default()
}

mod artifact;
mod cenotaph;
mod charm;
mod decimal_sat;
mod degree;
mod edict;
mod epoch;
mod etching;
mod flaw;
mod height;
mod pile;
mod rarity;
mod rune;
mod rune_id;
mod runestone;
mod sat;
mod sat_point;
mod spaced_rune;
mod terms;
pub mod varint;
