use super::*;
use alloc::string::String;
use codec::{Decode, Encode, MaxEncodedLen};
use core::str::FromStr;
use bitcoin::hashes::Hash;
use scale_info::TypeInfo;

#[derive(
	Copy,
	Clone,
	Debug,
	PartialEq,
	Ord,
	PartialOrd,
	Eq,
	Default,
	Decode,
	Encode,
	TypeInfo,
	MaxEncodedLen,
)]
pub struct SpacedRune {
	pub rune: Rune,
	pub spacers: u32,
}

#[derive(
	Copy,
	Clone,
	Debug,
	PartialEq,
	Ord,
	PartialOrd,
	Eq,
	Default,
	Decode,
	Encode,
	TypeInfo,
	MaxEncodedLen,
)]
pub struct Txid(pub [u8; 32]);

impl From<bitcoin::Txid> for Txid {
	fn from(value: bitcoin::Txid) -> Self {
		let v = value.to_byte_array();
		Txid(v)
	}
}

impl SpacedRune {
	pub fn new(rune: Rune, spacers: u32) -> Self {
		Self { rune, spacers }
	}
}

impl FromStr for SpacedRune {
	type Err = Error;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		let mut rune = String::new();
		let mut spacers = 0u32;

		for c in s.chars() {
			match c {
				'A'..='Z' => rune.push(c),
				'.' | '•' => {
					let flag = 1 << rune.len().checked_sub(1).ok_or(Error::LeadingSpacer)?;
					if spacers & flag != 0 {
						return Err(Error::DoubleSpacer);
					}
					spacers |= flag;
				},
				_ => return Err(Error::Character(c)),
			}
		}

		if 32 - spacers.leading_zeros() >= <usize as TryInto<u32>>::try_into(rune.len()).unwrap() {
			return Err(Error::TrailingSpacer);
		}

		Ok(SpacedRune { rune: rune.parse().map_err(Error::Rune)?, spacers })
	}
}

#[derive(Debug, PartialEq)]
pub enum Error {
	LeadingSpacer,
	TrailingSpacer,
	DoubleSpacer,
	Character(char),
	Rune(rune::Error),
}
