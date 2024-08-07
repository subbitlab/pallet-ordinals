use super::*;
use codec::{Decode, Encode, MaxEncodedLen};
use scale_info::TypeInfo;

#[derive(
	Default,
	Debug,
	PartialEq,
	Copy,
	Clone,
	Eq,
	Decode,
	Encode,
	TypeInfo,
	MaxEncodedLen,
)]
pub struct Terms {
	pub amount: Option<u128>,
	pub cap: Option<u128>,
	pub height: (Option<u64>, Option<u64>),
	pub offset: (Option<u64>, Option<u64>),
}
