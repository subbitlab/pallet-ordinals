use super::*;

#[derive(Eq, PartialEq, Debug, Default)]
pub struct Cenotaph {
	pub etching: Option<Rune>,
	pub flaw: Option<Flaw>,
	pub mint: Option<RuneId>,
}
