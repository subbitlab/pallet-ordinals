use super::*;

#[derive(Eq, PartialEq, Debug)]
pub enum Artifact {
	Cenotaph(Cenotaph),
	Runestone(Runestone),
}

impl Artifact {
	pub fn mint(&self) -> Option<RuneId> {
		match self {
			Self::Cenotaph(cenotaph) => cenotaph.mint,
			Self::Runestone(runestone) => runestone.mint,
		}
	}
}
