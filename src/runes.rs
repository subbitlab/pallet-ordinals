#[derive(Debug, PartialEq)]
pub enum MintError {
	Cap(u128),
	End(u64),
	Start(u64),
	Unmintable,
}

impl sp_std::fmt::Display for MintError {
	fn fmt(&self, f: &mut sp_std::fmt::Formatter<'_>) -> sp_std::fmt::Result {
		match self {
			MintError::Cap(cap) => write!(f, "limited to {cap} mints"),
			MintError::End(end) => write!(f, "mint ended on block {end}"),
			MintError::Start(start) => write!(f, "mint starts on block {start}"),
			MintError::Unmintable => write!(f, "not mintable"),
		}
	}
}
