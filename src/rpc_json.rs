use serde::Deserialize;


#[derive(Clone, PartialEq, Debug,Deserialize)]
pub struct GetBlockHeaderResult {
	pub height: usize,
}
