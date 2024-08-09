
use crate::*;
use bitcoin::block::Header;
use bitcoin::{Block, Transaction, Txid};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct BlockData {
	pub header: Header,
	pub txdata: Vec<(Transaction, Txid)>,
}


impl From<Block> for BlockData {
	fn from(block: Block) -> Self {
		BlockData {
			header: block.header,
			txdata: block
				.txdata
				.into_iter()
				.map(|transaction| {
					let txid = transaction.txid();
					(transaction, txid)
				})
				.collect(),
		}
	}
}
