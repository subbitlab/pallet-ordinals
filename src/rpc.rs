
use crate::*;
use bitcoin::hashes::Hash;
use bitcoin::{consensus, Block, BlockHash};
use ordinals::Txid;
use serde::{Deserialize, Serialize};
use sha2::{Digest};
use sp_runtime::offchain::http;
use sp_runtime::offchain::{Duration};
use sp_std::str::FromStr;
use crate::rpc_json::{GetBlockHeaderResult};
use hex::FromHex;
use thiserror_no_std::Error;
use crate::alloc::string::String;
use crate::alloc::string::ToString;
use sp_std::vec::Vec;
use sp_std::vec;
#[derive(Debug, Error)]
pub enum RpcError {
	#[error("IO error occured while calling {0} onto {1} due to {2}.")]
	Io(&'static str, String, String),
	#[error("Decoding response of {0} from {1} failed due to {2}.")]
	Decode(&'static str, String, String),
	#[error("Received an error of endpoint {0} from {1}: {2}.")]
	Endpoint(&'static str, String, String),
}

#[derive(Serialize, Debug, Clone)]
struct Payload {
	pub jsonrpc: &'static str,
	pub id: &'static str,
	pub method: &'static str,
	pub params: serde_json::Value,
}
#[derive(Deserialize, Serialize, Debug)]
struct Reply<R> {
	#[allow(dead_code)]
	pub id: String,
	pub error: Option<ErrorMsg>,
	pub result: Option<R>,
}

#[derive(Deserialize, Serialize, Debug)]
struct ErrorMsg {
	#[allow(dead_code)]
	code: i64,
	message: String,
}

/// [   0..1023] + [1024..2047] + [2048..3071] = 3072
/// [start, end] + [start, end] + [start, end] = total
fn split(end: u64, total: u64, limit: u64) -> (u64, u64) {
	let start = end + 1;
	let end = if start + limit >= total { total - 1 } else { start + limit - 1 };
	(start, end)
}

fn request_payload(endpoint: &'static str, params: serde_json::Value) -> Payload {
	Payload { jsonrpc: "1.0", id: "btc0", method: endpoint, params }
}

pub(crate) fn get_block_hash(url: &str, height: u32) -> Result<BlockHash> {
	let payload = request_payload("getblockhash", serde_json::json!([height]));
	let r = make_rpc::<String>(url, payload)?;
	let hash = BlockHash::from_str(&r).map_err(|e| {
		OrdError::Rpc(RpcError::Decode("getblockhash", url.to_string(), e.to_string()))
	})?;
	Ok(hash)
}

pub(crate) fn get_block_header(url: &str, hash: BlockHash) -> Result<GetBlockHeaderResult> {
	let paypload =
		request_payload("getblockheader", serde_json::json!([alloc::format!("{:x}", hash), true]));
	make_rpc::<GetBlockHeaderResult>(url, paypload)
}

pub fn make_rpc<R>(url: impl ToString, payload: Payload) -> Result<R>
where
	R: for<'a> Deserialize<'a> + sp_std::fmt::Debug,
{
	let deadline = sp_io::offchain::timestamp().add(Duration::from_millis(20000));
	let data = serde_json::to_string(&payload).unwrap();
	let url = url.to_string();
	let request =
		http::Request::post(url.as_str(), vec![data.clone()]).add_header("Content-Type", "application/json");
	let pending = request
		.deadline(deadline)
		.body(vec![data])
		.send()
		.map_err(|_| OrdError::OffchainHttp(http::Error::IoError))?;
	let response = pending
		.try_wait(deadline)
		.map_err(|_| OrdError::OffchainHttp(http::Error::DeadlineReached))?
		.map_err(|e| OrdError::OffchainHttp(e))?;
	if response.code != 200 {
		log::warn!("Unexpected status code: {}", response.code);
		return Err(OrdError::OffchainHttp(http::Error::Unknown));
	}
	let body = response.body().collect::<Vec<u8>>();
	let reply: Reply<R> = serde_json::from_slice(&body).map_err(|e| {
		OrdError::Rpc(RpcError::Decode(payload.method.as_ref(), url.to_string(), e.to_string()))
	})?;
	if reply.error.is_some() {
		return Err(OrdError::Rpc(RpcError::Endpoint(
			payload.method.as_ref(),
			url.to_string(),
			reply.error.map(|e| e.message).unwrap(),
		)));
	}
	reply.result.ok_or(OrdError::Rpc(RpcError::Decode(
		payload.method.as_ref(),
		url.to_string(),
		"No result".to_string(),
	)))
}
pub(crate) fn get_best_block_hash(url: &str) -> Result<BlockHash> {
	let payload = request_payload("getbestblockhash", serde_json::json!([]));
	let r = make_rpc::<String>(url, payload)?;
	let hash = BlockHash::from_str(&r).map_err(|e| {
		OrdError::Rpc(RpcError::Decode("getbestblockhash", url.to_string(), e.to_string()))
	})?;
	Ok(hash)
}

pub(crate) fn
get_block(url: &str, hash: BlockHash) -> Result<Block> {
	let payload = request_payload("getblock", serde_json::json!([alloc::format!("{:x}", hash), 0]));
	let hex: String = make_rpc(url, payload)?;
	use hex::FromHex;
	let hex = <Vec<u8>>::from_hex(hex)
		.map_err(|e| OrdError::Rpc(RpcError::Decode("getblock", url.to_string(), e.to_string())))?;
	consensus::encode::deserialize(&hex)
		.map_err(|e| OrdError::Rpc(RpcError::Decode("getblock", url.to_string(), e.to_string())))
}

#[derive(Debug, Error)]
pub enum OrdError {
	#[error("params: {0}")]
	Params(String),
	#[error("overflow")]
	Overflow,
	#[error("block verification")]
	BlockVerification(u32),
	#[error("index error: {0}")]
	Index(runes::MintError),
	#[error("rpc error: {0}")]
	Rpc(#[from] rpc::RpcError),
	#[error("offchain http error")]
	OffchainHttp(http::Error),
}


pub(crate) type Result<T> = sp_std::result::Result<T, OrdError>;

pub(crate) type OffchainWorkerRpcResult<T> = sp_std::result::Result<T, http::Error>;


