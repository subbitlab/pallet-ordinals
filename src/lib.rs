// We make sure this pallet uses `no_std` for compiling to Wasm.
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod index;
mod rpc;
mod rpc_json;
mod runes;
pub mod weights;

use crate::{
    index::{event::OrdEvent, lot::Lot},
    rpc::{OrdError, Result},
};
use alloc::string::{String, ToString};
use bitcoin::{constants::SUBSIDY_HALVING_INTERVAL, Network};
use frame_support::traits::BuildGenesisConfig;
use ordinals::{Height, Rune, RuneId, Terms};
use sp_runtime::traits::IdentifyAccount;
use sp_runtime::offchain::http;
use sp_std::{boxed::Box, collections::btree_map::BTreeMap, str::FromStr, vec::Vec};
use thiserror_no_std::Error;
use frame_system::offchain::{AppCrypto, CreateSignedTransaction, SendUnsignedTransaction, SignedPayload, Signer, SigningTypes};

pub use pallet::*;
pub use weights::*;

pub const REQUIRED_CONFIRMATIONS: u32 = 4;
pub const FIRST_HEIGHT: u32 = 839999;
pub const FIRST_BLOCK_HASH: &'static str =
    "0000000000000000000172014ba58d66455762add0512355ad651207918494ab";

pub const KEY_TYPE: KeyTypeId = KeyTypeId(*b"ordi");

/// Based on the above `KeyTypeId` we need to generate a pallet-specific crypto type wrappers.
/// We can use from supported crypto kinds (`sr25519`, `ed25519` and `ecdsa`) and augment
/// the types with this pallet-specific identifier.
pub mod sr25519 {
    mod app_sr25519 {
        use super::super::KEY_TYPE;
        use sp_runtime::app_crypto::{app_crypto, sr25519};
        app_crypto!(sr25519, KEY_TYPE);
    }

    sp_application_crypto::with_pair! {
		/// An octopus keypair using sr25519 as its crypto.
		pub type AuthorityPair = app_sr25519::Pair;
	}

    /// An octopus signature using sr25519 as its crypto.
    pub type AuthoritySignature = app_sr25519::Signature;

    /// An octopus identifier using sr25519 as its crypto.
    pub type AuthorityId = app_sr25519::Public;
}

pub mod ecdsa {
    mod app_ecdsa {
        use super::super::KEY_TYPE;
        use sp_runtime::app_crypto::{app_crypto, ecdsa};
        app_crypto!(ecdsa, KEY_TYPE);
    }

    sp_application_crypto::with_pair! {
		/// An octopus keypair using ecdsa as its crypto.
		pub type AuthorityPair = app_ecdsa::Pair;
	}

    /// An octopus signature using ecdsa as its crypto.
    pub type AuthoritySignature = app_ecdsa::Signature;

    /// An octopus identifier using ecdsa as its crypto.
    pub type AuthorityId = app_ecdsa::Public;
}

#[frame_support::pallet]
pub mod pallet {
    use super::*;
    use crate::index::entry::{Entry, OutPointValue, RuneBalance, TxidValue};
    use crate::index::RuneEntry;
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;
    use ordinals::{Artifact, Edict, Etching, Pile, Rune, RuneId, SpacedRune};

    #[pallet::config]
    pub trait Config: CreateSignedTransaction<Call<Self>> + frame_system::Config {
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
        type WeightInfo: WeightInfo;
        type MaxOutPointRuneBalancesLen: Get<u32>;
        type AppCrypto: AppCrypto<Self::Public, Self::Signature>;
        #[pallet::constant]
        type UnsignedPriority: Get<TransactionPriority>;
    }

    #[pallet::genesis_config]
    pub struct GenesisConfig<T> {
        pub phantom: sp_std::marker::PhantomData<T>,
        pub initial_rpc_url: String,
    }

    impl<T: Config> Default for GenesisConfig<T> {
        fn default() -> Self {
            Self {
                phantom: Default::default(),
                initial_rpc_url: "".to_string(),
            }
        }
    }

    #[pallet::storage]
    #[pallet::getter(fn outpoint_to_rune_balances)]
    pub type OutPointRuneBalances<T: Config> = StorageMap<
        _,
        Twox64Concat,
        OutPointValue,
        BoundedVec<RuneBalance, T::MaxOutPointRuneBalancesLen>,
        OptionQuery,
    >;

    #[pallet::storage]
    #[pallet::getter(fn rune_id_to_rune_entry)]
    pub type RuneIdToRuneEntry<T: Config> =
        StorageMap<_, Blake2_128Concat, RuneId, RuneEntry, OptionQuery>;

    #[pallet::storage]
    #[pallet::getter(fn rune_to_rune_id)]
    pub type RuneToRuneId<T: Config> = StorageMap<_, Blake2_128Concat, u128, RuneId, OptionQuery>;

    #[pallet::storage]
    #[pallet::getter(fn transaction_id_to_rune)]
    pub type TransactionIdToRune<T: Config> =
        StorageMap<_, Blake2_128Concat, TxidValue, u128, OptionQuery>;

    #[pallet::storage]
    #[pallet::getter(fn height_to_block_hash)]
    pub type HeightToBlockHash<T: Config> =
        StorageMap<_, Blake2_128Concat, u32, [u8; 32], OptionQuery>;

    #[pallet::storage]
    #[pallet::getter(fn highest_block)]
    pub type HighestHeight<T: Config> = StorageValue<_, (u32, [u8; 32]), OptionQuery>;

    #[pallet::storage]
    #[pallet::getter(fn get_url)]
    pub type RpcUrl<T: Config> = StorageValue<_, String, ValueQuery>;

    #[pallet::genesis_build]
    impl<T: Config> BuildGenesisConfig for GenesisConfig<T> {
        fn build(&self) {
            RpcUrl::<T>::put(self.initial_rpc_url.clone());
            let hash = BlockHash::from_str(FIRST_BLOCK_HASH)
                .expect("valid hash")
                .to_byte_array();
            HeightToBlockHash::<T>::insert(FIRST_HEIGHT, hash.clone());
            HighestHeight::<T>::put((FIRST_HEIGHT, hash));

            let rune = Rune(2055900680524219742);
            let id = RuneId { block: 1, tx: 0 };
            let etching = ordinals::Txid([0u8; 32]);

            RuneToRuneId::<T>::insert(rune.store(), id);

            RuneIdToRuneEntry::<T>::insert(
                id,
                RuneEntry {
                    block: id.block,
                    burned: 0,
                    divisibility: 0,
                    etching,
                    terms: Some(Terms {
                        amount: Some(1),
                        cap: Some(u128::MAX),
                        height: (
                            Some((SUBSIDY_HALVING_INTERVAL * 4).into()),
                            Some((SUBSIDY_HALVING_INTERVAL * 5).into()),
                        ),
                        offset: (None, None),
                    }),
                    mints: 0,
                    premine: 0,
                    spaced_rune: SpacedRune { rune, spacers: 128 },
                    symbol: Some('\u{29C9}'.to_string()),
                    timestamp: 0,
                    turbo: true,
                },
            );

            TransactionIdToRune::<T>::insert(etching.store(), rune.store());
        }
    }

    // The `Pallet` struct serves as a placeholder to implement traits, methods and dispatchables
    // (`Call`s) in this pallet.
    #[pallet::pallet]
    #[pallet::without_storage_info]
    pub struct Pallet<T>(_);

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {}

    #[pallet::error]
    pub enum Error<T> {
        NotInitial,
        BlockDeserialize,
        BlockIndexFailed,
        IllegalBlockHeight,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {

        #[transactional]
        #[pallet::weight(195_000_000)]
        pub fn submit_block(origin: OriginFor<T>,
            block_payload: BlockPayload<T::Public>,
            _signature: T::Signature,
        ) -> DispatchResultWithPostInfo {
            ensure_none(origin)?;
            let (current_height, _) = Self::highest_block().ok_or(Error::<T>::NotInitial)?;
            ensure!(current_height + 1 == block_payload.block_height, Error::<T>::IllegalBlockHeight);
            let block: BlockData = serde_json::from_slice(block_payload.block_bytes.as_slice()).map_err(|_| Error::<T>::BlockDeserialize)?;
            Self::index_block(block_payload.block_height, block).map_err(|_| Error::<T>::BlockIndexFailed)?;
            Ok(().into())
        }

    }

    #[pallet::validate_unsigned]
    impl<T: Config> ValidateUnsigned for Pallet<T> {
        type Call = Call<T>;

        fn validate_unsigned(source: TransactionSource, call: &Self::Call) -> TransactionValidity {
            // Firstly let's check that we call the right function.
            if let Call::submit_block { ref block_payload, ref signature } = call {
                let signature_valid =
                    SignedPayload::<T>::verify::<T::AppCrypto>(block_payload, signature.clone());
                if !signature_valid {
                    return InvalidTransaction::BadProof.into()
                }
                Self::validate_transaction_parameters(
                    block_payload.public.clone().into_account(),
                )
            } else {
                InvalidTransaction::Call.into()
            }
        }


    }

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        fn offchain_worker(block_number: BlockNumberFor<T>) {
            let highest_info = Self::highest_block();
            if highest_info.is_none() {
                return;
            }
            let (height, current) = highest_info.unwrap();

            match Self::get_best_from_rpc() {
                Ok((best, _)) => {
                    log::info!("our best = {}, their best = {}", height, best);
                    if height + REQUIRED_CONFIRMATIONS >= best {
                        return;
                    } else {
                        match Self::get_block(height + 1) {
                            Ok(block) => {
                                let current_hash =
                                    BlockHash::from_slice(current.as_slice()).unwrap();
                                if block.header.prev_blockhash != current_hash {
                                    log::info!(
											"reorg detected! our best = {}({:x}), the new block to be applied {:?}",
											height,
											current_hash,
											block.header
										  );
                                    return;
                                }
                                log::info!("indexing block {:?}", block.header);
                                let b = serde_json::to_vec(&block).unwrap();
                                let result = Signer::<T, T::AppCrypto>::all_accounts()
                                    .send_unsigned_transaction(
                                        |account| crate::BlockPayload {
                                            block_height: height + 1,
                                            block_bytes: b.clone(),
                                            public: account.public.clone(),
                                        },
                                        |payload, signature| Call::submit_block { block_payload: payload, signature },
                                    );

                            }
                            Err(e) => {
                                log::info!("error: {:?}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    log::info!("error: {:?}", e);
                }
            }
        }
    }

    impl<T: Config> Pallet<T> {

        fn validate_transaction_parameters(
            account_id: <T as frame_system::Config>::AccountId,
        ) -> TransactionValidity {
            // Let's make sure to reject transactions from the future.
            let current_block = <frame_system::Pallet<T>>::block_number();
   /*         if (current_block.clone() as u32) < block_number {
                return InvalidTransaction::Future.into()
            }
*/
            ValidTransaction::with_tag_prefix("PalletOrdinals")
                // We set base priority to 2**21 and hope it's included before any other
                // transactions in the pool.
                .priority(T::UnsignedPriority::get())
                // This transaction does not require anything else to go before into the pool.
                //.and_requires()
                // We set the `provides` tag to `account_id`. This makes
                // sure only one transaction produced by current validator will ever
                // get to the transaction pool and will end up in the block.
                // We can still have multiple transactions compete for the same "spot",
                // and the one with higher priority will replace other one in the pool.
                .and_provides(account_id)
                // The transaction is only valid for next 5 blocks. After that it's
                // going to be revalidated by the pool.
                .longevity(10)
                // It's fine to propagate that transaction to other peers, which means it can be
                // created even by nodes that don't produce blocks.
                // Note that sometimes it's better to keep it for yourself (if you are the block
                // producer), since for instance in some schemes others may copy your solution and
                // claim a reward.
                .propagate(true)
                .build()
        }
    }

    use bitcoin::blockdata::transaction::OutPoint;
    use bitcoin::hashes::Hash;
    use bitcoin::{BlockHash, Transaction};
    use frame_support::{pallet, transactional};
    use ordinals::{Runestone, Txid};

    impl<T: Config> Pallet<T> {
        pub(crate) fn increase_height(height: u32, hash: [u8; 32]) {
            HeightToBlockHash::<T>::insert(height, hash.clone());
            HighestHeight::<T>::put((height, hash));
        }

        fn set_beginning_block() {
            let mut hash = [0u8; 32];
            let hex = hex::decode(FIRST_BLOCK_HASH).unwrap();
            hash.copy_from_slice(hex.as_slice());
            Self::increase_height(FIRST_HEIGHT, hash);
        }

        #[allow(dead_code)]
        pub(crate) fn get_etching(txid: Txid) -> Result<Option<SpacedRune>> {
            let Some(rune) = Self::transaction_id_to_rune(Txid::store(txid)) else {
                return Ok(None);
            };
            let id = Self::rune_to_rune_id(rune).unwrap();
            let entry = Self::rune_id_to_rune_entry(id).unwrap();
            Ok(Some(entry.spaced_rune))
        }

        pub(crate) fn get_rune_balances_for_output(
            outpoint: OutPoint,
        ) -> Result<BTreeMap<SpacedRune, Pile>> {
            let balances_vec = Self::outpoint_to_rune_balances(OutPoint::store(outpoint));
            match balances_vec {
                Some(balances) => {
                    let mut result = BTreeMap::new();
                    for rune in balances.iter() {
                        let rune = *rune;

                        let entry = Self::rune_id_to_rune_entry(rune.id).unwrap();

                        result.insert(
                            entry.spaced_rune,
                            Pile {
                                amount: rune.balance,
                                divisibility: entry.divisibility,
                                symbol: entry.symbol.map(|s| {
                                    let v: Vec<char> = s.chars().collect();
                                    v[0]
                                }),
                            },
                        );
                    }
                    Ok(result)
                }
                None => Ok(BTreeMap::new()),
            }
        }

        pub fn init_rune() {
            Self::set_beginning_block();
            let rune = Rune(2055900680524219742);
            let id = RuneId { block: 1, tx: 0 };
            let etching = Txid([0u8; 32]);
            RuneToRuneId::<T>::insert(rune.store(), id);
            RuneIdToRuneEntry::<T>::insert(
                id,
                RuneEntry {
                    block: id.block,
                    burned: 0,
                    divisibility: 0,
                    etching,
                    terms: Some(Terms {
                        amount: Some(1),
                        cap: Some(u128::MAX),
                        height: (
                            Some((SUBSIDY_HALVING_INTERVAL * 4).into()),
                            Some((SUBSIDY_HALVING_INTERVAL * 5).into()),
                        ),
                        offset: (None, None),
                    }),
                    mints: 0,
                    premine: 0,
                    spaced_rune: SpacedRune { rune, spacers: 128 },
                    symbol: Some('\u{29C9}'.to_string()),
                    timestamp: 0,
                    turbo: true,
                },
            );
            TransactionIdToRune::<T>::insert(Txid::store(etching), rune.store());
        }
    }

    use crate::index::updater::BlockData;

    //updater
    impl<T: Config> Pallet<T> {
        pub fn get_block(height: u32) -> Result<BlockData> {
            let url = Self::get_url();
            let hash = rpc::get_block_hash(&url, height)?;
            let block = rpc::get_block(&url, hash)?;
            block
                .check_merkle_root()
                .then(|| BlockData::from(block))
                .ok_or(OrdError::BlockVerification(height))
        }

        fn unallocated(tx: &Transaction) -> Result<BTreeMap<RuneId, crate::index::lot::Lot>> {
            let mut unallocated: BTreeMap<RuneId, crate::index::lot::Lot> = BTreeMap::new();
            for input in &tx.input {
                let key = OutPoint::store(input.previous_output);
                let r = Self::outpoint_to_rune_balances(key.clone());
                OutPointRuneBalances::<T>::remove(key);

                if let Some(balances) = r {
                    for rune in balances.iter() {
                        let rune = *rune;
                        *unallocated.entry(rune.id).or_default() += rune.balance;
                    }
                }
            }
            Ok(unallocated)
        }

        pub fn get_best_from_rpc() -> Result<(u32, BlockHash)> {
            let url = Self::get_url();
            let hash = rpc::get_best_block_hash(&url)?;
            let header = rpc::get_block_header(&url, hash)?;
            Ok((header.height.try_into().expect("usize to u32"), hash))
        }
    }

    impl<T: Config> Pallet<T> {
        pub fn index_block(height: u32, block: BlockData) -> Result<()> {
            let mut updater = RuneUpdater {
                block_time: block.header.time,
                burned: BTreeMap::new(),
                event_handler: None,
                height,
                minimum: Rune::minimum_at_height(Network::Bitcoin, Height(height)),
            };
            for (i, (tx, txid)) in block.txdata.iter().enumerate() {
                let ordinals_txid = ordinals::Txid::from(txid.clone());
                Self::index_runes(&mut updater, u32::try_from(i).unwrap(), tx, ordinals_txid)?;
            }
            Self::update(updater)?;
            Self::increase_height(height, block.header.block_hash().to_byte_array());
            Ok(())
        }

        //runes updater integreate
        pub(super) fn index_runes(
            updater: &mut RuneUpdater,
            tx_index: u32,
            tx: &Transaction,
            txid: Txid,
        ) -> Result<()> {
            let artifact = Runestone::decipher(tx);
            let mut unallocated = Self::unallocated(tx)?;
            let mut allocated: Vec<BTreeMap<RuneId, Lot>> =
                alloc::vec![BTreeMap::new(); tx.output.len()];
            if let Some(artifact) = &artifact {
                if let Some(id) = artifact.mint() {
                    if let Some(amount) = Self::mint(&mut *updater, id)? {
                        *unallocated.entry(id).or_default() += amount;
                        let bitcoin_txid = bitcoin::Txid::from_slice(txid.0.as_slice()).unwrap();
                        if let Some(handler) = &updater.event_handler {
                            handler(crate::index::event::OrdEvent::RuneMinted {
                                block_height: updater.height,
                                txid: bitcoin_txid,
                                rune_id: id,
                                amount: amount.n(),
                            });
                        }
                    }
                }
                let etched = Self::etched(&mut *updater, tx_index, tx, artifact)?;
                if let Artifact::Runestone(runestone) = artifact {
                    if let Some((id, ..)) = etched {
                        *unallocated.entry(id).or_default() +=
                            runestone.etching.unwrap().premine.unwrap_or_default();
                    }

                    for Edict { id, amount, output } in runestone.edicts.iter().copied() {
                        let amount = Lot(amount);

                        // edicts with output values greater than the number of outputs
                        // should never be produced by the edict parser
                        let output = usize::try_from(output).unwrap();
                        assert!(output <= tx.output.len());

                        let id = if id == RuneId::default() {
                            let Some((id, ..)) = etched else {
                                continue;
                            };

                            id
                        } else {
                            id
                        };

                        let Some(balance) = unallocated.get_mut(&id) else {
                            continue;
                        };

                        let mut allocate = |balance: &mut Lot, amount: Lot, output: usize| {
                            if amount > 0 {
                                *balance -= amount;
                                *allocated[output].entry(id).or_default() += amount;
                            }
                        };

                        if output == tx.output.len() {
                            // find non-OP_RETURN outputs
                            let destinations = tx
                                .output
                                .iter()
                                .enumerate()
                                .filter_map(|(output, tx_out)| {
                                    (!tx_out.script_pubkey.is_op_return()).then_some(output)
                                })
                                .collect::<Vec<usize>>();

                            if !destinations.is_empty() {
                                if amount == 0 {
                                    // if amount is zero, divide balance between eligible outputs
                                    let amount = *balance / destinations.len() as u128;
                                    let remainder =
                                        usize::try_from(*balance % destinations.len() as u128)
                                            .unwrap();

                                    for (i, output) in destinations.iter().enumerate() {
                                        allocate(
                                            balance,
                                            if i < remainder { amount + 1 } else { amount },
                                            *output,
                                        );
                                    }
                                } else {
                                    // if amount is non-zero, distribute amount to eligible outputs
                                    for output in destinations {
                                        allocate(balance, amount.min(*balance), output);
                                    }
                                }
                            }
                        } else {
                            // Get the allocatable amount
                            let amount = if amount == 0 {
                                *balance
                            } else {
                                amount.min(*balance)
                            };

                            allocate(balance, amount, output);
                        }
                    }
                }

                if let Some((id, rune)) = etched {
                    Self::create_rune_entry(&mut *updater, txid, artifact, id, rune)?;
                }
            }

            let mut burned: BTreeMap<RuneId, Lot> = BTreeMap::new();

            if let Some(Artifact::Cenotaph(_)) = artifact {
                for (id, balance) in unallocated {
                    *burned.entry(id).or_default() += balance;
                }
            } else {
                let pointer = artifact
                    .map(|artifact| match artifact {
                        Artifact::Runestone(runestone) => runestone.pointer,
                        Artifact::Cenotaph(_) => unreachable!(),
                    })
                    .unwrap_or_default();

                // assign all un-allocated runes to the default output, or the first non
                // OP_RETURN output if there is no default
                if let Some(vout) = pointer
                    .map(|pointer| pointer as usize)
                    .inspect(|&pointer| assert!(pointer < allocated.len()))
                    .or_else(|| {
                        tx.output
                            .iter()
                            .enumerate()
                            .find(|(_vout, tx_out)| !tx_out.script_pubkey.is_op_return())
                            .map(|(vout, _tx_out)| vout)
                    })
                {
                    for (id, balance) in unallocated {
                        if balance > 0 {
                            *allocated[vout].entry(id).or_default() += balance;
                        }
                    }
                } else {
                    for (id, balance) in unallocated {
                        if balance > 0 {
                            *burned.entry(id).or_default() += balance;
                        }
                    }
                }
            }

            // update outpoint balances
            for (vout, balances) in allocated.into_iter().enumerate() {
                if balances.is_empty() {
                    continue;
                }

                // increment burned balances
                if tx.output[vout].script_pubkey.is_op_return() {
                    for (id, balance) in &balances {
                        *burned.entry(*id).or_default() += *balance;
                    }
                    continue;
                }

                // let mut balances = balances.into_iter().collect::<Vec<(RuneId, Lot)>>();

                // Sort balances by id so tests can assert balances in a fixed order
                // balances.sort();

                let bitcoin_txid = bitcoin::Txid::from_slice(txid.0.as_slice()).unwrap();
                let outpoint = OutPoint {
                    txid: bitcoin_txid,
                    vout: vout.try_into().unwrap(),
                };
                let mut vec = Vec::with_capacity(balances.len());

                for (id, balance) in balances {
                    vec.push(RuneBalance {
                        id,
                        balance: balance.0,
                    });
                    if let Some(handler) = &updater.event_handler {
                        let bitcoin_txid = bitcoin::Txid::from_slice(txid.0.as_slice()).unwrap();
                        handler(crate::index::event::OrdEvent::RuneTransferred {
                            outpoint,
                            block_height: updater.height,
                            txid: bitcoin_txid,
                            rune_id: id,
                            amount: balance.0,
                        });
                    }
                }

                //TODO	outpoint_to_rune_balances(|b| b.insert(outpoint.store(), vec).expect("MemoryOverflow"));
            }

            // increment entries with burned runes
            for (id, amount) in burned {
                *updater.burned.entry(id).or_default() += amount;

                if let Some(handler) = &updater.event_handler {
                    let bitcoin_txid = bitcoin::Txid::from_slice(txid.0.as_slice()).unwrap();
                    handler(crate::index::event::OrdEvent::RuneBurned {
                        block_height: updater.height,
                        txid: bitcoin_txid,
                        rune_id: id,
                        amount: amount.n(),
                    });
                }
            }

            Ok(())
        }

        fn create_rune_entry(
            updater: &mut RuneUpdater,
            txid: Txid,
            artifact: &Artifact,
            id: RuneId,
            rune: Rune,
        ) -> Result<()> {
            // crate::rune_to_rune_id(|r| r.insert(rune.store(), id)).expect("MemoryOverflow");
            TransactionIdToRune::<T>::insert(txid.store(), rune.0);

            let entry = match artifact {
                Artifact::Cenotaph(_) => RuneEntry {
                    block: id.block,
                    burned: 0,
                    divisibility: 0,
                    etching: txid,
                    terms: None, //TODO
                    mints: 0,
                    premine: 0,
                    spaced_rune: SpacedRune { rune, spacers: 0 },
                    symbol: None, //TODO
                    timestamp: updater.block_time.into(),
                    turbo: false,
                },
                Artifact::Runestone(Runestone { etching, .. }) => {
                    let Etching {
                        divisibility,
                        terms,
                        premine,
                        spacers,
                        symbol,
                        turbo,
                        ..
                    } = etching.unwrap();

                    RuneEntry {
                        block: id.block,
                        burned: 0,
                        divisibility: divisibility.unwrap_or_default(),
                        etching: txid,
                        terms,
                        mints: 0,
                        premine: premine.unwrap_or_default(),
                        spaced_rune: SpacedRune {
                            rune,
                            spacers: spacers.unwrap_or_default(),
                        },
                        symbol: symbol.map(|c| c.to_string()),
                        timestamp: updater.block_time.into(),
                        turbo,
                    }
                }
            };

            RuneIdToRuneEntry::<T>::insert(id, entry);

            let bitcoin_txid = bitcoin::Txid::from_slice(txid.0.as_slice()).unwrap();
            match &updater.event_handler {
                Some(handler) => handler(crate::index::event::OrdEvent::RuneEtched {
                    block_height: updater.height,
                    txid: bitcoin_txid,
                    rune_id: id,
                }),
                None => {}
            }
            Ok(())
        }

        fn etched(
            updater: &mut RuneUpdater,
            tx_index: u32,
            _tx: &Transaction,
            artifact: &Artifact,
        ) -> Result<Option<(RuneId, Rune)>> {
            let rune = match artifact {
                Artifact::Runestone(runestone) => match runestone.etching {
                    Some(etching) => etching.rune,
                    None => return Ok(None),
                },
                Artifact::Cenotaph(cenotaph) => match cenotaph.etching {
                    Some(rune) => Some(rune),
                    None => return Ok(None),
                },
            };

            let rune = if let Some(rune) = rune {
                if rune < updater.minimum || rune.is_reserved()
                // || crate::rune_to_rune_id(|r| r.get(&rune.0).is_some())
                // || !Self::tx_commits_to_rune(tx, rune).await?
                {
                    return Ok(None);
                }
                rune
            } else {
                Rune::reserved(updater.height.into(), tx_index)
            };

            Ok(Some((
                RuneId {
                    block: updater.height.into(),
                    tx: tx_index,
                },
                rune,
            )))
        }

        fn mint(updater: &mut RuneUpdater, id: RuneId) -> Result<Option<Lot>> {
            let Some(mut rune_entry) = Self::rune_id_to_rune_entry(&id) else {
                return Ok(None);
            };
            let Ok(amount) = rune_entry.mintable(updater.height.into()) else {
                return Ok(None);
            };
            rune_entry.mints += 1;
            RuneIdToRuneEntry::<T>::insert(id, rune_entry);
            Ok(Some(Lot(amount)))
        }

        pub fn update(updater: RuneUpdater) -> Result<()> {
            for (rune_id, burned) in updater.burned {
                let mut entry = Self::rune_id_to_rune_entry(rune_id.clone()).unwrap();
                entry.burned = entry.burned.checked_add(burned.n()).unwrap();
                RuneIdToRuneEntry::<T>::insert(rune_id, entry);
            }
            Ok(())
        }
    }
}

pub(crate) struct RuneUpdater {
    pub(crate) block_time: u32,
    pub(crate) burned: BTreeMap<RuneId, Lot>,
    pub(crate) event_handler: Option<Box<dyn Fn(OrdEvent)>>,
    pub(crate) height: u32,
    pub(crate) minimum: Rune,
}

use codec::{Encode, Decode};
use scale_info::TypeInfo;
use sp_core::crypto::KeyTypeId;

#[derive(Clone, Debug, Encode, Decode, TypeInfo, Eq, PartialEq)]
pub struct BlockPayload<Public> {
    pub block_height: u32,
    pub block_bytes: Vec<u8>,
    pub public: Public,
}

impl<T: SigningTypes> SignedPayload<T>
for BlockPayload<T::Public>
{
    fn public(&self) -> T::Public {
        self.public.clone()
    }
}
