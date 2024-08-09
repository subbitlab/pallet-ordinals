# pallet-ordinals


This project builds a utility substrate pallet to index RUNE UTXOs of bitcoin. It allows any parachains or substrate solochains to index RUNE UTXOs in their runtime using offchain worker.

## Features

- Initiate JSON-RPC requests to any bitcoin RPC server by offchain worker.
- Index and view bitcoin RUNE UTXOs in substrate chains.
- Validate each blocks from the RPC servers.

## Usage

Simply include this pallet to your runtime and set an URL indicates the bitcoin RPC server.

``` rust
    #[pallet::config]
    pub trait Config: CreateSignedTransaction<Call<Self>> + frame_system::Config {
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
        type WeightInfo: WeightInfo;
        type MaxOutPointRuneBalancesLen: Get<u32>;
        type AppCrypto: AppCrypto<Self::Public, Self::Signature>;
        #[pallet::constant]
        type UnsignedPriority: Get<TransactionPriority>;
    }
```

## License
This project is licensed under [LICENSE](MIT). The directory `crates` is originally from the [ord](https://github.com/ordinals/ord) which is licensed under [CC0](https://github.com/ordinals/ord/blob/master/LICENSE).
