use crate::types::{Asset, AssetId, Intent, OmnipoolAsset, StableSwapAsset};

// Incoming data representation that can be used over FFI if needed
pub type DataRepr = (
    u8,         // omnipool or stableswap
    AssetId,    // asset_id
    u128,       // reserve
    u128,       // hub reserve
    u8,         // decimals
    (u32, u32), // pool fee
    (u32, u32), // omnipool hub_fee
    AssetId,    // stableswap pool_id
    u128,       // stableswap amplification
    u128,       // stableswap shares
    u128,       // stableswap d
);
pub type IntentRepr = (u128, AssetId, AssetId, u128, u128, bool);

pub fn convert_intent_repr(intents: Vec<IntentRepr>) -> Vec<Intent> {
    intents
        .into_iter()
        .map(|v| {
            let (intent_id, asset_in, asset_out, amount_in, amount_out, partial) = v;
            Intent {
                intent_id,
                asset_in,
                asset_out,
                amount_in,
                amount_out,
                partial,
            }
        })
        .collect()
}

pub fn convert_data_repr(data: Vec<DataRepr>) -> Vec<Asset> {
    data.into_iter()
        .map(|v| {
            let (
                c,
                asset_id,
                reserve,
                hub_reserve,
                decimals,
                fee,
                hub_fee,
                pool_id,
                amp,
                shares,
                d,
            ) = v;
            match c {
                0 => Asset::Omnipool(OmnipoolAsset {
                    asset_id,
                    reserve,
                    hub_reserve,
                    decimals,
                    fee,
                    hub_fee,
                }),
                1 => Asset::StableSwap(StableSwapAsset {
                    pool_id,
                    asset_id,
                    reserve,
                    shares,
                    d,
                    decimals,
                    fee,
                    amplification: amp,
                }),
                _ => panic!("unsupported pool asset!"),
            }
        })
        .collect()
}
