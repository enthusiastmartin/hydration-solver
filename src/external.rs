use crate::types::{Asset, AssetId, Intent, OmnipoolAsset, StableSwapAsset};

// Incoming data representation that can be used over FFI if needed
pub type DataRepr = (
    u8,
    AssetId,
    u128,
    u128,
    u8,
    (u32, u32),
    (u32, u32),
    AssetId,
    u128,
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
            let (c, asset_id, reserve, hub_reserve, decimals, fee, hub_fee, pool_id, amp) = v;
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
                    decimals,
                    fee,
                    amplification: amp,
                }),
                _ => panic!("unsupported pool asset!"),
            }
        })
        .collect()
}
