use crate::types::AssetId;
use crate::types::FloatType;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(crate) struct OmnipoolAsset {
    pub asset_id: AssetId,
    pub decimals: u8,
    pub reserve: FloatType,
    pub hub_reserve: FloatType,
    pub fee: FloatType,
    pub protocol_fee: FloatType,
    pub hub_price: FloatType,
}

#[derive(Debug, Clone)]
pub(crate) struct Stablepool {
    pub pool_id: AssetId,
    pub assets: Vec<AssetId>,
    pub reserves: Vec<FloatType>,
    pub shares: FloatType,
    pub d: FloatType,
    pub fee: FloatType,
    pub amplification: u128,
}

impl Stablepool {
    pub fn ann(&self) -> FloatType {
        self.amplification as f64 * self.assets.len() as f64
    }
    pub fn reserve_sum(&self) -> FloatType {
        self.reserves.iter().sum()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AssetInfo {
    pub decimals: u8,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AmmStore {
    pub(crate) omnipool: BTreeMap<AssetId, OmnipoolAsset>,
    pub(crate) stablepools: BTreeMap<AssetId, Stablepool>,
    pub(crate) assets: BTreeMap<AssetId, AssetInfo>,
}

impl AmmStore {
    pub fn exists(&self, asset_id: AssetId) -> bool {
        self.assets.contains_key(&asset_id)
    }
    pub fn asset_info(&self, asset_id: AssetId) -> Option<&AssetInfo> {
        self.assets.get(&asset_id)
    }
}

pub(crate) fn process_data(info: Vec<crate::types::Asset>) -> AmmStore {
    let mut omnipool = BTreeMap::new();
    let mut stablepools: BTreeMap<AssetId, Stablepool> = BTreeMap::new();
    let mut assets = BTreeMap::new();
    for asset in info {
        match asset {
            crate::types::Asset::StableSwap(asset) => {
                let pool_id = asset.pool_id;
                let asset_id = asset.asset_id;
                let decimals = asset.decimals;
                let reserve = asset.reserve as f64 / 10u128.pow(decimals as u32) as f64;
                let fee = asset.fee.0 as f64 / asset.fee.1 as f64;
                let amplification = asset.amplification;

                stablepools
                    .entry(pool_id)
                    .and_modify(|pool| {
                        pool.assets.push(asset_id);
                        pool.reserves.push(reserve);
                    })
                    .or_insert(Stablepool {
                        pool_id,
                        assets: vec![asset_id],
                        reserves: vec![reserve],
                        fee,
                        amplification,
                    });

                assets.insert(asset_id, AssetInfo { decimals });
                assets.insert(pool_id, AssetInfo { decimals: 18 });
            }
            crate::types::Asset::Omnipool(asset) => {
                let asset_id = asset.asset_id;
                let decimals = asset.decimals;
                let reserve = asset.reserve_as_f64();
                let hub_reserve = asset.hub_reserve_as_f64();
                let fee = asset.fee_as_f64();
                let protocol_fee = asset.hub_fee_as_f64();
                let hub_price = if reserve > 0. {
                    hub_reserve / reserve
                } else {
                    0.
                };
                let asset_data = OmnipoolAsset {
                    asset_id,
                    decimals,
                    reserve,
                    hub_reserve,
                    fee,
                    protocol_fee,
                    hub_price,
                };
                omnipool.insert(asset_id, asset_data);
                assets.insert(asset_id, AssetInfo { decimals });
            }
        }
    }
    AmmStore {
        omnipool,
        stablepools,
        assets,
    }
}
