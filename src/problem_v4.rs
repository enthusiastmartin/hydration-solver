use crate::constants::{DEFAULT_PROFIT_TOKEN, HUB_ASSET_ID};
use crate::internal::{AmmStore, OmnipoolAsset};
use crate::to_f64_by_decimals;
use crate::types::{Asset, AssetId, FloatType, Intent, IntentId};
use anyhow::{anyhow, Result};
use clarabel::solver::{
    ExponentialConeT, NonnegativeConeT, SolverStatus, SupportedConeT, ZeroConeT,
};
use float_next_after::NextAfter;
use ndarray::{concatenate, s, Array1, Array2, ArrayBase, Axis, Ix1, OwnedRepr};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Neg;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum ProblemStatus {
    NotSolved,
    Solved,
    PrimalInfeasible,
    DualInfeasible,
    InsufficientProgress,
    NumericalError,
}

impl From<SolverStatus> for ProblemStatus {
    fn from(value: SolverStatus) -> Self {
        match value {
            SolverStatus::Solved => ProblemStatus::Solved,
            SolverStatus::AlmostSolved => ProblemStatus::Solved,
            SolverStatus::PrimalInfeasible => ProblemStatus::PrimalInfeasible,
            SolverStatus::DualInfeasible => ProblemStatus::DualInfeasible,
            SolverStatus::Unsolved => ProblemStatus::NotSolved,
            SolverStatus::InsufficientProgress => ProblemStatus::InsufficientProgress,
            //SolverStatus::NumericalError => ProblemStatus::NumericalError,
            _ => panic!("Unexpected solver status {:?}", value),
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct ICEProblemV4 {
    pub tkn_profit: AssetId,
    pub intent_ids: Vec<IntentId>,
    pub intents: Vec<Intent>,
    pub intent_amounts: Vec<(FloatType, FloatType)>,

    pub amm_store: AmmStore,

    pub all_asset_ids: Vec<AssetId>,
    pub omnipool_asset_ids: Vec<AssetId>,
    pub sigmas: Vec<usize>,
    pub auxiliaries: Vec<usize>,

    pub sigma_sum: usize,

    pub n: usize,           // number of omnipool assets
    pub asset_count: usize, // number of all assetsa ( N in python)
    pub m: usize,           // number of partial intents
    pub r: usize,           // number of full intents
    pub s: usize,           // number of stablepools /or other type of pools
    pub u: usize,           // Sum of auxiliaries

    pub o: Array2<FloatType>,
    pub rho: Array2<FloatType>,
    pub psi: Array2<FloatType>,
    pub share_indices: Vec<usize>,

    pub min_partial: FloatType,

    pub indicators: Option<Vec<usize>>,

    pub trading_asset_ids: Vec<AssetId>,
    pub partial_sell_maxs: Vec<FloatType>,
    pub initial_sell_maxs: Vec<FloatType>,
    pub partial_indices: Vec<usize>,
    pub full_indices: Vec<usize>,

    pub omnipool_directional_flags: Option<BTreeMap<AssetId, i8>>,
    pub amm_directional_flags: Option<BTreeMap<AssetId, i8>>,
    pub force_omnipool_approx: Option<BTreeMap<AssetId, AmmApprox>>,
    pub force_amm_approx: Option<Vec<Vec<AmmApprox>>>,

    pub last_omnipool_deltas: Option<BTreeMap<AssetId, FloatType>>,
    pub last_amm_deltas: Option<Vec<Vec<FloatType>>>,

    pub step_params: StepParams,
    pub fee_match: FloatType,
}

// Problem builder
impl ICEProblemV4 {
    pub fn new() -> Self {
        ICEProblemV4 {
            tkn_profit: DEFAULT_PROFIT_TOKEN,
            min_partial: 1.,
            fee_match: 0.,
            ..Default::default()
        }
    }

    pub fn with_intents(mut self, intents: Vec<Intent>) -> Self {
        self.intents = intents;
        self
    }

    pub fn with_amm_store(mut self, amm_store: AmmStore) -> Self {
        self.amm_store = amm_store;
        self
    }
}

impl ICEProblemV4 {
    pub fn prepare(&mut self) -> Result<()> {
        if self.intents.len() == 0 {
            return Err(anyhow!("Ice Problem: no intents provided"));
        }
        //Ensure tkn profit is omnipool asset
        if self.amm_store.omnipool.get(&self.tkn_profit).is_none() {
            return Err(anyhow!("Ice Problem: tkn profit is not omnipool asset"));
        }

        let intents_len = self.intents.len();

        let mut intents = Vec::with_capacity(intents_len);
        let mut intent_ids = Vec::with_capacity(intents_len);
        let mut intent_amounts = Vec::with_capacity(intents_len);
        let mut partial_sell_amounts = Vec::new();
        let mut partial_indices = Vec::new();
        let mut full_indices = Vec::new();
        let mut trading_tkns = BTreeSet::new();

        let asset_profit = DEFAULT_PROFIT_TOKEN;
        trading_tkns.insert(asset_profit);

        for (idx, intent) in self.intents.iter().enumerate() {
            let asset_in_info = self.amm_store.asset_info(intent.asset_in).ok_or(anyhow!(
                "Invalid intent - unknown token {:?}",
                intent.asset_in
            ))?;
            let asset_out_info = self.amm_store.asset_info(intent.asset_out).ok_or(anyhow!(
                "Invalid intent - unknown token {:?}",
                intent.asset_out
            ))?;
            intent_ids.push(intent.intent_id);
            let mut intent = intent.clone();

            let amount_in = to_f64_by_decimals!(intent.amount_in, asset_in_info.decimals);
            let amount_out = to_f64_by_decimals!(intent.amount_out, asset_out_info.decimals);


            let buy_amt_lrna_value = amount_out * self.price(HUB_ASSET_ID, intent.asset_out);
            let sell_amt_lrna_value = amount_in * self.price(HUB_ASSET_ID, intent.asset_in);
            
            if buy_amt_lrna_value < 1. && sell_amt_lrna_value < 1.{
                intent.partial = false;
            }

            intents.push(intent.clone());
            intent_amounts.push((amount_in, amount_out));

            if intent.partial {
                partial_indices.push(idx);
                partial_sell_amounts.push(amount_in);
            } else {
                full_indices.push(idx);
            }
            if intent.asset_in != HUB_ASSET_ID {
                trading_tkns.insert(intent.asset_in);
            }
            if intent.asset_out != HUB_ASSET_ID {
                //note: this should never happened, as it is not allowed to buy lrna!
                trading_tkns.insert(intent.asset_out);
            } else {
                return Err(anyhow!("It is not allowed to buy HUB_ASSET_ID!"));
            }
        }

        let mut sigmas = Vec::new();
        let mut ausiliaries = Vec::new();

        for stablepool in self.amm_store.stablepools.iter() {
            sigmas.push(stablepool.assets.len() + 1);
            ausiliaries.push(stablepool.assets.len() + 1);
        }

        let n = self.amm_store.omnipool.keys().count();
        let m = partial_indices.len();
        let r = full_indices.len();
        let s = self.amm_store.stablepools.len();
        let u = ausiliaries.iter().sum();
        let asset_count = self.amm_store.assets.len();
        let sigma_sum = sigmas.iter().sum();

        // this comes from the initial solution which we skipped,
        // so we intened to resolve all full intents
        //TODO: this should take input as init indicators and if set, do something - check python
        let indicators = None;

        let initial_sell_maxs = partial_sell_amounts.clone();

        self.intent_ids = intent_ids;
        self.intent_amounts = intent_amounts;
        self.n = n;
        self.m = m;
        self.r = r;
        self.s = s;
        self.u = u;
        self.asset_count = asset_count;
        self.sigma_sum = sigma_sum;
        self.sigmas = sigmas;
        self.auxiliaries = ausiliaries;
        self.indicators = indicators;
        self.trading_asset_ids = trading_tkns.into_iter().collect();
        self.partial_sell_maxs = partial_sell_amounts;
        self.initial_sell_maxs = initial_sell_maxs;
        self.partial_indices = partial_indices;
        self.full_indices = full_indices;
        self.omnipool_directional_flags = None;
        self.force_amm_approx = None;
        self.step_params = StepParams::default();
        self.all_asset_ids = self.amm_store.assets.keys().cloned().collect();
        self.omnipool_asset_ids = self.amm_store.omnipool.keys().cloned().collect();

        self.set_indicator_matrices();
        Ok(())
    }

    pub fn set_indicator_matrices(&mut self) {
        let mut o = Array2::<f64>::zeros((self.asset_count, self.n));
        let mut rho = Array2::<f64>::zeros((self.asset_count, self.sigma_sum));
        let mut psi = Array2::<f64>::zeros((self.asset_count, self.sigma_sum));

        for (i, &tkn) in self.all_asset_ids.iter().enumerate() {
            if let Some(j) = self.omnipool_asset_ids.iter().position(|&x| x == tkn) {
                o[(i, j)] = 1.0;
            }
        }

        let mut share_indices = Vec::new();
        let mut offset = 0;

        for amm in &self.amm_store.stablepools {
            if let Some(i) = self.all_asset_ids.iter().position(|&x| x == amm.pool_id) {
                share_indices.push(offset);
                rho[(i, offset)] = 1.0;

                for (k, &tkn) in amm.assets.iter().enumerate() {
                    if let Some(i) = self.all_asset_ids.iter().position(|&x| x == tkn) {
                        psi[(i, offset + k + 1)] = 1.0;
                    }
                }
                offset += amm.assets.len() + 1;
            }
        }

        self.o = o;
        self.rho = rho;
        self.psi = psi;
        self.share_indices = share_indices;
    }

    pub(crate) fn get_partial_intent_prices(&self) -> Vec<FloatType> {
        let mut prices = Vec::new();
        for &idx in self.partial_indices.iter() {
            let (amount_in, amount_out) = self.intent_amounts[idx];
            let price = amount_out / amount_in; //TODO: division by zero?!!
            prices.push(price);
        }
        prices
    }
}

impl ICEProblemV4 {
    pub(crate) fn get_partial_intents_amounts(&self) -> Vec<(FloatType, FloatType)> {
        self.partial_indices
            .iter()
            .map(|&idx| self.intent_amounts[idx])
            .collect()
    }

    pub(crate) fn get_full_intents_amounts(&self) -> Vec<(FloatType, FloatType)> {
        self.full_indices
            .iter()
            .map(|&idx| self.intent_amounts[idx])
            .collect()
    }

    pub fn get_full_intents(&self) -> Vec<&Intent> {
        self.full_indices
            .iter()
            .map(|&idx| &self.intents[idx])
            .collect()
    }

    pub fn get_partial_intents(&self) -> Vec<&Intent> {
        self.partial_indices
            .iter()
            .map(|&idx| &self.intents[idx])
            .collect()
    }

    pub(crate) fn get_omnipool_approx(&self, asset_id: AssetId) -> AmmApprox {
        if let Some(approx) = self.force_omnipool_approx.as_ref() {
            *approx.get(&asset_id).unwrap_or(&AmmApprox::None)
        } else {
            AmmApprox::None
        }
    }

    pub(crate) fn scale_obj_amt(&self, amt: FloatType) -> FloatType {
        let scaling = self.get_scaling();
        amt * scaling[&self.tkn_profit]
    }

    pub(crate) fn get_epsilon_tkn(&self) -> BTreeMap<AssetId, FloatType> {
        let mut r = BTreeMap::new();
        for asset_id in self.omnipool_asset_ids.iter() {
            let max_in = self.get_max_in()[&asset_id];
            let max_out = self.get_max_out()[&asset_id];
            let liquidity = self.get_asset_pool_data(*asset_id).reserve;
            let epsilon = max_in.abs().max(max_out.abs()) / liquidity;
            r.insert(*asset_id, epsilon);
        }
        r
    }

    pub fn get_amm_approx(&self, amm_idx: usize) -> Vec<AmmApprox> {
        if let Some(approx) = self.force_amm_approx.as_ref() {
            approx[amm_idx].clone()
        } else {
            panic!("No amm approx found!");
        }
    }

    pub fn get_c(&self) -> Array1<FloatType> {
        self.step_params._C.as_ref().cloned().unwrap()
    }

    pub fn get_b(&self) -> Array1<FloatType> {
        self.step_params._B.as_ref().cloned().unwrap()
    }

    pub fn get_s(&self) -> Vec<FloatType> {
        self.step_params._S.as_ref().cloned().unwrap()
    }
}

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum Direction {
    Sell,
    Buy,
    Both,
    Neither,
}

impl ICEProblemV4 {
    pub(crate) fn get_omnipool_directions(&self) -> BTreeMap<AssetId, Direction> {
        self.step_params
            .omnipool_directions
            .as_ref()
            .unwrap()
            .clone()
    }
    pub fn get_directions(&self) -> (BTreeMap<AssetId, Direction>, Vec<Vec<Direction>>) {
        (
            self.step_params
                .omnipool_directions
                .as_ref()
                .unwrap()
                .clone(),
            self.step_params.amm_directions.as_ref().unwrap().clone(),
        )
    }

    pub fn get_tkn_liquidity(&self, tkn: AssetId) -> FloatType {
        if self.is_omnipool_asset(tkn) {
            return self.amm_store.omnipool.get(&tkn).unwrap().reserve;
        } else {
            panic!("Not implemented yet!");
            //self.amm_store.stablepools.get(&tkn).unwrap().reserves[0]
        }
    }
    pub fn get_lrna_liquidity(&self, tkn: AssetId) -> FloatType {
        if self.is_omnipool_asset(tkn) {
            return self.amm_store.omnipool.get(&tkn).unwrap().hub_reserve;
        } else {
            panic!("Not omnipool asset!");
        }
    }
}

impl ICEProblemV4 {
    pub(crate) fn get_q(&self) -> Vec<FloatType> {
        self.step_params.q.as_ref().cloned().unwrap()
    }
    pub(crate) fn get_profit_A(&self) -> Array2<FloatType> {
        self.step_params.profit_a.as_ref().cloned().unwrap()
    }
    pub(crate) fn get_omnipool_asset_coefs(&self) -> &BTreeMap<AssetId, FloatType> {
        self.step_params.omnipool_asset_coefs.as_ref().unwrap()
    }
    pub(crate) fn get_omnipool_lrna_coefs(&self) -> &BTreeMap<AssetId, FloatType> {
        self.step_params.omnipool_lrna_coefs.as_ref().unwrap()
    }

    pub fn get_real_x(&self, x: Vec<FloatType>) -> Vec<FloatType> {
        let n = self.n;
        let m = self.m;
        let r = self.r;
        let N = self.asset_count;
        let sigma = self.sigma_sum;
        let s = self.s;
        let u = self.u;
        assert!(x.len() == 4 * n + 2 * sigma + u + m || x.len() == 4 * n + 2 * sigma + m + r);

        let scaling = self.get_scaling();
        let scaled_yi: Vec<FloatType> = (0..n).map(|i| x[i] * scaling[&1u32]).collect(); // Assuming 1u32 represents 'LRNA'
        let scaled_xi: Vec<FloatType> = self
            .omnipool_asset_ids
            .iter()
            .enumerate()
            .map(|(i, &tkn)| x[n + i] * scaling[&tkn])
            .collect();
        let scaled_lrna_lambda: Vec<FloatType> =
            (0..n).map(|i| x[2 * n + i] * scaling[&1u32]).collect();
        let scaled_lambda: Vec<FloatType> = self
            .omnipool_asset_ids
            .iter()
            .enumerate()
            .map(|(i, &tkn)| x[3 * n + i] * scaling[&tkn])
            .collect();

        let S_array = Array1::from_vec(self.get_s().clone());
        let x_scaling = (&self.rho + &self.psi).t().dot(&S_array);
        let mut scaled_x: Vec<FloatType> = x[4 * n..4 * n + sigma]
            .iter()
            .zip(x_scaling.iter())
            .map(|(x, scaling)| x * scaling)
            .collect();
        let scaled_l: Vec<FloatType> = x[4 * n + sigma..4 * n + 2 * sigma]
            .iter()
            .zip(x_scaling.iter())
            .map(|(x, scaling)| x * scaling)
            .collect();
        let mut scaled_d: Vec<FloatType> = Vec::new();
        let mut scaled_i: Vec<FloatType> = Vec::new();
        if x.len() == 4 * n + 2 * sigma + m + r {
            scaled_d = self
                .partial_indices
                .iter()
                .enumerate()
                .map(|(j, &idx)| x[4 * n + 2 * sigma + j] * scaling[&self.intents[idx].asset_in])
                .collect();
            scaled_i = (0..r).map(|l| x[4 * n + m + l]).collect();
        } else {
            scaled_d = self
                .partial_indices
                .iter()
                .enumerate()
                .map(|(j, &idx)| {
                    x[4 * n + 2 * sigma + u + j] * scaling[&self.intents[idx].asset_in]
                })
                .collect();
            let scaled_a: Vec<FloatType> = x[4 * n + 2 * sigma..4 * n + 2 * sigma + u]
                .iter()
                .map(|&a| a)
                .collect();
            scaled_i = Vec::new();
            scaled_x.extend(scaled_a);
        }
        let mut scaled_x = [
            scaled_yi,
            scaled_xi,
            scaled_lrna_lambda,
            scaled_lambda,
            scaled_x.clone(),
            scaled_l,
            scaled_d,
            scaled_i,
        ];
        scaled_x.concat()
    }

    pub fn get_scaled_x(&self, x: Vec<FloatType>) -> Vec<FloatType> {
        let n = self.n;
        let m = self.m;
        let r = self.r;
        let sigma = self.sigma_sum;
        assert!(x.len() == 4 * n + 3 * sigma + m || x.len() == 4 * n + 3 * sigma + m + r);

        let scaling = self.get_scaling();
        let scaled_yi: Vec<FloatType> = (0..n).map(|i| x[i] / scaling[&1u32]).collect(); // Assuming 1u32 represents 'LRNA'
        let scaled_xi: Vec<FloatType> = self
            .omnipool_asset_ids
            .iter()
            .enumerate()
            .map(|(i, &tkn)| x[n + i] / scaling[&tkn])
            .collect();
        let scaled_lrna_lambda: Vec<FloatType> =
            (0..n).map(|i| x[2 * n + i] / scaling[&1u32]).collect();
        let scaled_lambda: Vec<FloatType> = self
            .omnipool_asset_ids
            .iter()
            .enumerate()
            .map(|(i, &tkn)| x[3 * n + i] / scaling[&tkn])
            .collect();

        let S_array = Array1::from_vec(self.get_s().clone());
        let stableswap_scalars = (&self.rho + &self.psi).t().dot(&S_array);
        let scaled_X: Vec<FloatType> = x[4 * n..4 * n + sigma]
            .iter()
            .zip(stableswap_scalars.iter())
            .map(|(x, scaling)| x / scaling)
            .collect();
        let scaled_L: Vec<FloatType> = x[4 * n + sigma..4 * n + 2 * sigma]
            .iter()
            .zip(stableswap_scalars.iter())
            .map(|(x, scaling)| x / scaling)
            .collect();
        let scaled_a: Vec<FloatType> = x[4 * n + 2 * sigma..4 * n + 3 * sigma]
            .iter()
            .map(|&a| a)
            .collect();
        let scaled_d: Vec<FloatType> = self
            .partial_indices
            .iter()
            .enumerate()
            .map(|(j, &idx)| x[4 * n + 3 * sigma + j] / scaling[&self.intents[idx].asset_in])
            .collect();
        if x.len() == 4 * n + 3 * sigma + m + r {
            let scaled_I = (0..r).map(|l| x[4 * n + m + l]).collect::<Vec<FloatType>>();
            let mut scaled_x = vec![
                scaled_yi,
                scaled_xi,
                scaled_lrna_lambda,
                scaled_lambda,
                scaled_X,
                scaled_L,
                scaled_a,
                scaled_d,
                scaled_I,
            ];
            scaled_x.concat()
        } else {
            let mut scaled_x = vec![
                scaled_yi,
                scaled_xi,
                scaled_lrna_lambda,
                scaled_lambda,
                scaled_X,
                scaled_L,
                scaled_a,
                scaled_d,
            ];
            scaled_x.concat()
        }
    }
}
impl ICEProblemV4 {
    pub(crate) fn get_leftover_bounds(
        &self,
        allow_loss: bool,
        indices_to_keep: Option<Vec<usize>>,
    ) -> (Array2<FloatType>, Array1<FloatType>) {
        let k = 4 * self.n + 2 * self.sigma_sum + self.u + self.m;
        let profit_a = self.get_profit_A();
        let a3 = profit_a.slice(s![.., ..k]);
        let mut a3 = a3.neg();
        let i_coefs = profit_a.slice(s![.., k..]);
        let mut i_coefs = i_coefs.neg();
        if allow_loss {
            let profit_i = self
                .all_asset_ids
                .iter()
                .position(|&x| x == self.tkn_profit)
                .unwrap();
            let idx = profit_i + 1;
            let m = a3.nrows();
            assert!(idx < m, "Index out of bounds");
            let indices: Vec<_> = (0..m).filter(|&i| i != idx).collect();
            a3 = a3.select(Axis(0), &indices);
            i_coefs = i_coefs.select(Axis(0), &indices);
        }
        let a3_trimmed = if let Some(indices) = indices_to_keep {
            //a3.slice(s![.., indices]).to_owned()
            a3.select(Axis(1), &indices)
        } else {
            a3.to_owned()
        };
        let b3 = if self.r == 0 {
            Array1::zeros(a3_trimmed.shape()[0])
        } else {
            let indicators = self.get_indicators().unwrap();
            let i_array = Array1::from_iter(indicators.iter().map(|&x| x as f64));
            i_coefs.dot(&i_array).neg()
        };
        (a3_trimmed, b3)
    }
}

impl ICEProblemV4 {
    pub fn get_stableswap_bounds(
        &self,
        indices_to_keep: Option<&[usize]>,
    ) -> (Array2<f64>, Array1<f64>, Vec<SupportedConeT<FloatType>>) {
        let n = self.n;
        let sigma = self.sigma_sum;
        let u = self.u;
        let m = self.m;
        let k = 4 * n + 2 * sigma + u + m;

        // Initialize A5 and b5
        let mut A5 = Array2::zeros((0, k));
        let mut b5 = Array1::from_vec(vec![]);
        let mut cones5 = Vec::new();

        let C = self.get_c();
        let B = self.get_b();
        let share_indices = self.share_indices.clone();

        for (j, amm) in self.amm_store.stablepools.iter().enumerate() {
            // Check if amm is StableSwapPoolState (in Rust, enforced by type)
            let l = share_indices[j];
            let ann = amm.ann();
            let s0 = amm.shares;
            let D0 = amm.d;
            let n_amm = amm.assets.len();
            let sum_assets: f64 = amm.reserve_sum();
            let approx = self.get_amm_approx(j);
            let D0_prime = D0 * (1.0 - 1.0 / ann);
            let denom = sum_assets - D0_prime;

            // Shares constraint
            let (mut A5j, mut b5j, cone5j) = if approx[0] == AmmApprox::Linear {
                let mut A5j = Array2::zeros((1, k));
                A5j[[0, 4 * n + 2 * sigma + l]] = 1.0; // a_{j,0}
                A5j[[0, 4 * n + l]] = (1.0 + D0_prime / denom) * C[l] / s0; // delta_s
                for t in 1..=n_amm {
                    A5j[[0, 4 * n + l + t]] = -B[l + t] / denom; // delta_x_i
                }
                let b5j = Array1::from_vec(vec![0.0]);
                (A5j, b5j, ZeroConeT(1))
            } else {
                let mut A5j = Array2::zeros((3, k));
                let mut b5j = Array1::zeros(3);
                // x = a_{j,0}
                A5j[[0, 4 * n + 2 * sigma + l]] = -1.0;
                // y = 1 + C_jS_j / s_0
                A5j[[1, 4 * n + l]] = -C[l] / s0;
                b5j[1] = 1.0;
                // z = An^n / D_0 sum(x_i^0 + B_i X_i) + (1 - An^n)(1 + C_jS_j / s_0)
                A5j[[2, 4 * n + l]] = D0_prime * C[l] / denom / s0;
                for t in 1..=n_amm {
                    A5j[[2, 4 * n + l + t]] = -B[l + t] / denom;
                }
                b5j[2] = 1.0;
                (A5j, b5j, ExponentialConeT())
            };
            cones5.push(cone5j);

            // Asset constraints
            for t in 1..=n_amm {
                let x0 = amm.reserves[t - 1];
                let (A5jt, b5jt, cone5jt) = if approx[t] == AmmApprox::Linear {
                    let mut A5jt = Array2::zeros((1, k));
                    A5jt[[0, 4 * n + 2 * sigma + l + t]] = 1.0; // a_{j,t}
                    A5jt[[0, 4 * n + l]] = C[l] / s0; // delta_s
                    A5jt[[0, 4 * n + l + t]] = -B[l + t] / x0; // delta_x_i
                    let b5jt = Array1::from_vec(vec![0.0]);
                    (A5jt, b5jt, ZeroConeT(1))
                } else {
                    let mut A5jt = Array2::zeros((3, k));
                    let mut b5jt = Array1::zeros(3);
                    // x = a_{j,t}
                    A5jt[[0, 4 * n + 2 * sigma + l + t]] = -1.0;
                    // y = 1 + C_jS_j / s_0
                    A5jt[[1, 4 * n + l]] = -C[l] / s0;
                    b5jt[1] = 1.0;
                    // z = (x_t^0 + B_t X_t) / D_0
                    A5jt[[2, 4 * n + l + t]] = -B[l + t] / x0;
                    b5jt[2] = 1.0;
                    (A5jt, b5jt, ExponentialConeT())
                };
                cones5.push(cone5jt);
                A5j =
                    concatenate(Axis(0), &[A5j.view(), A5jt.view()]).expect("Failed to stack A5j");
                b5j =
                    concatenate(Axis(0), &[b5j.view(), b5jt.view()]).expect("Failed to concat b5j");
            }

            // Final constraint
            let mut A5j_final = Array2::zeros((1, k));
            for t in 0..=n_amm {
                A5j_final[[0, 4 * n + 2 * sigma + l + t]] = -1.0;
            }
            let b5j_final = Array1::from_vec(vec![0.0]);
            cones5.push(NonnegativeConeT(1));

            // Stack all parts
            A5 = concatenate(Axis(0), &[A5.view(), A5j.view(), A5j_final.view()])
                .expect("Failed to stack A5");
            b5 = concatenate(Axis(0), &[b5.view(), b5j.view(), b5j_final.view()])
                .expect("Failed to concat b5");
        }

        // Apply indices_to_keep if provided
        let A5_final = if let Some(indices) = indices_to_keep {
            A5.select(Axis(1), indices)
        } else {
            A5
        };

        (A5_final, b5, cones5)
    }
}

#[derive(Clone)]
pub struct SetupParams {
    pub indicators: Option<Vec<usize>>,
    pub omnipool_flags: Option<BTreeMap<AssetId, i8>>,
    pub amm_flags: Option<BTreeMap<AssetId, i8>>,
    pub sell_maxes: Option<Vec<FloatType>>,
    pub force_omnipool_approx: Option<BTreeMap<AssetId, AmmApprox>>,
    #[deprecated]
    pub force_amm_approx: Option<BTreeMap<AssetId, AmmApprox>>,
    pub force_amm_approx_vec: Option<Vec<Vec<AmmApprox>>>,
    pub rescale: bool,
    pub clear_sell_maxes: bool,
    pub clear_indicators: bool,
    pub clear_omnipool_approx: bool,
    pub clear_amm_approx: bool,

    pub omnipool_deltas: Option<BTreeMap<AssetId, FloatType>>,
    pub amm_deltas: Option<Vec<Vec<FloatType>>>,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum AmmApprox {
    Linear,
    Quadratic,
    Full,
    None,
}

impl SetupParams {
    pub fn new() -> Self {
        SetupParams {
            indicators: None,
            omnipool_flags: None,
            amm_flags: None,
            sell_maxes: None,
            force_omnipool_approx: None,
            force_amm_approx: None,
            force_amm_approx_vec: None,
            rescale: true,
            clear_sell_maxes: true,
            clear_indicators: true,
            clear_omnipool_approx: true,
            clear_amm_approx: false,
            omnipool_deltas: None,
            amm_deltas: None,
        }
    }
    pub fn with_indicators(mut self, indicators: Vec<usize>) -> Self {
        self.indicators = Some(indicators);
        self
    }
    pub fn with_flags(mut self, flags: BTreeMap<AssetId, i8>) -> Self {
        self.omnipool_flags = Some(flags);
        self
    }
    pub fn with_sell_maxes(mut self, sell_maxes: Vec<FloatType>) -> Self {
        self.sell_maxes = Some(sell_maxes);
        self
    }
    pub fn with_force_amm_approx(mut self, force_amm_approx: BTreeMap<AssetId, AmmApprox>) -> Self {
        self.force_amm_approx = Some(force_amm_approx);
        self
    }
    pub fn with_rescale(mut self, rescale: bool) -> Self {
        self.rescale = rescale;
        self
    }
    pub fn with_clear_sell_maxes(mut self, clear_sell_maxes: bool) -> Self {
        self.clear_sell_maxes = clear_sell_maxes;
        self
    }
    pub fn with_clear_indicators(mut self, clear_indicators: bool) -> Self {
        self.clear_indicators = clear_indicators;
        self
    }
    pub fn with_clear_amm_approx(mut self, clear_amm_approx: bool) -> Self {
        self.clear_omnipool_approx = clear_amm_approx;
        self
    }
    pub fn with_force_omnipool_approx(
        mut self,
        force_omnipool_approx: BTreeMap<AssetId, AmmApprox>,
    ) -> Self {
        self.force_omnipool_approx = Some(force_omnipool_approx);
        self
    }

    pub fn with_force_amm_approx_vec(mut self, force_amm_approx_vec: Vec<Vec<AmmApprox>>) -> Self {
        self.force_amm_approx_vec = Some(force_amm_approx_vec);
        self
    }
}

impl ICEProblemV4 {
    pub(crate) fn get_indicators(&self) -> Option<Vec<usize>> {
        self.indicators.as_ref().cloned()
    }
    pub(crate) fn get_indicators_len(&self) -> usize {
        if let Some(inds) = self.indicators.as_ref() {
            inds.iter().sum()
        } else {
            0
        }
    }

    pub(crate) fn get_asset_pool_data(&self, asset_id: AssetId) -> &OmnipoolAsset {
        self.amm_store.omnipool.get(&asset_id).unwrap()
    }

    pub fn is_omnipool_asset(&self, asset_id: AssetId) -> bool {
        self.amm_store.omnipool.contains_key(&asset_id)
    }

    pub(crate) fn price(&self, asset_a: AssetId, asset_b: AssetId) -> FloatType {
        if asset_a == HUB_ASSET_ID {
            let db = self.get_asset_pool_data(asset_b);
            return db.hub_reserve / db.reserve;
        }
        let da = self.get_asset_pool_data(asset_a);
        let db = self.get_asset_pool_data(asset_b);
        if asset_a == asset_b {
            1.0
        } else if asset_b == 1u32 {
            da.hub_price
        } else if asset_a == 1u32 {
            1. / db.hub_price
        } else {
            let da_hub_reserve = da.hub_reserve;
            let da_reserve = da.reserve;
            let db_hub_reserve = db.hub_reserve;
            let db_reserve = db.reserve;
            da_hub_reserve / da_reserve / db_hub_reserve * db_reserve
        }
    }

    pub(crate) fn set_up_problem(&mut self, params: SetupParams) {
        if let Some(new_indicators) = params.indicators {
            debug_assert_eq!(new_indicators.len(), self.r);
            self.indicators = Some(new_indicators);
        } else if params.clear_indicators {
            self.indicators = None;
        }
        if let Some(new_maxes) = params.sell_maxes {
            self.partial_sell_maxs = new_maxes;
        } else if params.clear_sell_maxes {
            self.partial_sell_maxs = self.initial_sell_maxs.clone();
        }

        if let Some(new_flags) = params.omnipool_flags {
            self.omnipool_directional_flags = Some(new_flags);
        } else {
            self.omnipool_directional_flags = None;
        }

        if let Some(flags) = params.amm_flags {
            self.amm_directional_flags = Some(flags);
        }

        if let Some(new_force_amm_approx) = params.force_omnipool_approx {
            self.force_omnipool_approx = Some(new_force_amm_approx);
        } else if params.clear_omnipool_approx {
            self.force_omnipool_approx = None;
        }

        if let Some(v) = params.force_amm_approx_vec {
            self.force_amm_approx = Some(v);
        } else if params.clear_amm_approx {
            self.force_amm_approx = None;
        }

        self.last_omnipool_deltas = params.omnipool_deltas;
        self.last_amm_deltas = params.amm_deltas;

        self.recalculate(params.rescale);
    }

    fn recalculate(&mut self, rescale: bool) {
        let mut step_params = StepParams::default();
        step_params.set_known_flow(self);
        step_params.set_max_in_out(self);
        //step_params.set_bounds(self);
        if rescale {
            step_params.set_scaling(self);
            step_params.set_omnipool_coefs(self);
        }
        step_params.set_directions(self);
        //step_params.set_omnipool_directions(self);
        step_params.set_tau_phi(self);
        step_params.set_coefficients(self);
        self.step_params = step_params;
    }

    pub(crate) fn get_intent(&self, idx: usize) -> &Intent {
        &self.intents[idx]
    }

    pub(crate) fn get_scaling(&self) -> &BTreeMap<AssetId, FloatType> {
        self.step_params.scaling.as_ref().unwrap()
    }

    pub(crate) fn get_max_in(&self) -> &BTreeMap<AssetId, FloatType> {
        self.step_params.max_in.as_ref().unwrap()
    }

    pub(crate) fn get_max_out(&self) -> &BTreeMap<AssetId, FloatType> {
        self.step_params.max_out.as_ref().unwrap()
    }

    pub(crate) fn get_partial_sell_maxs_scaled(&self) -> Vec<FloatType> {
        let mut partial_sell_maxes = self.partial_sell_maxs.clone();
        for (j, &idx) in self.partial_indices.iter().enumerate() {
            let intent = &self.intents[idx];
            let tkn = intent.asset_in;
            if tkn != 1u32 {
                let liquidity = self.amm_store.omnipool.get(&tkn).unwrap().reserve;
                partial_sell_maxes[j] = partial_sell_maxes[j].min(liquidity / 2.0);
            }
        }
        let scaling = self.get_scaling();
        partial_sell_maxes
            .iter()
            .enumerate()
            .map(|(j, &max)| max / scaling[&self.intents[self.partial_indices[j]].asset_in])
            .collect()
    }

    /*
    pub fn get_scaled_bounds(
        &self,
    ) -> (
        ndarray::Array1<FloatType>,
        ndarray::Array1<FloatType>,
        ndarray::Array1<FloatType>,
        ndarray::Array1<FloatType>,
        ndarray::Array1<FloatType>,
        ndarray::Array1<FloatType>,
        ndarray::Array1<FloatType>,
        ndarray::Array1<FloatType>,
    ) {
        let scaling = self.get_scaling();
        let lrna_scaling = scaling[&1u32.into()]; // Assuming 1u32 represents 'LRNA'

        let min_y = self.step_params.min_y.as_ref().unwrap();
        let max_y = self.step_params.max_y.as_ref().unwrap();
        let min_x = self.step_params.min_x.as_ref().unwrap();
        let max_x = self.step_params.max_x.as_ref().unwrap();
        let min_lrna_lambda = self.step_params.min_lrna_lambda.as_ref().unwrap();
        let max_lrna_lambda = self.step_params.max_lrna_lambda.as_ref().unwrap();
        let min_lambda = self.step_params.min_lambda.as_ref().unwrap();
        let max_lambda = self.step_params.max_lambda.as_ref().unwrap();

        let scaled_min_y = ndarray::Array1::from(
            min_y
                .iter()
                .map(|&val| val / lrna_scaling)
                .collect::<Vec<_>>(),
        );
        let scaled_max_y = ndarray::Array1::from(
            max_y
                .iter()
                .map(|&val| val / lrna_scaling)
                .collect::<Vec<_>>(),
        );
        let scaled_min_x = ndarray::Array1::from(
            self.trading_asset_ids
                .iter()
                .enumerate()
                .map(|(i, &tkn)| min_x[i] / scaling[&tkn])
                .collect::<Vec<_>>(),
        );
        let scaled_max_x = ndarray::Array1::from(
            self.trading_asset_ids
                .iter()
                .enumerate()
                .map(|(i, &tkn)| max_x[i] / scaling[&tkn])
                .collect::<Vec<_>>(),
        );
        let scaled_min_lrna_lambda = ndarray::Array1::from(
            min_lrna_lambda
                .iter()
                .map(|&val| val / lrna_scaling)
                .collect::<Vec<_>>(),
        );
        let scaled_max_lrna_lambda = ndarray::Array1::from(
            max_lrna_lambda
                .iter()
                .map(|&val| val / lrna_scaling)
                .collect::<Vec<_>>(),
        );
        let scaled_min_lambda = ndarray::Array1::from(
            self.trading_asset_ids
                .iter()
                .enumerate()
                .map(|(i, &tkn)| min_lambda[i] / scaling[&tkn])
                .collect::<Vec<_>>(),
        );
        let scaled_max_lambda = ndarray::Array1::from(
            self.trading_asset_ids
                .iter()
                .enumerate()
                .map(|(i, &tkn)| max_lambda[i] / scaling[&tkn])
                .collect::<Vec<_>>(),
        );

        (
            scaled_min_y,
            scaled_max_y,
            scaled_min_x,
            scaled_max_x,
            scaled_min_lrna_lambda,
            scaled_max_lrna_lambda,
            scaled_min_lambda,
            scaled_max_lambda,
        )
    }
     */
}

#[derive(Default, Clone, Debug)]
pub struct StepParams {
    pub known_flow: Option<BTreeMap<AssetId, (FloatType, FloatType)>>,
    pub max_in: Option<BTreeMap<AssetId, FloatType>>,
    pub max_out: Option<BTreeMap<AssetId, FloatType>>,
    pub min_in: Option<BTreeMap<AssetId, FloatType>>,
    pub min_out: Option<BTreeMap<AssetId, FloatType>>,
    pub scaling: Option<BTreeMap<AssetId, FloatType>>,
    pub omnipool_directions: Option<BTreeMap<AssetId, Direction>>,
    pub amm_directions: Option<Vec<Vec<Direction>>>,
    pub tau: Option<Array2<FloatType>>,
    pub phi: Option<Array2<FloatType>>,
    pub q: Option<Vec<FloatType>>,
    pub profit_a: Option<Array2<FloatType>>,
    pub _S: Option<Vec<FloatType>>,
    pub _C: Option<Array1<FloatType>>,
    pub _B: Option<Array1<FloatType>>,
    min_x: Option<Vec<FloatType>>,
    max_x: Option<Vec<FloatType>>,
    min_lambda: Option<Vec<FloatType>>,
    max_lambda: Option<Vec<FloatType>>,
    min_y: Option<Vec<FloatType>>,
    max_y: Option<Vec<FloatType>>,
    min_lrna_lambda: Option<Vec<FloatType>>,
    max_lrna_lambda: Option<Vec<FloatType>>,
    omnipool_lrna_coefs: Option<BTreeMap<AssetId, FloatType>>,
    omnipool_asset_coefs: Option<BTreeMap<AssetId, FloatType>>,
}

impl StepParams {
    fn set_known_flow(&mut self, problem: &ICEProblemV4) {
        let mut known_flow: BTreeMap<AssetId, (FloatType, FloatType)> = BTreeMap::new();

        // Add LRNA to known_flow
        known_flow.insert(HUB_ASSET_ID, (0.0, 0.0));

        // Initialize known_flow with zero values for all assets
        for &asset_id in problem.all_asset_ids.iter() {
            known_flow.insert(asset_id, (0.0, 0.0));
        }

        // Update known_flow based on full intents
        if let Some(I) = &problem.get_indicators() {
            assert_eq!(I.len(), problem.full_indices.len());
            for (i, &idx) in problem.full_indices.iter().enumerate() {
                if I[i] as f64 > 0.5 {
                    let intent = &problem.intents[idx];
                    let (sell_quantity, buy_quantity) = problem.intent_amounts[idx];
                    let tkn_sell = intent.asset_in;
                    let tkn_buy = intent.asset_out;
                    known_flow
                        .entry(tkn_sell)
                        .and_modify(|(amount_in, _)| {
                            *amount_in += sell_quantity;
                        })
                        .or_insert((sell_quantity, 0.0));

                    known_flow
                        .entry(tkn_buy)
                        .and_modify(|(_, amount_out)| {
                            *amount_out += buy_quantity;
                        })
                        .or_insert((0.0, buy_quantity));

                    /*
                    let entry = known_flow.entry(tkn_sell).or_insert((0.0, 0.0));
                    entry.0 = entry.0 + sell_quantity;

                    let entry = known_flow.entry(tkn_buy).or_insert((0.0, 0.0));
                    entry.1 = entry.1 + buy_quantity;

                     */
                }
            }
        }

        self.known_flow = Some(known_flow);
    }
    fn set_max_in_out(&mut self, problem: &ICEProblemV4) {
        let mut max_in: BTreeMap<AssetId, FloatType> = BTreeMap::new();
        let mut max_out: BTreeMap<AssetId, FloatType> = BTreeMap::new();
        let mut min_in: BTreeMap<AssetId, FloatType> = BTreeMap::new();
        let mut min_out: BTreeMap<AssetId, FloatType> = BTreeMap::new();

        for &asset_id in problem.all_asset_ids.iter() {
            max_in.insert(asset_id, 0.0);
            max_out.insert(asset_id, 0.0);
            min_in.insert(asset_id, 0.0);
            min_out.insert(asset_id, 0.0);
        }

        max_in.insert(1u32.into(), 0.0); // Assuming 1u32 represents 'LRNA'
        max_out.insert(1u32.into(), 0.0);
        min_in.insert(1u32.into(), 0.0);
        min_out.insert(1u32.into(), 0.0);

        for (i, &idx) in problem.partial_indices.iter().enumerate() {
            let intent = &problem.intents[idx];
            let (amount_in, amount_out) = problem.intent_amounts[idx];
            let tkn_sell = intent.asset_in;
            let tkn_buy = intent.asset_out;
            let sell_quantity = problem.partial_sell_maxs[i];
            let buy_quantity = amount_out / amount_in * sell_quantity;

            *max_in.get_mut(&tkn_sell).unwrap() += sell_quantity;
            *max_out.get_mut(&tkn_buy).unwrap() += if buy_quantity != 0.0 {
                buy_quantity.next_after(FloatType::INFINITY)
            } else {
                0.0
            };
        }

        if problem.get_indicators().is_none() {
            for &idx in problem.full_indices.iter() {
                let intent = &problem.intents[idx];
                let (sell_quantity, buy_quantity) = problem.intent_amounts[idx];
                let tkn_sell = intent.asset_in;
                let tkn_buy = intent.asset_out;

                *max_in.get_mut(&tkn_sell).unwrap() += sell_quantity;
                *max_out.get_mut(&tkn_buy).unwrap() += buy_quantity;
            }
        }

        for (&tkn, &(in_flow, out_flow)) in self.known_flow.as_ref().unwrap().iter() {
            *max_in.get_mut(&tkn).unwrap() += in_flow - out_flow;
            *min_in.get_mut(&tkn).unwrap() += in_flow - out_flow;
            *max_out.get_mut(&tkn).unwrap() -= in_flow - out_flow;
            *min_out.get_mut(&tkn).unwrap() -= in_flow - out_flow;
        }

        let fees: BTreeMap<AssetId, FloatType> = problem
            .trading_asset_ids
            .iter()
            .map(|&tkn| (tkn, problem.get_asset_pool_data(tkn).fee))
            .collect();

        for &tkn in problem.trading_asset_ids.iter() {
            *max_in.get_mut(&tkn).unwrap() = max_in[&tkn].max(0.0);
            *min_in.get_mut(&tkn).unwrap() = min_in[&tkn].max(0.0);
            *max_out.get_mut(&tkn).unwrap() = (max_out[&tkn] / (1.0 - fees[&tkn])).max(0.0);
            *min_out.get_mut(&tkn).unwrap() = (min_out[&tkn] / (1.0 - fees[&tkn])).max(0.0);
        }

        *max_out.get_mut(&1u32.into()).unwrap() = 0.0; // Assuming 1u32 represents 'LRNA'
        *min_out.get_mut(&1u32.into()).unwrap() = 0.0;
        *max_in.get_mut(&1u32.into()).unwrap() = max_in[&1u32.into()].max(0.0);
        *min_in.get_mut(&1u32.into()).unwrap() = min_in[&1u32.into()].max(0.0);

        self.max_in = Some(max_in);
        self.max_out = Some(max_out);
        self.min_in = Some(min_in);
        self.min_out = Some(min_out);
    }

    fn set_scaling(&mut self, problem: &ICEProblemV4) {
        let mut scaling: BTreeMap<AssetId, FloatType> = BTreeMap::new();

        // Initialize scaling with zero values for all assets
        for &asset_id in problem.all_asset_ids.iter() {
            scaling.insert(asset_id, 0.0);
        }

        // Initialize scaling for LRNA
        scaling.insert(1u32.into(), 0.0); // Assuming 1u32 represents 'LRNA'

        for &tkn in problem.all_asset_ids.iter() {
            let max_in = self.max_in.as_ref().unwrap()[&tkn];
            let max_out = self.max_out.as_ref().unwrap()[&tkn];
            scaling.insert(tkn, max_in.max(max_out));

            if scaling[&tkn] == 0.0 && tkn != problem.tkn_profit {
                scaling.insert(tkn, 1.0);
            }

            if problem.is_omnipool_asset(tkn) {
                // Set scaling for LRNA equal to scaling for asset, adjusted by spot price
                let omnipool_data = problem.get_asset_pool_data(tkn);
                let scalar = scaling[&tkn] * omnipool_data.hub_reserve / omnipool_data.reserve;
                scaling.insert(1u32.into(), scaling[&1u32.into()].max(scalar));

                if let Some(omnipool_deltas) = &problem.last_omnipool_deltas {
                    //self._scaling[tkn] = max(self._scaling[tkn], abs(self._last_omnipool_deltas[tkn]))
                    scaling.insert(tkn, scaling[&tkn].max(omnipool_deltas[&tkn].abs()));
                }

                // Raise scaling for tkn_profit to scaling for asset, adjusted by spot price, if needed
                let scalar_profit = scaling[&tkn] * problem.price(tkn, problem.tkn_profit);
                scaling.insert(
                    problem.tkn_profit,
                    scaling[&problem.tkn_profit].max(scalar_profit / 10_000f64),
                );
            }
        }

        for stablepool in problem.amm_store.stablepools.iter() {
            let mut max_scale = scaling[&stablepool.pool_id];
            for &tkn in stablepool.assets.iter() {
                max_scale = max_scale.max(scaling[&tkn]);
            }
            scaling.insert(stablepool.pool_id, max_scale);
            for &tkn in stablepool.assets.iter() {
                scaling.insert(tkn, max_scale);
            }
        }

        if let Some(amm_deltas) = &problem.last_amm_deltas {
            for (i, amm) in problem.amm_store.stablepools.iter().enumerate() {
                assert_eq!(amm.assets.len() + 1, amm_deltas[i].len());
                scaling.insert(
                    amm.pool_id,
                    scaling[&amm.pool_id].max(amm_deltas[i][0].abs()),
                );
                for (j, tkn) in amm.assets.iter().enumerate() {
                    scaling.insert(*tkn, scaling[tkn].max(amm_deltas[i][j + 1].abs()));
                }
            }
        }

        self.scaling = Some(scaling);
    }
    fn set_omnipool_coefs(&mut self, problem: &ICEProblemV4) {
        let mut amm_lrna_coefs: BTreeMap<AssetId, FloatType> = BTreeMap::new();
        let mut amm_asset_coefs: BTreeMap<AssetId, FloatType> = BTreeMap::new();

        let scaling = self.scaling.as_ref().unwrap();
        for &tkn in problem.omnipool_asset_ids.iter() {
            let omnipool_data = problem.get_asset_pool_data(tkn);
            amm_lrna_coefs.insert(tkn, scaling[&1u32.into()] / omnipool_data.hub_reserve); // Assuming 1u32 represents 'LRNA'
            amm_asset_coefs.insert(tkn, scaling[&tkn] / omnipool_data.reserve);
        }

        self.omnipool_lrna_coefs = Some(amm_lrna_coefs);
        self.omnipool_asset_coefs = Some(amm_asset_coefs);
    }
}

impl StepParams {
    pub fn set_directions(&mut self, problem: &ICEProblemV4) {
        let mut omnipool_directions = BTreeMap::new();
        let mut amm_directions = vec![];
        /*
                for tkn in self._omnipool_directional_flags:
            if self._omnipool_directional_flags[tkn] == -1:
                self._omnipool_directions[tkn] = "sell"
            elif self._omnipool_directional_flags[tkn] == 1:
                self._omnipool_directions[tkn] = "buy"
            elif self._omnipool_directional_flags[tkn] == 0:
                self._omnipool_directions[tkn] = "neither"

        self._amm_directions = []
        for l in self._amm_directional_flags:
            new_list = []
            for f in l:
                if f == -1:
                    new_list.append("sell")
                elif f == 1:
                    new_list.append("buy")
                elif f == 0:
                    new_list.append("neither")
            self._amm_directions.append(new_list)

         */
        if let Some(flags) = &problem.omnipool_directional_flags {
            for (&tkn, &flag) in flags.iter() {
                match flag {
                    -1 => {
                        omnipool_directions.insert(tkn, Direction::Sell);
                    }
                    1 => {
                        omnipool_directions.insert(tkn, Direction::Buy);
                    }
                    0 => {
                        omnipool_directions.insert(tkn, Direction::Neither);
                    }
                    _ => {}
                }
            }
        }

        if let Some(flags) = &problem.amm_directional_flags {
            for (&pool_id, flag) in flags.iter() {
                let mut new_list = Vec::new();
                match flag {
                    -1 => {
                        new_list.push(Direction::Sell);
                    }
                    1 => {
                        new_list.push(Direction::Buy);
                    }
                    0 => {
                        new_list.push(Direction::Neither);
                    }
                    _ => {}
                }
                amm_directions.push(new_list);
            }
        }

        self.omnipool_directions = Some(omnipool_directions);
        self.amm_directions = Some(amm_directions);
    }
}

type A1T = ArrayBase<OwnedRepr<FloatType>, Ix1>;

impl StepParams {
    pub fn set_tau_phi(&mut self, problem: &ICEProblemV4) {
        let big_n = problem.asset_count;
        let m = problem.m;
        let r = problem.r;

        let mut tau1 = ndarray::Array2::zeros((big_n + 1, m));
        let mut phi1 = ndarray::Array2::zeros((big_n + 1, m));
        let mut tau2 = ndarray::Array2::zeros((big_n + 1, r));
        let mut phi2 = ndarray::Array2::zeros((big_n + 1, r));

        let mut tkn_list = vec![1u32];
        tkn_list.extend(problem.all_asset_ids.iter().cloned());

        for (j, &idx) in problem.partial_indices.iter().enumerate() {
            let intent = &problem.intents[idx];
            let tkn_sell = intent.asset_in;
            let tkn_buy = intent.asset_out;
            let tkn_sell_idx = tkn_list.iter().position(|&tkn| tkn == tkn_sell).unwrap();
            let tkn_buy_idx = tkn_list.iter().position(|&tkn| tkn == tkn_buy).unwrap();
            tau1[(tkn_sell_idx, j)] = 1.;
            phi1[(tkn_buy_idx, j)] = 1.;
        }
        for (l, &idx) in problem.full_indices.iter().enumerate() {
            let intent = &problem.intents[idx];
            let tkn_sell = intent.asset_in;
            let tkn_buy = intent.asset_out;
            let tkn_sell_idx = tkn_list.iter().position(|&tkn| tkn == tkn_sell).unwrap();
            let tkn_buy_idx = tkn_list.iter().position(|&tkn| tkn == tkn_buy).unwrap();
            tau2[(tkn_sell_idx, l)] = 1.;
            phi2[(tkn_buy_idx, l)] = 1.;
        }

        self.tau = Some(ndarray::concatenate![Axis(1), tau1, tau2]);
        self.phi = Some(ndarray::concatenate![Axis(1), phi1, phi2]);
    }

    pub fn set_coefficients(&mut self, problem: &ICEProblemV4) {
        // profit calculations
        let n = problem.n;
        let m = problem.m;
        let r = problem.r;

        // y_i are net LRNA into Omnipool
        let profit_lrna_y_coefs: ArrayBase<OwnedRepr<FloatType>, Ix1> = -ndarray::Array1::ones(n);
        // x_i are net assets into Omnipool
        let profit_lrna_x_coefs: ArrayBase<OwnedRepr<FloatType>, Ix1> = ndarray::Array1::zeros(n);
        // lrna_lambda_i are LRNA amounts coming out of Omnipool
        let profit_lrna_lrna_lambda_coefs: A1T = ndarray::Array::from(
            problem
                .omnipool_asset_ids
                .iter()
                .map(|&tkn| -problem.get_asset_pool_data(tkn).protocol_fee)
                .collect::<Vec<_>>(),
        );

        let profit_lrna_lambda_coefs: A1T = ndarray::Array1::zeros(n);
        let profit_lrna_big_x_coefs: A1T = ndarray::Array1::zeros(problem.sigma_sum);
        let profit_lrna_l_coefs: A1T = ndarray::Array1::zeros(problem.sigma_sum);
        let profit_lrna_a_coefs: A1T = ndarray::Array1::zeros(problem.u);

        let lrna_d_coefs = self.tau.as_ref().unwrap().row(0).clone().to_vec();
        let profit_lrna_d_coefs = ndarray::Array::from(lrna_d_coefs[..m].to_vec());

        let sell_amts: Vec<FloatType> = problem
            .full_indices
            .iter()
            .map(|&idx| problem.intent_amounts[idx].0)
            .collect();
        let profit_lrna_I_coefs: Vec<FloatType> = lrna_d_coefs[m..]
            .to_vec()
            .iter()
            .zip(sell_amts.iter())
            .map(|(&tau, &sell_amt)| tau * sell_amt / self.scaling.as_ref().unwrap()[&1u32])
            .collect(); //TODO: set scaling sets initial value to 0 for lrna;;verify if not division by zero

        /*
        let profit_lrna_coefs = ndarray::concatenate![
            Axis(0),
            profit_lrna_y_coefs,
            profit_lrna_x_coefs,
            profit_lrna_lrna_lambda_coefs,
            profit_lrna_lambda_coefs,
            profit_lrna_d_coefs,
            profit_lrna_I_coefs
        ];
         */
        let mut profit_lrna_coefs = vec![];
        profit_lrna_coefs.extend(profit_lrna_y_coefs);
        profit_lrna_coefs.extend(profit_lrna_x_coefs);
        profit_lrna_coefs.extend(profit_lrna_lrna_lambda_coefs);
        profit_lrna_coefs.extend(profit_lrna_lambda_coefs);
        profit_lrna_coefs.extend(profit_lrna_big_x_coefs);
        profit_lrna_coefs.extend(profit_lrna_l_coefs);
        profit_lrna_coefs.extend(profit_lrna_a_coefs);
        profit_lrna_coefs.extend(profit_lrna_d_coefs);
        profit_lrna_coefs.extend(profit_lrna_I_coefs);

        // leftover must be higher than required fees
        let fees: Vec<FloatType> = problem
            .omnipool_asset_ids
            .iter()
            .map(|&tkn| problem.get_asset_pool_data(tkn).fee)
            .collect();

        let buffer_fee = 0.00001f64;
        let stableswap_fees = problem
            .amm_store
            .stablepools
            .iter()
            .map(|pool| {
                let fee = pool.fee + buffer_fee - problem.fee_match;
                vec![fee; pool.assets.len()]
            })
            .collect::<Vec<Vec<FloatType>>>()
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<FloatType>>();

        let partial_intent_prices: Vec<FloatType> = problem.get_partial_intent_prices();
        let profit_y_coefs = ndarray::Array2::zeros((problem.asset_count, n));
        let mut profit_x_coefs = ndarray::Array2::zeros((problem.asset_count, n));
        //let profit_x_coefs = -Array2::<FloatType>::eye(n);
        let profit_lrna_lambda_coefs = ndarray::Array2::zeros((problem.asset_count, n));
        let mut profit_lambda_coefs = ndarray::Array2::zeros((problem.asset_count, n));

        for (i, tkn) in problem.all_asset_ids.iter().enumerate() {
            if problem.is_omnipool_asset(*tkn) {
                let j = problem
                    .omnipool_asset_ids
                    .iter()
                    .position(|&id| id == *tkn)
                    .unwrap();
                profit_x_coefs[(i, j)] = -1.0;
                profit_lambda_coefs[(i, j)] = problem.fee_match - fees[j];
            }
        }
        /*
        let profit_lambda_coefs = -Array2::<FloatType>::from_diag(&Array1::from(
            fees.iter()
                .map(|&fee| fee - problem.fee_match)
                .collect::<Vec<_>>(),
        ));

         */

        let profit_X_coefs = problem.rho.clone() - problem.psi.clone();

        //TODO: this fails with incompatible shapes
        //let stableswap_fees_flat_arr = Array1::from_vec(stableswap_fees.clone());
        //let neg_fees = - &stableswap_fees_flat_arr;
        //let profit_L_coefs = &problem.psi * &neg_fees.insert_axis(Axis(0));
        let profit_L_coefs = ndarray::Array2::zeros((problem.asset_count, problem.sigma_sum));
        let profit_a_coefs = ndarray::Array2::zeros((problem.asset_count, problem.u));

        let scaling = self.scaling.as_ref().unwrap();
        let scaling_vars: Vec<FloatType> = problem
            .partial_indices
            .iter()
            .enumerate()
            .map(|(j, &idx)| {
                let intent = &problem.intents[idx];
                partial_intent_prices[j] * scaling[&intent.asset_in] / scaling[&intent.asset_out]
            })
            .collect();

        let vars_scaled = scaling_vars
            .iter()
            .map(|&v| v * 1.0 / (1.0 - problem.fee_match))
            .collect::<Vec<f64>>();

        let phi = self.phi.as_ref().unwrap();
        let tau = self.tau.as_ref().unwrap();
        let scaled_phi = phi.slice(s![1.., ..m]).to_owned() * &Array1::from(vars_scaled.clone());
        let profit_d_coefs = tau.slice(s![1.., ..m]).to_owned() - scaled_phi;

        let buy_amts: Vec<FloatType> = problem
            .full_indices
            .iter()
            .map(|&idx| problem.intent_amounts[idx].1)
            .collect();
        let buy_scaled = buy_amts
            .iter()
            .map(|&v| v * 1.0 / (1.0 - problem.fee_match))
            .collect::<Vec<_>>();

        let phi = self.phi.as_ref().unwrap();
        let scaled_phi = phi.slice(s![1.., m..]).to_owned() * &Array1::from(buy_scaled.clone());
        let scaled_tau = tau.slice(s![1.., m..]).to_owned() * &Array1::from(sell_amts.clone());
        let unscaled_diff = scaled_tau - scaled_phi;
        let scalars: Vec<FloatType> = problem
            .all_asset_ids
            .iter()
            .map(|&tkn| scaling[&tkn])
            .collect();
        let un_size = unscaled_diff.shape()[0];
        let scalars = Array2::from_shape_vec((un_size, 1), scalars).unwrap();
        let i_coefs = unscaled_diff / scalars;

        let l = profit_lrna_coefs.len();
        let profit_A_LRNA = Array2::from_shape_vec((1, l), profit_lrna_coefs).unwrap();


        let profit_A_assets = ndarray::concatenate![
            Axis(1),
            profit_y_coefs,
            profit_x_coefs,
            profit_lrna_lambda_coefs,
            profit_lambda_coefs,
            profit_X_coefs,
            profit_L_coefs,
            profit_a_coefs,
            profit_d_coefs,
            i_coefs,
        ];

        let profit_A = Some(ndarray::concatenate![
            Axis(0),
            profit_A_LRNA,
            profit_A_assets
        ]);
        self.profit_a = profit_A.clone();

        let profit_tkn_idx = problem
            .all_asset_ids
            .iter()
            .position(|&tkn| tkn == problem.tkn_profit)
            .unwrap();
        self.q = Some(profit_A.as_ref().unwrap().row(profit_tkn_idx + 1).to_vec());
        /*
        self._S = np.array([self._scaling[tkn] for tkn in self.asset_list])
        self._C = self._rho.T @ self._S
        self._B = self._psi.T @ self._S
         */
        self._S = Some(
            problem
                .all_asset_ids
                .iter()
                .map(|&tkn| scaling[&tkn])
                .collect::<Vec<FloatType>>(),
        );
        self._C = Some(
            problem
                .rho
                .clone()
                .t()
                .dot(&Array1::from(self._S.as_ref().unwrap().clone())),
        );
        self._B = Some(
            problem
                .psi
                .clone()
                .t()
                .dot(&Array1::from(self._S.as_ref().unwrap().clone())),
        );
    }
}
