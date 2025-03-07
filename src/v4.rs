#![allow(non_snake_case)]

use crate::constants::{FLOAT_INF, HUB_ASSET_ID, ROUND_TOLERANCE};
use crate::internal::{process_data, Stablepool};
use crate::problem_v4::{
    AmmApprox, Direction, ICEProblemV4 as ICEProblem, ProblemStatus, SetupParams,
};
use crate::types::{AssetId, Balance, FloatType, Intent, ResolvedIntent, SolverResult};
use anyhow::{anyhow, Result};
use clarabel::algebra::*;
use clarabel::solver::*;
use highs::{HighsModelStatus, Sense};
use ndarray::{s, Array1, Array2, Axis};
use std::collections::BTreeMap;
use std::ops::Neg;

fn calculate_scaling(
    intents: &[Intent],
    intent_amounts: &[(f64, f64)],
    asset_ids: &[AssetId],
    omnipool_reserves: &[f64],
    omnipool_hub_reserves: &[f64],
) -> BTreeMap<AssetId, f64> {
    let mut scaling = BTreeMap::new();
    scaling.insert(1u32.into(), f64::INFINITY);

    for (idx, intent) in intents.iter().enumerate() {
        if intent.asset_in != 1u32 {
            let a = intent.asset_in;
            let sq = intent_amounts[idx].0;
            scaling
                .entry(a)
                .and_modify(|v| *v = v.max(sq))
                .or_insert(sq);
        }
        if intent.asset_out != 1u32 {
            let a = intent.asset_out;
            let sq = intent_amounts[idx].1;
            scaling
                .entry(a)
                .and_modify(|v| *v = v.max(sq))
                .or_insert(sq);
        }
    }

    for ((asset_id, reserve), hub_reserve) in asset_ids
        .iter()
        .zip(omnipool_reserves.iter())
        .zip(omnipool_hub_reserves.iter())
    {
        scaling
            .entry(*asset_id)
            .and_modify(|v| *v = v.min(*reserve))
            .or_insert(1.0);
        let scalar = (scaling.get(asset_id).unwrap() * *hub_reserve) / *reserve;
        scaling
            .entry(1u32)
            .and_modify(|v| *v = v.min(scalar))
            .or_insert(scalar);
    }

    scaling
}

fn calculate_tau_phi(
    intents: &[Intent],
    asset_ids: &[AssetId],
    scaling: &BTreeMap<AssetId, f64>,
) -> (CscMatrix, CscMatrix) {
    let n = asset_ids.len();
    let m = intents.len();
    let mut tau = CscMatrix::zeros((n, m));
    let mut phi = CscMatrix::zeros((n, m));
    for (j, intent) in intents.iter().enumerate() {
        let sell_i = asset_ids
            .iter()
            .position(|&tkn| tkn == intent.asset_in)
            .unwrap();
        let buy_i = asset_ids
            .iter()
            .position(|&tkn| tkn == intent.asset_out)
            .unwrap();
        tau.set_entry((sell_i, j), 1.);
        let s = scaling.get(&intent.asset_in).unwrap() / scaling.get(&intent.asset_out).unwrap();
        phi.set_entry((buy_i, j), s);
    }
    (tau, phi)
}
fn convert_to_balance(a: f64, dec: u8) -> Balance {
    let factor = 10u128.pow(dec as u32);
    (a * factor as f64) as Balance
}

// note that intent_deltas are < 0
fn prepare_resolved_intents(
    intents: &[Intent],
    asset_decimals: &BTreeMap<AssetId, u8>,
    converted_intent_amounts: &[(f64, f64)],
    intent_deltas: &[f64],
    intent_prices: &[f64],
    tolerance: f64,
) -> Vec<ResolvedIntent> {
    let mut resolved_intents = Vec::new();

    for (idx, delta_in) in intent_deltas.iter().enumerate() {
        debug_assert!(
            converted_intent_amounts[idx].0 >= -delta_in,
            "delta in is too high!"
        );
        let accepted_tolerance_amount = converted_intent_amounts[idx].0 * tolerance;
        let remainder = converted_intent_amounts[idx].0 + delta_in; // note that delta in is < 0
        let (amount_in, amount_out) = if remainder < accepted_tolerance_amount {
            // Do not leave dust, resolve the whole intent amount
            (intents[idx].amount_in, intents[idx].amount_out)
        } else if -delta_in <= accepted_tolerance_amount {
            // Do not trade dust
            (0u128, 0u128)
        } else {
            // just resolve solver amounts
            let amount_in = -delta_in;
            let amount_out = intent_prices[idx] * amount_in;
            (
                convert_to_balance(
                    amount_in,
                    *asset_decimals.get(&intents[idx].asset_in).unwrap(),
                ),
                convert_to_balance(
                    amount_out,
                    *asset_decimals.get(&intents[idx].asset_out).unwrap(),
                ),
            )
        };

        if amount_in == 0 || amount_out == 0 {
            continue;
        }
        let resolved_intent = ResolvedIntent {
            intent_id: intents[idx].intent_id,
            amount_in,
            amount_out,
        };
        resolved_intents.push(resolved_intent);
    }

    resolved_intents
}

fn round_solution(intents: &[(f64, f64)], intent_deltas: Vec<f64>, tolerance: f64) -> Vec<f64> {
    let mut deltas = Vec::new();
    for i in 0..intents.len() {
        // don't leave dust in intent due to rounding error
        if intents[i].0 + intent_deltas[i] < tolerance * intents[i].0 {
            deltas.push(-(intents[i].0));
        // don't trade dust amount due to rounding error
        } else if -intent_deltas[i] <= tolerance * intents[i].0 {
            deltas.push(0.);
        } else {
            deltas.push(intent_deltas[i]);
        }
    }
    deltas
}

fn add_buy_deltas(
    intents: Vec<(FloatType, FloatType)>,
    sell_deltas: Vec<FloatType>,
) -> Vec<(FloatType, FloatType)> {
    let mut deltas = Vec::new();
    for (i, (amount_in, amount_out)) in intents.iter().enumerate() {
        let sell_delta = sell_deltas[i];
        let buy_delta = -sell_delta * amount_out / amount_in;
        deltas.push((sell_delta, buy_delta));
    }
    deltas
}

fn diags(n: usize, m: usize, data: Vec<f64>) -> CscMatrix {
    let mut res = CscMatrix::zeros((n, m));
    for i in 0..n {
        res.set_entry((i, i), data[i]);
    }
    res
}

pub struct SolverV4;

impl SolverV4 {
    pub fn solve(
        intents: Vec<Intent>,
        pool_data: Vec<crate::types::Asset>,
    ) -> Result<SolverResult> {
        if intents.is_empty() {
            return Ok(SolverResult {
                resolved_intents: vec![],
            });
        }
        // atm we support only omnipool assets - let's prepare those
        let store = process_data(pool_data);
        let mut problem = ICEProblem::new()
            .with_intents(intents)
            .with_amm_store(store);
        problem.prepare()?;

        let (n, m, r, sigma) = (problem.n, problem.m, problem.r, problem.sigma_sum);

        dbg!(n, m, r, sigma);

        //dbg!(problem.indicators);

        let inf = FLOAT_INF;

        let k_milp = 4 * n + 3 * sigma + m + r;
        let mut Z_L = -inf;
        let mut Z_U = inf;
        let mut best_status = ProblemStatus::NotSolved;

        let mut y_best: Vec<usize> = Vec::new(); //TODO: this seems to be different
        let mut best_intent_deltas: Vec<FloatType> = Vec::with_capacity(m); // m size
        let mut best_omnipool_deltas: BTreeMap<AssetId, FloatType> = BTreeMap::new(); // should be m size

        let mut best_amm_deltas: Vec<Vec<FloatType>> = vec![];
        //let milp_ob = -inf;

        // Force small 	trades to execute
        // note this comes from initial solution which we skip for now
        // so nothing is mandatory just yet, but let;s prepare

        let exec_indices: Vec<usize> = vec![];
        let mut mandatory_indicators = vec![0; r];
        for &i in &exec_indices {
            mandatory_indicators[i] = 1;
        }

        let bk: Vec<usize> = mandatory_indicators
            .iter()
            .enumerate()
            .filter(|&(_, &val)| val == 1)
            .map(|(idx, _)| idx + 4 * n + 3 * sigma + m)
            .collect();

        let mut new_a = Array2::<f64>::zeros((1, k_milp));
        for &i in &bk {
            new_a[[0, i]] = 1.0;
        }

        let mut new_a_upper = Array1::from_elem(1, inf);
        let mut new_a_lower = Array1::from_elem(1, bk.len() as f64);

        let mut Z_U_archive = vec![];
        let mut Z_L_archive = vec![];
        let indicators = problem.get_indicators().unwrap_or(vec![0; r]);
        let mut x_list = Array2::<f64>::zeros((0, 4 * n + 3 * sigma + m));

        let mut iter_indicators = indicators.clone();

        for _i in 0..5 {
            let params = SetupParams::new().with_indicators(iter_indicators.clone());
            problem.set_up_problem(params);
            let (omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas) =
                find_good_solution(&problem, true, true, true, true);

            if obj < Z_U && dual_obj <= 0.0 {
                Z_U = obj;
                y_best = iter_indicators.clone();
                best_amm_deltas = amm_deltas.clone();
                best_omnipool_deltas = omnipool_deltas.clone();
                best_intent_deltas = intent_deltas.clone();
                best_status = status;
            }

            if status != ProblemStatus::PrimalInfeasible && status != ProblemStatus::DualInfeasible
            {
                //TODO: verify if this is correct
                let x2 = Array2::from_shape_vec((1, 4 * n + 3 * sigma + m), x).unwrap();
                x_list = ndarray::concatenate![Axis(0), x_list, x2];
            }

            // Get new cone constraint from current indicators
            let BK: Vec<usize> = iter_indicators
                .iter()
                .enumerate()
                .filter(|&(_, &val)| val == 1)
                .map(|(idx, _)| idx + 4 * n + 3 * sigma + m) // TODO: idx here is not really correct?!!
                .collect();
            let NK: Vec<usize> = iter_indicators
                .iter()
                .enumerate()
                .filter(|&(_, &val)| val == 0)
                .map(|(idx, _)| idx + 4 * n + 3 * sigma + m) // TODO: idx here is not really correct?!!
                .collect();
            let mut IC_A = Array2::<f64>::zeros((1, k_milp));
            for &i in &BK {
                IC_A[[0, i]] = 1.0;
            }
            for &i in &NK {
                IC_A[[0, i]] = -1.0;
            }
            let IC_upper = Array1::from_elem(1, BK.len() as f64 - 1.);
            let IC_lower = Array1::from_elem(1, -FLOAT_INF);

            // Add cone constraint to A, A_upper, A_lower
            let A = ndarray::concatenate![ndarray::Axis(0), new_a.view(), IC_A.view()];
            let A_upper =
                ndarray::concatenate![ndarray::Axis(0), new_a_upper.view(), IC_upper.view()];
            let A_lower =
                ndarray::concatenate![ndarray::Axis(0), new_a_lower.view(), IC_lower.view()];

            problem.set_up_problem(SetupParams::new());
            let (
                omnipool_deltas,
                partial_intent_deltas,
                indicators,
                s_new_a,
                s_new_a_upper,
                s_new_a_lower,
                milp_obj,
                valid,
                amm_deltas,
            ) = solve_inclusion_problem(
                &problem,
                Some(x_list.clone()),
                Some(Z_U),
                Some(-FLOAT_INF),
                Some(A),
                Some(A_upper),
                Some(A_lower),
            );

            if !valid {
                break;
            }
            iter_indicators = indicators;
            new_a = s_new_a;
            new_a_upper = s_new_a_upper;
            new_a_lower = s_new_a_lower;
            Z_L = Z_L.max(milp_obj);
            Z_U_archive.push(Z_U);
            Z_L_archive.push(Z_L);
        }
        if best_status != ProblemStatus::Solved {
            // no solution found
            return Err(anyhow!("Best status not solved: {:?}", best_status));
        }

        let sell_deltas = round_solution(
            &problem.get_partial_intents_amounts(),
            best_intent_deltas,
            ROUND_TOLERANCE,
        );
        let partial_deltas_with_buys =
            add_buy_deltas(problem.get_partial_intents_amounts(), sell_deltas);

        let full_deltas_with_buys = problem
            .get_full_intents_amounts()
            .iter()
            .enumerate()
            .map(|(l, (amount_in, amount_out))| {
                if y_best[l] == 1 {
                    (-amount_in, *amount_out)
                } else {
                    (0., 0.)
                }
            })
            .collect::<Vec<_>>();

        let mut deltas = vec![None; m + r];
        for (i, delta) in problem.partial_indices.iter().enumerate() {
            deltas[problem.partial_indices[i]] = Some(partial_deltas_with_buys[i]);
        }
        for (i, delta) in problem.full_indices.iter().enumerate() {
            deltas[problem.full_indices[i]] = Some(full_deltas_with_buys[i]);
        }

        //TODO: add this
        //let (deltas_final, obj) = add_small_trades(&problem, deltas);

        // Construct resolved intents
        let mut resolved_intents = Vec::new();

        for (idx, intent_delta) in deltas.iter().enumerate() {
            if let Some((delta_in, delta_out)) = intent_delta {
                let intent = &problem.intents[idx];
                let converted_intent_amount = problem.intent_amounts[idx];
                debug_assert!(
                    converted_intent_amount.0 >= -delta_in,
                    "delta in is too high!"
                );

                let accepted_tolerance_amount = converted_intent_amount.0 * ROUND_TOLERANCE;
                let remainder = converted_intent_amount.0 + delta_in; // note that delta in is < 0
                let (amount_in, amount_out) = if remainder < accepted_tolerance_amount {
                    // Do not leave dust, resolve the whole intent amount
                    (intent.amount_in, intent.amount_out)
                } else if -delta_in <= accepted_tolerance_amount {
                    // Do not trade dust
                    (0u128, 0u128)
                } else {
                    // just resolve solver amounts
                    let amount_in = -delta_in;
                    let amount_out = *delta_out;
                    (
                        convert_to_balance(
                            amount_in,
                            problem.get_asset_pool_data(intent.asset_in).decimals,
                        ),
                        convert_to_balance(
                            amount_out,
                            problem.get_asset_pool_data(intent.asset_out).decimals,
                        ),
                    )
                };

                if amount_in == 0 || amount_out == 0 {
                    continue;
                }
                let resolved_intent = ResolvedIntent {
                    intent_id: problem.intent_ids[idx],
                    amount_in,
                    amount_out,
                };
                resolved_intents.push(resolved_intent);
            }
        }

        Ok(SolverResult { resolved_intents })
    }
}

fn solve_inclusion_problem(
    p: &ICEProblem,
    x_real_list: Option<Array2<f64>>, // NLP solution
    upper_bound: Option<f64>,
    lower_bound: Option<f64>,
    old_A: Option<Array2<f64>>,
    old_A_upper: Option<Array1<f64>>,
    old_A_lower: Option<Array1<f64>>,
) -> (
    BTreeMap<AssetId, f64>,
    Vec<Option<f64>>,
    Vec<usize>,
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    f64,
    bool,
    Vec<Vec<FloatType>>,
) {
    let asset_list = p.all_asset_ids.clone();
    let omnipool_asset_list = p.omnipool_asset_ids.clone();
    let tkn_list = vec![1u32]
        .into_iter()
        .chain(asset_list.iter().cloned())
        .collect::<Vec<_>>();
    let (n, m, r, sigma) = (p.n, p.m, p.r, p.sigma_sum);
    let k = 4 * n + 3 * sigma + m + r;

    let scaling = p.get_scaling();
    let x_list = x_real_list.map(|x| x.map_axis(Axis(1), |row| p.get_scaled_x(row.to_vec())));

    let inf = f64::INFINITY;

    let upper_bound = upper_bound.unwrap_or(inf);
    let lower_bound = lower_bound.unwrap_or(-inf);

    let partial_intent_sell_amts = p.get_partial_sell_maxs_scaled();

    let mut max_lambda_d = BTreeMap::new();
    let mut max_lrna_lambda_d = BTreeMap::new();
    let mut max_y_d = BTreeMap::new();
    let mut min_y_d = BTreeMap::new();
    let mut max_x_d = BTreeMap::new();
    let mut min_x_d = BTreeMap::new();

    for tkn in omnipool_asset_list.iter() {
        max_lambda_d.insert(
            tkn.clone(),
            p.get_asset_pool_data(*tkn).reserve / scaling.get(tkn).unwrap() / 2.0,
        );
        max_lrna_lambda_d.insert(
            tkn.clone(),
            p.get_asset_pool_data(*tkn).hub_reserve / scaling.get(&HUB_ASSET_ID).unwrap() / 2.0,
        );
        max_y_d.insert(tkn.clone(), *max_lrna_lambda_d.get(tkn).unwrap());
        min_y_d.insert(tkn.clone(), -max_lrna_lambda_d.get(tkn).unwrap());
        max_x_d.insert(tkn.clone(), *max_lambda_d.get(tkn).unwrap());
        min_x_d.insert(tkn.clone(), -max_lambda_d.get(tkn).unwrap());
    }

    let max_in = p.get_max_in();
    let max_out = p.get_max_out();

    for tkn in omnipool_asset_list.iter() {
        if *tkn != p.tkn_profit {
            /*
            max_x_d.insert(
                tkn.clone(),
                max_in.get(&tkn).unwrap() / scaling.get(&tkn).unwrap() * 2.0,
            );
            min_x_d.insert(
                tkn.clone(),
                -max_out.get(&tkn).unwrap() / scaling.get(&tkn).unwrap() * 2.0,
            );

             */
            max_lambda_d.insert(tkn.clone(), -min_x_d.get(&tkn).unwrap());
            /*
            let max_y_unscaled = max_out.get(&tkn).unwrap()
                * p.get_asset_pool_data(*tkn).hub_reserve
                / (p.get_asset_pool_data(*tkn).reserve - max_out.get(&tkn).unwrap())
                + max_in.get(&HUB_ASSET_ID).unwrap();
            max_y_d.insert(
                tkn.clone(),
                max_y_unscaled / scaling.get(&HUB_ASSET_ID).unwrap(),
            );
            min_y_d.insert(
                tkn.clone(),
                -max_in.get(&tkn).unwrap() * p.get_asset_pool_data(*tkn).hub_reserve
                    / (p.get_asset_pool_data(*tkn).reserve + max_in.get(&tkn).unwrap())
                    / scaling.get(&HUB_ASSET_ID).unwrap(),
            );
            max_lrna_lambda_d.insert(tkn.clone(), -min_y_d.get(&tkn).unwrap());

             */
        }
    }
    /*
     min_y = np.array([min_y_d[tkn] for tkn in op_asset_list])
    max_y = np.array([max_y_d[tkn] for tkn in op_asset_list])
    min_x = np.array([min_x_d[tkn] for tkn in op_asset_list])
    max_x = np.array([max_x_d[tkn] for tkn in op_asset_list])
    min_lrna_lambda = np.zeros(n)
    max_lrna_lambda = np.array([max_lrna_lambda_d[tkn] for tkn in op_asset_list])
    min_lambda = np.zeros(n)
    max_lambda = np.array([max_lambda_d[tkn] for tkn in op_asset_list])
     */

    let min_y = Array1::from_iter(
        omnipool_asset_list
            .iter()
            .map(|tkn| min_y_d.get(tkn).unwrap().clone()),
    );
    let max_y = Array1::from_iter(
        omnipool_asset_list
            .iter()
            .map(|tkn| max_y_d.get(tkn).unwrap().clone()),
    );
    let min_x = Array1::from_iter(
        omnipool_asset_list
            .iter()
            .map(|tkn| min_x_d.get(tkn).unwrap().clone()),
    );
    let max_x = Array1::from_iter(
        omnipool_asset_list
            .iter()
            .map(|tkn| max_x_d.get(tkn).unwrap().clone()),
    );
    let min_lrna_lambda = Array1::zeros(n);
    let max_lrna_lambda = Array1::from_iter(
        omnipool_asset_list
            .iter()
            .map(|tkn| max_lrna_lambda_d.get(tkn).unwrap().clone()),
    );
    let min_lambda = Array1::zeros(n);
    let max_lambda = Array1::from_iter(
        omnipool_asset_list
            .iter()
            .map(|tkn| max_lambda_d.get(tkn).unwrap().clone()),
    );

    /*

    let (
        mut min_y,
        mut max_y,
        mut min_x,
        mut max_x,
        mut min_lrna_lambda,
        mut max_lrna_lambda,
        mut min_lambda,
        mut max_lambda,
    ) = p.get_scaled_bounds();
    let profit_i = asset_list
        .iter()
        .position(|tkn| tkn == &p.tkn_profit)
        .unwrap();
    max_x[profit_i] = inf;
    max_y[profit_i] = inf;
    min_lambda[profit_i] = 0.0;
    min_lrna_lambda[profit_i] = 0.0;

    min_y = min_y.clone() - 1.1 * min_y.abs();
    min_x = min_x.clone() - 1.1 * min_x.abs();
    min_lrna_lambda = min_lrna_lambda.clone() - 1.1 * min_lrna_lambda.abs();
    min_lambda = min_lambda.clone() - 1.1 * min_lambda.abs();
    max_y = max_y.clone() + 1.1 * max_y.abs();
    max_x = max_x.clone() + 1.1 * max_x.abs();
    max_lrna_lambda = max_lrna_lambda.clone() + 1.1 * max_lrna_lambda.abs();
    max_lambda = max_lambda.clone() + 1.1 * max_lambda.abs();

     */

    /*
    max_L = np.array([])
    for amm in p.amm_list:
        max_L = np.append(max_L, amm.shares)
        for tkn in amm.asset_list:
            max_L = np.append(max_L, amm.liquidity[tkn])

    B = p.get_B()
    C = p.get_C()
    max_L = max_L / (B + C)

    min_L = np.zeros(sigma)
    min_X = [-x for x in max_L]
    max_X = [inf] * sigma
    min_a = [-inf] * sigma
    max_a = [inf] * sigma
     */

    let mut max_L = vec![];
    for amm in p.amm_store.stablepools.iter() {
        max_L.push(amm.shares);
        for reserve in amm.reserves.iter() {
            max_L.push(*reserve)
        }
    }
    let B = p.get_b();
    let C = p.get_c();
    let max_L = ndarray::Array1::from_vec(max_L);
    let max_L = max_L / (B.clone() + C.clone());
    //max_L = max_L.iter().map(|&v| v / (B + C)).collect::<Vec<_>>();

    let min_L = Array1::<f64>::zeros(sigma);
    let min_X = Array1::from_iter(max_L.iter().map(|&x| -x));
    let max_X = Array1::<f64>::from_elem(sigma, inf);
    let min_a = Array1::<f64>::from_elem(sigma, -inf);
    let max_a = Array1::<f64>::from_elem(sigma, inf);

    /*
     lower = np.concatenate([min_y, min_x, min_lrna_lambda, min_lambda, min_X, min_L, min_a, [0] * (m + r)])
    upper = np.concatenate([max_y, max_x, max_lrna_lambda, max_lambda, max_X, max_L, max_a, partial_intent_sell_amts, [1] * r])

     */

    let lower = ndarray::concatenate![
        Axis(0),
        min_y.view(),
        min_x.view(),
        min_lrna_lambda.view(),
        min_lambda.view(),
        min_X.view(),
        min_L.view(),
        min_a.view(),
        Array1::zeros(m + r).view()
    ];

    let upper = ndarray::concatenate![
        Axis(0),
        max_y.view(),
        max_x.view(),
        max_lrna_lambda.view(),
        max_lambda.view(),
        max_X.view(),
        Array1::from(max_L).view(),
        max_a.view(),
        partial_intent_sell_amts,
        Array1::ones(r).view()
    ];

    let mut S = Array2::<f64>::zeros((n, k));
    let mut S_upper = Array1::<f64>::zeros(n);
    let x_zero = Array1::<f64>::zeros(4 * n + 3 * sigma + m);
    let mut offset = 0;
    for (s, amm) in p.amm_store.stablepools.iter().enumerate() {
        let D0_prime = amm.d - amm.d / amm.ann();
        let s0 = amm.shares;
        let c = C[offset];
        let sum_assets = amm.reserves.iter().sum::<f64>();
        let denom = sum_assets - D0_prime;
        let a0 = x_list.as_ref().unwrap()[s][4 * n + 2 * sigma + offset];
        let X0 = x_list.as_ref().unwrap()[s][4 * n + offset];
        let exp = (a0 / (1.0 + c * X0 / s0)).exp();
        let term = (c / s0 - c * a0 / (s0 + c)) * exp;
        let mut S_row = Array2::<f64>::zeros((1, k));
        let grad_s = term + c * D0_prime / (s0 * denom);
        let grads_i = amm
            .assets
            .iter()
            .enumerate()
            .map(|(idx, tkn)| -B.clone()[offset + idx])
            .collect::<Vec<_>>();
        let grad_a = exp;
        S_row[[0, 4 * n + offset]] = grad_s;
        S_row[[0, 4 * n + 2 * sigma + offset]] = grad_a;
        for (l, tkn) in amm.assets.iter().enumerate() {
            S_row[[0, 4 * n + offset + l + 1]] = grads_i[l] / denom;
        }
        let grad_dot_x = grad_s * X0
            + grad_a * a0
            + grads_i
                .iter()
                .zip(amm.assets.iter().enumerate())
                .map(|(&grad, (idx, tkn))| grad * x_list.as_ref().unwrap()[s][4 * n + offset + idx])
                .sum::<f64>();
        let sum_deltas = amm
            .assets
            .iter()
            .enumerate()
            .zip(1..)
            .map(|((idx, tkn), l)| {
                B[offset + l] * x_list.as_ref().unwrap()[s][4 * n + offset + idx]
            })
            .sum::<f64>();
        let g_neg =
            (1.0 + c * X0 / s0) * exp - sum_deltas / denom + D0_prime * c * X0 / (denom * s0) - 1.0;
        let S_row_upper = Array1::from_elem(1, grad_dot_x + g_neg);
        S = ndarray::concatenate![Axis(0), S.view(), S_row.view()];
        S_upper = ndarray::concatenate![Axis(0), S_upper.view(), S_row_upper.view()];
        for (l, tkn) in amm.assets.iter().enumerate() {
            let mut S_row = Array2::<f64>::zeros((1, k));
            let grad_s = term;
            let grad_a = exp;
            let grad_x = -B[offset + l + 1] / amm.reserves[l];
            S_row[[0, 4 * n + offset]] = grad_s;
            S_row[[0, 4 * n + 2 * sigma + offset + l + 1]] = grad_a;
            S_row[[0, 4 * n + offset + l + 1]] = grad_x;
            let ai = x_list.as_ref().unwrap()[s][4 * n + 2 * sigma + offset + l + 1];
            let grad_dot_x = grad_s * X0
                + grad_a * ai
                + grad_x * x_list.as_ref().unwrap()[s][4 * n + offset + l + 1];
            let g_neg = (1.0 + c * X0 / s0) * exp
                - B[offset + l + 1] * x_list.as_ref().unwrap()[s][4 * n + offset + l + 1]
                    / amm.reserves[l]
                - 1.0;
            let S_row_upper = Array1::from_elem(1, grad_dot_x + g_neg);
            S = ndarray::concatenate![Axis(0), S.view(), S_row.view()];
            S_upper = ndarray::concatenate![Axis(0), S_upper.view(), S_row_upper.view()];
        }
        offset += 1 + amm.assets.len();
    }
    /*
    S_lower = np.array([-inf]*len(S_upper))

    # need top level Stableswap constraint
    A_amm = np.zeros((p.s, k))
    offset = 0
    for i, amm in enumerate(p.amm_list):
        for j in range(1 + len(amm.asset_list)):
            A_amm[i, 4*n + 2*sigma + offset + j] = 1
        offset += 1 + len(amm.asset_list)
    A_amm_upper = np.array([inf]*p.s)
    A_amm_lower = np.zeros(p.s)

    # asset leftover must be above zero
    A3 = p.get_profit_A()
    A3_upper = np.array([inf]*(N+1))
    A3_lower = np.zeros(N+1)
     */

    let S_lower = Array1::<f64>::from_elem(S_upper.len(), -inf);

    let mut A_amm = Array2::<f64>::zeros((p.s, k));
    let mut offset = 0;
    for (i, amm) in p.amm_store.stablepools.iter().enumerate() {
        for j in 0..1 + amm.assets.len() {
            A_amm[[i, 4 * n + 2 * sigma + offset + j]] = 1.0;
        }
        offset += 1 + amm.assets.len();
    }
    let A_amm_upper = Array1::<f64>::from_elem(p.s, inf);
    let A_amm_lower = Array1::<f64>::zeros(p.s);

    let A3 = p.get_profit_A();
    let A3_upper = Array1::<f64>::from_elem(n + 1, inf);
    let A3_lower = Array1::<f64>::zeros(n + 1);

    let mut A5 = Array2::<f64>::zeros((2 * n, k));
    for i in 0..n {
        A5[[i, i]] = 1.0;
        A5[[i, 2 * n + i]] = 1.0;
        A5[[n + i, n + i]] = 1.0;
        A5[[n + i, 3 * n + i]] = 1.0;
    }
    let A5_upper = Array1::<f64>::from_elem(2 * n, inf);
    let A5_lower = Array1::<f64>::zeros(2 * n);

    /*
    # inequality constraints: X_j + L_j >= 0
    A7 = np.zeros((sigma, k))
    for i in range(sigma):
        A7[i, 4*n + i] = 1
        A7[i, 4*n + sigma + i] = 1
    A7_upper = np.array([inf] * sigma)
    A7_lower = np.zeros(sigma)
    # A7 = np.zeros((0,k))
    # A7_upper = np.array([])
    # A7_lower = np.array([])

    # optimized value must be lower than best we have so far, higher than lower bound
    A8 = np.zeros((1, k))
    q = p.get_q()
    A8[0, :] = -q
    A8_upper = np.array([upper_bound / scaling[p.tkn_profit]])
    A8_upper = np.array([upper_bound/10 / scaling[p.tkn_profit]])
    A8_lower = np.array([lower_bound / scaling[p.tkn_profit]])
     */

    let mut A7 = Array2::<f64>::zeros((sigma, k));
    for i in 0..sigma {
        A7[[i, 4 * n + i]] = 1.0;
        A7[[i, 4 * n + sigma + i]] = 1.0;
    }
    let A7_upper = Array1::<f64>::from_elem(sigma, inf);
    let A7_lower = Array1::<f64>::zeros(sigma);

    let mut A8 = Array2::<f64>::zeros((1, k));
    let q = p.get_q();
    let q_a = ndarray::Array1::from(q.clone());
    A8.row_mut(0).assign(&(-q_a));
    let A8_upper = Array1::from_elem(1, upper_bound / 10. / scaling[&p.tkn_profit]);
    let A8_lower = Array1::from_elem(1, lower_bound / scaling[&p.tkn_profit]);

    /*
        if old_A is None:
        old_A = np.zeros((0, k))
    if old_A_upper is None:
        old_A_upper = np.array([])
    if old_A_lower is None:
        old_A_lower = np.array([])
    assert len(old_A_upper) == len(old_A_lower) == old_A.shape[0]
    A = np.vstack([old_A, S, A_amm, A3, A5, A7, A8])
    A_upper = np.concatenate([old_A_upper, S_upper, A_amm_upper, A3_upper, A5_upper, A7_upper, A8_upper])
    A_lower = np.concatenate([old_A_lower, S_lower, A_amm_lower, A3_lower, A5_lower, A7_lower, A8_lower])
     */

    let old_A = old_A.unwrap_or_else(|| Array2::<f64>::zeros((0, k)));
    let old_A_upper = old_A_upper.unwrap_or_else(|| Array1::<f64>::zeros(0));
    let old_A_lower = old_A_lower.unwrap_or_else(|| Array1::<f64>::zeros(0));

    let A = ndarray::concatenate![
        Axis(0),
        old_A.view(),
        S.view(),
        A_amm.view(),
        A3.view(),
        A5.view(),
        A7.view(),
        A8.view()
    ];
    let A_upper = ndarray::concatenate![
        Axis(0),
        old_A_upper.view(),
        S_upper.view(),
        A_amm_upper.view(),
        A3_upper.view(),
        A5_upper.view(),
        A7_upper.view(),
        A8_upper.view()
    ];
    let A_lower = ndarray::concatenate![
        Axis(0),
        old_A_lower.view(),
        S_lower.view(),
        A_amm_lower.view(),
        A3_lower.view(),
        A5_lower.view(),
        A7_lower.view(),
        A8_lower.view()
    ];

    /*
    nonzeros = []
    start = [0]
    a = []
    for i in range(A.shape[0]):
        row_nonzeros = np.where(A[i, :] != 0)[0]
        nonzeros.extend(row_nonzeros)
        start.append(len(nonzeros))
        a.extend(A[i, row_nonzeros])
     */

    let mut nonzeros = vec![];
    let mut start = vec![0];
    let mut a = vec![];
    for i in 0..A.shape()[0] {
        let row_nonzeros = A
            .index_axis(Axis(0), i)
            .into_shape((k,))
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        nonzeros.extend(row_nonzeros.clone());
        start.push(nonzeros.len());
        a.extend(row_nonzeros.iter().map(|&idx| A[[i, idx]]));
    }

    let mut pb = highs::RowProblem::new();

    let mut col_cost = vec![];
    for (idx, &v) in q.iter().enumerate() {
        let lower_bound = lower[idx];
        let upper_bound = upper[idx];
        let x = pb.add_column(-v, lower_bound..upper_bound);
        col_cost.push(x);
    }

    for (idx, row) in A.outer_iter().enumerate() {
        let v = row.to_vec();
        // now zip v with col_cost
        let v = v
            .iter()
            .zip(col_cost.iter())
            .map(|(a, b)| (*b, *a))
            .collect::<Vec<_>>();
        let lower_bound = A_lower[idx];
        let upper_bound = A_upper[idx];
        pb.add_row(lower_bound..upper_bound, v);
    }
    let mut model = pb.optimise(Sense::Minimise);
    model.set_option("small_matrix_value", 1e-12);
    model.set_option("primal_feasibility_tolerance", 1e-10);
    model.set_option("dual_feasibility_tolerance", 1e-10);
    model.set_option("mip_feasibility_tolerance", 1e-10);

    let solved = model.solve();
    let status = solved.status();
    let solution = solved.get_solution();
    let x_expanded = solution.columns().to_vec();
    let value_valid = status != HighsModelStatus::Infeasible;

    /*

    //TODO: should we integrality and options ?!! seems to work without that
    lp.integrality = vec![highs::VarType::Continuous; 4 * n + m]
        .into_iter()
        .chain(vec![highs::VarType::Integer; r])
        .collect();
    let options = h.get_options();
    options.small_matrix_value = 1e-12;
    options.primal_feasibility_tolerance = 1e-10;
    options.dual_feasibility_tolerance = 1e-10;
    options.mip_feasibility_tolerance = 1e-10;
    let status = h.get_model_status();
    let solution = h.get_solution();
    let info = h.get_info();
    let basis = h.get_basis();
    let value_valid = solution.value_valid,
    let status  = status.to_string(),
    let x_expanded = solution.col_value;
     */

    let mut new_omnipool_deltas = BTreeMap::new();

    for (i, tkn) in p.omnipool_asset_ids.iter().enumerate() {
        new_omnipool_deltas.insert(*tkn, x_expanded[n + i] * scaling[tkn]);
    }

    let mut new_amm_deltas = vec![];
    let mut exec_partial_intent_deltas = vec![None; m];
    let mut exec_full_intent_flags = vec![];

    let mut offset = 0;
    for amm in p.amm_store.stablepools.iter() {
        let mut deltas = vec![x_expanded[4 * n + offset] * scaling[&amm.pool_id]];
        for (l, tkn) in amm.assets.iter().enumerate() {
            deltas.push(x_expanded[4 * n + offset + l + 1] * scaling[tkn]);
        }
        new_amm_deltas.push(deltas);
        offset += amm.assets.len() + 1;
    }

    for i in 0..m {
        exec_partial_intent_deltas[i] = Some(
            -x_expanded[4 * n + 3 * sigma + i]
                * scaling[&p.get_intent(p.partial_indices[i]).asset_in],
        );
    }

    for i in 0..r {
        exec_full_intent_flags.push(if x_expanded[4 * n + 3 * sigma + m + i] > 0.5 {
            1
        } else {
            0
        });
    }

    let save_A = old_A.clone();
    let save_A_upper = old_A_upper.clone();
    let save_A_lower = old_A_lower.clone();

    let score = -q.dot(&x_expanded) * scaling[&p.tkn_profit];

    (
        new_omnipool_deltas,
        exec_partial_intent_deltas,
        exec_full_intent_flags,
        save_A,
        save_A_upper,
        save_A_lower,
        score,
        value_valid,
        new_amm_deltas,
    )

    /*
    (
        new_amm_deltas,
        exec_partial_intent_deltas,
        exec_full_intent_flags,
        save_A,
        save_A_upper,
        save_A_lower,
        -q.clone().dot(&x_expanded) * scaling[&p.tkn_profit],
        value_valid,
    )

         */
}

fn find_good_solution(
    problem: &ICEProblem,
    scale_trade_max: bool,
    approx_amm_eqs: bool,
    do_directional_run: bool,
    allow_loss: bool,
) -> (
    BTreeMap<AssetId, f64>,
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
    ProblemStatus,
    Vec<Vec<FloatType>>,
) {
    /*

    n, m, r = p.n, p.m, p.r
    N, u, s, sigma = p.N, p.u, p.s, p.sigma
    force_omnipool_approx = {tkn: "linear" for tkn in p.omnipool.asset_list}
    force_amm_approx = [["linear" for _ in range(len(amm.asset_list) + 1)] for amm in p.amm_list]
    p.set_up_problem(clear_I=False, force_omnipool_approx=force_omnipool_approx, force_amm_approx=force_amm_approx)
    omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas = _find_solution_unrounded(p, allow_loss=allow_loss)
    # if partial trade size is much higher than executed trade, lower trade max
    if scale_trade_max:
        trade_pcts = [max(-intent_deltas[i],0) / m if m > 0 else 0 for i, m in enumerate(p.partial_sell_maxs)]
    else:
        trade_pcts = [1] * len(p.partial_sell_maxs)
    trade_pcts = trade_pcts + [1 for _ in range(r)]
     */

    let mut p: ICEProblem = problem.clone();
    let (n, m, r) = (p.n, p.m, p.r);
    let (big_n, u, s, sigma) = (p.asset_count, p.u, p.s, p.sigma_sum);

    let mut force_omnipool_approx = BTreeMap::new();
    for tkn in &p.omnipool_asset_ids {
        force_omnipool_approx.insert(*tkn, AmmApprox::Linear);
    }

    let mut force_amm_approx = vec![];

    for pool in p.amm_store.stablepools.iter() {
        let mut approx = vec![];
        for _ in 0..pool.assets.len() + 1 {
            approx.push(AmmApprox::Linear);
        }
        force_amm_approx.push(approx);
    }

    let setup_params = SetupParams::new()
        .with_clear_indicators(false)
        .with_force_omnipool_approx(force_omnipool_approx.clone())
        .with_force_amm_approx_vec(force_amm_approx.clone());

    p.set_up_problem(setup_params);

    let (mut omnipool_deltas, intent_deltas, x, obj, dual_obj, status, mut amm_deltas) =
        find_solution_unrounded(&p, allow_loss);

    let mut trade_pcts: Vec<f64> = if scale_trade_max {
        p.partial_sell_maxs
            .iter()
            .enumerate()
            .map(|(i, &m)| if m > 0.0 { -intent_deltas[i] / m } else { 0.0 })
            .collect()
    } else {
        vec![1.0; p.partial_sell_maxs.len()]
    };
    trade_pcts.extend(vec![1.0; r]);

    let mut approx_adjusted_ct = 0;
    if approx_amm_eqs
        && status != ProblemStatus::PrimalInfeasible
        && status != ProblemStatus::DualInfeasible
    {
        let mut omnipool_pcts = BTreeMap::new();
        for tkn in &p.omnipool_asset_ids {
            omnipool_pcts.insert(*tkn, omnipool_deltas[tkn].abs() / p.get_tkn_liquidity(*tkn));
        }

        for tkn in &p.omnipool_asset_ids {
            if force_omnipool_approx[tkn] == AmmApprox::Linear && omnipool_pcts[tkn] > 1e-6 {
                force_omnipool_approx.insert(*tkn, AmmApprox::Quadratic);
                approx_adjusted_ct += 1;
            }
            if force_omnipool_approx[tkn] == AmmApprox::Quadratic && omnipool_pcts[tkn] > 1e-3 {
                force_omnipool_approx.insert(*tkn, AmmApprox::Full);
                approx_adjusted_ct += 1;
            }
        }

        let mut stableswap_pcts = vec![];
        for (i, amm) in p.amm_store.stablepools.iter().enumerate() {
            let mut pcts = vec![];
            let sum_delta_x = amm_deltas[i].iter().skip(1).sum::<f64>();
            pcts.push(sum_delta_x / amm.d);
            pcts.extend(
                amm_deltas[i]
                    .iter()
                    .skip(1)
                    .zip(amm.assets.iter().skip(1))
                    .enumerate()
                    .map(|(idx, (delta, tkn))| delta.abs() / amm.reserves[idx + 1]),
            );
            stableswap_pcts.push(pcts);
        }

        for (s, amm) in p.amm_store.stablepools.iter().enumerate() {
            if force_amm_approx[s][0] == AmmApprox::Linear && stableswap_pcts[s][0] > 1e-5 {
                force_amm_approx[s][0] = AmmApprox::Full;
                approx_adjusted_ct += 1;
            }
            for (j, tkn) in amm.assets.iter().enumerate() {
                if force_amm_approx[s][j + 1] == AmmApprox::Linear
                    && stableswap_pcts[s][j + 1] > 1e-5
                {
                    force_amm_approx[s][j + 1] = AmmApprox::Full;
                    approx_adjusted_ct += 1;
                }
            }
        }
    }

    for _ in 0..100 {
        let trade_pcts_nonzero_max = p
            .partial_sell_maxs
            .iter()
            .enumerate()
            .filter(|(i, &m)| m > 0.0)
            .map(|(i, _)| -intent_deltas[i] / m as f64)
            .collect::<Vec<_>>();
        let min_value = trade_pcts_nonzero_max
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if (trade_pcts_nonzero_max.len() == 0 || min_value >= 0.1) && approx_adjusted_ct == 0 {
            break;
        }
        let (new_maxes, zero_ct) = if trade_pcts_nonzero_max.len() > 0 && min_value < 0.1 {
            scale_down_partial_intents(&p, &trade_pcts, 10f64)
        } else {
            (None, 0)
        };

        //TODO: set up problem here !!! Part 2
        let mut params = if let Some(nm) = new_maxes {
            SetupParams::new().with_sell_maxes(nm)
        } else {
            SetupParams::new()
        };
        let params = params
            .with_clear_indicators(false)
            .with_force_omnipool_approx(force_omnipool_approx.clone())
            .with_force_amm_approx_vec(force_amm_approx.clone());
        p.set_up_problem(params);

        // set deltas
        p.last_omnipool_deltas = Some(omnipool_deltas.clone());
        p.last_amm_deltas = Some(amm_deltas.clone());

        println!("------------");

        let (omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas) =
            find_solution_unrounded(&p, allow_loss);

        if scale_trade_max {
            trade_pcts = p
                .partial_sell_maxs
                .iter()
                .enumerate()
                .map(|(i, &m)| if m > 0.0 { -intent_deltas[i] / m } else { 0.0 })
                .collect();
        }

        if approx_amm_eqs
            && status != ProblemStatus::PrimalInfeasible
            && status != ProblemStatus::DualInfeasible
        {
            let mut omnipool_pcts = BTreeMap::new();
            for tkn in &p.omnipool_asset_ids {
                omnipool_pcts.insert(*tkn, omnipool_deltas[tkn].abs() / p.get_tkn_liquidity(*tkn));
            }

            approx_adjusted_ct = 0;
            for tkn in &p.omnipool_asset_ids {
                if force_omnipool_approx[tkn] == AmmApprox::Linear && omnipool_pcts[tkn] > 1e-6 {
                    force_omnipool_approx.insert(*tkn, AmmApprox::Quadratic);
                    approx_adjusted_ct += 1;
                }
                if force_omnipool_approx[tkn] == AmmApprox::Quadratic && omnipool_pcts[tkn] > 1e-3 {
                    force_omnipool_approx.insert(*tkn, AmmApprox::Full);
                    approx_adjusted_ct += 1;
                }
            }

            let mut stableswap_pcts = vec![];
            for (i, amm) in p.amm_store.stablepools.iter().enumerate() {
                let mut pcts = vec![];
                let sum_delta_x = amm_deltas[i].iter().skip(1).sum::<f64>();
                pcts.push(sum_delta_x / amm.d);
                pcts.extend(
                    amm_deltas[i]
                        .iter()
                        .skip(1)
                        .zip(amm.assets.iter().skip(1))
                        .enumerate()
                        .map(|(idx, (delta, tkn))| delta.abs() / amm.reserves[idx + 1]),
                );
                stableswap_pcts.push(pcts);
            }

            for (s, amm) in p.amm_store.stablepools.iter().enumerate() {
                if force_amm_approx[s][0] == AmmApprox::Linear && stableswap_pcts[s][0] > 1e-5 {
                    force_amm_approx[s][0] = AmmApprox::Full;
                    approx_adjusted_ct += 1;
                }
                for (j, tkn) in amm.assets.iter().enumerate() {
                    if force_amm_approx[s][j + 1] == AmmApprox::Linear
                        && stableswap_pcts[s][j + 1] > 1e-5
                    {
                        force_amm_approx[s][j + 1] = AmmApprox::Full;
                        approx_adjusted_ct += 1;
                    }
                }
            }
        }
    }

    /*
    //TODO: this does not anything yet in python
    if do_directional_run {
        let (omnipool_flags, amm_flags) = get_directional_flags(&omnipool_deltas, &amm_deltas);

        let setup_params = SetupParams::new()
            .with_omnipool_flags(omnipool_flags)
            .with_clear_indicators(false)
            .with_clear_sell_maxes(false)
            .with_clear_omnipool_approx(false)
            .with_clear_amm_approx(false)
            .with_omnipool_deltas(omnipool_deltas)
            .with_amm_deltas(amm_deltas)
            .with_amm_flags(amm_flags);

        p.set_up_problem(setup_params);

        let (omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas) =
            find_solution_unrounded(&p, allow_loss);
    }

     */

    if status == ProblemStatus::PrimalInfeasible || status == ProblemStatus::DualInfeasible {
        let amm_deltas = vec![];
        return (
            BTreeMap::new(),
            vec![0.0; p.partial_indices.len()],
            vec![0.; 4 * n + m],
            0.0,
            0.0,
            ProblemStatus::Solved,
            amm_deltas,
        );
    }

    let mut x_unscaled = p.get_real_x(x.iter().cloned().collect());
    for i in 0..n {
        let tkn = p.omnipool_asset_ids[i];
        if (x_unscaled[i] / p.get_lrna_liquidity(tkn)).abs() < 1e-11
            || (x_unscaled[n + i] / p.get_tkn_liquidity(tkn)).abs() < 1e-11
        {
            x_unscaled[i] = 0.0;
            x_unscaled[n + i] = 0.0;
            x_unscaled[2 * n + i] = 0.0;
            x_unscaled[3 * n + i] = 0.0;
            *omnipool_deltas.get_mut(&tkn).unwrap() = 0.0;
        }
    }

    let mut offset = 0;
    let stablepools = &p.amm_store.stablepools;
    for (i, amm_delta) in amm_deltas.iter_mut().enumerate() {
        if (amm_delta[0] / stablepools[i].shares).abs() < 1e-11 {
            x_unscaled[4 * n + offset] = 0.0;
            x_unscaled[4 * n + sigma + offset] = 0.0;
            x_unscaled[4 * n + 2 * sigma + offset] = 0.0;
            amm_delta[0] = 0.0;
        }
        for (j, tkn) in stablepools[i].assets.iter().enumerate() {
            if (amm_delta[j + 1] / stablepools[i].reserves[j]).abs() < 1e-11 {
                amm_delta[j + 1] = 0.0;
                x_unscaled[4 * n + offset + j + 1] = 0.0;
                x_unscaled[4 * n + sigma + offset + j + 1] = 0.0;
                x_unscaled[4 * n + 2 * sigma + offset + j + 1] = 0.0;
            }
        }
        offset += stablepools[i].assets.len() + 1;
    }

    (
        omnipool_deltas,
        intent_deltas,
        x_unscaled,
        obj,
        dual_obj,
        status,
        amm_deltas,
    )
}

fn find_solution_unrounded(
    p: &ICEProblem,
    allow_loss: bool,
) -> (
    BTreeMap<AssetId, f64>,
    Vec<f64>,
    Array2<f64>,
    f64,
    f64,
    ProblemStatus,
    Vec<Vec<f64>>,
) {
    if p.indicators.is_none() {
        panic!("dont know why yet i should panic here")
    }

    let full_intents = p.get_full_intents();
    let partial_intents = p.get_partial_intents();
    let stablepools = p.amm_store.stablepools.clone();

    let asset_list = p.all_asset_ids.clone();
    let (n, m, r) = (p.n, p.m, p.r);
    let (big_n, sigma, s, u) = (p.asset_count, p.sigma_sum, p.s, p.u);

    /*
    if p.get_indicators_len() as f64 + p.partial_sell_maxs.iter().sum::<f64>() == 0.0 {
        return (
            p.trading_asset_ids.iter().map(|&tkn| (tkn, 0.0)).collect(),
            vec![0.0; p.partial_indices.len()],
            Array2::zeros((4 * p.n + p.m, 1)),
            0.0,
            0.0,
            ProblemStatus::Solved,
        );
    }
     */

    //let full_intents = &p.full_intents;
    let partial_intents_len = p.partial_indices.len();
    let trading_asset_ids = &p.trading_asset_ids;
    let (n, m, r) = (p.n, p.m, p.r);

    /*
    if partial_intents_len + p.get_indicators_len() == 0 {
        return (
            asset_list.iter().map(|&tkn| (tkn, 0.0)).collect(),
            vec![],
            Array2::zeros((4 * n, 1)),
            0.0,
            0.0,
            ProblemStatus::Solved,
        );
    }
     */

    let (omnipool_directions, amm_directions) = p.get_directions();
    let k = 4 * n + 2 * sigma + m + u;
    let mut indices_to_keep: Vec<usize> = (0..k).collect();

    /*
    for &tkn in directions.keys() {
        if directions[&tkn] == Direction::Sell || directions[&tkn] == Direction::Neither {
            indices_to_keep
                .retain(|&i| i != 2 * n + asset_list.iter().position(|&x| x == tkn).unwrap());
        }
        if directions[&tkn] == Direction::Buy || directions[&tkn] == Direction::Neither {
            indices_to_keep
                .retain(|&i| i != 3 * n + asset_list.iter().position(|&x| x == tkn).unwrap());
        }
        if directions[&tkn] == Direction::Neither {
            indices_to_keep.retain(|&i| i != asset_list.iter().position(|&x| x == tkn).unwrap());
            indices_to_keep
                .retain(|&i| i != n + asset_list.iter().position(|&x| x == tkn).unwrap());
        }
    }

     */

    let k_real = indices_to_keep.len();
    let P_trimmed = CscMatrix::zeros((k_real, k_real));
    let q_all = ndarray::Array::from(p.get_q());

    let objective_I_coefs = q_all.slice(s![k..]);
    let objective_I_coefs = objective_I_coefs.neg();
    let q = q_all.slice(s![..k]);
    let q = q.neg();
    let q_trimmed: Vec<f64> = indices_to_keep.iter().map(|&i| q[i]).collect();

    //let diff_coefs = Array2::<f64>::zeros((2 * n + m, 2 * n));
    //let nonzero_coefs = -Array2::<f64>::eye(2 * n + m);
    //let A1 = ndarray::concatenate![Axis(1), diff_coefs, nonzero_coefs];

    let mut A1 = Array2::<f64>::zeros((0, k));
    let mut cones1 = vec![];
    /*
    profit_i = p.asset_list.index(p.tkn_profit)
    op_tradeable_indices = [i for i in range(n) if p.omnipool.asset_list[i] in p.trading_tkns]
    if profit_i not in op_tradeable_indices:
        op_tradeable_indices.append(profit_i)
    for i in range(n):
        tkn = p.omnipool.asset_list[i]
        if tkn in omnipool_directions and tkn in p._last_omnipool_deltas:
            delta_pct = p._last_omnipool_deltas[tkn] / p.omnipool.liquidity[tkn]  # possibly round to zero
        else:
            delta_pct = 1  # to avoid causing any rounding
        if i in op_tradeable_indices and abs(delta_pct) > 1e-11:  # we need lambda_i >= 0, lrna_lambda_i >= 0
            A1i = np.zeros((2, k))
            A1i[0, 3 * n + i] = -1  # lambda_i
            A1i[1, 2 * n + i] = -1  # lrna_lambda_i
            cone1i = cb.NonnegativeConeT(2)
            if p.omnipool.asset_list[i] in omnipool_directions:
                if omnipool_directions[p.omnipool.asset_list[i]] == "buy":  # we need y_i <= 0, x_i >= 0
                    A1i_dir = np.zeros((2, k))
                    A1i_dir[0, i] = 1
                    A1i_dir[1, n + i] = -1
                    A1i = np.vstack([A1i, A1i_dir])
                    cone1i = cb.NonnegativeConeT(4)
                elif omnipool_directions[p.omnipool.asset_list[i]] == "sell":  # we need y_i >= 0, x_i <= 0
                    A1i_dir = np.zeros((2, k))
                    A1i_dir[0, i] = -1
                    A1i_dir[1, n + i] = 1
                    A1i = np.vstack([A1i, A1i_dir])
                    cone1i = cb.NonnegativeConeT(4)

        else:  # we need y_i = 0, x_i = 0, lambda_i = 0, lrna_lambda_i = 0
            A1i = np.zeros((4, k))
            A1i[0, i] = 1  # y_i
            A1i[1, n + i] = 1  # x_i
            A1i[2, 2 * n + i] = 1  # lrna_lambda_i
            A1i[3, 3 * n + i] = 1  # lambda_i
            cone1i = cb.ZeroConeT(4)
        A1 = np.vstack([A1, A1i])
        cones1.append(cone1i)
     */

    let profit_i = trading_asset_ids
        .iter()
        .position(|&x| x == p.tkn_profit)
        .unwrap();
    let mut op_tradeable_indices: Vec<usize> = (0..n)
        .filter(|&i| {
            let asset_id = p.omnipool_asset_ids[i];
            trading_asset_ids.contains(&asset_id)
        })
        .collect();
    if !op_tradeable_indices.contains(&profit_i) {
        op_tradeable_indices.push(profit_i);
    }

    for i in 0..n {
        let tkn = p.omnipool_asset_ids[i];
        dbg!(&p.last_omnipool_deltas);

        let delta_pct = if let Some(delta) = p.last_omnipool_deltas.as_ref() {
            if let Some(d) = delta.get(&tkn) {
                if omnipool_directions.contains_key(&tkn) {
                    *d / p.get_tkn_liquidity(tkn)
                } else {
                    1.0
                }
            } else {
                1.0
            }
        } else {
            1.0
        };
        let (A1i, cone1i) = if op_tradeable_indices.contains(&i) && delta_pct.abs() > 1e-11 {
            let mut A1i = Array2::<f64>::zeros((2, k));
            A1i[[0, 3 * n + i]] = -1.0;
            A1i[[1, 2 * n + i]] = -1.0;
            let mut cone1i = NonnegativeConeT(2);
            if let Some(direction) = omnipool_directions.get(&tkn) {
                if direction == &Direction::Buy {
                    let mut A1i_dir = Array2::<f64>::zeros((2, k));
                    A1i_dir[[0, i]] = 1.0;
                    A1i_dir[[1, n + i]] = -1.0;
                    A1i = ndarray::concatenate![Axis(0), A1i, A1i_dir];
                    cone1i = NonnegativeConeT(4);
                } else if direction == &Direction::Sell {
                    let mut A1i_dir = Array2::<f64>::zeros((2, k));
                    A1i_dir[[0, i]] = -1.0;
                    A1i_dir[[1, n + i]] = 1.0;
                    A1i = ndarray::concatenate![Axis(0), A1i, A1i_dir];
                    cone1i = NonnegativeConeT(4);
                }
            }
            (A1i, cone1i)
        } else {
            let mut A1i = Array2::<f64>::zeros((4, k));
            A1i[[0, i]] = 1.0;
            A1i[[1, n + i]] = 1.0;
            A1i[[2, 2 * n + i]] = 1.0;
            A1i[[3, 3 * n + i]] = 1.0;
            let cone1i = ZeroConeT(4);
            (A1i, cone1i)
        };
        A1 = ndarray::concatenate![Axis(0), A1, A1i];
        cones1.push(cone1i);
    }

    /*
    offset = 0
    for i, amm in enumerate(amm_list):
        if len(amm_directions) > 0 and len(p._last_amm_deltas) > 0:
            delta_pct = p._last_amm_deltas[i][0] / amm.shares  # possibly round to zero
        else:
            delta_pct = 1  # to avoid causing any rounding
        if amm.unique_id in p.trading_tkns and abs(delta_pct) > 1e-11:
            A1i = np.zeros((1, k))
            A1i[0, 4 * n + sigma + offset] = -1
            cones1.append(cb.NonnegativeConeT(1))
            if len(amm_directions) > 0:
                if amm_directions[i][0] == "buy":  # X0 >= 0
                    A1i_dir = np.zeros((1, k))
                    A1i_dir[0, 4 * n + offset] = -1
                    A1i = np.vstack([A1i, A1i_dir])
                    cones1.append(cb.NonnegativeConeT(1))
                elif amm_directions[i][0] == "sell":  # X0 <= 0
                    A1i_dir = np.zeros((1, k))
                    A1i_dir[0, 4 * n + offset] = 1
                    A1i = np.vstack([A1i, A1i_dir])
                    cones1.append(cb.NonnegativeConeT(1))
        else:
            A1i = np.zeros((2, k))
            A1i[0, 4 * n + offset] = 1
            A1i[1, 4 * n + sigma + offset] = 1
            cones1.append(cb.ZeroConeT(2))
        for j, tkn in enumerate(amm.asset_list):
            if len(amm_directions) > 0 and len(p._last_amm_deltas) > 0:
                delta_pct = p._last_amm_deltas[i][j+1] / amm.liquidity[tkn]  # possibly round to zero
            else:
                delta_pct = 1  # to avoid causing any rounding
            if tkn in p.trading_tkns and abs(delta_pct) > 1e-11:
                A1ij = np.zeros((1, k))
                A1ij[0, 4 * n + sigma + offset + j + 1] = -1
                cones1.append(cb.NonnegativeConeT(1))
                if len(amm_directions) > 0:
                    if amm_directions[i][j+1] == "buy":  #Xj >= 0
                        A1ij_dir = np.zeros((1, k))
                        A1ij_dir[0, 4 * n + offset + j + 1] = -1
                        A1ij = np.vstack([A1ij, A1ij_dir])
                        cones1.append(cb.NonnegativeConeT(1))
                    elif amm_directions[i][j+1] == "sell":  #Xj <= 0
                        A1ij_dir = np.zeros((1, k))
                        A1ij_dir[0, 4 * n + offset + j + 1] = 1
                        A1ij = np.vstack([A1ij, A1ij_dir])
                        cones1.append(cb.NonnegativeConeT(1))
            else:
                A1ij = np.zeros((2, k))
                A1ij[0, 4 * n + offset + j + 1] = 1
                A1ij[1, 4 * n + sigma + offset + j + 1] = 1
                cones1.append(cb.ZeroConeT(2))
            A1i = np.vstack([A1i, A1ij])
        A1 = np.vstack([A1, A1i])
        offset += len(amm.asset_list) + 1
     */

    let mut offset = 0;

    dbg!(&amm_directions);

    for (i, amm) in stablepools.iter().enumerate() {
        let delta_pct = if let Some(delta) = &p.last_amm_deltas {
            delta[i][0] / amm.shares
        } else {
            1.0
        };
        let mut A1i = Array2::<f64>::zeros((1, k));
        if trading_asset_ids.contains(&amm.pool_id) && delta_pct.abs() > 1e-11 {
            A1i = Array2::<f64>::zeros((1, k));
            A1i[[0, 4 * n + sigma + offset]] = -1.0;
            cones1.push(NonnegativeConeT(1));
            if amm_directions.len() > 0 {
                let directions = amm_directions[i].clone();
                if directions[0] == Direction::Buy {
                    let mut A1i_dir = Array2::<f64>::zeros((1, k));
                    A1i_dir[[0, 4 * n + offset]] = -1.0;
                    A1i = ndarray::concatenate![Axis(0), A1i, A1i_dir];
                    cones1.push(NonnegativeConeT(1));
                } else if directions[0] == Direction::Sell {
                    let mut A1i_dir = Array2::<f64>::zeros((1, k));
                    A1i_dir[[0, 4 * n + offset]] = 1.0;
                    A1i = ndarray::concatenate![Axis(0), A1i, A1i_dir];
                    cones1.push(NonnegativeConeT(1));
                }
            }
        } else {
            A1i = Array2::<f64>::zeros((2, k));
            A1i[[0, 4 * n + offset]] = 1.0;
            A1i[[1, 4 * n + sigma + offset]] = 1.0;
            cones1.push(ZeroConeT(2));
        }
        for (j, tkn) in amm.assets.iter().enumerate() {
            let delta_pct = if let Some(delta) = &p.last_amm_deltas {
                delta[i][j + 1] / p.get_tkn_liquidity(*tkn)
            } else {
                1.0
            };
            let mut A1ij = Array2::<f64>::zeros((1, k));
            if trading_asset_ids.contains(tkn) && delta_pct.abs() > 1e-11 {
                A1ij[[0, 4 * n + sigma + offset + j + 1]] = -1.0;
                cones1.push(NonnegativeConeT(1));
                let directions = amm_directions[i].clone();
                if directions[j + 1] == Direction::Buy {
                    let mut A1ij_dir = Array2::<f64>::zeros((1, k));
                    A1ij_dir[[0, 4 * n + offset + j + 1]] = -1.0;
                    A1ij = ndarray::concatenate![Axis(0), A1ij, A1ij_dir];
                    cones1.push(NonnegativeConeT(1));
                } else if directions[j + 1] == Direction::Sell {
                    let mut A1ij_dir = Array2::<f64>::zeros((1, k));
                    A1ij_dir[[0, 4 * n + offset + j + 1]] = 1.0;
                    A1ij = ndarray::concatenate![Axis(0), A1ij, A1ij_dir];
                    cones1.push(NonnegativeConeT(1));
                }
            } else {
                A1ij = Array2::<f64>::zeros((2, k));
                A1ij[[0, 4 * n + offset + j + 1]] = 1.0;
                A1ij[[1, 4 * n + sigma + offset + j + 1]] = 1.0;
                cones1.push(ZeroConeT(2));
            }
            A1i = ndarray::concatenate![Axis(0), A1i, A1ij];
        }
        A1 = ndarray::concatenate![Axis(0), A1, A1i];
        offset += amm.assets.len() + 1;
    }

    //A1_trimmed = A1[:, indices_to_keep]
    let A1_trimmed = A1.select(Axis(1), &indices_to_keep);
    /*
    let rows_to_keep: Vec<usize> = (0..2 * n + m)
        .filter(|&i| indices_to_keep.contains(&(2 * n + i)))
        .collect();
    let A1_trimmed = A1
        .select(Axis(0), &rows_to_keep)
        .select(Axis(1), &indices_to_keep);
    let cone1 = NonnegativeConeT(A1_trimmed.shape()[0]);
     */
    let b1 = Array1::<f64>::zeros(A1_trimmed.shape()[0]);

    // intent variables are constrained from above, and from below by 0
    let amm_coefs = Array2::<f64>::zeros((2 * m, k - m));
    //let d_coefs = Array2::<f64>::eye(m);
    let d_coefs = ndarray::concatenate![
        Axis(0),
        Array2::<FloatType>::eye(m),
        Array2::<FloatType>::eye(m).neg()
    ];
    let A2 = ndarray::concatenate![Axis(1), amm_coefs, d_coefs];
    //let b2 = Array1::from(p.get_partial_sell_maxs_scaled());
    //b2 = np.concatenate([np.array(p.get_partial_sell_maxs_scaled()), np.zeros(m)])
    let b2 = ndarray::concatenate![
        Axis(0),
        p.get_partial_sell_maxs_scaled(),
        Array1::<FloatType>::zeros(m)
    ];
    let A2_trimmed = A2.select(Axis(1), &indices_to_keep);
    let cone2 = NonnegativeConeT(A2_trimmed.shape()[0]);

    let (A3_trimmed, b3) = p.get_leftover_bounds(allow_loss, Some(indices_to_keep.clone()));
    let cone3 = NonnegativeConeT(A3_trimmed.shape()[0]);

    let mut A4 = Array2::<f64>::zeros((0, k));
    let mut b4 = Array1::<f64>::zeros(0);
    let mut cones4 = vec![];
    let epsilon_tkn = p.get_epsilon_tkn();

    dbg!(&epsilon_tkn);

    for i in 0..n {
        let tkn = p.omnipool_asset_ids[i];
        let approx = p.get_omnipool_approx(tkn);
        /*
        let approx =
            if approx == AmmApprox::None && epsilon_tkn[&tkn] <= 1e-6 && tkn != p.tkn_profit {
                AmmApprox::Linear
            } else if approx == AmmApprox::None && epsilon_tkn[&tkn] <= 1e-3 {
                AmmApprox::Quadratic
            } else {
                approx
            };

         */

        let (A4i, b4i, cone) = match approx {
            AmmApprox::Linear => {
                if !omnipool_directions.contains_key(&tkn) {
                    let c1 = 1.0 / (1.0 + epsilon_tkn[&tkn]);
                    let c2 = 1.0 / (1.0 - epsilon_tkn[&tkn]);
                    let mut A4i = Array2::<f64>::zeros((2, k));
                    A4i[[0, i]] = -p.get_omnipool_lrna_coefs()[&tkn];
                    A4i[[0, n + i]] = -p.get_omnipool_asset_coefs()[&tkn] * c1;
                    A4i[[1, i]] = -p.get_omnipool_lrna_coefs()[&tkn];
                    A4i[[1, n + i]] = -p.get_omnipool_asset_coefs()[&tkn] * c2;
                    (A4i, Array1::<f64>::zeros(2), NonnegativeConeT(2))
                } else {
                    let c = if omnipool_directions[&tkn] == Direction::Sell {
                        1.0 / (1.0 - epsilon_tkn[&tkn])
                    } else {
                        1.0 / (1.0 + epsilon_tkn[&tkn])
                    };
                    let mut A4i = Array2::<f64>::zeros((1, k));
                    A4i[[0, i]] = -p.get_omnipool_lrna_coefs()[&tkn];
                    A4i[[0, n + i]] = -p.get_omnipool_asset_coefs()[&tkn] * c;
                    (A4i, Array1::<f64>::zeros(1), ZeroConeT(1))
                }
            }
            AmmApprox::Quadratic => {
                let mut A4i = Array2::<f64>::zeros((3, k));
                A4i[[1, i]] = -p.get_omnipool_lrna_coefs()[&tkn];
                A4i[[1, n + i]] = -p.get_omnipool_asset_coefs()[&tkn];
                A4i[[2, n + i]] = -p.get_omnipool_asset_coefs()[&tkn];
                (A4i, ndarray::array![1.0, 0.0, 0.0], PowerConeT(0.5))
            }
            _ => {
                let mut A4i = Array2::<f64>::zeros((3, k));
                A4i[[0, i]] = -p.get_omnipool_lrna_coefs()[&tkn];
                A4i[[1, n + i]] = -p.get_omnipool_asset_coefs()[&tkn];
                (A4i, Array1::<f64>::ones(3), PowerConeT(0.5))
            }
        };

        A4 = ndarray::concatenate![Axis(0), A4, A4i];
        //b4.append(Axis(0),(&b4i).into());
        b4 = ndarray::concatenate![Axis(0), b4, b4i];
        cones4.push(cone);
    }

    let A4_trimmed = A4.select(Axis(1), &indices_to_keep);

    let (A5_trimmed, b5, cones5) = p.get_stableswap_bounds(indices_to_keep.clone());

    let mut A6 = Array2::<f64>::zeros((0, k));
    /*
       for i in range(n):
       A6i = np.zeros((2, k))
       A6i[0, i] = -1  # lrna_lambda + yi >= 0
       A6i[0, 2*n+i] = -1  # lrna_lambda + yi >= 0
       A6i[1, n+i] = -1  # lambda + xi >= 0
       A6i[1, 3*n+i] = -1  # lambda + xi >= 0
       A6 = np.vstack([A6, A6i])

    */
    for i in 0..n {
        let mut A6i = Array2::<f64>::zeros((2, k));
        A6i[[0, i]] = -1.0;
        A6i[[0, 2 * n + i]] = -1.0;
        A6i[[1, n + i]] = -1.0;
        A6i[[1, 3 * n + i]] = -1.0;
        A6 = ndarray::concatenate![Axis(0), A6, A6i];
    }
    let A6_trimmed = A6.select(Axis(1), &indices_to_keep);
    let b6 = Array1::<f64>::zeros(A6.shape()[0]);
    let cone6 = NonnegativeConeT(A6.shape()[0]);

    let mut A7 = Array2::<f64>::zeros((0, k));
    for i in 0..sigma {
        let mut A7i = Array2::<f64>::zeros((1, k));
        A7i[[0, 4 * n + i]] = -1.0;
        A7i[[0, 4 * n + sigma + i]] = -1.0;
        A7 = ndarray::concatenate![Axis(0), A7, A7i];
    }
    let A7_trimmed = A7.select(Axis(1), &indices_to_keep);
    let b7 = Array1::<f64>::zeros(A7.shape()[0]);
    let cone7 = NonnegativeConeT(A7.shape()[0]);

    let A = ndarray::concatenate![
        Axis(0),
        A1_trimmed,
        A2_trimmed,
        A3_trimmed,
        A4_trimmed,
        A5_trimmed,
        A6_trimmed,
        A7_trimmed
    ];

    // convert a1_trimmed to vec of vec<f64>, note that to_vec does not exist
    let shape = A1_trimmed.shape();
    let mut a_vec = Vec::new();
    for idx in 0..shape[0] {
        let a1_q = A1_trimmed.select(Axis(0), &[idx]);
        let (v, _) = a1_q.into_raw_vec_and_offset();
        a_vec.push(v);
    }
    let A1_trimmed = a_vec;
    let A1_trimmed = CscMatrix::from(&A1_trimmed);

    // convert a2 trimmed
    let shape = A2_trimmed.shape();
    let mut a_vec = Vec::new();
    for idx in 0..shape[0] {
        let a2_q = A2_trimmed.select(Axis(0), &[idx]);
        let (v, _) = a2_q.into_raw_vec_and_offset();
        a_vec.push(v);
    }
    let A2_trimmed = a_vec;
    let A2_trimmed = CscMatrix::from(&A2_trimmed);

    // convert a3 trimmed
    let shape = A3_trimmed.shape();
    let mut a_vec = Vec::new();
    for idx in 0..shape[0] {
        let a3_q = A3_trimmed.select(Axis(0), &[idx]);
        let (v, _) = a3_q.into_raw_vec_and_offset();
        a_vec.push(v);
    }
    let A3_trimmed = a_vec;
    let A3_trimmed = CscMatrix::from(&A3_trimmed);

    // convert a4 trimmed
    let shape = A4_trimmed.shape();
    let mut a_vec = Vec::new();
    for idx in 0..shape[0] {
        let a4_q = A4_trimmed.select(Axis(0), &[idx]);
        let (v, _) = a4_q.into_raw_vec_and_offset();
        a_vec.push(v);
    }
    let A4_trimmed = a_vec;
    let A4_trimmed = CscMatrix::from(&A4_trimmed);

    // Convert a5 trimmed
    let shape = A5_trimmed.shape();
    let mut a_vec = Vec::new();
    for idx in 0..shape[0] {
        let a5_q = A5_trimmed.select(Axis(0), &[idx]);
        let (v, _) = a5_q.into_raw_vec_and_offset();
        a_vec.push(v);
    }
    let A5_trimmed = a_vec;
    let A5_trimmed = CscMatrix::from(&A5_trimmed);

    // Convert a6 trimmed
    let shape = A6_trimmed.shape();
    let mut a_vec = Vec::new();
    for idx in 0..shape[0] {
        let a6_q = A6_trimmed.select(Axis(0), &[idx]);
        let (v, _) = a6_q.into_raw_vec_and_offset();
        a_vec.push(v);
    }
    let A6_trimmed = a_vec;
    let A6_trimmed = CscMatrix::from(&A6_trimmed);

    // Convert a7 trimmed
    let shape = A7_trimmed.shape();
    let mut a_vec = Vec::new();
    for idx in 0..shape[0] {
        let a7_q = A7_trimmed.select(Axis(0), &[idx]);
        let (v, _) = a7_q.into_raw_vec_and_offset();
        a_vec.push(v);
    }
    let A7_trimmed = a_vec;
    let A7_trimmed = CscMatrix::from(&A7_trimmed);

    let A = if A2_trimmed.n != 0 {
        CscMatrix::vcat(&A1_trimmed, &A2_trimmed)
    } else {
        A1_trimmed
    };
    let A = CscMatrix::vcat(&A, &A3_trimmed);
    let A = CscMatrix::vcat(&A, &A4_trimmed);
    //TODO: in some cases it results in A5 with shape 0,0 - so can we just excklude it ?
    let A = if A5_trimmed.n != 0 {
        CscMatrix::vcat(&A, &A5_trimmed)
    } else {
        A
    };
    let A = if A6_trimmed.n != 0 {
        CscMatrix::vcat(&A, &A6_trimmed)
    } else {
        A
    };
    let A = if A7_trimmed.n != 0 {
        CscMatrix::vcat(&A, &A7_trimmed)
    } else {
        A
    };
    let b = ndarray::concatenate![Axis(0), b1, b2, b3, b4, b5, b6, b7];

    let mut cones = vec![];
    cones.extend(cones1.into_iter());
    cones.push(cone2);
    cones.push(cone3);
    cones.extend(cones4.into_iter());
    cones.extend(cones5.into_iter());
    cones.push(cone6);
    cones.push(cone7);

    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .build()
        .unwrap();
    let mut solver = DefaultSolver::new(&P_trimmed, &q_trimmed, &A, &b.to_vec(), &cones, settings);
    solver.solve();
    let x = solver.solution.x;
    let status = solver.solution.status;
    let solve_time = solver.solution.solve_time;
    let obj_value = solver.solution.obj_val;
    let obj_value_dual = solver.solution.obj_val_dual;
    //println!("status: {:?}", status);
    //println!("time: {:?}", solve_time);

    let mut new_omnipool_deltas = BTreeMap::new();
    let mut new_amm_deltas = vec![];
    let mut exec_intent_deltas = vec![0.0; partial_intents_len];
    let mut x_expanded = vec![0.0; k];
    for (i, &index) in indices_to_keep.iter().enumerate() {
        x_expanded[index] = x[i];
    }
    let x_scaled = p.get_real_x(x_expanded.clone());
    for i in 0..n {
        let tkn = p.all_asset_ids[i];
        new_omnipool_deltas.insert(tkn, x_scaled[n + i]);
    }
    for j in 0..partial_intents_len {
        exec_intent_deltas[j] = -x_scaled[4 * n + 2 * sigma + u + j];
    }

    let mut offset = 0;
    for amm in p.amm_store.stablepools.iter() {
        let mut deltas = vec![x_scaled[4 * n + offset]];
        for t in 0..amm.assets.len() {
            deltas.push(x_scaled[4 * n + offset + t + 1]);
        }
        new_amm_deltas.push(deltas);
        offset += amm.assets.len() + 1;
    }

    let obj_offset = if let Some(I) = p.get_indicators() {
        let v = I.iter().map(|&x| x as f64).collect::<Vec<_>>();
        objective_I_coefs.to_vec().dot(&v)
    } else {
        0.0
    };
    (
        new_omnipool_deltas,
        exec_intent_deltas,
        Array2::from_shape_vec((k, 1), x_expanded).unwrap(),
        p.scale_obj_amt(obj_value + obj_offset),
        p.scale_obj_amt(obj_value_dual + obj_offset),
        status.into(),
        new_amm_deltas,
    )
}

fn scale_down_partial_intents(
    p: &ICEProblem,
    trade_pcts: &[f64],
    scale: f64,
) -> (Option<Vec<f64>>, usize) {
    let mut zero_ct = 0;
    let mut intent_sell_maxs = p.partial_sell_maxs.clone();

    for (i, &m) in p.partial_sell_maxs.iter().enumerate() {
        let old_sell_quantity = m * trade_pcts[i];
        let mut new_sell_max = m / scale;

        if old_sell_quantity < new_sell_max {
            let partial_intent_idx = p.partial_indices[i];
            let intent = p.intents[partial_intent_idx].clone();
            let tkn = intent.asset_in;
            let sell_amt_lrna_value = new_sell_max * p.get_asset_pool_data(tkn).hub_price;

            if sell_amt_lrna_value < p.min_partial {
                new_sell_max = 0.0;
                zero_ct += 1;
            }
            intent_sell_maxs[i] = new_sell_max;
        }
    }

    (Some(intent_sell_maxs), zero_ct)
}

fn get_directional_flags(amm_deltas: &BTreeMap<AssetId, f64>) -> BTreeMap<AssetId, i8> {
    let mut flags = BTreeMap::new();
    for (&tkn, &delta) in amm_deltas.iter() {
        let flag = if delta > 0.0 {
            1
        } else if delta < 0.0 {
            -1
        } else {
            0
        };
        flags.insert(tkn, flag);
    }
    flags
}
