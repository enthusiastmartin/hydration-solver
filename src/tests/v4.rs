use crate::tests::load_amm_state;
use crate::types::*;
use crate::v4::SolverV4;
use std::time::Instant;

const DATA_STRIPPED: &str = r##"[{"Omnipool":{"asset_id":0,"reserve":86787892196719287820,"hub_reserve":39349774149749914,"decimals":12,"fee":[1500,1000000],"hub_fee":[500,1000000]}},{"Omnipool":{"asset_id":28,"reserve":4556265753708959318053,"hub_reserve":9612298398114154,"decimals":15,"fee":[1500,1000000],"hub_fee":[500,1000000]}},{"Omnipool":{"asset_id":33,"reserve":6993437475117384536714797,"hub_reserve":13119005909651208,"decimals":18,"fee":[1500,1000000],"hub_fee":[500,1000000]}},{"Omnipool":{"asset_id":102,"reserve":14934592932069234854488578,"hub_reserve":589363778311062935,"decimals":18,"fee":[1500,1000000],"hub_fee":[500,1000000]}},{"Omnipool":{"asset_id":8,"reserve":5086493193287732516,"hub_reserve":29474085676606977,"decimals":12,"fee":[1500,1000000],"hub_fee":[500,1000000]}},{"Omnipool":{"asset_id":1000765,"reserve":10415120493009989839,"hub_reserve":36587988309690706,"decimals":18,"fee":[1500,1000000],"hub_fee":[500,1000000]}}]"##;

#[test]
fn simple_scenario() {
    let intents: &str = r##"[
  {
    "intent_id": 0,
    "asset_in": 33,
    "asset_out": 28,
    "amount_in": 1341239791268921631091765,
    "amount_out": 1073349042192081223680,
    "partial": true
  },
  {
    "intent_id": 1,
    "asset_in": 102,
    "asset_out": 28,
    "amount_in": 1692553304022051096953112,
    "amount_out": 28494216431468577554432,
    "partial": false
  },
  {
    "intent_id": 2,
    "asset_in": 8,
    "asset_out": 1000765,
    "amount_in": 226596254840150354,
    "amount_out": 336389845986943552,
    "partial": false
  }
]"##;

    let intents = serde_json::from_str::<Vec<Intent>>(intents).unwrap();
    let data = load_amm_state();

    let accepted = vec![0, 33, 28, 102, 8, 1000765];
    let data = data
        .into_iter()
        .filter(|asset| match asset {
            Asset::Omnipool(v) => accepted.contains(&v.asset_id),
            Asset::StableSwap(v) => accepted.contains(&v.asset_id),
        })
        .collect::<Vec<Asset>>();

    let solution = SolverV4::solve(intents, data).unwrap();
    dbg!(solution.resolved_intents);
}

#[test]
fn solver_should_find_solution_for_one_small_amount_partial_intent() {
    let data = load_amm_state();
    let intents = vec![Intent {
        intent_id: 0,
        asset_in: 0u32,
        asset_out: 27u32,
        amount_in: 100_000_000_000_000,
        amount_out: 1_149_000_000_000,
        partial: true,
    }];
    let solution = SolverV4::solve(intents, data).unwrap();
    let expected_solution = vec![ResolvedIntent {
        intent_id: 0,
        amount_in: 100_000_000_000_000,
        amount_out: 1_149_000_000_000,
    }];
    assert_eq!(solution.resolved_intents, expected_solution);
}

#[test]
fn solver_should_find_solution_for_one_large_amount_partial_intent() {
    let data = load_amm_state();
    let intents = vec![Intent {
        intent_id: 0,
        asset_in: 0u32,
        asset_out: 27u32,
        amount_in: 1_000_000_000_000_000_000,
        amount_out: 1_149_000_000_000_000,
        partial: true,
    }];
    let solution = SolverV4::solve(intents, data).unwrap();
    let expected_solution = vec![ResolvedIntent {
        intent_id: 0,
        amount_in: 1_000_000_000_000_000_000,
        amount_out: 1_149_000_000_000_000,
    }];
    assert_eq!(solution.resolved_intents, expected_solution);
}
#[test]
fn solver_should_find_solution_for_one_large_amount_full_intent() {
    let data = load_amm_state();
    let intents = vec![Intent {
        intent_id: 0,
        asset_in: 0u32,
        asset_out: 27u32,
        amount_in: 1_000_000_000_000_000_000,
        amount_out: 1_149_000_000_000_000,
        partial: false,
    }];
    let solution = SolverV4::solve(intents, data).unwrap();
    let expected_solution = vec![ResolvedIntent {
        intent_id: 0,
        amount_in: 1_000_000_000_000_000_000,
        amount_out: 1_149_000_000_000_000,
    }];
    assert_eq!(solution.resolved_intents, expected_solution);
}

#[test]
fn solver_should_find_solution_for_two_intents() {
    let data = load_amm_state();
    let intents = vec![
        Intent {
            intent_id: 0,
            asset_in: 0u32,
            asset_out: 27u32,
            amount_in: 1_000_000_000_000_000_000,
            amount_out: 1_149_000_000_000_000,
            partial: false,
        },
        Intent {
            intent_id: 1,
            asset_in: 20,
            asset_out: 8,
            amount_in: 165_453_758_222_187_283_838,
            amount_out: 2808781311006261193,
            partial: true,
        },
    ];
    let start = Instant::now();
    let solution = SolverV4::solve(intents, data).unwrap();
    let duration = start.elapsed();
    println!("Time elapsed in solve() is: {:?}", duration);
    let expected_solution = vec![ResolvedIntent {
        intent_id: 0,
        amount_in: 1_000_000_000_000_000_000,
        amount_out: 1_149_000_000_000_000,
    }];
    assert_eq!(solution.resolved_intents, expected_solution);
}

#[test]
fn solver_should_find_solution_for_two_partial_intents() {
    let data = load_amm_state();
    let intents = vec![
        Intent {
            intent_id: 0,
            asset_in: 12,
            asset_out: 14,
            amount_in: 9206039265427194,
            amount_out: 1,
            partial: true,
        },
        Intent {
            intent_id: 1,
            asset_in: 28,
            asset_out: 8,
            amount_in: 1076105965030805693,
            amount_out: 1,
            partial: true,
        },
    ];
    let start = Instant::now();
    let solution = SolverV4::solve(intents, data).unwrap();
    let duration = start.elapsed();
    println!("Time elapsed in solve() is: {:?}", duration);
    let expected_solution = vec![
        ResolvedIntent {
            intent_id: 0,
            amount_in: 9206039265427194,
            amount_out: 1,
        },
        ResolvedIntent {
            intent_id: 1,
            amount_in: 1076105965030805693,
            amount_out: 1,
        },
    ];
    assert_eq!(solution.resolved_intents, expected_solution);
}

#[test]
fn solver_should_work_with_stableswap_intent() {
    let data = load_amm_state();
    let intents = vec![
        Intent {
            intent_id: 0,
            asset_in: 13,
            asset_out: 5,
            amount_in: 514888002332937478066650,
            amount_out: 664083505362373041510455118870258,
            partial: false,
        },
        Intent {
            intent_id: 1,
            asset_in: 20,
            asset_out: 14,
            amount_in: 165665617143487433531,
            amount_out: 12177733280754553178994,
            partial: true,
        },
        Intent {
            intent_id: 2,
            asset_in: 0,
            asset_out: 16,
            amount_in: 25528234672916292207,
            amount_out: 871403327041354,
            partial: false,
        },
        Intent {
            intent_id: 3,
            asset_in: 100,
            asset_out: 101,
            amount_in: 303603756622822659947591,
            amount_out: 20555903343957624238452664953,
            partial: false,
        },
    ];
    let start = Instant::now();
    let solution = SolverV4::solve(intents, data).unwrap();
    let duration = start.elapsed();
    println!("Time elapsed in solve() is: {:?}", duration);
    let expected_solution = vec![
        ResolvedIntent {
            intent_id: 0,
            amount_in: 1_000_000_000_000_000_000,
            amount_out: 1_149_000_000_000_000,
        },
        ResolvedIntent {
            intent_id: 1,
            amount_in: 36895351807444140032,
            amount_out: 626344035537618048,
        },
    ];
    assert_eq!(solution.resolved_intents, expected_solution);
}

#[test]
fn solver_should_fail_when_it_contains_non_existing_trade_assets() {
    let data = load_amm_state();
    let intents = vec![
        Intent {
            intent_id: 0,
            asset_in: 123344513,
            asset_out: 5,
            amount_in: 514888002332937478066650,
            amount_out: 664083505362373041510455118870258,
            partial: false,
        },
        Intent {
            intent_id: 1,
            asset_in: 20,
            asset_out: 14,
            amount_in: 165665617143487433531,
            amount_out: 12177733280754553178994,
            partial: true,
        },
        Intent {
            intent_id: 2,
            asset_in: 0,
            asset_out: 16,
            amount_in: 25528234672916292207,
            amount_out: 871403327041354,
            partial: false,
        },
        Intent {
            intent_id: 3,
            asset_in: 100,
            asset_out: 101,
            amount_in: 303603756622822659947591,
            amount_out: 20555903343957624238452664953,
            partial: false,
        },
    ];
    let start = Instant::now();
    let solution = SolverV4::solve(intents, data);
    let duration = start.elapsed();
    println!("Time elapsed in solve() is: {:?}", duration);
    assert!(solution.is_err());
}
