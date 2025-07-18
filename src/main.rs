use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};


use clap::{Parser};
use comfy_table::{Table, Cell};
use comfy_table::presets::UTF8_FULL_CONDENSED;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use float_ord::FloatOrd;
use itertools::Itertools;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(name="v_s_rs", about="Computes V(S) function based on trajectories.")]
struct Args {
    /// Dataset to parse (one JSON array per line)
    #[arg(long)]
    dataset: String,

    /// Which field(s) contains the `state` (can repeat)
    #[arg(long = "state-field")]
    state_field: Vec<String>,

    /// Which field contains the immediate `reward`
    #[arg(long = "reward-field")]
    reward_field: String,

    /// If provided, performs every-state-visit estimation
    #[arg(long)]
    every_visit: bool,

    /// Discount factor for future rewards
    #[arg(long, default_value_t = 1.0)]
    discount: f64,

    /// Confidence interval (e.g. 95)
    #[arg(long, default_value_t = 95.0)]
    ci: f64,

    /// Number of decimal places to round to
    #[arg(long, default_value_t = 3)]
    round: usize,

    /// Minimum observations filter
    #[arg(long = "min-observations", default_value_t = 0)]
    min_obs: usize,

    /// Upper Windsorize bound
    #[arg(long = "ub-windsorize")]
    ub_wind: Option<f64>,

    /// Lower Windsorize bound
    #[arg(long = "lb-windsorize")]
    lb_wind: Option<f64>,

    /// Random seed
    #[arg(long = "random-seed")]
    seed: Option<u64>,

    /// Number of bootstrap samples for CI
    #[arg(long = "bootstrap-samples", default_value_t = 2000)]
    samples: usize,

    /// Estimator: "mean" or "median"
    #[arg(long, value_parser = ["mean","median"], default_value = "mean")]
    estimator: String,

    /// Number of permutations/bootstraps for comparison test
    #[arg(long = "comparison-test")]
    comparison_test: Option<usize>,

    /// Comma-separated list of fields to group on for comparison
    #[arg(long = "comparison-group-fields")]
    comparison_group_fields: Option<String>,

    /// Compute P(B >= A) instead of P(B > A)
    #[arg(long = "probability-gte")]
    probability_gte: bool,
}

fn permutation_test(
    a_data: &[f64],
    b_data: &[f64],
    rng: &mut impl Rng,
    func: fn(&[f64]) -> f64,
    b_perm: usize,
    two_sided: bool,
) -> (f64, f64) {
    let obs = func(a_data) - func(b_data);
    let mut pool = Vec::with_capacity(a_data.len() + b_data.len());
    pool.extend_from_slice(a_data);
    pool.extend_from_slice(b_data);
    let n = a_data.len();
    let mut stats = Vec::with_capacity(b_perm);

    for _ in 0..b_perm {
        pool.shuffle(rng);
        let a_slice = &pool[..n];
        let b_slice = &pool[n..];
        stats.push(func(a_slice) - func(b_slice));
    }

    let p_val = if two_sided {
        stats.iter().filter(|&&x| x.abs() >= obs.abs()).count() as f64 / b_perm as f64
    } else {
        stats.iter().filter(|&&x| x >= obs).count() as f64 / b_perm as f64
    };

    (obs, p_val)
}

fn bootstrap_p_a_b(
    a_data: &[f64],
    b_data: &[f64],
    rng: &mut impl Rng,
    samples: usize,
    gte: bool,
) -> f64 {
    let seed = rng.random::<u64>();
    let count = (0..samples).into_par_iter().map(|idx| {
        let mut prng = XorShiftRng::seed_from_u64(seed + idx as u64);
        let &a = a_data.choose(&mut prng).unwrap();
        let &b = b_data.choose(&mut prng).unwrap();
        if gte {
            if a >= b { 1 } else { 0 }
        } else {
            if a > b { 1 } else { 0 }
        }
    }).sum::<usize>();
    count as f64 / samples as f64
}

#[derive(Copy,Clone,Debug)]
enum Stat {
    Mean,
    Median
}

fn median(it: impl Iterator<Item=f64>) -> f64 {
    let mut vs = it.collect::<Vec<_>>();
    vs.sort_by_key(|v| FloatOrd(*v));

    let mid = vs.len() / 2;
    if vs.len() % 2 == 0 {
        (vs[mid - 1] + vs[mid]) / 2.0
    } else {
        vs[mid]
    }
}

fn bootstrap_ci(
    data: &[f64],
    rng: &mut impl Rng,
    stat: Stat,
    samples: usize,
    alpha: f64,
) -> (f64, f64) {
    let n = data.len();
    let seed = rng.random::<u64>();
    let mut stats: Vec<_> = (0..samples).into_par_iter()
        .map(|si| {
            let mut prng = XorShiftRng::seed_from_u64(seed + si as u64);
            let it = (0..n).map(|_| *data.choose(&mut prng).unwrap());
            match stat {
                Stat::Mean => it.sum::<f64>() / n as f64,
                Stat::Median => median(it)
            }
        })
        .collect();
    stats.sort_by_key(|s| FloatOrd(*s));
    let lo_idx = ((alpha / 2.0) * (samples as f64)).floor() as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * (samples as f64)).ceil().min((samples - 1) as f64) as usize;
    (stats[lo_idx], stats[hi_idx])
}

fn compute_returns(
    orders: &[Value],
    rets: &mut Vec<f64>,
    reward_field: &str,
    discount: f64,
    ub: Option<f64>,
    lb: Option<f64>,
) {
    let mut G = 0.0;
    rets.clear();
    for o in orders.iter().rev() {
        let mut r = o[reward_field].as_f64().unwrap_or(0.0);
        if let Some(u) = ub { r = r.min(u) }
        if let Some(l) = lb { r = r.max(l) }
        G = discount * G + r;
        rets.push(G);
    }
    rets.reverse();
}

fn get_state(order: &Value, fields: &[String]) -> Vec<String> {
    fields.iter().map(|f| {
        match &order[f] {
            Value::String(s) => s.into(),
            field => field.to_string()
        }
    }).collect()
}

fn aggregate_states(args: &Args) -> HashMap<Vec<String>, Vec<f64>> {
    let file = File::open(&args.dataset).expect("cannot open dataset");
    let reader = BufReader::new(file);

    let mut map: HashMap<Vec<String>, Vec<f64>> = HashMap::new();

    let mut rets = Vec::new();
    for line in reader.lines().map(Result::unwrap) {
        let orders: Vec<Value> = serde_json::from_str(&line).unwrap();
        compute_returns(&orders, &mut rets, &args.reward_field,
                                   args.discount, args.ub_wind, args.lb_wind);

        let mut seen: HashSet<Vec<String>> = HashSet::new();
        for (o, &ret) in orders.iter().zip(&rets) {
            let state = get_state(o, &args.state_field);
            if args.every_visit || seen.insert(state.clone()) {
                map.entry(state).or_default().push(ret);
            }
        }
    }
    map
}

fn format_row(
    a_state: &[String],
    b_state: &[String],
    a_len: usize,
    b_len: usize,
    obs: f64,
    p_val: f64,
    p_b_gt_a: f64,
    digits: usize,
    ci_cut: f64,
) -> Vec<Cell> {
    let suffix = if p_val <= ci_cut + 1e-8 { "*" } else { "" };
    vec![
        Cell::new(a_state.join("|")),
        Cell::new(b_state.join("|")),
        Cell::new(a_len.to_string()),
        Cell::new(b_len.to_string()),
        Cell::new(format!("{:.1$}", obs, digits)),
        Cell::new(format!("{:.1$}{2}", p_val, digits, suffix)),
        Cell::new(format!("{:.1$}", p_b_gt_a, digits)),
    ]
}

fn comparison_tests(
    state_values: &HashMap<Vec<String>, Vec<f64>>,
    group_key: &dyn Fn(&Vec<String>) -> Vec<String>,
    ci_cut: f64,
    rng: &mut impl Rng,
    func: fn(&[f64]) -> f64,
    permutations: usize,
    digits: usize,
    min_obs: usize,
    prob_gte: bool,
) {
    // Group states
    let mut groups: HashMap<_, Vec<_>> = HashMap::new();
    let mut sv: Vec<_> = state_values
        .into_iter()
        .collect();

    sv.sort_by_key(|kv| (*kv.0).clone());
    for (state, data) in sv {
        if data.len() < min_obs { continue; }
        let key = group_key(state);
        groups.entry(key).or_default().push((state.clone(), data.clone()));
    }

    // Build table
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED).apply_modifier(UTF8_ROUND_CORNERS);

    let mut hdr = vec!["A","B","|A|","|B|","E[B] - E[A]","P-Value"];
    hdr.push(if prob_gte { "P(B >= A)" } else { "P(B > A)"});

    table.set_header(hdr.iter().map(|h| Cell::new(*h)).collect::<Vec<_>>());

    // Materialize the different permutations so we can do it in parallel.
    type C = (Vec<String>, Vec<f64>);
    let mut groups: Vec<_> = groups.iter().map(|(grp, datasets)| {
        let tups: Vec<(&C, &C)> = datasets.iter().tuple_combinations().collect();
        (grp, datasets, tups)
    }).collect();

    groups.sort_by_key(|(k, _, _)| k.clone());

    let seed = rng.random::<u64>();
    let row_groups: Vec<_> = groups.into_par_iter().enumerate().map(|(i, (_grp, _items, tups))| {

        let rows: Vec<_> = tups.par_iter().enumerate().map(|(ti, (a, b))| {
            let mut rng = XorShiftRng::seed_from_u64(seed + (i << 8 + ti) as u64);
            let (a_state, a_data) = a;
            let (b_state, b_data) = b;
            let (obs, p_val) = permutation_test(b_data, a_data, &mut rng, func, permutations, true);
            let p_b_gt_a = bootstrap_p_a_b(b_data, a_data, &mut rng, permutations, prob_gte);
            format_row(a_state, b_state, a_data.len(), b_data.len(),
                                 obs, p_val, p_b_gt_a, digits, ci_cut)
        }).collect();

        let mut rows_all = Vec::with_capacity(tups.len() + 1);
        rows_all.push(vec![Cell::new("────────────────────────────")]);
        rows_all.extend(rows.into_iter());
        rows_all
    }).collect();

    for rows in row_groups.into_iter() {
        for row in rows.into_iter() {
            table.add_row(row);
        }
    }

    println!("{}", table);
}

fn main() {
    let args = Args::parse();
    let mut rng = args.seed
        .map(XorShiftRng::seed_from_u64)
        .unwrap_or_else(||XorShiftRng::seed_from_u64(20252025));

    // 1) Aggregate returns by state
    let state_values = aggregate_states(&args);

    // 2) Print main V(s) table
    let ci_cut = 1.0 - args.ci / 100.0;
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED).apply_modifier(UTF8_ROUND_CORNERS);
    let mut hdr = args.state_field.clone();
    hdr.push("Observations".into());
    hdr.push("Lower Bound".into());
    hdr.push(format!("Expected {}", args.reward_field));
    hdr.push("Upper Bound".into());
    table.set_header(hdr.iter().map(|h| Cell::new(h)).collect::<Vec<_>>());

    // Sort states lexicographically
    let mut keys: Vec<_> = state_values.keys().cloned().collect();
    keys.sort();
    for state in keys {
        let data = &state_values[&state];
        if data.len() < args.min_obs { continue; }
        let mean = data.iter().sum::<f64>() / (data.len() as f64);
        let (lb, ub) = bootstrap_ci(data, &mut rng,
            Stat::Mean,
            args.samples,
            ci_cut);

        let mut row = state.iter().map(|s| Cell::new(s)).collect::<Vec<_>>();
        row.push(Cell::new(data.len().to_string()));
        row.push(Cell::new(format!("{:.1$}", lb, args.round)));
        row.push(Cell::new(format!("{:.1$}", mean, args.round)));
        row.push(Cell::new(format!("{:.1$}", ub, args.round)));
        table.add_row(row);
    }
    println!("{}", table);

    // 3) If requested: run pairwise comparison tests
    if let Some(nperm) = args.comparison_test {
        let func: fn(&[f64]) -> f64 = match args.estimator.as_str() {
            "median" => |d| {
                median(d.iter().cloned())
            },
            _ => |d| d.iter().sum::<f64>() / (d.len() as f64),
        };

        // prepare grouping function
        let group_fn: Box<dyn Fn(&Vec<String>) -> Vec<String>> =
            if let Some(ref s) = args.comparison_group_fields {
                let idxs: Vec<usize> = s.split(',')
                    .map(|i| i.parse().expect("Need to be an integer")).collect();
                Box::new(move |st: &Vec<String>| {
                    idxs.iter().map(|&i| st[i].clone()).collect()
                })
            } else {
                Box::new(|_: &Vec<String>| vec![])
            };

        comparison_tests(
            &state_values,
            &*group_fn,
            ci_cut,
            &mut rng,
            func,
            nperm,
            args.round,
            args.min_obs,
            args.probability_gte,
        );
    }
}
