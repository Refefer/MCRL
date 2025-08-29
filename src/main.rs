//! # V(S) Computation CLI
//!
//! This file implements a command‑line tool that reads a dataset of trajectories
//! (each line is a JSON array of state/action/reward objects) and computes the
//! expected discounted return for each unique state configuration.  It also
//! provides a suite of statistical inference utilities – bootstrap confidence
//! intervals, permutation tests, and a Bayesian bootstrap – to compare the
//! values of two groups of states.
//!
//! The implementation is deliberately split into small, well‑named functions so
//! that the overall architecture is clear:
//!
//! * **Argument parsing** – `clap` derives a struct (`Args`) that holds all
//!   command‑line options.
//! * **Data ingestion** – `aggregate_states` streams the file line‑by‑line,
//!   computes returns for each trajectory, and aggregates them by a user‑defined
//!   state key.
//! * **Statistical utilities** – a collection of generic functions for
//!   permutation testing, bootstrapping, and Bayesian bootstrapping.
//! * **Result presentation** – two tables are printed using `comfy_table` –
//!   one for the raw V(s) estimates and a second (optional) for pair‑wise
//!   comparison tests.
//!
//! The comments below aim to explain the *why* behind each component, the
//! assumptions made (e.g. independence of returns, the meaning of "windsorizing"
//! for outlier handling, the mathematical basis of permutation tests), and the
//! performance considerations (parallelism with Rayon, deterministic RNG, etc.).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use clap::{Parser}; // Command‑line argument parsing.
use comfy_table::{Table, Cell}; // Pretty‑printing tables to the terminal.
use comfy_table::presets::UTF8_FULL_CONDENSED;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use float_ord::FloatOrd; // Allows ordering of floating point numbers (NaN handling).
use itertools::Itertools; // Provides iterator adapters like `tuple_combinations`.
use rand::prelude::*; // RNG traits.
use rand_xorshift::XorShiftRng; // Fast, deterministic RNG that can be seeded.
use rand_distr::Dirichlet; // Dirichlet distribution for the Bayesian bootstrap.
use rayon::prelude::*; // Data‑parallelism.
use serde_json::Value; // JSON handling.

/// Command‑line interface definition.
///
/// `clap` will automatically generate help text and parse the arguments into
/// this struct.  Each field maps to a CLI flag and includes documentation that
/// appears in `--help`.
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

    /// If provided, performs every‑state‑visit estimation (otherwise first‑visit)
    #[arg(long)]
    every_visit: bool,

    /// Discount factor for future rewards (γ in reinforcement learning)
    #[arg(long, default_value_t = 1.0)]
    discount: f64,

    /// Confidence interval (e.g. 95) – later converted to α = 1 - ci/100
    #[arg(long, default_value_t = 95.0)]
    ci: f64,

    /// Number of decimal places to round numbers when printing
    #[arg(long, default_value_t = 3)]
    round: usize,

    /// Minimum observations filter – states with fewer samples are omitted
    #[arg(long = "min-observations", default_value_t = 0)]
    min_obs: usize,

    /// Upper Windsorize bound – caps extreme high rewards
    #[arg(long = "ub-windsorize")]
    ub_wind: Option<f64>,

    /// Lower Windsorize bound – floors extreme low rewards
    #[arg(long = "lb-windsorize")]
    lb_wind: Option<f64>,

    /// Random seed – enables reproducible bootstraps/permutations
    #[arg(long = "random-seed")]
    seed: Option<u64>,

    /// Number of bootstrap samples for CI computation
    #[arg(long = "bootstrap-samples", default_value_t = 2000)]
    samples: usize,

    /// Estimator choice – "mean" or "median" for the central tendency
    #[arg(long, value_parser = ["mean","median"], default_value = "mean")]
    estimator: String,

    /// Number of permutation/bootstraps for pairwise comparison test
    #[arg(long = "comparison-test")]
    comparison_test: Option<usize>,

    /// Alpha parameter for Bayesian probability.  α = 1 yields a uniform prior.
    /// α < 1 is optimistic (puts more weight on higher returns), α > 1 is more
    /// empirical (concentrates probability mass around observed data).
    #[arg(long = "alpha")]
    alpha: Option<f64>,

    /// Comma‑separated list of fields to group on for comparison (e.g. "0,2")
    #[arg(long = "comparison-group-fields")]
    comparison_group_fields: Option<String>,

    /// Compute P(B ≥ A) instead of the strict inequality P(B > A)
    #[arg(long = "probability-gte")]
    probability_gte: bool,
}

/* ---------------------------------------------------------------------------
 *  Permutation Test
 * ---------------------------------------------------------------------------
 * A permutation test evaluates the null hypothesis that two samples come from
 * the same distribution.  By repeatedly shuffling the combined data and
 * recomputing the test statistic (here: difference of a user‑provided function
 * – typically the mean or median), we obtain an empirical distribution of the
 * statistic under the null.  The p‑value is the proportion of shuffled
 * statistics that are at least as extreme as the observed one.
 *
 * Assumptions:
 *   * Exchangeability – the samples are identically distributed under H0.
 *   * Independence between observations (no autocorrelation).
 *   * The statistic function is deterministic and operates on a slice.
 *
 * The implementation is deliberately generic: `func` can be any fn(&[f64]) -> f64.
 */
fn permutation_test(
    a_data: &[f64],
    b_data: &[f64],
    rng: &mut impl Rng,
    func: fn(&[f64]) -> f64,
    b_perm: usize,
    two_sided: bool,
) -> (f64, f64) {
    // Observed statistic (difference between groups).
    let obs = func(a_data) - func(b_data);
    // Pool both groups together for shuffling.
    let mut pool = Vec::with_capacity(a_data.len() + b_data.len());
    pool.extend_from_slice(a_data);
    pool.extend_from_slice(b_data);
    let n = a_data.len(); // Size of the first group.
    let mut stats = Vec::with_capacity(b_perm);

    for _ in 0..b_perm {
        // In‑place Fisher‑Yates shuffle.
        pool.shuffle(rng);
        let a_slice = &pool[..n];
        let b_slice = &pool[n..];
        stats.push(func(a_slice) - func(b_slice));
    }

    // Compute p‑value based on the chosen tail.
    let p_val = if two_sided {
        stats.iter().filter(|&&x| x.abs() >= obs.abs()).count() as f64 / b_perm as f64
    } else {
        stats.iter().filter(|&&x| x >= obs).count() as f64 / b_perm as f64
    };

    (obs, p_val)
}

/* ---------------------------------------------------------------------------
 *  Bootstrap for P(E[B] > E[A])
 * ---------------------------------------------------------------------------
 * The function `bootstrap_p_e_a_b` estimates the probability that the *expected*
 * value of group B exceeds that of group A.  It repeatedly draws bootstrap
 * samples (with replacement) from each group, computes the sample means, and
 * counts how often the B mean is larger.  This is a classic non‑parametric
 * approach that approximates the posterior probability of the inequality under
 * a uniform prior on the empirical distribution.
 */
fn bootstrap_p_e_a_b(
    a_data: &[f64],
    b_data: &[f64],
    rng: &mut impl Rng,
    samples: usize,
) -> f64 {
    // A deterministic seed offset ensures reproducibility across parallel runs.
    let seed = rng.gen::<u64>();
    let count = (0..samples).into_par_iter().map(|idx| {
        let mut prng = XorShiftRng::seed_from_u64(seed + idx as u64);

        let a_sample_mean: f64 = (0..a_data.len())
            .map(|_| *a_data.choose(&mut prng).unwrap())
            .sum::<f64>() / a_data.len() as f64;

        let b_sample_mean: f64 = (0..b_data.len())
            .map(|_| *b_data.choose(&mut prng).unwrap())
            .sum::<f64>() / b_data.len() as f64;

        if a_sample_mean > b_sample_mean { 1 } else { 0 }
    }).sum::<usize>();

    count as f64 / samples as f64
}

/* ---------------------------------------------------------------------------
 *  Bootstrap for P(B > A) (or ≥) – direct comparison of draws
 * ---------------------------------------------------------------------------
 * Instead of comparing sample means we compare *single* bootstrap draws.
 * This yields a Monte‑Carlo estimate of the probability that a randomly drawn
 * value from B exceeds a randomly drawn value from A.
 */
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

/* ---------------------------------------------------------------------------
 *  Core statistical building blocks
 * ---------------------------------------------------------------------------
 */
#[derive(Copy,Clone,Debug)]
enum Stat {
    Mean,
    Median
}

/// Compute the median of an iterator of f64 values.
/// The iterator is collected, sorted with `FloatOrd` (handles NaN) and the
/// middle element(s) are returned.
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

/* ---------------------------------------------------------------------------
 *  Bootstrap confidence interval
 * ---------------------------------------------------------------------------
 * Returns a (lower, upper) tuple that encloses `(1‑α)` of the bootstrap
 * distribution of the chosen statistic (mean or median).  The algorithm:
 *   1. Draw `samples` bootstrap resamples (with replacement).
 *   2. Compute the statistic for each resample.
 *   3. Sort the results and take the α/2 and 1‑α/2 quantiles.
 * The function is deliberately parallel using Rayon for speed on large
 * datasets.
 */
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

/* ---------------------------------------------------------------------------
 *  Bayesian Bootstrap
 * ---------------------------------------------------------------------------
 * The Bayesian bootstrap (Rubin, 1981) treats the empirical distribution as a
 * Dirichlet‑distributed random measure with concentration parameter α.  Sampling
 * from a Dirichlet yields random weights that sum to 1; the weighted average of
 * the observed data then constitutes a draw from the posterior over the true
 * distribution.
 *
 * In this implementation we fix a subsample size of 1 000.  When the original
 * dataset is smaller than that, we sample *with* replacement; otherwise we take a
 * subsample *without* replacement to preserve the empirical distribution.
 */
fn bayesian_bootstrap(
    a_data: &[f64],
    alpha: f64,
    n_bootstrap: usize,
    rng: &mut impl Rng,
) -> Vec<f64> {
    // Fixed subsample size (chosen empirically for a good trade‑off).
    const subsample_size: usize = 1_000;

    let n = a_data.len();
    // Dirichlet concentration vector – same α for each component.
    let alpha = [alpha; subsample_size];
    let dist = Dirichlet::new(alpha).expect("Error with alpha selected for Dirichlet distribution!");
    let mut results = vec![0f64; n_bootstrap];

    let seed = rng.random::<u64>();

    // Decide whether to sample with replacement based on data size.
    let replacement = n < subsample_size;
    results.par_iter_mut().enumerate().for_each(|(i, vi)| {
        let mut prng = XorShiftRng::seed_from_u64(seed + i as u64);
        let mut sample = [1.0; subsample_size];
        if replacement {
            // Sample *with* replacement; each draw picks a random element.
            sample.iter_mut().for_each(|si| {
                *si = *a_data.choose(&mut prng).unwrap();
            });
        } else {
            // Sample *without* replacement using `choose_multiple`.
            for (b, si) in a_data.choose_multiple(&mut prng, sample.len()).zip(sample.iter_mut()) {
                *si = *b;
            }
        }
        // Draw Dirichlet weights and compute weighted sum.
        let weights = dist.sample(&mut prng);
        *vi = weights.into_iter().zip(sample.into_iter()).map(|(w, s)| w * s).sum::<f64>();
    });

    results

}

/// Probability that a random draw from `a_data` exceeds a draw from `b_data`
/// under the Bayesian bootstrap.
fn bayesian_bootstrap_p_a_b(
    a_data: &[f64],
    b_data: &[f64],
    alpha: f64,
    n_bootstrap: usize,
    rng: &mut impl Rng
) -> f64 {
    let a_boot = bayesian_bootstrap(a_data, alpha, n_bootstrap, rng);
    let b_boot = bayesian_bootstrap(b_data, alpha, n_bootstrap, rng);
    let successes = a_boot.into_iter().zip(b_boot.into_iter()).map(|(ai, bi)| {
        if (ai - bi) > 0f64 {1f64} else {0f64}
    }).sum::<f64>();
    successes / n_bootstrap as f64
}

/* ---------------------------------------------------------------------------
 *  Return computation for a single trajectory
 * ---------------------------------------------------------------------------
 * The function iterates a trajectory *backwards* to compute the discounted
 * return `G_t = r_t + γ * G_{t+1}`.  This mirrors the standard reinforcement‑
 * learning definition of the return.  Optional Windsorizing clamps each reward
 * before accumulation – a simple robustification against outliers.
 */
fn compute_returns(
    orders: &[Value],
    rets: &mut Vec<f64>,
    reward_field: &str,
    discount: f64,
    ub: Option<f64>,
    lb: Option<f64>,
) {
    let mut G = 0.0;
    for o in orders.iter().rev() {
        let mut r = o[reward_field].as_f64().unwrap_or(0.0);
        if let Some(u) = ub { r = r.min(u) }
        if let Some(l) = lb { r = r.max(l) }
        G = discount * G + r;
        rets.push(G);
    }
    // Reverse the vector so that `rets[i]` corresponds to the i‑th step.
    rets.reverse();
}

/* ---------------------------------------------------------------------------
 *  State extraction
 * ---------------------------------------------------------------------------
 * `fields` holds the list of JSON keys that together define a *state*.
 * Each field value is converted to a string – numbers become their textual
 * representation, strings are kept as‑is.  The resulting `Vec<String>` is used as
 * a hash map key throughout the program.
 */
fn get_state(order: &Value, fields: &[String]) -> Vec<String> {
    fields.iter().map(|f| {
        match &order[f] {
            Value::String(s) => s.into(),
            field => field.to_string()
        }
    }).collect()
}

/* ---------------------------------------------------------------------------
 *  Aggregation of returns by state
 * ---------------------------------------------------------------------------
 * Reads the dataset line‑by‑line so memory usage stays bounded even for huge
 * files.  For each trajectory we compute discounted returns, then associate each
 * return with the corresponding state key.  The `every_visit` flag controls
 * whether we use every occurrence of a state (first‑visit vs. every‑visit MC).
 */
fn aggregate_states(args: &Args) -> HashMap<Vec<String>, Vec<f64>> {
    let file = File::open(&args.dataset).expect("cannot open dataset");
    let reader = BufReader::new(file);

    let mut map: HashMap<Vec<String>, Vec<f64>> = HashMap::new();

    let mut rets = Vec::new();
    for line in reader.lines().map(Result::unwrap) {
        let orders: Vec<Value> = serde_json::from_str(&line).unwrap();
        rets.clear();
        compute_returns(&orders, &mut rets, &args.reward_field,
                                   args.discount, args.ub_wind, args.lb_wind);

        // Track which states we have already seen in this episode for first‑visit.
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

/* ---------------------------------------------------------------------------
 *  Table row formatting for comparison tests
 * ---------------------------------------------------------------------------
 * Centralises the logic that builds a vector of `Cell`s; this makes the
 * construction of the `comfy_table` rows in `comparison_tests` compact.
 */
fn format_row(
    a_state: &[String],
    b_state: &[String],
    a_len: usize,
    b_len: usize,
    obs: f64,
    p_val: f64,
    p_b_gt_a: f64,
    p_e_b_gt_a: f64,
    bp_b_gt_a: f64,
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
        Cell::new(format!("{:.1$}", p_e_b_gt_a, digits)),
        Cell::new(format!("{:.1$}", bp_b_gt_a, digits)),
    ]
}

/* ---------------------------------------------------------------------------
 *  Pairwise comparison tests across groups of states
 * ---------------------------------------------------------------------------
 * For each user‑defined group (e.g., all states that share certain fields) we
 * generate all unordered pairs of states, compute a suite of statistical tests,
 * and emit a formatted row.  The heavy lifting is performed in parallel.
 */
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
    alpha: f64
) {
    // ---------------------------------------------------------------------
    // Group states by the user‑provided key extractor.
    // ---------------------------------------------------------------------
    let mut groups: HashMap<_, Vec<_>> = HashMap::new();
    let mut sv: Vec<_> = state_values
        .into_iter()
        .collect();

    // Sort deterministic for reproducible output.
    sv.sort_by_key(|kv| (*kv.0).clone());
    for (state, data) in sv {
        if data.len() < min_obs { continue; }
        let key = group_key(state);
        groups.entry(key).or_default().push((state.clone(), data.clone()));
    }

    // ---------------------------------------------------------------------
    // Build the output table.
    // ---------------------------------------------------------------------
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED).apply_modifier(UTF8_ROUND_CORNERS);

    let mut hdr = vec!["A", "B", "|A|", "|B|", "E[B] - E[A]", "P-Value"];
    if prob_gte {
        hdr.push("P(B >= A)");
    } else {
        hdr.push("P(B > A)");
    }
    hdr.push("P(E[B] > E[A])");
    let bayes_col = format!("Bayes P(B > A, alpha={})", alpha);
    hdr.push(&bayes_col);

    table.set_header(hdr.iter().map(|h| Cell::new(*h)).collect::<Vec<_>>());

    // ---------------------------------------------------------------------
    // Materialise pairs per group; then process each group in parallel.
    // ---------------------------------------------------------------------
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
            let p_b_gt_a  = bootstrap_p_a_b(b_data, a_data, &mut rng, permutations, prob_gte);
            let p_e_b_gt_a  = bootstrap_p_e_a_b(b_data, a_data, &mut rng, permutations);
            let bp_b_gt_a = bayesian_bootstrap_p_a_b(b_data, a_data, alpha, permutations, &mut rng);
            format_row(a_state, b_state, a_data.len(), b_data.len(),
                                 obs, p_val, p_b_gt_a, p_e_b_gt_a, bp_b_gt_a, digits, ci_cut)
        }).collect();

        // Insert a visual separator row before each group.
        let mut rows_all = Vec::with_capacity(tups.len() + 1);
        rows_all.push(vec![Cell::new("────────────────────────────")]);
        rows_all.extend(rows.into_iter());
        rows_all
    }).collect();

    // Render all rows.
    for rows in row_groups.into_iter() {
        for row in rows.into_iter() {
            table.add_row(row);
        }
    }

    println!("{}", table);
}

/* ---------------------------------------------------------------------------
 *  Main entry point
 * ---------------------------------------------------------------------------
 * 1. Parse arguments.
 * 2. Initialise a deterministic RNG (either user‑provided seed or a hard‑coded
 *    fallback for reproducibility across runs).
 * 3. Aggregate returns per state.
 * 4. Print the V(s) table with bootstrap confidence intervals.
 * 5. Optionally run the pairwise comparison suite.
 */
fn main() {
    let args = Args::parse();
    let mut rng = args.seed
        .map(XorShiftRng::seed_from_u64)
        .unwrap_or_else(||XorShiftRng::seed_from_u64(20252025));

    // ---------------------------------------------------------------------
    // 1) Aggregate returns by state.
    // ---------------------------------------------------------------------
    let state_values = aggregate_states(&args);

    // ---------------------------------------------------------------------
    // 2) Print main V(s) table.
    // ---------------------------------------------------------------------
    let ci_cut = 1.0 - args.ci / 100.0; // Convert CI percentage to α.
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED).apply_modifier(UTF8_ROUND_CORNERS);
    let mut hdr = args.state_field.clone();
    hdr.push("Observations".into());
    hdr.push("Lower Bound".into());
    hdr.push(format!("Expected {}", args.reward_field));
    hdr.push("Upper Bound".into());
    table.set_header(hdr.iter().map(|h| Cell::new(h)).collect::<Vec<_>>());

    // Sort states lexicographically for deterministic output.
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

    // ---------------------------------------------------------------------
    // 3) If requested, run pairwise comparison tests.
    // ---------------------------------------------------------------------
    if let Some(nperm) = args.comparison_test {
        let func: fn(&[f64]) -> f64 = match args.estimator.as_str() {
            "median" => |d| {
                median(d.iter().cloned())
            },
            _ => |d| d.iter().sum::<f64>() / (d.len() as f64),
        };

        // Prepare grouping function based on optional user‑provided index list.
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
            args.alpha.unwrap_or(1.0)
        );
    }
}
