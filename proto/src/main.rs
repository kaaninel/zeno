use anyhow::{bail, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::path::PathBuf;

use zeno_proto::config::ZenoConfig;
use zeno_proto::data::ByteDataset;
use zeno_proto::model::ZenoAgent;
use zeno_proto::train;

fn select_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = Device::new_cuda(0) {
            println!("Using CUDA device");
            return Ok(dev);
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(dev) = Device::new_metal(0) {
            println!("Using Metal device");
            return Ok(dev);
        }
    }
    println!("Using CPU device");
    Ok(Device::Cpu)
}

fn print_usage() {
    println!("Zeno Prototype 1 — Core Agent + Trie Memory PoC");
    println!();
    println!("Usage:");
    println!("  zeno-proto train --phase <2|3|4|5> [--data-dir <path>] [--steps <N>] [--lr <F>]");
    println!("  zeno-proto train-all [--data-dir <path>]");
    println!("  zeno-proto eval [--data-dir <path>] [--memory]");
    println!("  zeno-proto info");
    println!();
    println!("Options:");
    println!("  --data-dir <path>  Directory with training text files (default: ../docs)");
    println!("  --steps <N>        Override step count for the phase");
    println!("  --lr <F>           Override learning rate");
    println!("  --memory           Enable memory for evaluation");
    println!("  --save <path>      Save model weights after training");
    println!("  --load <path>      Load model weights before training/eval");
}

fn parse_args() -> Result<CliArgs> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        bail!("No command provided");
    }

    let command = args[1].as_str();
    let mut cli = CliArgs {
        command: command.to_string(),
        phase: None,
        data_dir: PathBuf::from("../docs"),
        steps: None,
        lr: None,
        use_memory: false,
        save_path: None,
        load_path: None,
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--phase" => {
                i += 1;
                cli.phase = Some(args[i].parse()?);
            }
            "--data-dir" => {
                i += 1;
                cli.data_dir = PathBuf::from(&args[i]);
            }
            "--steps" => {
                i += 1;
                cli.steps = Some(args[i].parse()?);
            }
            "--lr" => {
                i += 1;
                cli.lr = Some(args[i].parse()?);
            }
            "--memory" => {
                cli.use_memory = true;
            }
            "--save" => {
                i += 1;
                cli.save_path = Some(PathBuf::from(&args[i]));
            }
            "--load" => {
                i += 1;
                cli.load_path = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                bail!("Unknown argument: {other}");
            }
        }
        i += 1;
    }

    Ok(cli)
}

struct CliArgs {
    command: String,
    phase: Option<u32>,
    data_dir: PathBuf,
    steps: Option<usize>,
    lr: Option<f64>,
    use_memory: bool,
    save_path: Option<PathBuf>,
    load_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = parse_args()?;
    let device = select_device()?;
    let cfg = ZenoConfig::default_proto();

    // Create model
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let agent = ZenoAgent::new(vb, &cfg)?;

    // Print param count
    let total_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("Model: {} params ({:.1}K)", total_params, total_params as f64 / 1000.0);

    // Load weights if requested
    if let Some(ref path) = cli.load_path {
        println!("Loading weights from {}", path.display());
        varmap.load(path)?;
    }

    // Load dataset
    let dataset = ByteDataset::from_directory(&cli.data_dir, cfg.context_window)?;
    println!("Dataset: {} chunks from {}", dataset.len(), cli.data_dir.display());

    if dataset.is_empty() {
        bail!("No training data found in {}", cli.data_dir.display());
    }

    match cli.command.as_str() {
        "train" => {
            let phase = cli.phase.unwrap_or_else(|| {
                eprintln!("Error: --phase required for train command");
                std::process::exit(1);
            });

            match phase {
                2 => {
                    let mut pcfg = train::Phase2Config::default();
                    if let Some(steps) = cli.steps { pcfg.steps = steps; }
                    if let Some(lr) = cli.lr { pcfg.lr = lr; }
                    train::train_phase2(&agent, &varmap, &dataset, &cfg, &pcfg, &device)?;
                }
                3 => {
                    let mut pcfg = train::Phase3Config::default();
                    if let Some(steps) = cli.steps { pcfg.steps = steps; }
                    if let Some(lr) = cli.lr { pcfg.lr = lr; }
                    train::train_phase3(&agent, &varmap, &dataset, &cfg, &pcfg, &device)?;
                }
                4 => {
                    let mut pcfg = train::Phase4Config::default();
                    if let Some(steps) = cli.steps { pcfg.steps = steps; }
                    if let Some(lr) = cli.lr { pcfg.lr = lr; }
                    train::train_phase4(&agent, &varmap, &dataset, &cfg, &pcfg, &device)?;
                }
                5 => {
                    let mut pcfg = train::Phase5Config::default();
                    if let Some(lr) = cli.lr { pcfg.lr_base = lr; }
                    if let Some(steps) = cli.steps { pcfg.steps_per_sub = steps; }
                    train::train_phase5(&agent, &varmap, &dataset, &cfg, &pcfg, &device)?;
                }
                _ => bail!("Invalid phase: {}. Use 2, 3, 4, or 5.", phase),
            }

            // Save if requested
            if let Some(ref path) = cli.save_path {
                println!("Saving weights to {}", path.display());
                varmap.save(path)?;
            }
        }

        "train-all" => {
            println!("Running all training phases sequentially...\n");

            let mut p2cfg = train::Phase2Config::default();
            if let Some(steps) = cli.steps { p2cfg.steps = steps; }
            train::train_phase2(&agent, &varmap, &dataset, &cfg, &p2cfg, &device)?;
            println!();

            let mut p3cfg = train::Phase3Config::default();
            if let Some(steps) = cli.steps { p3cfg.steps = steps; }
            train::train_phase3(&agent, &varmap, &dataset, &cfg, &p3cfg, &device)?;
            println!();

            let mut p4cfg = train::Phase4Config::default();
            if let Some(steps) = cli.steps { p4cfg.steps = steps; }
            train::train_phase4(&agent, &varmap, &dataset, &cfg, &p4cfg, &device)?;
            println!();

            let mut p5cfg = train::Phase5Config::default();
            if let Some(steps) = cli.steps { p5cfg.steps_per_sub = steps; }
            train::train_phase5(&agent, &varmap, &dataset, &cfg, &p5cfg, &device)?;

            if let Some(ref path) = cli.save_path {
                println!("\nSaving weights to {}", path.display());
                varmap.save(path)?;
            }
        }

        "eval" => {
            train::evaluate(&agent, &dataset, &cfg, &device, cli.use_memory)?;
        }

        "info" => {
            println!("\nConfig: {:?}", cfg);
            println!("\nParam breakdown:");
            let data = varmap.data().lock().unwrap();
            let mut groups: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for (name, var) in data.iter() {
                let prefix = name.split('.').next().unwrap_or(name).to_string();
                *groups.entry(prefix).or_insert(0) += var.elem_count();
            }
            let mut sorted: Vec<_> = groups.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            for (name, count) in &sorted {
                println!("  {:<25} {:>8} ({:.1}K)", name, count, *count as f64 / 1000.0);
            }
            println!("  {:<25} {:>8} ({:.1}K)", "TOTAL", total_params, total_params as f64 / 1000.0);
        }

        _ => {
            print_usage();
            bail!("Unknown command: {}", cli.command);
        }
    }

    Ok(())
}
