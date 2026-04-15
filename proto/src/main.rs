use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use zeno_proto::config::ZenoConfig;
use zeno_proto::data::ByteDataset;
use zeno_proto::model::ZenoAgent;
use zeno_proto::train::{self, CheckpointConfig};
use zeno_proto::visualizer::TrieVisualizerRuntime;

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
    println!("  zeno-proto train --phase <2|3|4|5> [options]");
    println!("  zeno-proto train-all [options]");
    println!("  zeno-proto eval [--data-dir <path>] [--memory] [--load <path>]");
    println!("  zeno-proto visualize --input <path> [--follow]");
    println!("  zeno-proto info");
    println!();
    println!("Options:");
    println!("  --data-dir <path>         Training text files directory (default: data)");
    println!("  --input <path>            Visualizer JSONL input for visualize");
    println!("  --steps <N>               Override step count for the phase");
    println!("  --lr <F>                  Override learning rate");
    println!("  --memory                  Enable memory for evaluation");
    println!("  --follow                  Follow appended visualizer events");
    println!("  --save <path>             Save model weights after training");
    println!("  --load <path>             Load model weights before training/eval");
    println!(
        "  --checkpoint-dir <path>   Directory for periodic checkpoints (default: checkpoints)"
    );
    println!("  --checkpoint-every <N>    Save checkpoint every N steps (0 = phase end only)");
    println!("  --trie-events <path|->    Stream sparse trie events as JSONL");
}

fn parse_args() -> Result<CliArgs> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        bail!("No command provided");
    }
    if matches!(args[1].as_str(), "--help" | "-h") {
        print_usage();
        std::process::exit(0);
    }

    let command = args[1].as_str();
    let mut cli = CliArgs {
        command: command.to_string(),
        phase: None,
        data_dir: PathBuf::from("data"),
        steps: None,
        lr: None,
        use_memory: false,
        save_path: None,
        load_path: None,
        checkpoint_dir: PathBuf::from("checkpoints"),
        checkpoint_every: 1000,
        trie_events: None,
        visualize_input: None,
        follow: false,
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
            "--input" => {
                i += 1;
                cli.visualize_input = Some(args[i].clone());
            }
            "--lr" => {
                i += 1;
                cli.lr = Some(args[i].parse()?);
            }
            "--memory" => {
                cli.use_memory = true;
            }
            "--follow" => {
                cli.follow = true;
            }
            "--save" => {
                i += 1;
                cli.save_path = Some(PathBuf::from(&args[i]));
            }
            "--load" => {
                i += 1;
                cli.load_path = Some(PathBuf::from(&args[i]));
            }
            "--checkpoint-dir" => {
                i += 1;
                cli.checkpoint_dir = PathBuf::from(&args[i]);
            }
            "--checkpoint-every" => {
                i += 1;
                cli.checkpoint_every = args[i].parse()?;
            }
            "--trie-events" => {
                i += 1;
                cli.trie_events = Some(args[i].clone());
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
    checkpoint_dir: PathBuf,
    checkpoint_every: usize,
    trie_events: Option<String>,
    visualize_input: Option<String>,
    follow: bool,
}

fn main() -> Result<()> {
    let cli = parse_args()?;
    if cli.command == "visualize" {
        return zeno_proto::visualizer_tui::run(
            cli.visualize_input
                .as_deref()
                .context("visualize requires --input <path>")?,
            cli.follow,
        );
    }
    let device = select_device()?;
    let cfg = ZenoConfig::default_proto();

    // Set up Ctrl+C handler
    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = interrupted.clone();
    ctrlc::set_handler(move || {
        eprintln!("\n⚠ Interrupt received — saving checkpoint and exiting...");
        interrupted_clone.store(true, Ordering::Relaxed);
    })
    .expect("Failed to set Ctrl+C handler");

    let ckpt = CheckpointConfig {
        checkpoint_dir: Some(cli.checkpoint_dir.clone()),
        checkpoint_every: cli.checkpoint_every,
        interrupted: interrupted.clone(),
    };
    let visualizer = match cli.trie_events.as_deref() {
        Some("-") => Some(TrieVisualizerRuntime::stdout()),
        Some(path) => Some(TrieVisualizerRuntime::create(&PathBuf::from(path))?),
        None => None,
    };

    // Create model
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let agent = ZenoAgent::new(vb, &cfg)?;

    // Print param count
    let total_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!(
        "Model: {} params ({:.1}K)",
        total_params,
        total_params as f64 / 1000.0
    );

    // Load weights if requested
    if let Some(ref path) = cli.load_path {
        println!("Loading weights from {}", path.display());
        varmap.load(path)?;
    }

    // Load dataset
    let dataset = ByteDataset::from_directory(&cli.data_dir, cfg.context_window)?;
    println!(
        "Dataset: {} chunks from {}",
        dataset.len(),
        cli.data_dir.display()
    );

    if dataset.is_empty() {
        bail!("No training data found in {}", cli.data_dir.display());
    }

    if let Some(visualizer) = visualizer.as_ref() {
        visualizer.emit_session_started(&cli.command, cli.phase, cli.use_memory, &cfg)?;
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
                    if let Some(steps) = cli.steps {
                        pcfg.steps = steps;
                    }
                    if let Some(lr) = cli.lr {
                        pcfg.lr = lr;
                    }
                    train::train_phase2(
                        &agent,
                        &varmap,
                        &dataset,
                        &cfg,
                        &pcfg,
                        &ckpt,
                        &device,
                        visualizer.as_ref(),
                    )?;
                }
                3 => {
                    let mut pcfg = train::Phase3Config::default();
                    if let Some(steps) = cli.steps {
                        pcfg.steps = steps;
                    }
                    if let Some(lr) = cli.lr {
                        pcfg.lr = lr;
                    }
                    train::train_phase3(
                        &agent,
                        &varmap,
                        &dataset,
                        &cfg,
                        &pcfg,
                        &ckpt,
                        &device,
                        visualizer.as_ref(),
                    )?;
                }
                4 => {
                    let mut pcfg = train::Phase4Config::default();
                    if let Some(steps) = cli.steps {
                        pcfg.steps = steps;
                    }
                    if let Some(lr) = cli.lr {
                        pcfg.lr = lr;
                    }
                    train::train_phase4(
                        &agent,
                        &varmap,
                        &dataset,
                        &cfg,
                        &pcfg,
                        &ckpt,
                        &device,
                        visualizer.as_ref(),
                    )?;
                }
                5 => {
                    let mut pcfg = train::Phase5Config::default();
                    if let Some(lr) = cli.lr {
                        pcfg.lr_base = lr;
                    }
                    if let Some(steps) = cli.steps {
                        pcfg.steps_per_sub = steps;
                    }
                    train::train_phase5(
                        &agent,
                        &varmap,
                        &dataset,
                        &cfg,
                        &pcfg,
                        &ckpt,
                        &device,
                        visualizer.as_ref(),
                    )?;
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
            if let Some(steps) = cli.steps {
                p2cfg.steps = steps;
            }
            train::train_phase2(
                &agent,
                &varmap,
                &dataset,
                &cfg,
                &p2cfg,
                &ckpt,
                &device,
                visualizer.as_ref(),
            )?;
            if interrupted.load(Ordering::Relaxed) {
                println!("\nTraining interrupted after Phase 2.");
                return Ok(());
            }
            println!();

            let mut p3cfg = train::Phase3Config::default();
            if let Some(steps) = cli.steps {
                p3cfg.steps = steps;
            }
            train::train_phase3(
                &agent,
                &varmap,
                &dataset,
                &cfg,
                &p3cfg,
                &ckpt,
                &device,
                visualizer.as_ref(),
            )?;
            if interrupted.load(Ordering::Relaxed) {
                println!("\nTraining interrupted after Phase 3.");
                return Ok(());
            }
            println!();

            let mut p4cfg = train::Phase4Config::default();
            if let Some(steps) = cli.steps {
                p4cfg.steps = steps;
            }
            train::train_phase4(
                &agent,
                &varmap,
                &dataset,
                &cfg,
                &p4cfg,
                &ckpt,
                &device,
                visualizer.as_ref(),
            )?;
            if interrupted.load(Ordering::Relaxed) {
                println!("\nTraining interrupted after Phase 4.");
                return Ok(());
            }
            println!();

            let mut p5cfg = train::Phase5Config::default();
            if let Some(steps) = cli.steps {
                p5cfg.steps_per_sub = steps;
            }
            train::train_phase5(
                &agent,
                &varmap,
                &dataset,
                &cfg,
                &p5cfg,
                &ckpt,
                &device,
                visualizer.as_ref(),
            )?;

            if let Some(ref path) = cli.save_path {
                println!("\nSaving weights to {}", path.display());
                varmap.save(path)?;
            }
        }

        "eval" => {
            train::evaluate(
                &agent,
                &dataset,
                &cfg,
                &device,
                cli.use_memory,
                visualizer.as_ref(),
            )?;
        }

        "info" => {
            println!("\nConfig: {:?}", cfg);
            println!("\nParam breakdown:");
            let data = varmap.data().lock().unwrap();
            let mut groups: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for (name, var) in data.iter() {
                let prefix = name.split('.').next().unwrap_or(name).to_string();
                *groups.entry(prefix).or_insert(0) += var.elem_count();
            }
            let mut sorted: Vec<_> = groups.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            for (name, count) in &sorted {
                println!(
                    "  {:<25} {:>8} ({:.1}K)",
                    name,
                    count,
                    *count as f64 / 1000.0
                );
            }
            println!(
                "  {:<25} {:>8} ({:.1}K)",
                "TOTAL",
                total_params,
                total_params as f64 / 1000.0
            );
        }

        _ => {
            print_usage();
            bail!("Unknown command: {}", cli.command);
        }
    }

    if let Some(visualizer) = visualizer.as_ref() {
        visualizer.emit_session_completed(&cli.command)?;
    }

    Ok(())
}
