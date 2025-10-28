import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.logging import RichHandler
from rich.panel import Panel
from rich import print as rprint

import utils
import model

sys.stdout.reconfigure(line_buffering=True)

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("train_mulh")


def save_model(in_model, epoch, out_dir, optimizer, loss):
    """Save model checkpoint with error handling and logging."""
    try:
        save_name = 'epoch{}.pt'.format(epoch)
        out_path = os.path.join(out_dir, save_name)

        logger.info(f"Saving model checkpoint: [cyan]{save_name}[/cyan]", extra={"markup": True})

        if not os.path.exists(out_dir):
            logger.warning(f"Output directory [yellow]{out_dir}[/yellow] doesn't exist, creating it", extra={"markup": True})
            os.makedirs(out_dir, exist_ok=True)

        torch.save({
            'model_state_dict': in_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'epoch': epoch
        }, out_path)

        file_size = os.path.getsize(out_path) / (1024 * 1024)  # Size in MB
        logger.info(f"[green]✓[/green] Checkpoint saved successfully: {out_path} ({file_size:.2f} MB)", extra={"markup": True})

    except PermissionError as e:
        logger.error(f"[red]✗[/red] Permission denied when saving model to {out_path}: {e}", extra={"markup": True})
        raise
    except OSError as e:
        logger.error(f"[red]✗[/red] OS error when saving model: {e}", extra={"markup": True})
        raise
    except Exception as e:
        logger.error(f"[red]✗[/red] Unexpected error saving model checkpoint: {e}", extra={"markup": True})
        raise


def valid(in_model, val_dataset, batch_size, device, samp_rate=None):
    """Validate model with rich progress tracking and error handling."""
    try:
        logger.info("[bold blue]Starting validation phase...[/bold blue]", extra={"markup": True})

        if samp_rate is None:
            test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            logger.info(f"Using full validation dataset: {len(val_dataset)} samples")
        else:
            subset_size = len(val_dataset) // samp_rate
            subset_sampler = SubsetRandomSampler(range(subset_size))
            test_loader = DataLoader(val_dataset, batch_size=batch_size,
                                     shuffle=False, sampler=subset_sampler)
            logger.info(f"Using sampled validation dataset: {subset_size} samples (rate: 1/{samp_rate})")

        in_model.to(device)
        in_model.eval()
        loss1, loss2, loss3 = 0.0, 0.0, 0.0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            val_task = progress.add_task("[cyan]Validating...", total=len(test_loader))

            with torch.no_grad():
                for i, (images, disp) in enumerate(test_loader):
                    try:
                        images = images.to(device)
                        disp = disp.to(device)
                        disp_out = in_model(images)
                        loss1 += utils.abs_rel_error(disp_out, disp)
                        loss2 += utils.D1_metric(disp_out, disp)
                        loss3 += utils.RMSE(disp_out, disp)
                        progress.update(val_task, advance=1)

                    except RuntimeError as e:
                        logger.error(f"[red]Runtime error in validation batch {i}: {e}[/red]", extra={"markup": True})
                        raise
                    except Exception as e:
                        logger.error(f"[red]Unexpected error in validation batch {i}: {e}[/red]", extra={"markup": True})
                        raise

        loss1 /= len(test_loader)
        loss2 /= len(test_loader)
        loss3 /= len(test_loader)

        # Create a rich table for validation results
        results_table = Table(title="Validation Metrics", show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan", justify="left")
        results_table.add_column("Value", style="green", justify="right")

        results_table.add_row("AbsRel", f"{loss1:.4f}")
        results_table.add_row("D1", f"{loss2:.4f}")
        results_table.add_row("RMSE", f"{loss3:.4f}")

        console.print(results_table)
        logger.info("[green]✓[/green] Validation completed successfully", extra={"markup": True})

    except Exception as e:
        logger.error(f"[red]✗[/red] Validation failed: {e}", extra={"markup": True})
        raise


def train(in_model, train_dataset, val_dataset, args):
    """Train model with comprehensive logging and error handling."""
    try:
        console.print(Panel.fit("[bold green]Starting Training Process[/bold green]", border_style="green"))

        # Setup data loader
        if args.sample_rate is None:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            logger.info(f"Using full training dataset: {len(train_dataset)} samples")
        else:
            subset_size = len(train_dataset) // args.sample_rate
            subset_sampler = SubsetRandomSampler(range(subset_size))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=subset_sampler)
            logger.info(f"Sample rate: {args.sample_rate}, using {subset_size} samples")

        batch_num = len(train_dataset) // args.batch_size
        logger.info(f"Total batches per epoch: {batch_num}")

        # Setup optimizer and criterion
        optimizer = optim.Adam(in_model.parameters(), lr=args.learning_rate)
        logger.info(f"Initialized Adam optimizer with learning rate: {args.learning_rate}")

        criterion = utils.SmoothLoss(1)
        logger.info("Using SmoothLoss criterion")

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(args.device)
        logger.info(f"Using device: [bold cyan]{device}[/bold cyan]", extra={"markup": True})

        # Create save directory
        if not os.path.exists(args.save_dir):
            logger.info(f"Creating save directory: [yellow]{args.save_dir}[/yellow]", extra={"markup": True})
            try:
                os.makedirs(args.save_dir)
            except OSError as e:
                logger.error(f"[red]Failed to create save directory: {e}[/red]", extra={"markup": True})
                raise

        # Load checkpoint if provided
        if args.checkpoint_path is not None:
            try:
                assert os.path.exists(args.checkpoint_path), f"{args.checkpoint_path} does not exist!"
                logger.info(f"Loading checkpoint from: [cyan]{args.checkpoint_path}[/cyan]", extra={"markup": True})

                trained_model = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
                trained_model = utils.remove_module_prefix(trained_model)
                in_model.load_state_dict(trained_model['model_state_dict'])

                if 'optimizer_state_dict' in trained_model:
                    in_model.to(device)  # move the model to GPU before optimizer declaration
                    optimizer.load_state_dict(trained_model['optimizer_state_dict'])
                    logger.info("Loaded optimizer state from checkpoint")

                checkpoint_epoch = trained_model.get('epoch', 'unknown')
                checkpoint_loss = trained_model.get('loss', 'unknown')
                logger.info(f"[green]✓[/green] Checkpoint loaded: epoch={checkpoint_epoch}, loss={checkpoint_loss}", extra={"markup": True})

            except FileNotFoundError as e:
                logger.error(f"[red]Checkpoint file not found: {e}[/red]", extra={"markup": True})
                raise
            except Exception as e:
                logger.error(f"[red]Failed to load checkpoint: {e}[/red]", extra={"markup": True})
                raise

        # Setup model for training
        in_model = torch.nn.DataParallel(in_model)
        logger.info("Model wrapped with DataParallel for multi-GPU training")

        in_model.train()
        in_model.to(device)
        logger.info("[green]Model moved to device and set to training mode[/green]", extra={"markup": True})

        # Training phase
        train_epoch = args.total_epochs - args.val_epochs
        logger.info(f"Training configuration: {train_epoch} training epochs, {args.val_epochs} validation epochs")

        console.print(Panel(f"[bold cyan]Phase 1: Training ({train_epoch} epochs)[/bold cyan]\nLearning Rate: {args.learning_rate}", border_style="cyan"))

        for epo in range(train_epoch):
            running_loss = 0.0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                train_task = progress.add_task(
                    f"[green]Epoch {epo}/{train_epoch-1}",
                    total=len(train_loader),
                    loss=0.0
                )

                try:
                    for i, (images, disparities) in enumerate(train_loader):
                        try:
                            images = images.to(device)
                            disparities = disparities.to(device)
                            outputs = in_model(images)
                            loss = criterion(outputs, disparities)

                            # backward pass and optimize
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            running_loss += loss.item()
                            current_loss = running_loss / (i + 1)
                            progress.update(train_task, advance=1, loss=current_loss)

                        except RuntimeError as e:
                            logger.error(f"[red]Runtime error in training batch {i}: {e}[/red]", extra={"markup": True})
                            raise
                        except Exception as e:
                            logger.error(f"[red]Error in training batch {i}: {e}[/red]", extra={"markup": True})
                            raise

                except Exception as e:
                    logger.error(f"[red]Training failed at epoch {epo}: {e}[/red]", extra={"markup": True})
                    raise

            final_loss = running_loss / (i + 1)
            logger.info(f"Epoch {epo} completed - Average Loss: [yellow]{final_loss:.4f}[/yellow]", extra={"markup": True})

        # Validation phase with lower learning rate
        logger.info(f"Switching to validation phase with learning rate: {args.learning_rate_val}")
        optimizer = optim.Adam(in_model.parameters(), lr=args.learning_rate_val)

        console.print(Panel(f"[bold magenta]Phase 2: Validation & Fine-tuning ({args.val_epochs} epochs)[/bold magenta]\nLearning Rate: {args.learning_rate_val}", border_style="magenta"))

        for epo in range(train_epoch, args.total_epochs):
            running_loss = 0.0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                train_task = progress.add_task(
                    f"[magenta]Epoch {epo}/{args.total_epochs-1}",
                    total=len(train_loader),
                    loss=0.0
                )

                try:
                    for i, (images, disparities) in enumerate(train_loader):
                        try:
                            images = images.to(device)
                            disparities = disparities.to(device)
                            outputs = in_model(images)
                            loss = criterion(outputs, disparities)

                            # backward pass and optimize
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            running_loss += loss.item()
                            current_loss = running_loss / (i + 1)
                            progress.update(train_task, advance=1, loss=current_loss)

                        except RuntimeError as e:
                            logger.error(f"[red]Runtime error in validation epoch batch {i}: {e}[/red]", extra={"markup": True})
                            raise
                        except Exception as e:
                            logger.error(f"[red]Error in validation epoch batch {i}: {e}[/red]", extra={"markup": True})
                            raise

                except Exception as e:
                    logger.error(f"[red]Validation epoch {epo} failed: {e}[/red]", extra={"markup": True})
                    raise

            running_loss = running_loss / (i + 1)
            logger.info(f"Epoch {epo} completed - Average Loss: [yellow]{running_loss:.4f}[/yellow]", extra={"markup": True})

            # Run validation
            valid(in_model, val_dataset, args.batch_size, device, args.sample_rate)

            # Save model checkpoint
            save_model(in_model, epo, args.save_dir, optimizer, running_loss)

        console.print(Panel.fit("[bold green]✓ Training Finished Successfully![/bold green]", border_style="green"))
        logger.info("[bold green]Training completed successfully[/bold green]", extra={"markup": True})

    except KeyboardInterrupt:
        logger.warning("[yellow]Training interrupted by user (Ctrl+C)[/yellow]", extra={"markup": True})
        raise
    except Exception as e:
        logger.error(f"[red]✗[/red] Training failed with error: {e}", extra={"markup": True})
        raise


if __name__ == "__main__":
    try:
        # Print header
        console.print("\n")
        console.rule("[bold blue]Multi-Head Depth Training Script[/bold blue]")
        console.print("\n")

        parser = argparse.ArgumentParser()

        parser.add_argument('--checkpoint_path', '-c', type=str, default=None,
                            help='Path to the trained check point. Only required for finetuning based on trained model.')
        parser.add_argument('--dataset', '-d', type=str, default='sceneflow',
                            choices=['sceneflow', 'ADT', 'DTU', 'Middlebury'],
                            help='Name of training dataset: sceneflow, ADT, DTU or Middlebury')
        parser.add_argument('--data_path', '-p', type=str,
                            help='Path to the dataset')
        parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                            help='Device to run the model')
        parser.add_argument('--save_dir', '-s', type=str, default='ckpt',
                            help='File path to save the check point.')
        parser.add_argument('--batch_size', '-b', type=int, default=10)
        parser.add_argument('--total_epochs', '-e', type=int, default=100)
        parser.add_argument('--val_epochs', '-v', type=int, default=20)
        parser.add_argument('--sample_rate', type=int, default=None,
                            help='Sample rate of the dataset. The length of the dataset is divided by it.')
        parser.add_argument('--learning_rate', '-lr', type=float, default=4e-4,
                            help='Learning rate of the model in training phase.')
        parser.add_argument('--learning_rate_val', '-lrv', type=float, default=4e-4,
                            help='Learning rate of the model in validation phase.')

        arguments = parser.parse_args()

        # Display configuration
        config_table = Table(title="Training Configuration", show_header=True, header_style="bold cyan")
        config_table.add_column("Parameter", style="cyan", justify="left")
        config_table.add_column("Value", style="yellow", justify="left")

        config_table.add_row("Dataset", arguments.dataset)
        config_table.add_row("Data Path", arguments.data_path if arguments.data_path else "Not specified")
        config_table.add_row("Device", arguments.device)
        config_table.add_row("Save Directory", arguments.save_dir)
        config_table.add_row("Batch Size", str(arguments.batch_size))
        config_table.add_row("Total Epochs", str(arguments.total_epochs))
        config_table.add_row("Validation Epochs", str(arguments.val_epochs))
        config_table.add_row("Sample Rate", str(arguments.sample_rate) if arguments.sample_rate else "Full dataset")
        config_table.add_row("Training LR", str(arguments.learning_rate))
        config_table.add_row("Validation LR", str(arguments.learning_rate_val))
        config_table.add_row("Checkpoint Path", arguments.checkpoint_path if arguments.checkpoint_path else "None")

        console.print(config_table)
        console.print("\n")

        # Validate arguments
        if arguments.total_epochs < arguments.val_epochs:
            logger.error("[red]Total number of epochs should be greater than the number of validation epochs[/red]", extra={"markup": True})
            exit(1)

        if not arguments.data_path:
            logger.error("[red]Data path is required! Use --data_path or -p to specify[/red]", extra={"markup": True})
            exit(1)

        if not os.path.exists(arguments.data_path):
            logger.error(f"[red]Data path does not exist: {arguments.data_path}[/red]", extra={"markup": True})
            exit(1)

        # Load datasets with error handling
        logger.info(f"Loading [cyan]{arguments.dataset}[/cyan] dataset...", extra={"markup": True})

        try:
            if arguments.dataset == 'sceneflow':
                train_dataset = utils.SceneFlowDataset(arguments.data_path, stereo=True)
                val_dataset = utils.SceneFlowDataset(arguments.data_path, train=False, stereo=True)
            elif arguments.dataset == 'ADT':
                train_dataset = utils.ADT(arguments.data_path, train=True)
                val_dataset = utils.ADT(arguments.data_path, train=False)
            elif arguments.dataset == 'DTU':
                train_dataset = utils.DTU(arguments.data_path, train='train', output_homo=False)
                val_dataset = utils.DTU(arguments.data_path, train='test', output_homo=False)
            elif arguments.dataset == 'Middlebury':
                train_dataset = utils.Middlebury(arguments.data_path)
                val_dataset = utils.Middlebury(arguments.data_path)
            else:
                logger.error(f"[red]Dataset '{arguments.dataset}' not recognized[/red]", extra={"markup": True})
                exit(1)

            logger.info(f"[green]✓[/green] Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples", extra={"markup": True})

        except FileNotFoundError as e:
            logger.error(f"[red]Dataset files not found: {e}[/red]", extra={"markup": True})
            exit(1)
        except Exception as e:
            logger.error(f"[red]Failed to load dataset: {e}[/red]", extra={"markup": True})
            logger.exception("Dataset loading error details:")
            exit(1)

        # Initialize model
        logger.info("Initializing [cyan]Stereo_MulH[/cyan] model...", extra={"markup": True})
        try:
            input_model = model.Stereo_MulH()
            logger.info("[green]✓[/green] Model initialized successfully", extra={"markup": True})
        except Exception as e:
            logger.error(f"[red]Failed to initialize model: {e}[/red]", extra={"markup": True})
            logger.exception("Model initialization error details:")
            exit(1)

        # Start training
        train(input_model, train_dataset, val_dataset, arguments)

    except KeyboardInterrupt:
        console.print("\n")
        logger.warning("[yellow]Program interrupted by user (Ctrl+C)[/yellow]", extra={"markup": True})
        exit(130)
    except SystemExit as e:
        # Propagate exit codes
        raise
    except Exception as e:
        console.print("\n")
        logger.error(f"[red bold]Fatal error occurred:[/red bold] {e}", extra={"markup": True})
        logger.exception("Full traceback:")
        exit(1)
