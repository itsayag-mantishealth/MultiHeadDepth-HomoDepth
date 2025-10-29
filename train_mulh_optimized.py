"""
Multi-Head Depth Estimation Training Script (Optimized)

This script trains a stereo depth estimation model using the Stereo_MulH architecture.
Supports multiple datasets (SceneFlow, ADT, DTU, Middlebury) with TensorBoard logging,
validation debug visualizations, and checkpoint management.

Features:
- Two-phase training (main training + fine-tuning with lower learning rate)
- TensorBoard integration for metrics and debug images
- Automatic best model tracking based on training loss
- Rich console output with progress bars and tables
- Validation debug images with disparity visualizations

Usage:
    python train_mulh.py --data_path /path/to/data --dataset sceneflow --batch_size 10
"""

import argparse
import os
from typing import Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.logging import RichHandler
from rich.panel import Panel
import logging
import sys

import utils
import model

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Enable line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)

# Initialize rich console for beautiful output
console = Console()

# Setup structured logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("train_mulh")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    checkpoint_type: str = "epoch"
) -> None:
    """
    Save model checkpoint with all necessary state information.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        save_path: Full path where checkpoint will be saved
        checkpoint_type: Type of checkpoint (epoch/latest/best) for logging
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save checkpoint with all training state
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch
    }, save_path)

    # Log successful save with file size
    file_size = os.path.getsize(save_path) / (1024 * 1024)  # Convert to MB
    logger.info(
        f"[green]✓[/green] {checkpoint_type.capitalize()} checkpoint saved: "
        f"{os.path.basename(save_path)} ({file_size:.2f} MB)",
        extra={"markup": True}
    )


def create_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    sample_rate: Optional[int] = None,
    data_percentage: Optional[float] = None,
    shuffle: bool = True
) -> Tuple[DataLoader, int]:
    """
    Create DataLoader with optional data subsampling.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for training
        sample_rate: Use 1/sample_rate of data (mutually exclusive with data_percentage)
        data_percentage: Use percentage of data (mutually exclusive with sample_rate)
        shuffle: Whether to shuffle data (ignored if sampler is used)

    Returns:
        Tuple of (DataLoader, actual_dataset_size)
    """
    if sample_rate is None and data_percentage is None:
        # Use full dataset with shuffling
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        size = len(dataset)
        logger.info(f"Using full dataset: {size} samples")
    elif sample_rate is not None:
        # Use sample rate to determine subset size
        subset_size = len(dataset) // sample_rate
        subset_sampler = SubsetRandomSampler(range(subset_size))
        loader = DataLoader(dataset, batch_size=batch_size, sampler=subset_sampler)
        logger.info(f"Using sampled dataset: {subset_size} samples (rate: 1/{sample_rate})")
        size = subset_size
    else:  # data_percentage is not None
        # Use percentage to determine subset size
        subset_size = int(len(dataset) * data_percentage)
        subset_sampler = SubsetRandomSampler(range(subset_size))
        loader = DataLoader(dataset, batch_size=batch_size, sampler=subset_sampler)
        logger.info(f"Using {data_percentage*100:.1f}% of data: {subset_size}/{len(dataset)} samples")
        size = subset_size

    return loader, size


# ============================================================================
# VALIDATION
# ============================================================================

def validate(
    model: torch.nn.Module,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: torch.device,
    sample_rate: Optional[int] = None,
    data_percentage: Optional[float] = None,
    save_dir: Optional[str] = None,
    epoch: Optional[int] = None,
    tensorboard_writer: Optional[SummaryWriter] = None
) -> float:
    """
    Run validation and optionally create debug visualizations.

    Args:
        model: Model to validate
        val_dataset: Validation dataset
        batch_size: Batch size for validation
        device: Device to run validation on
        sample_rate: Optional sampling rate for faster validation
        data_percentage: Optional data percentage for faster validation
        save_dir: Directory to save debug images
        epoch: Current epoch (for logging and filenames)
        tensorboard_writer: Optional TensorBoard writer for logging

    Returns:
        Primary validation loss (AbsRel metric)
    """
    logger.info("[bold blue]Starting validation phase...[/bold blue]", extra={"markup": True})

    # Create validation data loader
    val_loader, _ = create_data_loader(
        val_dataset, batch_size, sample_rate, data_percentage, shuffle=False
    )

    # Set model to evaluation mode
    model.to(device)
    model.eval()

    # Initialize metric accumulators
    abs_rel_error, d1_metric, rmse_metric = 0.0, 0.0, 0.0

    # Validation loop with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        val_task = progress.add_task("[cyan]Validating...", total=len(val_loader))

        # Disable gradient computation for validation
        with torch.no_grad():
            for images, disparities in val_loader:
                # Move data to device
                images = images.to(device)
                disparities = disparities.to(device)

                # Forward pass
                disparity_predictions = model(images)

                # Compute validation metrics
                abs_rel_error += utils.abs_rel_error(disparity_predictions, disparities)
                d1_metric += utils.D1_metric(disparity_predictions, disparities)
                rmse_metric += utils.RMSE(disparity_predictions, disparities)

                progress.update(val_task, advance=1)

    # Compute average metrics
    num_batches = len(val_loader)
    abs_rel_error /= num_batches
    d1_metric /= num_batches
    rmse_metric /= num_batches

    # Display validation results in a table
    results_table = Table(title="Validation Metrics", show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="cyan", justify="left")
    results_table.add_column("Value", style="green", justify="right")
    results_table.add_row("AbsRel (↓)", f"{abs_rel_error:.4f}")
    results_table.add_row("D1 (↓)", f"{d1_metric:.4f}")
    results_table.add_row("RMSE (↓)", f"{rmse_metric:.4f}")
    console.print(results_table)

    # Log metrics to TensorBoard
    if tensorboard_writer is not None and epoch is not None:
        tensorboard_writer.add_scalar('validation/AbsRel', abs_rel_error, epoch)
        tensorboard_writer.add_scalar('validation/D1', d1_metric, epoch)
        tensorboard_writer.add_scalar('validation/RMSE', rmse_metric, epoch)

    # Create debug visualization images
    if save_dir is not None and epoch is not None:
        logger.info("[cyan]Creating validation debug images...[/cyan]", extra={"markup": True})
        utils.create_validation_debug_images(
            model=model,
            val_dataset=val_dataset,
            device=device,
            save_dir=save_dir,
            epoch=epoch,
            num_samples=10,
            stereo=True,
            tensorboard_writer=tensorboard_writer
        )

    logger.info("[green]✓[/green] Validation completed successfully", extra={"markup": True})

    # Return primary validation metric (AbsRel)
    return abs_rel_error


# ============================================================================
# TRAINING EPOCH
# ============================================================================

def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    phase_name: str = "Training"
) -> float:
    """
    Run one training epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs (for display)
        phase_name: Name of training phase (for display)

    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0

    # Training loop with progress bar
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
        task_color = "green" if "Training" in phase_name else "magenta"
        train_task = progress.add_task(
            f"[{task_color}]Epoch {epoch}/{total_epochs-1}",
            total=len(train_loader),
            loss=0.0
        )

        for batch_idx, (images, disparities) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            disparities = disparities.to(device)

            # Forward pass: compute model predictions
            outputs = model(images)
            loss = criterion(outputs, disparities)

            # Backward pass: compute gradients and update parameters
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters

            # Track running loss for logging
            running_loss += loss.item()
            current_loss = running_loss / (batch_idx + 1)
            progress.update(train_task, advance=1, loss=current_loss)

    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    logger.info(
        f"Epoch {epoch} completed - Average Loss: [yellow]{avg_loss:.4f}[/yellow]",
        extra={"markup": True}
    )

    return avg_loss


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    args: argparse.Namespace
) -> None:
    """
    Main training loop with two-phase training strategy.

    Phase 1: Training phase with normal learning rate
    Phase 2: Fine-tuning phase with reduced learning rate

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        args: Command-line arguments containing all hyperparameters
    """
    console.print(Panel.fit("[bold green]Starting Training Process[/bold green]", border_style="green"))

    # ========================================================================
    # SETUP: Data, Device, Optimizer, Criterion
    # ========================================================================

    # Create training data loader
    train_loader, dataset_size = create_data_loader(
        train_dataset,
        args.batch_size,
        args.sample_rate,
        args.data_percentage,
        shuffle=True
    )

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: [bold cyan]{device}[/bold cyan]", extra={"markup": True})

    # Initialize optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = utils.SmoothLoss(1)  # Smooth L1 loss with delta=1
    logger.info(f"Optimizer: Adam (lr={args.learning_rate})")
    logger.info("Loss criterion: SmoothLoss")

    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: [yellow]{args.save_dir}[/yellow]", extra={"markup": True})

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logging to: [cyan]{tensorboard_dir}[/cyan]", extra={"markup": True})

    # Track best model (based on training loss)
    best_loss = float('inf')

    # ========================================================================
    # CHECKPOINT LOADING (if provided)
    # ========================================================================

    if args.checkpoint_path is not None:
        assert os.path.exists(args.checkpoint_path), f"Checkpoint not found: {args.checkpoint_path}"
        logger.info(f"Loading checkpoint: [cyan]{args.checkpoint_path}[/cyan]", extra={"markup": True})

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        checkpoint = utils.remove_module_prefix(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            model.to(device)  # Move model to device before loading optimizer
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")

        # Log checkpoint info
        ckpt_epoch = checkpoint.get('epoch', 'unknown')
        ckpt_loss = checkpoint.get('loss', 'unknown')
        logger.info(f"[green]✓[/green] Checkpoint loaded: epoch={ckpt_epoch}, loss={ckpt_loss}", extra={"markup": True})

    # ========================================================================
    # MODEL PREPARATION
    # ========================================================================

    # Wrap model with DataParallel for multi-GPU training
    model = torch.nn.DataParallel(model)
    model.to(device)
    logger.info("Model wrapped with DataParallel for multi-GPU training")

    # ========================================================================
    # PHASE 1: MAIN TRAINING
    # ========================================================================

    train_epochs = args.total_epochs - args.val_epochs
    logger.info(f"Training: {train_epochs} epochs, Fine-tuning: {args.val_epochs} epochs")

    console.print(Panel(
        f"[bold cyan]Phase 1: Training ({train_epochs} epochs)[/bold cyan]\n"
        f"Learning Rate: {args.learning_rate}",
        border_style="cyan"
    ))

    for epoch in range(train_epochs):
        # Train for one epoch
        epoch_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, train_epochs, "Training"
        )

        # Log training metrics to TensorBoard
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/learning_rate', args.learning_rate, epoch)

        # Run validation after each epoch
        val_loss = validate(
            model, val_dataset, args.batch_size, device,
            args.sample_rate, args.data_percentage,
            args.save_dir, epoch, writer
        )

        # Save latest checkpoint (overwritten every epoch)
        latest_path = os.path.join(args.save_dir, 'latest.pt')
        save_checkpoint(model, optimizer, epoch, epoch_loss, latest_path, "latest")

        # Save best model if current loss is better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, epoch_loss, best_path, "best")
            logger.info(f"[green]✓[/green] New best model (loss: {best_loss:.4f})", extra={"markup": True})

    # ========================================================================
    # PHASE 2: FINE-TUNING WITH LOWER LEARNING RATE
    # ========================================================================

    logger.info(f"Switching to fine-tuning phase (lr: {args.learning_rate_val})")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_val)

    console.print(Panel(
        f"[bold magenta]Phase 2: Fine-tuning ({args.val_epochs} epochs)[/bold magenta]\n"
        f"Learning Rate: {args.learning_rate_val}",
        border_style="magenta"
    ))

    for epoch in range(train_epochs, args.total_epochs):
        # Train for one epoch
        epoch_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.total_epochs, "Fine-tuning"
        )

        # Log training metrics to TensorBoard
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/learning_rate', args.learning_rate_val, epoch)

        # Run validation after each epoch
        val_loss = validate(
            model, val_dataset, args.batch_size, device,
            args.sample_rate, args.data_percentage,
            args.save_dir, epoch, writer
        )

        # Save epoch checkpoint (kept for all fine-tuning epochs)
        epoch_path = os.path.join(args.save_dir, f'epoch{epoch}.pt')
        save_checkpoint(model, optimizer, epoch, epoch_loss, epoch_path, "epoch")

        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, 'latest.pt')
        save_checkpoint(model, optimizer, epoch, epoch_loss, latest_path, "latest")

        # Save best model if current loss is better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, epoch_loss, best_path, "best")
            logger.info(f"[green]✓[/green] New best model (loss: {best_loss:.4f})", extra={"markup": True})

    # ========================================================================
    # CLEANUP
    # ========================================================================

    writer.close()
    logger.info("TensorBoard writer closed")

    console.print(Panel.fit(
        "[bold green]✓ Training Finished Successfully![/bold green]",
        border_style="green"
    ))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Head Depth Estimation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--dataset', '-d', type=str, default='sceneflow',
        choices=['sceneflow', 'ADT', 'DTU', 'Middlebury'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data_path', '-p', type=str, required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--sample_rate', type=int, default=None,
        help='Use 1/sample_rate of data (mutually exclusive with --data_percentage)'
    )
    parser.add_argument(
        '--data_percentage', type=float, default=None,
        help='Percentage of data to use (0.0-1.0, mutually exclusive with --sample_rate)'
    )

    # Model arguments
    parser.add_argument(
        '--checkpoint_path', '-c', type=str, default=None,
        help='Path to checkpoint for fine-tuning'
    )

    # Training arguments
    parser.add_argument(
        '--batch_size', '-b', type=int, default=10,
        help='Batch size for training'
    )
    parser.add_argument(
        '--total_epochs', '-e', type=int, default=100,
        help='Total number of training epochs'
    )
    parser.add_argument(
        '--val_epochs', '-v', type=int, default=20,
        help='Number of fine-tuning epochs (with lower learning rate)'
    )
    parser.add_argument(
        '--learning_rate', '-lr', type=float, default=4e-4,
        help='Learning rate for main training phase'
    )
    parser.add_argument(
        '--learning_rate_val', '-lrv', type=float, default=4e-4,
        help='Learning rate for fine-tuning phase'
    )

    # System arguments
    parser.add_argument(
        '--device', type=str, default='cuda', choices=['cpu', 'cuda'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--save_dir', '-s', type=str, default='ckpt',
        help='Directory to save checkpoints and logs'
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments and exit if invalid."""
    errors = []

    # Check mutually exclusive data sampling options
    if args.sample_rate is not None and args.data_percentage is not None:
        errors.append("Cannot use both --sample_rate and --data_percentage")

    # Validate data_percentage range
    if args.data_percentage is not None:
        if args.data_percentage <= 0.0 or args.data_percentage > 1.0:
            errors.append("--data_percentage must be in range (0.0, 1.0]")

    # Validate epoch configuration
    if args.total_epochs < args.val_epochs:
        errors.append("--total_epochs must be >= --val_epochs")

    # Validate data path
    if not os.path.exists(args.data_path):
        errors.append(f"Data path does not exist: {args.data_path}")

    # Print errors and exit if any
    if errors:
        for error in errors:
            logger.error(f"[red]✗ {error}[/red]", extra={"markup": True})
        exit(1)


def display_configuration(args: argparse.Namespace) -> None:
    """Display training configuration in a table."""
    config_table = Table(
        title="Training Configuration",
        show_header=True,
        header_style="bold cyan"
    )
    config_table.add_column("Parameter", style="cyan", justify="left")
    config_table.add_column("Value", style="yellow", justify="left")

    # Add configuration rows
    config_table.add_row("Dataset", args.dataset)
    config_table.add_row("Data Path", args.data_path)
    config_table.add_row("Device", args.device)
    config_table.add_row("Save Directory", args.save_dir)
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Total Epochs", str(args.total_epochs))
    config_table.add_row("Fine-tuning Epochs", str(args.val_epochs))

    # Data sampling configuration
    if args.sample_rate:
        config_table.add_row("Data Sampling", f"1/{args.sample_rate} of data")
    elif args.data_percentage:
        config_table.add_row("Data Sampling", f"{args.data_percentage*100:.1f}% of data")
    else:
        config_table.add_row("Data Sampling", "Full dataset")

    config_table.add_row("Training LR", str(args.learning_rate))
    config_table.add_row("Fine-tuning LR", str(args.learning_rate_val))
    config_table.add_row("Checkpoint", args.checkpoint_path if args.checkpoint_path else "None")

    console.print(config_table)
    console.print("\n")


def load_datasets(args: argparse.Namespace) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load training and validation datasets based on arguments."""
    logger.info(f"Loading [cyan]{args.dataset}[/cyan] dataset...", extra={"markup": True})

    # Load appropriate dataset
    if args.dataset == 'sceneflow':
        train_ds = utils.SceneFlowDataset(args.data_path, train=True, stereo=True)
        val_ds = utils.SceneFlowDataset(args.data_path, train=False, stereo=True)
    elif args.dataset == 'ADT':
        train_ds = utils.ADT(args.data_path, train=True)
        val_ds = utils.ADT(args.data_path, train=False)
    elif args.dataset == 'DTU':
        train_ds = utils.DTU(args.data_path, train='train', output_homo=False)
        val_ds = utils.DTU(args.data_path, train='test', output_homo=False)
    elif args.dataset == 'Middlebury':
        train_ds = utils.Middlebury(args.data_path)
        val_ds = utils.Middlebury(args.data_path)
    else:
        logger.error(f"[red]Unknown dataset: {args.dataset}[/red]", extra={"markup": True})
        exit(1)

    logger.info(
        f"[green]✓[/green] Loaded: {len(train_ds)} train samples, {len(val_ds)} val samples",
        extra={"markup": True}
    )

    return train_ds, val_ds


def main():
    """Main entry point."""
    # Print header
    console.print("\n")
    console.rule("[bold blue]Multi-Head Depth Training Script (Optimized)[/bold blue]")
    console.print("\n")

    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args)
    display_configuration(args)

    # Load datasets
    train_dataset, val_dataset = load_datasets(args)

    # Initialize model
    logger.info("Initializing [cyan]Stereo_MulH[/cyan] model...", extra={"markup": True})
    depth_model = model.Stereo_MulH()
    logger.info("[green]✓[/green] Model initialized", extra={"markup": True})

    # Start training
    train(depth_model, train_dataset, val_dataset, args)


if __name__ == "__main__":
    main()
