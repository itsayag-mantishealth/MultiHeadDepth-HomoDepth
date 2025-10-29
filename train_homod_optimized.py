"""
Homography and Depth Estimation Training Script (Optimized)

This script trains a joint homography and depth estimation model using the HomoDepth architecture.
The model learns to estimate both homography transformations and disparity maps from stereo images.

Key Features:
- Joint homography and depth estimation with uncertainty weighting
- Two-phase training (main training + fine-tuning with lower learning rate)
- TensorBoard integration for metrics and debug images
- Automatic best model tracking based on training loss
- Validation debug images with disparity visualizations

Loss Components:
- Homography loss: MSE on normalized homography parameters
- Disparity loss: Smooth L1 loss on disparity predictions
- Uncertainty weighting: Learned s1 and s2 parameters balance the two losses

Usage:
    python train_homod.py --data_path /path/to/DTU --batch_size 10 --total_epochs 100
"""

import argparse
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
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
logger = logging.getLogger("train_homod")

# Homography normalization bounds (dataset-specific for DTU)
HOMO_MIN = [0.55, -0.2, -96, -0.35, 0.85, -56, 0.9]
HOMO_MAX = [1.05, 0.4, -15, 0.25, 1.2, 128, 1]


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
    abs_rel_error, rmse_metric, d1_metric = 0.0, 0.0, 0.0

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
            for images, homographies, disparities in val_loader:
                # Move data to device
                images = images.to(device)
                disparities = disparities.to(device)

                # Forward pass: model returns (homo_norm, homo, disparity)
                _, _, disparity_predictions = model(images)

                # Compute validation metrics (only on disparity)
                abs_rel_error += utils.abs_rel_error(disparity_predictions, disparities)
                rmse_metric += utils.RMSE(disparity_predictions, disparities)
                d1_metric += utils.D1_metric(disparity_predictions, disparities)

                progress.update(val_task, advance=1)

    # Compute average metrics
    num_batches = len(val_loader)
    abs_rel_error /= num_batches
    rmse_metric /= num_batches
    d1_metric /= num_batches

    # Display validation results in a table
    results_table = Table(title="Validation Metrics", show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="cyan", justify="left")
    results_table.add_column("Value", style="green", justify="right")
    results_table.add_row("AbsRel (↓)", f"{abs_rel_error:.4f}")
    results_table.add_row("RMSE (↓)", f"{rmse_metric:.4f}")
    results_table.add_row("D1 (↓)", f"{d1_metric:.4f}")
    console.print(results_table)

    # Log metrics to TensorBoard
    if tensorboard_writer is not None and epoch is not None:
        tensorboard_writer.add_scalar('validation/AbsRel', abs_rel_error, epoch)
        tensorboard_writer.add_scalar('validation/RMSE', rmse_metric, epoch)
        tensorboard_writer.add_scalar('validation/D1', d1_metric, epoch)

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
    mse_criterion: nn.Module,
    smooth_criterion: nn.Module,
    wmse_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    homo_min: list,
    homo_max: list,
    phase_name: str = "Training"
) -> Tuple[float, float]:
    """
    Run one training epoch with joint homography and depth estimation.

    The model learns two tasks simultaneously:
    1. Homography estimation (geometric transformation between views)
    2. Disparity estimation (depth from stereo)

    The total loss is a weighted sum with learnable uncertainty parameters:
    L = L_homo / (2 * exp(s1)) + L_disp / (2 * exp(s2)) + (s1 + s2) / 2

    Args:
        model: Model to train (HomoDepth)
        train_loader: Training data loader
        optimizer: Optimizer for parameter updates
        mse_criterion: MSE loss for normalized homography
        smooth_criterion: Smooth L1 loss for disparity
        wmse_criterion: Weighted MSE loss for homography (for logging only)
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs (for display)
        homo_min: Minimum values for homography normalization
        homo_max: Maximum values for homography normalization
        phase_name: Name of training phase (for display)

    Returns:
        Tuple of (average_l1_loss, average_wmse_loss)
    """
    model.train()
    running_l1_loss = 0.0  # Disparity loss
    running_wmse_loss = 0.0  # Homography loss (weighted MSE)

    # Training loop with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("L1: {task.fields[l1]:.4f} | WMSE: {task.fields[wmse]:.4f}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_color = "green" if "Training" in phase_name else "magenta"
        train_task = progress.add_task(
            f"[{task_color}]Epoch {epoch}/{total_epochs-1}",
            total=len(train_loader),
            l1=0.0,
            wmse=0.0
        )

        for batch_idx, (images, homographies, disparities) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            homographies = homographies.to(device)
            disparities = disparities.to(device)

            # ================================================================
            # FORWARD PASS
            # ================================================================
            # Model returns three outputs:
            # 1. homo_norm_pred: Normalized homography prediction (for MSE loss)
            # 2. homo_pred: Unnormalized homography prediction (for WMSE evaluation)
            # 3. disp_pred: Disparity prediction
            homo_norm_pred, homo_pred, disp_pred = model(images)

            # Normalize ground truth homography to [0, 1] range for stable training
            homo_norm_gt = utils.homo2norm(homographies, homo_max, homo_min).to(device)

            # ================================================================
            # COMPUTE LOSSES
            # ================================================================
            # Homography loss: MSE on normalized parameters
            homo_loss = mse_criterion(homo_norm_pred, homo_norm_gt)

            # Disparity loss: Smooth L1 loss
            disp_loss = smooth_criterion(disp_pred, disparities)

            # Total loss with uncertainty weighting (learned s1, s2 parameters)
            # Lower uncertainty (higher exp(s)) → higher weight for that task
            # The regularization term (s1 + s2)/2 prevents uncertainties from growing too large
            total_loss = (
                homo_loss / (2 * torch.exp(model.s1)) +  # Weighted homography loss
                disp_loss / (2 * torch.exp(model.s2)) +  # Weighted disparity loss
                (model.s1 + model.s2) / 2  # Regularization on uncertainty parameters
            )

            # ================================================================
            # BACKWARD PASS
            # ================================================================
            optimizer.zero_grad()  # Clear previous gradients
            total_loss.backward()  # Compute gradients via backpropagation
            optimizer.step()  # Update model parameters

            # ================================================================
            # LOGGING METRICS
            # ================================================================
            # Compute WMSE on unnormalized homography for interpretability
            wmse_loss = wmse_criterion(homo_pred, homographies)
            running_wmse_loss += wmse_loss.item()
            running_l1_loss += disp_loss.item()

            # Update progress bar
            current_l1 = running_l1_loss / (batch_idx + 1)
            current_wmse = running_wmse_loss / (batch_idx + 1)
            progress.update(train_task, advance=1, l1=current_l1, wmse=current_wmse)

    # Compute average losses for the epoch
    avg_l1_loss = running_l1_loss / len(train_loader)
    avg_wmse_loss = running_wmse_loss / len(train_loader)

    logger.info(
        f"Epoch {epoch} completed - L1: [yellow]{avg_l1_loss:.4f}[/yellow], "
        f"WMSE: [yellow]{avg_wmse_loss:.4f}[/yellow]",
        extra={"markup": True}
    )

    return avg_l1_loss, avg_wmse_loss


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
        model: Model to train (HomoDepth)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        args: Command-line arguments containing all hyperparameters
    """
    console.print(Panel.fit(
        "[bold green]Starting Homography + Depth Training[/bold green]",
        border_style="green"
    ))

    # ========================================================================
    # SETUP: Data, Device, Optimizer, Criteria
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

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    logger.info(f"Optimizer: Adam (lr={args.learning_rate})")

    # Initialize loss criteria
    mse_criterion = nn.MSELoss()  # For normalized homography
    smooth_criterion = utils.SmoothLoss(1)  # For disparity (delta=1)
    wmse_criterion = utils.WMSELoss(50, device)  # For homography evaluation
    logger.info("Loss criteria: MSE (homo), SmoothL1 (disp), WMSE (eval)")

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
        logger.info(
            f"[green]✓[/green] Checkpoint loaded: epoch={ckpt_epoch}, loss={ckpt_loss}",
            extra={"markup": True}
        )

    # ========================================================================
    # MODEL PREPARATION
    # ========================================================================

    # Move model to device and set to training mode
    model.to(device)
    model.train()
    logger.info("Model initialized for training")

    # Log learnable uncertainty parameters
    logger.info(f"Initial uncertainty parameters: s1={model.s1.item():.4f}, s2={model.s2.item():.4f}")

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
        l1_loss, wmse_loss = train_epoch(
            model, train_loader, optimizer,
            mse_criterion, smooth_criterion, wmse_criterion,
            device, epoch, train_epochs, HOMO_MIN, HOMO_MAX, "Training"
        )

        # Log training metrics to TensorBoard
        writer.add_scalar('train/l1_loss', l1_loss, epoch)
        writer.add_scalar('train/wmse', wmse_loss, epoch)
        writer.add_scalar('train/learning_rate', args.learning_rate, epoch)
        writer.add_scalar('train/uncertainty_s1', model.s1.item(), epoch)
        writer.add_scalar('train/uncertainty_s2', model.s2.item(), epoch)

        # Run validation after each epoch
        val_loss = validate(
            model, val_dataset, args.batch_size, device,
            args.sample_rate, args.data_percentage,
            args.save_dir, epoch, writer
        )

        # Save latest checkpoint (overwritten every epoch)
        latest_path = os.path.join(args.save_dir, 'latest.pt')
        save_checkpoint(model, optimizer, epoch, l1_loss, latest_path, "latest")

        # Save best model if current loss is better
        if l1_loss < best_loss:
            best_loss = l1_loss
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, l1_loss, best_path, "best")
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
        l1_loss, wmse_loss = train_epoch(
            model, train_loader, optimizer,
            mse_criterion, smooth_criterion, wmse_criterion,
            device, epoch, args.total_epochs, HOMO_MIN, HOMO_MAX, "Fine-tuning"
        )

        # Log training metrics to TensorBoard
        writer.add_scalar('train/l1_loss', l1_loss, epoch)
        writer.add_scalar('train/learning_rate', args.learning_rate_val, epoch)
        writer.add_scalar('train/uncertainty_s1', model.s1.item(), epoch)
        writer.add_scalar('train/uncertainty_s2', model.s2.item(), epoch)

        # Run validation after each epoch
        val_loss = validate(
            model, val_dataset, args.batch_size, device,
            args.sample_rate, args.data_percentage,
            args.save_dir, epoch, writer
        )

        # Save epoch checkpoint (kept for all fine-tuning epochs)
        epoch_path = os.path.join(args.save_dir, f'epoch{epoch}.pt')
        save_checkpoint(model, optimizer, epoch, l1_loss, epoch_path, "epoch")

        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, 'latest.pt')
        save_checkpoint(model, optimizer, epoch, l1_loss, latest_path, "latest")

        # Save best model if current loss is better
        if l1_loss < best_loss:
            best_loss = l1_loss
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, l1_loss, best_path, "best")
            logger.info(f"[green]✓[/green] New best model (loss: {best_loss:.4f})", extra={"markup": True})

    # ========================================================================
    # CLEANUP
    # ========================================================================

    # Log final uncertainty parameters
    logger.info(
        f"Final uncertainty parameters: s1={model.s1.item():.4f}, s2={model.s2.item():.4f}"
    )

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
        description="Train HomoDepth model (joint homography and depth estimation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--data_path', '-p', type=str, required=True,
        help='Path to DTU dataset directory'
    )
    parser.add_argument(
        '--sample_rate', '-sr', type=int, default=None,
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
    config_table.add_row("Dataset", "DTU (HomoDepth)")
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


def main():
    """Main entry point."""
    # Print header
    console.print("\n")
    console.rule("[bold blue]HomoDepth Training Script (Optimized)[/bold blue]")
    console.print("\n")

    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args)
    display_configuration(args)

    # Load DTU dataset
    logger.info("Loading [cyan]DTU[/cyan] dataset...", extra={"markup": True})
    train_dataset = utils.DTU(args.data_path, train='train')
    val_dataset = utils.DTU(args.data_path, train='test')
    logger.info(
        f"[green]✓[/green] Loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples",
        extra={"markup": True}
    )

    # Initialize HomoDepth model
    logger.info("Initializing [cyan]HomoDepth[/cyan] model...", extra={"markup": True})
    homodepth_model = model.HomoDepth()
    logger.info("[green]✓[/green] Model initialized", extra={"markup": True})

    # Start training
    train(homodepth_model, train_dataset, val_dataset, args)


if __name__ == "__main__":
    main()
