import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
from datetime import datetime
import json
import logging
from pathlib import Path

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
logger = logging.getLogger("eval_sceneflow")


def compute_metrics(pred_disp, gt_disp, threshold=3.0):
    """
    Compute stereo matching metrics.

    Args:
        pred_disp: Predicted disparity map
        gt_disp: Ground truth disparity map
        threshold: Threshold for D1 metric (default: 3.0 pixels)

    Returns:
        Dictionary of metrics
    """
    # Create mask for valid pixels (non-zero, non-inf, non-nan)
    mask = (gt_disp > 0) & (~torch.isinf(gt_disp)) & (~torch.isnan(gt_disp))

    if mask.sum() == 0:
        logger.warning("No valid pixels in ground truth disparity")
        return {
            'abs_rel': float('nan'),
            'sq_rel': float('nan'),
            'rmse': float('nan'),
            'rmse_log': float('nan'),
            'd1': float('nan'),
            'mae': float('nan'),
            'num_valid_pixels': 0
        }

    pred_disp_valid = pred_disp[mask]
    gt_disp_valid = gt_disp[mask]

    # Absolute Relative Error
    abs_rel = torch.mean(torch.abs(pred_disp_valid - gt_disp_valid) / gt_disp_valid)

    # Square Relative Error
    sq_rel = torch.mean(((pred_disp_valid - gt_disp_valid) ** 2) / gt_disp_valid)

    # RMSE (Root Mean Square Error)
    rmse = torch.sqrt(torch.mean((pred_disp_valid - gt_disp_valid) ** 2))

    # RMSE log
    rmse_log = torch.sqrt(torch.mean((torch.log(pred_disp_valid + 1e-8) - torch.log(gt_disp_valid + 1e-8)) ** 2))

    # D1 metric: percentage of pixels with error > threshold
    error = torch.abs(pred_disp_valid - gt_disp_valid)
    d1 = torch.mean((error > threshold).float()) * 100.0

    # Mean Absolute Error
    mae = torch.mean(torch.abs(pred_disp_valid - gt_disp_valid))

    # Threshold accuracies
    thresh = torch.maximum((gt_disp_valid / pred_disp_valid), (pred_disp_valid / gt_disp_valid))
    a1 = (thresh < 1.25).float().mean() * 100.0
    a2 = (thresh < 1.25 ** 2).float().mean() * 100.0
    a3 = (thresh < 1.25 ** 3).float().mean() * 100.0

    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'd1': d1.item(),
        'mae': mae.item(),
        'a1': a1.item(),
        'a2': a2.item(),
        'a3': a3.item(),
        'num_valid_pixels': mask.sum().item()
    }


def create_disparity_visualization(left_img, pred_disp, gt_disp, metrics, sample_idx, output_path):
    """
    Create a visualization comparing predicted and ground truth disparity.

    Args:
        left_img: Left input image tensor (C, H, W)
        pred_disp: Predicted disparity (1, H, W)
        gt_disp: Ground truth disparity (1, H, W)
        metrics: Dictionary of computed metrics
        sample_idx: Sample index
        output_path: Path to save the visualization
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Convert tensors to numpy
        if left_img.shape[0] == 6:  # Stereo pair
            left_img = left_img[:3]  # Take only left image

        left_img_np = left_img.cpu().numpy().transpose(1, 2, 0)
        left_img_np = (left_img_np - left_img_np.min()) / (left_img_np.max() - left_img_np.min())

        pred_disp_np = pred_disp.squeeze().cpu().numpy()
        gt_disp_np = gt_disp.squeeze().cpu().numpy()

        # Create error map
        error_map = np.abs(pred_disp_np - gt_disp_np)
        mask = (gt_disp_np > 0) & (~np.isinf(gt_disp_np)) & (~np.isnan(gt_disp_np))
        error_map[~mask] = 0

        # Row 1: Input, Ground Truth, Prediction
        axes[0, 0].imshow(left_img_np)
        axes[0, 0].set_title(f'Left Input Image\nSample {sample_idx}', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        im1 = axes[0, 1].imshow(gt_disp_np, cmap='jet', vmin=0, vmax=max(gt_disp_np.max(), pred_disp_np.max()))
        axes[0, 1].set_title('Ground Truth Disparity', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(pred_disp_np, cmap='jet', vmin=0, vmax=max(gt_disp_np.max(), pred_disp_np.max()))
        axes[0, 2].set_title('Predicted Disparity', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Row 2: Error Map, Histogram, Metrics Table
        im3 = axes[1, 0].imshow(error_map, cmap='hot', vmin=0, vmax=min(error_map.max(), 20))
        axes[1, 0].set_title('Absolute Error Map', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Error histogram
        valid_errors = error_map[mask]
        axes[1, 1].hist(valid_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Absolute Error (pixels)', fontsize=10)
        axes[1, 1].set_ylabel('Frequency', fontsize=10)
        axes[1, 1].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, min(valid_errors.max(), 20))

        # Metrics table
        axes[1, 2].axis('off')
        metric_text = (
            f"Metrics:\n"
            f"{'─' * 30}\n"
            f"AbsRel:      {metrics['abs_rel']:.4f}\n"
            f"SqRel:       {metrics['sq_rel']:.4f}\n"
            f"RMSE:        {metrics['rmse']:.4f}\n"
            f"RMSE (log):  {metrics['rmse_log']:.4f}\n"
            f"MAE:         {metrics['mae']:.4f}\n"
            f"D1 (>3px):   {metrics['d1']:.2f}%\n"
            f"{'─' * 30}\n"
            f"δ < 1.25:    {metrics['a1']:.2f}%\n"
            f"δ < 1.25²:   {metrics['a2']:.2f}%\n"
            f"δ < 1.25³:   {metrics['a3']:.2f}%\n"
            f"{'─' * 30}\n"
            f"Valid Pixels: {metrics['num_valid_pixels']:,}"
        )
        axes[1, 2].text(0.1, 0.5, metric_text, fontsize=11, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.debug(f"Saved visualization to {output_path}")

    except Exception as e:
        logger.error(f"Failed to create visualization for sample {sample_idx}: {e}")
        plt.close('all')


def generate_html_report(metrics_list, avg_metrics, args, output_dir, timestamp):
    """Generate an HTML report with evaluation results."""
    try:
        html_path = output_dir / f"evaluation_report_{timestamp}.html"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SceneFlow Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 20px 0;
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
        }}
        .info-item {{
            display: flex;
            justify-content: space-between;
        }}
        .info-label {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .info-value {{
            color: #34495e;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metrics-table th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        .metrics-table td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .metrics-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .metrics-table .metric-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .metrics-table .metric-value {{
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .medium {{ color: #f39c12; font-weight: bold; }}
        .bad {{ color: #e74c3c; font-weight: bold; }}
        .visualizations {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .viz-item {{
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            transition: transform 0.2s;
        }}
        .viz-item:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .viz-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .viz-caption {{
            padding: 10px;
            background-color: #f8f9fa;
            text-align: center;
            font-size: 14px;
            color: #2c3e50;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .summary-box h3 {{
            margin-top: 0;
            font-size: 24px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 SceneFlow Evaluation Report</h1>

        <div class="info-grid">
            <div class="info-item">
                <span class="info-label">Model Path:</span>
                <span class="info-value">{args.model_path}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Dataset Path:</span>
                <span class="info-value">{args.data_path}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Device:</span>
                <span class="info-value">{args.device}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Batch Size:</span>
                <span class="info-value">{args.batch_size}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Samples Evaluated:</span>
                <span class="info-value">{len(metrics_list)}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Evaluation Date:</span>
                <span class="info-value">{timestamp}</span>
            </div>
        </div>

        <div class="summary-box">
            <h3>📊 Overall Performance Summary</h3>
            <p style="font-size: 16px; margin: 10px 0;">
                Average metrics computed across {len(metrics_list)} samples from the SceneFlow validation dataset.
            </p>
        </div>

        <h2>Average Metrics</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="metric-name">AbsRel</td>
                    <td class="metric-value">{avg_metrics['abs_rel']:.4f}</td>
                    <td>Absolute Relative Error (lower is better)</td>
                </tr>
                <tr>
                    <td class="metric-name">SqRel</td>
                    <td class="metric-value">{avg_metrics['sq_rel']:.4f}</td>
                    <td>Square Relative Error (lower is better)</td>
                </tr>
                <tr>
                    <td class="metric-name">RMSE</td>
                    <td class="metric-value">{avg_metrics['rmse']:.4f}</td>
                    <td>Root Mean Square Error (lower is better)</td>
                </tr>
                <tr>
                    <td class="metric-name">RMSE (log)</td>
                    <td class="metric-value">{avg_metrics['rmse_log']:.4f}</td>
                    <td>RMSE in log space (lower is better)</td>
                </tr>
                <tr>
                    <td class="metric-name">MAE</td>
                    <td class="metric-value">{avg_metrics['mae']:.4f}</td>
                    <td>Mean Absolute Error in pixels (lower is better)</td>
                </tr>
                <tr>
                    <td class="metric-name">D1 (&gt;3px)</td>
                    <td class="metric-value">{avg_metrics['d1']:.2f}%</td>
                    <td>Percentage of pixels with error &gt; 3px (lower is better)</td>
                </tr>
                <tr>
                    <td class="metric-name">δ &lt; 1.25</td>
                    <td class="metric-value">{avg_metrics['a1']:.2f}%</td>
                    <td>Accuracy: % of pixels with ratio &lt; 1.25 (higher is better)</td>
                </tr>
                <tr>
                    <td class="metric-name">δ &lt; 1.25²</td>
                    <td class="metric-value">{avg_metrics['a2']:.2f}%</td>
                    <td>Accuracy: % of pixels with ratio &lt; 1.56 (higher is better)</td>
                </tr>
                <tr>
                    <td class="metric-name">δ &lt; 1.25³</td>
                    <td class="metric-value">{avg_metrics['a3']:.2f}%</td>
                    <td>Accuracy: % of pixels with ratio &lt; 1.95 (higher is better)</td>
                </tr>
            </tbody>
        </table>

        <h2>Sample Visualizations</h2>
        <p>Showing results for {min(args.num_visualizations, len(metrics_list))} samples (displaying input, ground truth, prediction, and error analysis).</p>

        <div class="visualizations">
"""

        # Add visualizations
        viz_files = sorted(output_dir.glob("sample_*.png"))[:args.num_visualizations]
        for viz_file in viz_files:
            sample_num = viz_file.stem.replace('sample_', '')
            html_content += f"""
            <div class="viz-item">
                <img src="{viz_file.name}" alt="Sample {sample_num}">
                <div class="viz-caption">Sample {sample_num}</div>
            </div>
"""

        html_content += """
        </div>

        <p class="timestamp">Report generated on """ + timestamp + """</p>
    </div>
</body>
</html>
"""

        with open(html_path, 'w') as f:
            f.write(html_content)

        logger.info(f"[green]✓[/green] HTML report saved to: [cyan]{html_path}[/cyan]", extra={"markup": True})
        return html_path

    except Exception as e:
        logger.error(f"[red]Failed to generate HTML report: {e}[/red]", extra={"markup": True})
        return None


def evaluate_model(args):
    """Main evaluation function."""
    try:
        console.print("\n")
        console.rule("[bold blue]SceneFlow Model Evaluation[/bold blue]")
        console.print("\n")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"eval_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: [cyan]{output_dir}[/cyan]", extra={"markup": True})

        # Setup device
        device = torch.device(args.device)
        logger.info(f"Using device: [bold cyan]{device}[/bold cyan]", extra={"markup": True})

        # Load dataset
        logger.info(f"Loading SceneFlow validation dataset from: [cyan]{args.data_path}[/cyan]", extra={"markup": True})
        try:
            val_dataset = utils.SceneFlowDataset(args.data_path, train=False, stereo=True)
            logger.info(f"[green]✓[/green] Dataset loaded: {len(val_dataset)} samples", extra={"markup": True})
        except Exception as e:
            logger.error(f"[red]✗ Failed to load dataset: {e}[/red]", extra={"markup": True})
            return

        # Create dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        logger.info(f"Batch size: {args.batch_size}, Number of batches: {len(val_loader)}")

        # Load model
        logger.info(f"Loading model from: [cyan]{args.model_path}[/cyan]", extra={"markup": True})
        try:
            if not os.path.exists(args.model_path):
                logger.error(f"[red]✗ Model checkpoint not found: {args.model_path}[/red]", extra={"markup": True})
                return

            eval_model = model.Stereo_MulH()
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                loss = checkpoint.get('loss', 'unknown')
                logger.info(f"Checkpoint info - Epoch: {epoch}, Loss: {loss}")
            else:
                state_dict = checkpoint
                logger.warning("Checkpoint doesn't contain metadata (epoch, loss)")

            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = utils.remove_module_prefix({'model_state_dict': state_dict})['model_state_dict']

            eval_model.load_state_dict(state_dict)
            eval_model.to(device)
            eval_model.eval()

            logger.info("[green]✓[/green] Model loaded successfully", extra={"markup": True})

        except Exception as e:
            logger.error(f"[red]✗ Failed to load model: {e}[/red]", extra={"markup": True})
            logger.exception("Model loading error details:")
            return

        # Evaluate
        console.print(Panel("[bold green]Starting Evaluation[/bold green]", border_style="green"))

        all_metrics = []
        sample_idx = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            eval_task = progress.add_task("[cyan]Evaluating...", total=len(val_loader))

            with torch.no_grad():
                for batch_idx, (images, gt_disp) in enumerate(val_loader):
                    try:
                        images = images.to(device)
                        gt_disp = gt_disp.to(device)

                        # Predict
                        pred_disp = eval_model(images)

                        # Compute metrics for each sample in batch
                        for i in range(images.shape[0]):
                            metrics = compute_metrics(pred_disp[i], gt_disp[i], threshold=args.d1_threshold)
                            all_metrics.append(metrics)

                            # Create visualization for first N samples
                            if sample_idx < args.num_visualizations:
                                viz_path = output_dir / f"sample_{sample_idx:04d}.png"
                                create_disparity_visualization(
                                    images[i],
                                    pred_disp[i],
                                    gt_disp[i],
                                    metrics,
                                    sample_idx,
                                    viz_path
                                )

                            sample_idx += 1

                        progress.update(eval_task, advance=1)

                    except Exception as e:
                        logger.error(f"[red]Error processing batch {batch_idx}: {e}[/red]", extra={"markup": True})
                        continue

        # Compute average metrics
        logger.info("[bold blue]Computing average metrics...[/bold blue]", extra={"markup": True})

        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = float('nan')

        # Display results in rich table
        results_table = Table(title="Evaluation Results", show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan", justify="left")
        results_table.add_column("Value", style="green", justify="right")
        results_table.add_column("Description", style="white", justify="left")

        results_table.add_row("AbsRel", f"{avg_metrics['abs_rel']:.4f}", "Absolute Relative Error")
        results_table.add_row("SqRel", f"{avg_metrics['sq_rel']:.4f}", "Square Relative Error")
        results_table.add_row("RMSE", f"{avg_metrics['rmse']:.4f}", "Root Mean Square Error")
        results_table.add_row("RMSE (log)", f"{avg_metrics['rmse_log']:.4f}", "RMSE in log space")
        results_table.add_row("MAE", f"{avg_metrics['mae']:.4f}", "Mean Absolute Error")
        results_table.add_row("D1 (>3px)", f"{avg_metrics['d1']:.2f}%", "% pixels with error > 3px")
        results_table.add_row("δ < 1.25", f"{avg_metrics['a1']:.2f}%", "Threshold accuracy")
        results_table.add_row("δ < 1.25²", f"{avg_metrics['a2']:.2f}%", "Threshold accuracy")
        results_table.add_row("δ < 1.25³", f"{avg_metrics['a3']:.2f}%", "Threshold accuracy")

        console.print(results_table)

        # Save metrics to JSON
        json_path = output_dir / "metrics.json"
        results_data = {
            'average_metrics': avg_metrics,
            'per_sample_metrics': all_metrics,
            'evaluation_info': {
                'model_path': args.model_path,
                'data_path': args.data_path,
                'num_samples': len(all_metrics),
                'device': str(device),
                'timestamp': timestamp
            }
        }

        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"[green]✓[/green] Metrics saved to: [cyan]{json_path}[/cyan]", extra={"markup": True})

        # Generate HTML report
        html_path = generate_html_report(all_metrics, avg_metrics, args, output_dir, timestamp)

        # Summary
        console.print(Panel.fit(
            f"[bold green]✓ Evaluation Complete![/bold green]\n\n"
            f"Evaluated {len(all_metrics)} samples\n"
            f"Results saved to: [cyan]{output_dir}[/cyan]\n"
            f"Visualizations: {min(args.num_visualizations, len(all_metrics))} samples\n"
            f"HTML Report: [cyan]{html_path.name if html_path else 'Failed'}[/cyan]",
            border_style="green"
        ))

    except Exception as e:
        logger.error(f"[red bold]✗ Evaluation failed: {e}[/red bold]", extra={"markup": True})
        logger.exception("Evaluation error details:")
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Evaluate Stereo Multi-Head model on SceneFlow dataset")

        parser.add_argument('--model_path', '-m', type=str, required=True,
                          help='Path to the trained model checkpoint (.pt file)')
        parser.add_argument('--data_path', '-d', type=str, required=True,
                          help='Path to the SceneFlow dataset')
        parser.add_argument('--output_dir', '-o', type=str, default='eval_results',
                          help='Directory to save evaluation results')
        parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                          help='Device to run evaluation')
        parser.add_argument('--batch_size', '-b', type=int, default=1,
                          help='Batch size for evaluation')
        parser.add_argument('--num_workers', type=int, default=4,
                          help='Number of data loading workers')
        parser.add_argument('--num_visualizations', '-n', type=int, default=20,
                          help='Number of sample visualizations to generate')
        parser.add_argument('--d1_threshold', type=float, default=3.0,
                          help='Threshold for D1 metric (in pixels)')

        args = parser.parse_args()

        # Display configuration
        config_table = Table(title="Evaluation Configuration", show_header=True, header_style="bold cyan")
        config_table.add_column("Parameter", style="cyan", justify="left")
        config_table.add_column("Value", style="yellow", justify="left")

        config_table.add_row("Model Path", args.model_path)
        config_table.add_row("Data Path", args.data_path)
        config_table.add_row("Output Directory", args.output_dir)
        config_table.add_row("Device", args.device)
        config_table.add_row("Batch Size", str(args.batch_size))
        config_table.add_row("Num Workers", str(args.num_workers))
        config_table.add_row("Visualizations", str(args.num_visualizations))
        config_table.add_row("D1 Threshold", f"{args.d1_threshold} pixels")

        console.print(config_table)
        console.print("\n")

        # Run evaluation
        evaluate_model(args)

    except KeyboardInterrupt:
        console.print("\n")
        logger.warning("[yellow]Evaluation interrupted by user (Ctrl+C)[/yellow]", extra={"markup": True})
    except Exception as e:
        console.print("\n")
        logger.error(f"[red bold]Fatal error: {e}[/red bold]", extra={"markup": True})
        logger.exception("Full traceback:")
