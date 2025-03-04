import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random

from model.medtok import MedTok
from data.dataset import MedicalCodeDataset
from data.dataloader import create_dataloader
from utils.config import MedTokConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MEDTOK model")
    
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--codebook_size", type=int, default=12000, help="Codebook size")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--text_encoder_model", type=str, default="bert-base-uncased", 
                        help="Text encoder model name")
    parser.add_argument("--text_encoder_type", type=str, default="weighted_pooling", 
                        choices=["default", "weighted_pooling"], help="Text encoder type")
    parser.add_argument("--graph_encoder_type", type=str, default="hierarchical", 
                        choices=["default", "gat", "hierarchical"], help="Graph encoder type")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device (cuda or cpu)")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_config(args):
    """Create config object from arguments."""
    config = MedTokConfig()
    
    # Update config with command line arguments
    config.codebook_size = args.codebook_size
    config.embedding_dim = args.embedding_dim
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_epochs = args.epochs
    config.seed = args.seed
    config.device = args.device
    config.text_encoder_model = args.text_encoder_model
    config.save_dir = os.path.join(args.output_dir, "checkpoints")
    config.log_dir = os.path.join(args.output_dir, "logs")
    
    # Create a few additional properties
    config.text_encoder_type = args.text_encoder_type
    config.graph_encoder_type = args.graph_encoder_type
    
    return config


def train_epoch(model, dataloader, optimizer, scheduler, config, epoch, writer):
    """
    Train the model for one epoch.
    
    Args:
        model: The MEDTOK model
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration object
        epoch: Current epoch number
        writer: TensorBoard writer
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    total_loss = 0.0
    total_q_loss = 0.0
    total_packing_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        graph_features = batch["graph_features"].to(config.device)
        graph_edge_index = batch["graph_edge_index"].to(config.device)
        graph_batch = batch["graph_batch"].to(config.device)
        
        # Forward pass
        outputs = model(
            input_ids, 
            graph_features, 
            graph_edge_index, 
            graph_batch
        )
        
        # Get losses
        loss = outputs["total_loss"]
        q_loss = outputs["quantization_loss"]
        packing_loss = outputs["packing_loss"]
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        total_q_loss += q_loss.item()
        total_packing_loss += packing_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": loss.item(),
            "q_loss": q_loss.item(),
            "pack_loss": packing_loss.item()
        })
        
        # Log to TensorBoard every 100 steps
        if batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Loss/quantization", q_loss.item(), global_step)
            writer.add_scalar("Loss/packing", packing_loss.item(), global_step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_q_loss = total_q_loss / len(dataloader)
    avg_packing_loss = total_packing_loss / len(dataloader)
    
    # Log epoch metrics
    writer.add_scalar("Epoch/loss", avg_loss, epoch)
    writer.add_scalar("Epoch/q_loss", avg_q_loss, epoch)
    writer.add_scalar("Epoch/packing_loss", avg_packing_loss, epoch)
    
    return avg_loss


def validate(model, dataloader, config, epoch, writer):
    """
    Validate the model on the validation set.
    
    Args:
        model: The MEDTOK model
        dataloader: Validation dataloader
        config: Configuration object
        epoch: Current epoch number
        writer: TensorBoard writer
    
    Returns:
        Average validation loss
    """
    model.eval()
    
    total_loss = 0.0
    total_q_loss = 0.0
    total_packing_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Validation {epoch}/{config.num_epochs}")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            graph_features = batch["graph_features"].to(config.device)
            graph_edge_index = batch["graph_edge_index"].to(config.device)
            graph_batch = batch["graph_batch"].to(config.device)
            
            # Forward pass
            outputs = model(
                input_ids, 
                graph_features, 
                graph_edge_index, 
                graph_batch
            )
            
            # Get losses
            loss = outputs["total_loss"]
            q_loss = outputs["quantization_loss"]
            packing_loss = outputs["packing_loss"]
            
            # Update metrics
            total_loss += loss.item()
            total_q_loss += q_loss.item()
            total_packing_loss += packing_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "val_loss": loss.item(),
                "val_q_loss": q_loss.item(),
                "val_pack_loss": packing_loss.item()
            })
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_q_loss = total_q_loss / len(dataloader)
    avg_packing_loss = total_packing_loss / len(dataloader)
    
    # Log validation metrics
    writer.add_scalar("Validation/loss", avg_loss, epoch)
    writer.add_scalar("Validation/q_loss", avg_q_loss, epoch)
    writer.add_scalar("Validation/packing_loss", avg_packing_loss, epoch)
    
    return avg_loss


def visualize_codebook(model, writer, epoch, config):
    """
    Visualize the codebook vectors using PCA or t-SNE.
    
    Args:
        model: The MEDTOK model
        writer: TensorBoard writer
        epoch: Current epoch number
        config: Configuration object
    """
    from sklearn.decomposition import PCA
    
    # Get the codebook weights
    codebook = model.codebook.codebook.weight.detach().cpu().numpy()
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    codebook_2d = pca.fit_transform(codebook)
    
    # Create scatter plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    
    # Plot text-specific region
    plt.scatter(
        codebook_2d[:config.text_specific_size, 0],
        codebook_2d[:config.text_specific_size, 1],
        color='blue',
        alpha=0.6,
        label='Text-specific'
    )
    
    # Plot graph-specific region
    plt.scatter(
        codebook_2d[config.text_specific_size:config.text_specific_size+config.graph_specific_size, 0],
        codebook_2d[config.text_specific_size:config.text_specific_size+config.graph_specific_size, 1],
        color='green',
        alpha=0.6,
        label='Graph-specific'
    )
    
    # Plot shared region
    plt.scatter(
        codebook_2d[config.text_specific_size+config.graph_specific_size:, 0],
        codebook_2d[config.text_specific_size+config.graph_specific_size:, 1],
        color='red',
        alpha=0.6,
        label='Shared'
    )
    
    plt.legend()
    plt.title(f'Codebook Visualization (Epoch {epoch})')
    
    # Save figure to TensorBoard
    writer.add_figure("Codebook/pca", plt.gcf(), epoch)
    plt.close()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    # Create config
    config = create_config(args)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Create datasets and dataloaders
    train_dataset = MedicalCodeDataset(args.data_dir, config, split="train")
    val_dataset = MedicalCodeDataset(args.data_dir, config, split="val")
    
    train_dataloader = create_dataloader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_dataloader = create_dataloader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Create model
    model = MedTok(config).to(config.device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * config.num_epochs
    
    # Create learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_steps / total_steps
    )
    
    # Train the model
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, config, epoch, writer)
        
        # Validate
        val_loss = validate(model, val_dataloader, config, epoch, writer)
        
        # Visualize codebook (every 5 epochs)
        if epoch % 5 == 0:
            visualize_codebook(model, writer, epoch, config)
        
        # Save the model if it has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                },
                os.path.join(config.save_dir, 'best_model.pt')
            )
            print(f"Epoch {epoch}: New best model saved (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                },
                os.path.join(config.save_dir, f'checkpoint_epoch_{epoch}.pt')
            )
    
    # Save the final model
    torch.save(
        {
            'epoch': config.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        },
        os.path.join(config.save_dir, 'final_model.pt')
    )
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
