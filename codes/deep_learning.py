import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from uuid import uuid4




def make_it_tensor(
    array: list | np.ndarray | pd.Series | pd.DataFrame
) -> torch.Tensor:
    if isinstance(array, list):
        array = np.array(array)
    
    if isinstance(array, pd.Series) or isinstance(array, pd.DataFrame):
        array = array.to_numpy()
    
    
    return torch.from_numpy(array.astype(np.float32))
    

def make_loader(X, y, batch_size=64, shuffle=False):
    # DataFrame -> Numpy -> Tensor (DataFrame cannot change to Tensor directly)
    X = make_it_tensor(X)
    y = make_it_tensor(y)
    
    # (samples,) -> (samples, 1)
    # For matching the dimension of output from CNN-LSTM of (64, 1) 
    # While original y's shape is (64,) only
    if y.ndim == 1:
        y = y.unsqueeze(1)
        
    # Match answer with preidction
    dataset = TensorDataset(X, y)
    
    # Make it a dataloader
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def train(model, data_loader, loss_fn, optimizer, epochs=1, device="cpu"):
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    model.train()
    model = model.to(device)
    
    total_loss = 0.0
    total_sample = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            prediction = model(X_batch)
            
            loss = loss_fn(prediction, y_batch)
            
            loss.backward()
            
            optimizer.step()
            
            batch_size = X_batch.shape[0]
            batch_loss = loss.item() * batch_size
            total_loss += batch_loss
            total_sample += batch_size
            epoch_loss += batch_loss
            epoch_samples += batch_size

        if epoch_samples:
            print(
                f"Epoch {epoch + 1}/{epochs} - training average loss: "
                f"{epoch_loss / epoch_samples:.6f}",
                flush=True,
            )
        
    if total_sample == 0:
        raise ValueError("Missing data (sample) for training")
    
    return total_loss / total_sample 


def evaluation(
    model: nn.Module, 
    data_loader: DataLoader, 
    loss_fn: nn.Module,
    device: str = "cpu"
) -> tuple[float, np.ndarray]:
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_sample = 0
    
    # For evaluation.py's df
    eval_predictions = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            
            # reshape(-1) is a wild card that flattens multi dimension 
            # inputs into 1 dimension
            eval_predictions.append(predictions.cpu().reshape(-1))
            
            loss = loss_fn(predictions, y_batch)
            
            batch_size = X_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_sample += batch_size
    
    # input: [[1.0], [2.0].....] --> concat and change to ndarray()
    if total_sample == 0 or len(eval_predictions) == 0: 
        raise ValueError("Required more data to get the desired results")
    
    all_predictions = torch.cat(eval_predictions).numpy()
    
    return total_loss / total_sample, all_predictions




def save_model(
    model,
    model_name,
    mean,
    std,
    train_start,
    train_end,
    feature_columns,
    *,
    checkpoint_dir: str | Path | None = None,
    metadata: dict | None = None,
):
    """Save the supplied final model without overwriting an earlier run.

    Metadata should contain plain Python values so the checkpoint remains
    compatible with the default weights-only loader.
    """
    checkpoint_dir = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else Path(__file__).resolve().parent.parent / "checkpoints"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_path = checkpoint_dir / (
        f"{Path(model_name).stem}_{timestamp}_{uuid4().hex[:8]}.pt"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": np.asarray(mean).tolist(),
            "std": np.asarray(std).tolist(),
            "feature_columns": list(feature_columns),
            "train_start": train_start,
            "train_end": train_end,
            "metadata": dict(metadata) if metadata is not None else {},
        },
        model_path,
    )

    print(f"Model is saved: {model_path.resolve()}")
    return model_path


def load_model(
    initial_model,
    model_path: str | Path,
    map_location="cpu",
    weights_only=True,
    device="cpu",
):
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model path doesn't exist: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=map_location,
        weights_only=weights_only,
    )

    initial_model.load_state_dict(checkpoint["model_state_dict"])
    model = initial_model.to(device)
    model.eval()

    return (
        model,
        np.asarray(checkpoint["mean"]),
        np.asarray(checkpoint["std"]),
        checkpoint["feature_columns"],
        checkpoint["train_start"],
        checkpoint["train_end"],
    )
