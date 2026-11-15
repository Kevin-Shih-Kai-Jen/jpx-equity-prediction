import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path




def make_it_tensor(
    array: list | np.ndarray | pd.Series | pd.DataFrame
) -> torch.Tensor:
    if isinstance(array, list):
        array = np.array(array)
    
    if isinstance(array, pd.Series) or isinstance(array, pd.DataFrame):
        array = array.to_numpy()
    
    
    return torch.from_numpy(array.astype(np.float32))
    

def make_loader(X, y, batch_size=64, shuffle=False, dates=None):
    if dates is not None:
        if shuffle:
            raise ValueError(
                "Daily loader currently requires shuffle=False"
            )
        return make_daily_loader(X, y, dates)
    
    
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
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available() # Make the data in the CPU memory that is faster to move data to GPU
    )
    

def make_daily_loader(X, y, dates):
    X = make_it_tensor(X)
    y = make_it_tensor(y)
    
    if y.ndim == 1:
        y = y.unsqueeze(-1) # Make it (N, 1)
    
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("Stage_1 requires y.shape == (N, 1)")
    
    if not len(X) == len(y) == len(dates):
        raise ValueError("X, y, dates must have same shape")
    
    dates_groups = {}
    
    for i, date in enumerate(dates):
        date = pd.Timestamp(date)
        dates_groups.setdefault(date, []).append(i)
        
    batches = [
        dates_groups[date]
        for date in sorted(dates_groups)
    ]
    
    dataset = TensorDataset(X, y)
    
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=batches
    )
    
    # Record how data's orders are changed.
    # In evaluation batches, the order will be restored again
    loader.changed_data_orders = torch.tensor(
        [i for batch_indicies in batches for i in batch_indicies],
        dtype=torch.long
    )

    return loader

def train(
    model, 
    data_loader, 
    loss_fn, 
    optimizer, 
    epochs=1, 
    device="cpu",
    patience: int=10,
    min_change: float=0.00001
):
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    model = model.to(device)
    model.train()
    
    total_loss = 0.0
    total_sample = 0
    best_loss = float("inf")
    count = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_sample = 0

        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            prediction, loss = model(X_batch, loss_fn, y_batch)
           
            loss.backward()
            
            optimizer.step()
            
            batch_size = X_batch.shape[0]
            
            total_loss += loss.item() * batch_size
            total_sample += batch_size

            epoch_loss += loss.item() * batch_size
            epoch_sample += batch_size
        
        if epoch_sample == 0:
            raise ValueError("Missing data (sample) for training")
        
        avg_loss = epoch_loss / epoch_sample

        # Patience
        if best_loss - avg_loss > min_change:
            best_loss = avg_loss
            count = 0
        else:
            count += 1
        
        if count >= patience:
            print(f"The model has not improved for {patience} epochs, so we stop training")
            break
    
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
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            predictions = model(X_batch, None, None)
            
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
    
    # Daily bases -> Stock Bases
    # To match get_eval_df(all_pred) which is stock bases sorting
    if hasattr(data_loader, "changed_data_orders"):
        order = torch.argsort(data_loader.changed_data_orders)
        all_predictions = all_predictions[order]
    
    return total_loss / total_sample, all_predictions


def save_model(
    model,
    model_name,
    mean,
    std,
    train_start,
    train_end,
    feature_columns,
):
    model_path = Path("checkpoints") / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": np.asarray(mean).tolist(),
            "std": np.asarray(std).tolist(),
            "feature_columns": list(feature_columns),
            "train_start": train_start,
            "train_end": train_end,
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
    