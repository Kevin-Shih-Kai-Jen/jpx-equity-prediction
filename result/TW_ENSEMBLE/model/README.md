# TW_ENSEMBLE model checkpoints

`JPX_ENSEMBLE_initial.pt` is the exact Japanese starting checkpoint; `TW_ENSEMBLE_final.pt` is the final selected Taiwan checkpoint.

The final archive contains `model_state_dict`, `scaler` and `config`. Architecture: `ENSEMBLE(6, 128, 1)` in [`../source/models.py`](../source/models.py). Use `torch.load(..., map_location="cpu", weights_only=True)` and a strict state-dictionary load. Preserve the source feature construction and frozen scaler for inference.

Only F3 epoch 1 was accepted as an update. All later selected weights were carried forward. Neither file is an optimizer-resume checkpoint or a live trading system. See the complete report for temporal, data and execution limitations.
