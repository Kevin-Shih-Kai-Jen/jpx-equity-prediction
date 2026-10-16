"""Small CPU-only regression checks; no market data or full backtest is used.

Run from the repository root: python -m unittest discover -s tests -v
The pipeline test uses real preprocessing, CNN-LSTM training and checkpoints.
Only CSV input, split dates and the 400-stock portfolio metric are substituted.
"""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

import JPX_deep_learning_models as application
import deep_learning
import evaluation
import main
import models


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_two_folds_train_continuously_and_save_only_final_model(self):
        torch.manual_seed(23)
        dates = pd.bdate_range("2024-01-02", periods=50)
        day = np.arange(len(dates), dtype=float)
        raw = pd.DataFrame({
            "SecuritiesCode": 1301,
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": 100.0 + day,
            "High": 103.0 + day,
            "Low": 99.0 + day,
            "Close": 101.0 + day,
            "Volume": 1000.0 + day,
            "AdjustmentFactor": 1.0,
            "ExpectedDividend": 0.0,
            "Target": np.sin(day) / 100.0,
        })
        splits = [
            [dates[0], dates[45], dates[45], dates[47]],
            [dates[0], dates[47], dates[47], dates[49]],
        ]
        original_train = deep_learning.train
        original_save = deep_learning.save_model
        trained = []
        evaluated = []

        def train_and_record(model, loader, loss_fn, optimizer, **kwargs):
            loss = original_train(model, loader, loss_fn, optimizer, **kwargs)
            self.assertTrue(np.isfinite(loss))
            trained.append({
                "model": model,
                "optimizer": optimizer,
                "samples": len(loader.dataset),
                "epochs": kwargs["epochs"],
                "steps": {int(state["step"]) for state in optimizer.state.values()},
            })
            return loss

        def portfolio_metric(context, predictions, lookback):
            scored = evaluation.get_eval_df(context, predictions, lookback)
            fold = len(evaluated)
            expected_dates = dates[45 + 2 * fold:47 + 2 * fold]
            self.assertEqual(scored["Date"].tolist(), expected_dates.tolist())
            self.assertTrue(np.isfinite(scored["Prediction"]).all())
            evaluated.append(scored)
            return {"earnings": 0.0, "sharpe": 0.0, "rank_IC": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            data_path = Path(directory) / "custom prices.csv"

            def save_final(**kwargs):
                self.assertEqual(len(trained), len(splits))
                self.assertEqual(len(evaluated), len(splits))
                return original_save(**kwargs, checkpoint_dir=checkpoint_dir)

            output = io.StringIO()
            with (
                patch.object(application, "main_loader", side_effect=lambda *a, **k: raw.copy()) as loader,
                patch.object(application.utils, "create_start_end_date", return_value=splits),
                patch.object(torch.backends.mps, "is_available", return_value=False),
                patch.object(deep_learning, "train", side_effect=train_and_record),
                patch.object(evaluation, "overall_performance", side_effect=portfolio_metric),
                patch.object(deep_learning, "save_model", side_effect=save_final) as save,
                contextlib.redirect_stdout(output),
            ):
                path = application.model_2(data_path=data_path)

            self.assertEqual(loader.call_count, 2)
            for call in loader.call_args_list:
                self.assertEqual(call.args[0], data_path)
            save.assert_called_once()
            self.assertEqual(list(checkpoint_dir.iterdir()), [path])
            self.assertEqual(path.suffix, ".pt")
            self.assertIs(trained[0]["model"], trained[1]["model"])
            self.assertIs(trained[0]["optimizer"], trained[1]["optimizer"])
            self.assertEqual([run["samples"] for run in trained], [2, 4])
            self.assertEqual([run["epochs"] for run in trained], [20, 20])
            self.assertEqual([run["steps"] for run in trained], [{20}, {40}])
            self.assertEqual(output.getvalue().count("Epoch 20/20"), 2)

            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            metadata = checkpoint["metadata"]
            self.assertEqual(checkpoint["train_start"], dates[0].strftime("%Y-%m-%d"))
            self.assertEqual(checkpoint["train_end"], dates[44].strftime("%Y-%m-%d"))
            self.assertEqual(metadata["fold"], 2)
            self.assertEqual(metadata["total_folds"], 2)
            self.assertEqual(metadata["epochs_per_fold"], 20)
            self.assertEqual(metadata["lookback"], 42)
            self.assertEqual(metadata["model_config"], {
                "num_features": 6, "cnn_out_channels": 64,
                "kernel_size": 3, "hidden_size": 100,
            })
            for key, date in {
                "split_train_end_exclusive": dates[47],
                "validation_start_inclusive": dates[47],
                "validation_end_exclusive": dates[49],
                "scaler_fit_start_inclusive": dates[0],
                "scaler_fit_end_inclusive": dates[46],
            }.items():
                self.assertEqual(metadata[key], date.strftime("%Y-%m-%d"))

            loaded = main.use_loaded_model(path)
            self.assertEqual(len(loaded), 6)
            self.assertEqual(loaded[3], application.FEAUTRES)
            self.assertEqual(loaded[4:], (checkpoint["train_start"], checkpoint["train_end"]))
            final_scaler_data = raw.iloc[:47][application.FEAUTRES]
            np.testing.assert_allclose(loaded[1], final_scaler_data.mean())
            expected_scale = final_scaler_data.std(ddof=0).replace(0, 1)
            np.testing.assert_allclose(loaded[2], expected_scale)
            probe = torch.randn(2, 42, 6)
            with torch.inference_mode():
                torch.testing.assert_close(
                    trained[-1]["model"].eval()(probe), loaded[0](probe),
                    rtol=0, atol=0,
                )

    def test_repeated_checkpoint_saves_preserve_both_runs(self):
        model = models.CNN_LSTM(6, cnn_out_channels=4, kernel_size=3, hidden_size=4)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = (model, "CNN_LSTM", np.zeros(6), np.ones(6),
                    "2024-01-02", "2024-03-04", application.FEAUTRES)
            first = deep_learning.save_model(*args, checkpoint_dir=directory)
            first_bytes = first.read_bytes()
            with torch.no_grad():
                model.linear.bias.add_(1.0)
            second = deep_learning.save_model(*args, checkpoint_dir=directory)
            self.assertNotEqual(first, second)
            self.assertEqual(first.suffix, ".pt")
            self.assertEqual(second.suffix, ".pt")
            self.assertEqual(first.read_bytes(), first_bytes)
            self.assertEqual(len(list(Path(directory).glob("*.pt"))), 2)
            self.assertNotEqual(second.read_bytes(), first_bytes)

    def test_cli_passes_custom_and_default_csv_paths(self):
        custom = Path("input") / "my prices.csv"
        for arguments, expected in [
            (["--data", str(custom)], custom),
            ([], ROOT / "data" / "stock_prices.csv"),
        ]:
            with self.subTest(arguments=arguments):
                with patch.object(sys, "argv", ["main.py", *arguments]), patch.object(main, "model_2") as run:
                    main.main()
                run.assert_called_once_with(data_path=expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
