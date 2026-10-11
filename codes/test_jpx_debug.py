"""JPX debug exercises: tests only, no replacement implementation.

Place this file beside the six uploaded .py modules. No CSV is needed.
Run selected rounds with:
    python -m unittest -v test_jpx_debug.TensorTests
    python -m unittest -v test_jpx_debug.DataTests
    python -m unittest -v test_jpx_debug.ModelTests
    python -m unittest -v test_jpx_debug.TrainingTests
    python -m unittest -v test_jpx_debug.PipelineTests
Run all checks with:
    python -m unittest -v test_jpx_debug

Contracts used by this exercise:
* make_it_tensor preserves shape and returns numeric torch.float32 data.
* create_time_sequence returns windows and one scalar label per window.
* train processes ONE complete epoch by default and returns sample-weighted loss.
* The current model_1 configuration uses window=1, so it must support that length.
  If you intentionally change that design, revise the window-1 test explicitly.
* Cleanup must reject or exclude an invalid required price; an invalid split
  factor must be explicitly rejected or repaired to a valid positive value.

These tests use synthetic examples, not JPX market ground truth. They do not
verify the effective date of AdjustmentFactor, real data quality, chronological
train/validation splitting, or a model's investment performance. A passing test
is evidence only for the case tested, not proof that the entire project is correct.
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data_loader
import data_processor as processor
import features
import deep_learning as learning
import JPX_deep_learning_models as application

# All examples are tiny and CPU-only.
torch.set_num_threads(1)

FEATURES = ["Open", "High", "Low", "Close", "Volume", "ExpectedDividend"]


def tiny_frame(n: int = 4) -> pd.DataFrame:
    """Raw-looking, finite data with no split and no dividend event."""
    return pd.DataFrame({
        "SecuritiesCode": [1001] * n,
        "Date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "Open": np.arange(n, dtype=float) + 100,
        "High": np.arange(n, dtype=float) + 102,
        "Low": np.arange(n, dtype=float) + 99,
        "Close": np.arange(n, dtype=float) + 101,
        "Volume": np.arange(n, dtype=float) * 100 + 1000,
        "AdjustmentFactor": np.ones(n),
        "ExpectedDividend": np.zeros(n),
        "Target": np.arange(n, dtype=float) / 100,
    })


class TensorTests(unittest.TestCase):
    """Round 1: actual input types, numeric dtype, and shape preservation."""

    def assert_conversion(self, value, expected) -> None:
        tensor = learning.make_it_tensor(value)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32,
                         "Use one explicit numeric dtype for model inputs.")
        np.testing.assert_allclose(tensor.detach().cpu().numpy(),
                                   np.asarray(expected, dtype=np.float32))
        self.assertEqual(tuple(tensor.shape), np.asarray(expected).shape,
                         "Conversion must not flatten or reorder axes.")

    def test_01_series(self):
        self.assert_conversion(pd.Series([1., 2.]), [1., 2.])

    def test_02_dataframe(self):
        self.assert_conversion(pd.DataFrame([[1., 2.], [3., 4.]]),
                               [[1., 2.], [3., 4.]])

    def test_03_numpy_array(self):
        self.assert_conversion(np.array([[1., 2.], [3., 4.]]),
                               [[1., 2.], [3., 4.]])

    def test_04_scalar_list(self):
        self.assert_conversion([1., 2.], [1., 2.])

    def test_05_list_of_dataframe_windows(self):
        windows = [pd.DataFrame([[1., 2.], [3., 4.]]),
                   pd.DataFrame([[5., 6.], [7., 8.]])]
        self.assert_conversion(windows, [[[1., 2.], [3., 4.]],
                                         [[5., 6.], [7., 8.]]])


class SequenceTests(unittest.TestCase):
    """Control tests: preserve the windowing logic that is already working."""

    def test_01_stock_boundaries_sorting_feature_order_and_last_target(self):
        a = tiny_frame(3)
        b = tiny_frame(3)
        a["Close"], a["Target"] = [10., 20., 30.], [.1, .2, .3]
        b["SecuritiesCode"] = 2002
        b["Close"], b["Target"] = [110., 120., 130.], [1.1, 1.2, 1.3]
        # Deliberately mix stocks, dates, and index labels.
        df = pd.concat([a, b], ignore_index=True).iloc[[5, 1, 3, 0, 4, 2]]
        X, y = features.create_time_sequence(df, time_sequence=2,
                                              col_names=["Close"])
        self.assertEqual(len(X), 4)
        np.testing.assert_allclose(np.asarray(X),
                                   [[[10.], [20.]], [[20.], [30.]],
                                    [[110.], [120.]], [[120.], [130.]]])
        np.testing.assert_allclose(y, [.2, .3, 1.2, 1.3])

    def test_02_window_one_retains_all_rows(self):
        df = tiny_frame(3)
        X, y = features.create_time_sequence(df, 1, ["Close", "Open"])
        self.assertEqual(np.asarray(X).shape, (3, 1, 2))
        np.testing.assert_allclose(np.asarray(X)[0, 0],
                                   df.loc[0, ["Close", "Open"]].to_numpy(dtype=float))
        np.testing.assert_allclose(y, df["Target"])

    def test_03_stock_shorter_than_window(self):
        X, y = features.create_time_sequence(tiny_frame(2), 3, ["Close"])
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)


class DataTests(unittest.TestCase):
    """Round 2: data invariants, row labels, and validation coverage."""

    def test_01_no_split_keeps_original_prices_and_volume(self):
        raw = tiny_frame()
        out = processor.correct_prices_with_factors(raw)
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            np.testing.assert_allclose(out[column], raw[column],
                err_msg=f"No adjustment event: {column} should not change.")

    def test_02_input_dataframe_is_not_mutated(self):
        raw = tiny_frame()
        saved = raw.copy(deep=True)
        processor.correct_prices_with_factors(raw)
        pd.testing.assert_frame_equal(raw, saved)

    def test_03_nonconsecutive_index_labels(self):
        raw = tiny_frame(3)
        raw.index = [10, 20, 30]
        raw.loc[20, "Close"] = -1
        out = features.get_valid_rows(raw, list(raw.columns))
        self.assertEqual(out["Date"].tolist(), raw.iloc[[0, 2]]["Date"].tolist())

    def test_04_reordered_labels_must_not_remove_the_wrong_row(self):
        raw = tiny_frame(2)
        raw.index = [1, 0]
        raw["Close"] = [-1., 100.]
        out = features.get_valid_rows(raw, list(raw.columns))
        self.assertEqual(out["Close"].tolist(), [100.],
                         "Index labels and positions are not interchangeable.")

    def test_05_required_open_cannot_remain_nan(self):
        raw = tiny_frame(3)
        raw.loc[1, "Open"] = np.nan
        try:
            out = features.get_valid_rows(raw, list(raw.columns))
        except ValueError:
            return  # Explicit rejection is also a valid chosen policy.
        self.assertTrue(out["Open"].notna().all(),
                        "Listing Open as a feature does not create a validation rule.")

    def test_06_nonpositive_factors_are_rejected_or_repaired(self):
        for factor in [0., -0.5]:
            with self.subTest(factor=factor):
                raw = tiny_frame(3)
                raw.loc[0, "AdjustmentFactor"] = factor
                try:
                    out = processor.correct_prices_with_factors(raw)
                except ValueError:
                    continue
                factors = out["AdjustmentFactor"].to_numpy()
                self.assertTrue(np.isfinite(factors).all() and (factors > 0).all(),
                                "A finite numeric factor can still be invalid.")
                self.assertTrue(np.isfinite(out["Close"]).all())

    def test_07_get_features_reports_missing_columns(self):
        # This checks the explicit ValueError contract written in get_features,
        # not whether pandas eventually raises some other exception.
        with self.assertRaises(ValueError):
            features.get_features(tiny_frame(), ["Close", "NotAColumn"])

    def test_08_metadata_prints_column_names_not_a_method_object(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            data_loader.print_loaded_data(tiny_frame())
        self.assertIn(str(tiny_frame().columns.tolist()), capture.getvalue())


class ModelTests(unittest.TestCase):
    """Round 3: axes, pooling limits, dropout semantics, and configuration."""

    @staticmethod
    def small_model():
        return learning.CNN_LSTM(num_features=6, cnn_out_channels=4,
                                 kernel_size=1, hidden_size=4)

    def test_01_window_two_forward(self):
        model = self.small_model().eval()
        with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
            pred = model(torch.ones(3, 2, 6))
        self.assertEqual(tuple(pred.shape), (3, 1))
        self.assertTrue(torch.isfinite(pred).all().item())

    def test_02_current_window_one_design_must_run(self):
        model = self.small_model().eval()
        with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
            pred = model(torch.ones(3, 1, 6))
        self.assertEqual(tuple(pred.shape), (3, 1))

    def test_03_no_dropout1d_on_two_dimensional_batch_head(self):
        model = self.small_model().train()
        seen_dimensions = []
        handles = []
        for module in model.modules():
            if isinstance(module, nn.Dropout1d):
                handles.append(module.register_forward_pre_hook(
                    lambda _module, inputs: seen_dimensions.append(inputs[0].ndim)))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model(torch.ones(4, 4, 6))
        finally:
            for handle in handles:
                handle.remove()
        self.assertNotIn(2, seen_dimensions,
                         "For Dropout1d, a 2D input is (channels, length), not (batch, features).")

    def test_04_configuration_uses_feature_count_not_row_count(self):
        # Stop at construction, before allocation/training. No real CSV is read.
        class ReachedConstructor(Exception):
            pass

        captured = {}
        def record_constructor(*args, **kwargs):
            captured["num_features"] = kwargs.get("num_features", args[0] if args else None)
            raise ReachedConstructor

        with patch.object(application, "main_loader", return_value=tiny_frame(8)), \
             patch.object(application.deep_learning, "make_loader", return_value=object()), \
             patch.object(application.deep_learning, "CNN_LSTM", side_effect=record_constructor), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ReachedConstructor):
                application.model_1()
        self.assertEqual(captured["num_features"], 6,
                         "Adding rows must not change the number of input features.")


class CountingModel(nn.Module):
    """A deliberately simple model isolates loop/loss bugs from CNN bugs."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.))
        self.calls = 0
        self.grad_modes = []

    def forward(self, x):
        self.calls += 1
        self.grad_modes.append(torch.is_grad_enabled())
        return x.new_zeros((x.shape[0], 1)) + self.bias


def fixed_loader():
    # Three batches of size 2, 2, 1. The correct sample-weighted MAE is 2.6.
    return DataLoader(TensorDataset(torch.ones(5, 1),
                                   torch.tensor([[1.], [1.], [3.], [3.], [5.]])),
                      batch_size=2, shuffle=False)


class TrainingTests(unittest.TestCase):
    """Round 4: full epoch, accumulation, empty data, and eval mode."""

    def test_01_training_processes_every_batch(self):
        model = CountingModel()
        loader = fixed_loader()
        learning.train(model, loader, nn.L1Loss(),
                       torch.optim.SGD(model.parameters(), lr=0))
        self.assertEqual(model.calls, len(loader),
                         "A return inside the loop terminates the whole function.")

    def test_02_training_returns_sample_weighted_mean(self):
        model = CountingModel()
        result = learning.train(model, fixed_loader(), nn.L1Loss(),
                                torch.optim.SGD(model.parameters(), lr=0))
        # lr=0 makes every batch's expected loss independent of optimization.
        self.assertAlmostEqual(result, 2.6, places=6,
                               msg="Accumulate all batch sums before dividing by all samples.")

    def test_03_empty_training_loader_has_clear_error(self):
        model = CountingModel()
        empty = DataLoader(TensorDataset(torch.empty(0, 1), torch.empty(0, 1)),
                           batch_size=2)
        with self.assertRaises(ValueError):
            learning.train(model, empty, nn.L1Loss(),
                           torch.optim.SGD(model.parameters(), lr=0))

    def test_04_evaluation_mean_and_no_grad_control(self):
        model = CountingModel()
        result = learning.evaluation(model, fixed_loader(), nn.L1Loss())
        self.assertAlmostEqual(result, 2.6, places=6)
        self.assertEqual(model.calls, 3)
        self.assertFalse(model.training)
        self.assertEqual(model.grad_modes, [False, False, False])

    def test_05_evaluation_empty_loader_control(self):
        empty = DataLoader(TensorDataset(torch.empty(0, 1), torch.empty(0, 1)),
                           batch_size=2)
        with self.assertRaises(ValueError):
            learning.evaluation(CountingModel(), empty, nn.L1Loss())


class ExactShapeL1(nn.Module):
    """Fail before broadcasting can silently compare different samples."""

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise AssertionError(
                f"Loss shape mismatch: prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}. Each sample needs exactly one matching target.")
        return nn.functional.l1_loss(prediction, target)


class PipelineTests(unittest.TestCase):
    """Integration checks: real window -> loader -> training/evaluation calls.

    These naturally depend on the conversion checks passing first. No cleaning,
    scaling, market-data validation, or chronological splitting is tested here.
    """

    def make_sequence_loader(self):
        X, y = features.create_time_sequence(tiny_frame(4), 2, FEATURES)
        return learning.make_loader(X, y, batch_size=2, shuffle=False)

    def test_01_train_loss_shapes_match(self):
        model = CountingModel()
        value = learning.train(model, self.make_sequence_loader(), ExactShapeL1(),
                               torch.optim.SGD(model.parameters(), lr=0))
        self.assertTrue(np.isfinite(value))

    def test_02_eval_loss_shapes_match(self):
        value = learning.evaluation(CountingModel(), self.make_sequence_loader(),
                                    ExactShapeL1())
        self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main(verbosity=2)
