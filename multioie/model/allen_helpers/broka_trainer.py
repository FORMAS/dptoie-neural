import logging

import os
import time
import traceback
import datetime
from typing import Dict, Any, Tuple

import torch
import torch.optim.lr_scheduler

from allennlp.common.checks import ConfigurationError
from allennlp.common import util as common_util
from allennlp.training import util as training_util, GradientDescentTrainer
import torch.distributed as dist

from multioie.model.allen_helpers.my_metric_tracker import MyMetricTracker

logger = logging.getLogger(__name__)


class BrokaTrainer(GradientDescentTrainer):
    def __init__(self, *args, **kwargs):
        self.max_tokens_per_batch = kwargs["max_tokens_per_batch"]
        self._batch_num_total = 0
        del kwargs["max_tokens_per_batch"]
        del kwargs["min_improvement"]
        super().__init__(*args, **kwargs)
        self._metric_tracker = MyMetricTracker(kwargs["patience"])

    def _try_train(self) -> Tuple[Dict[str, Any], int]:
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._primary and self._checkpointer is not None:
                self._checkpointer.save_checkpoint(epoch, self, save_model_only=True)

            # Wait for the primary process to finish saving the model checkpoint
            if self._distributed:
                dist.barrier()

            # get peak of memory usage
            for key, value in train_metrics.items():
                if key.startswith("gpu_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)
                elif key.startswith("worker_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            this_epoch_val_metric: float = 0.0
            if self._validation_data_loader is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, val_reg_loss, num_batches = self._validation_loss(epoch)

                    # It is safe again to wait till the validation is done. This is
                    # important to get the metrics right.
                    if self._distributed:
                        dist.barrier()

                    val_metrics = training_util.get_metrics(
                        self.model,
                        val_loss,
                        val_reg_loss,
                        batch_loss=None,
                        batch_reg_loss=None,
                        num_batches=num_batches,
                        reset=True,
                        world_size=self._world_size,
                        cuda_device=self.cuda_device,
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = self._metric_tracker.combined_score(val_metrics)
                    self._metric_tracker.add_metrics(val_metrics)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._primary:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"),
                    metrics,
                )

            # If we do not have a validation data, use the training metrics.
            if self._validation_data_loader is None:
                fake_metrics = {
                    "loss": metrics["training_loss"],
                    "f1-measure-overall": metrics["training_f1-measure-overall"],
                }
                this_epoch_val_metric = self._metric_tracker.combined_score(fake_metrics)
                self._metric_tracker.add_metrics(fake_metrics)

                if self._metric_tracker.should_stop_early():
                    logger.info("Ran out of patience.  Stopping training.")
                    break
            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric)

            if self._primary and self._checkpointer is not None:
                self._checkpointer.save_checkpoint(
                    epoch, self, is_best_so_far=self._metric_tracker.is_best_so_far()
                )

            # Wait for the primary process to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

            for callback in self._callbacks:
                callback.on_epoch(self, metrics=metrics, epoch=epoch, is_primary=self._primary)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1
        else:
            epoch = self._num_epochs - 1

        # Load the best model state before returning
        best_model_state = (
            None if self._checkpointer is None else self._checkpointer.best_model_state()
        )
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics, epoch
