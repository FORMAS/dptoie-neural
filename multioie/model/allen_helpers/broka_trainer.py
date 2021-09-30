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
        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        training_start_time = None

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._epochs_completed < self._start_after_epochs_completed:
                # We're still catching up with the checkpoint, so we do nothing.
                # Note that we have to call _train_epoch() even when we know the epoch is skipped. We have to
                # read from the data loader, because the data loader and dataset readers might use randomness,
                # and we have to make sure we consume exactly the same instances in exactly the same way every
                # time we train, even when starting from a checkpoint, so that we update the randomness
                # generators in the same way each time.
                self._epochs_completed += 1
                self._batches_in_epoch_completed = 0
                continue
            if training_start_time is None:
                training_start_time = epoch_start_time

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
            metrics["training_epochs"] = self._epochs_completed
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
                    # "f1-measure-overall": metrics["training_f1-measure-overall"],
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
            for callback in self._callbacks:
                callback.on_epoch(self, metrics=metrics, epoch=epoch, is_primary=self._primary)

            self._epochs_completed += 1
            self._batches_in_epoch_completed = 0

            # The checkpointer saves state from the learning rate scheduler, momentum scheduler, moving
            # average, and callbacks, so we have to make sure those are updated before we save the
            # checkpoint here.
            if self._primary and self._checkpointer is not None:
                self._checkpointer.maybe_save_checkpoint(
                    self, self._epochs_completed, self._batches_in_epoch_completed
                )
            # Wait for the primary process to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

            if self._primary and self._serialization_dir and self._metric_tracker.is_best_so_far():
                self._best_model_filename = os.path.join(self._serialization_dir, "best.th")
                if self._moving_average is None:
                    torch.save(self.model.state_dict(), self._best_model_filename)
                else:
                    self._moving_average.assign_average_value()
                    try:
                        torch.save(self.model.state_dict(), self._best_model_filename)
                    finally:
                        self._moving_average.restore()
            # Wait for the primary process to finish saving the best
            if self._distributed:
                dist.barrier()

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if self._metric_tracker.should_stop_early():
                logger.info("Ran out of patience. Stopping training.")
                break

            if epoch < self._num_epochs - 1:
                time_per_epoch = training_elapsed_time / (
                    (epoch + 1) - self._start_after_epochs_completed
                )
                # Note: If the first non-skipped epoch is half skipped (because it was checkpointed half-way
                # through), then this estimate is going to be optimistic.
                estimated_time_remaining = (
                    time_per_epoch * self._num_epochs
                ) - training_elapsed_time
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)
        else:
            epoch = self._num_epochs - 1

        # Load the best model state before returning
        if self._best_model_filename is None or self._metric_tracker.is_best_so_far():
            self._finalize_model()
        else:
            # The model we're loading here has already been finalized.
            self.model.load_state_dict(torch.load(self._best_model_filename))

        return metrics, epoch
