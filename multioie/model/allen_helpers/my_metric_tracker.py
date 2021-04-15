from typing import Optional, Iterable, Dict, Any, Union, List
from collections import deque

from allennlp.common.checks import ConfigurationError
from allennlp.training.metric_tracker import MetricTracker


class MyMetricTracker(MetricTracker):
    def __init__(
        self,
        patience: Optional[int] = None,
    ) -> None:
        self._patience = patience
        self._best_so_far: Optional[float] = None
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = True
        self._epoch_number = 0
        self.best_epoch: Optional[int] = None
        self.best_epoch_metrics: Dict[str, float] = {}

        self.tracked_metrics = []

        # self.tracked_metrics.append((100.0, "f1-measure-overall"))
        self.tracked_metrics.append((-1.0, "loss"))
