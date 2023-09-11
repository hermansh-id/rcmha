from abc import ABC
import torch


class StateModule:
    def __init__(self):
        pass

    def create_state(self) -> any:
        raise NotImplementedError

    def set_state(self, data: any):
        raise NotImplementedError

    def on_epoch_start(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError


class Metric(StateModule, ABC):
    def track(self):
        pass


class AccuracyState:
    """
    State class for tracking accuracy metric.

    Attributes:
        samples (int): The total number of samples processed.
        correct (int): The total number of correct predictions.

    Example:
        >>> state = AccuracyState()
        >>> state.samples = 100
        >>> state.correct = 80
    """
    samples: int = 0
    correct: int = 0

    def reset(self):
        """
        Reset the accuracy state by setting samples and correct counts to zero.
        """
        self.samples = 0
        self.correct = 0

class Accuracy(Metric):
    """
    A PyTorch metric for calculating accuracy.

    This metric computes the accuracy of model predictions compared to target values.

    Args:
        ignore_index (int, optional): The index to ignore when calculating accuracy. Default is -1.

    Example:
        >>> accuracy_metric = Accuracy(ignore_index=-1)
        >>> predictions = torch.tensor([1, 2, 3, 4])
        >>> targets = torch.tensor([1, 2, 0, 4])
        >>> accuracy_metric(predictions, targets)
        >>> accuracy = accuracy_metric.compute()
    """

    data = AccuracyState

    def __init__(self, ignore_index: int = -1):
        """
        Initializes a new instance of the Accuracy metric.

        Args:
            ignore_index (int, optional): The index to ignore when calculating accuracy. Default is -1.
        """
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        """
        Update the accuracy metric with model predictions and target values.

        Args:
            output (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.

        Example:
            >>> accuracy_metric = Accuracy(ignore_index=-1)
            >>> predictions = torch.tensor([1, 2, 3, 4])
            >>> targets = torch.tensor([1, 2, 0, 4])
            >>> accuracy_metric(predictions, targets)
        """
        output = output.view(-1, output.shape[-1])
        target = target.view(-1)
        pred = output.argmax(dim=-1)
        mask = target == self.ignore_index
        pred.masked_fill_(mask, self.ignore_index)
        n_masked = mask.sum().item()
        self.data.correct += pred.eq(target).sum().item() - n_masked
        self.data.samples += len(target) - n_masked

    def create_state(self):
        """
        Create a new AccuracyState instance.

        Returns:
            AccuracyState: A new instance of the AccuracyState class.

        Example:
            >>> state = self.create_state()
        """
        return AccuracyState()

    def set_state(self, data: any):
        """
        Set the internal state of the accuracy metric.

        Args:
            data (any): The state data to set.

        Example:
            >>> data = AccuracyState()
            >>> self.set_state(data)
        """
        self.data = data

    def on_epoch_start(self):
        """
        Reset the accuracy state at the beginning of an epoch.

        Example:
            >>> self.on_epoch_start()
        """
        self.data.reset()

    def on_epoch_end(self):
        """
        Track and record accuracy at the end of an epoch.

        Example:
            >>> self.on_epoch_end()
        """
        self.track()

    def track(self):
        """
        Calculate and return the accuracy value.

        Returns:
            float: The calculated accuracy.

        Example:
            >>> accuracy = self.track()
        """
        if self.data.samples == 0:
            return
        return (self.data.correct / self.data.samples)

