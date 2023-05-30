from typing import Callable, Tuple
import torch


class Task:
    """This is a general class for tasks, which can be used for
    both regression and classification."""

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
        """Sample num_samples points from the task's distribution.
        Returns a tuple (x, y, loss_fct), where 
        x is the input tensor,
        y is the label tensor,
        loss_fct is a loss function."""
        raise NotImplementedError

    def fn(self, x: torch.Tensor) -> torch.Tensor:
        """The function to be learned (for regression)."""
        raise NotImplementedError


class Sampler:
    """A sampler for tasks."""

    @property
    def task_name(self) -> str:
        """The name of the task."""
        raise NotImplementedError

    def sample_task(self) -> Task:
        """Sample a task from the sampler's distribution."""
        raise NotImplementedError


class SinusoidTask(Task):
    """A sinusoid task."""
    _a: float
    _b: float

    def __init__(self, a: float, b: float):
        self._a = a
        self._b = b

    @torch.no_grad()
    def fn(self, x: torch.Tensor) -> torch.Tensor:
        return self._a * torch.sin(x + self._b)

    @torch.no_grad()
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
        x = torch.rand((num_samples, 1)) * 10 - 5  # Sample x in [-5, 5]
        y = self.fn(x)  # calculate f(x)
        loss_fct = torch.nn.MSELoss()  # use MSE loss
        return x, y, loss_fct

    
class SinusoidSampler(Sampler):
    """A sampler for sinusoid tasks."""
    _amplitude_range: Tuple[float, float]
    _phase_range: Tuple[float, float]

    def __init__(self, amplitude_range: Tuple[float, float] = (0.1, 5.0), phase_range: Tuple[float, float] = (0., torch.pi)):
        self._amplitude_range = amplitude_range
        self._phase_range = phase_range

    @property
    def task_name(self) -> str:
        return f"SinusoidTask"

    @torch.no_grad()
    def sample_task(self) -> SinusoidTask:
        a = torch.rand(1).item() * (self._amplitude_range[1] - self._amplitude_range[0]) + self._amplitude_range[0]
        b = torch.rand(1).item() * (self._phase_range[1] - self._phase_range[0]) + self._phase_range[0]
        return SinusoidTask(a, b)


class Poly5Task(Task):

    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float):
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._e = e
        self._f = f

    @property
    def task_name(self) -> str:
        return f"Poly5Task"

    @torch.no_grad()
    def fn(self, x: torch.Tensor) -> torch.Tensor:
        return self._a * x**5 + self._b * x**4 + self._c * x**3 + self._d * x**2 + self._e * x + self._f

    @torch.no_grad()
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
        x = torch.rand((num_samples, 1)) * 10 - 5  # Sample x in [-5, 5]
        y = self.fn(x)  # calculate f(x)
        loss_fct = torch.nn.MSELoss()  # use MSE loss
        return x, y, loss_fct

    
class Poly5Sampler(Sampler):
    """A sampler for polynomial of degree 5 tasks."""
    _range: Tuple[float, float]

    def __init__(self, range: Tuple[float, float] = (-5.0, 5.0)):
        self._range = range

    @torch.no_grad()
    def sample_task(self) -> Poly5Task:
        a = torch.rand(1).item() * (self._range[1] - self._range[0]) + self._range[0]
        b = torch.rand(1).item() * (self._range[1] - self._range[0]) + self._range[0]
        c = torch.rand(1).item() * (self._range[1] - self._range[0]) + self._range[0]
        d = torch.rand(1).item() * (self._range[1] - self._range[0]) + self._range[0]
        e = torch.rand(1).item() * (self._range[1] - self._range[0]) + self._range[0]
        f = torch.rand(1).item() * (self._range[1] - self._range[0]) + self._range[0]
        return Poly5Task(a, b, c, d, e, f)