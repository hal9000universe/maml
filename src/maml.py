import higher
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import copy

from tasks import Task, Sampler, SinusoidSampler


# define task
sampler: Sampler = SinusoidSampler()


def sample_task() -> Task:
    """Sample a task from the sampler's distribution."""
    return sampler.sample_task()


def maml(model: nn.Module, meta_opt: optim.Optimizer, inner_steps: int, n_iters: int, num_task_samples: int, learning_rate: float, num_tasks: int = 10, num_meta_task_samples: int = 25, device: str = 'cpu'):
    """
    Algorithm from https://arxiv.org/pdf/1703.03400v3.pdf (MAML for Few-Shot Supervised Learning)
    """
    for epoch in tqdm(range(n_iters)): 

        # Meta update
        meta_opt.zero_grad()  # reset meta gradients
        for i in range(num_tasks):
            opt = torch.optim.SGD(model.parameters(), lr=learning_rate)  # initialize the inner optimizer
            # Inner update
            with higher.innerloop_ctx(model, opt, copy_initial_weights=True) as (f_model, diffopt):
                task = sample_task()  # sample a task ti ~ p(T)

                for step in range(inner_steps):
                    x_batch, target, loss_fct = task.sample(num_task_samples)  # sample K examples from ti
                    loss = loss_fct(model(x_batch.to(device)), target.to(device))  # compute loss on these K examples
                    diffopt.step(loss)  # update the model weights to adadted weights \theta_i_prime

                x_meta, y_meta, loss_fct = task.sample(num_meta_task_samples)  # sample examples from ti for meta update

                # Compute \nabla_theta L(f_theta_i_prime) for task ti
                loss = loss_fct(model(x_meta.to(device)), y_meta.to(device))
                loss.backward()

        meta_opt.step()  # meta step - update the initial weights \theta 

    return model


def fine_tune(model: nn.Module, task: Task, num_steps: int, num_task_samples: int, learning_rate: float, device: str = 'cpu') -> nn.Module:
    """
    Fine-tune the model on the task
    """
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)  # initialize the inner optimizer
    for epoch in range(num_steps):
        x_batch, target, loss_fct = task.sample(num_task_samples)  # sample K examples from ti
        loss = loss_fct(model(x_batch.to(device)), target.to(device))  # compute loss on these K examples
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


if __name__ == "__main__":
    # general settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    load: bool = True
    training: bool = False
    save_fig: bool = False

    # MAML settings
    inner_steps: int = 1
    n_iters: int = 70000
    num_task_samples: int = 10
    meta_learning_rate: float = 1e-3
    learning_rate: float = 2e-2
    num_tasks: int = 10
    num_meta_task_samples: int = 25

    # Fine-tuning settings
    ft_num_steps: int = 50
    ft_num_task_samples: int = 10
    ft_learning_rate: float = 2e-2

    # Model
    if load:
        model: nn.Module = torch.load(f'maml - {sampler.task_name}.pt')
    else:
        model: nn.Module = nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU(), nn.Linear(40, 1)).to(device)

    # Optimizer
    meta_optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=meta_learning_rate)

    # Training
    if training:
        model = maml(
            model=model, 
            meta_opt=meta_optimizer,
            inner_steps=inner_steps,
            n_iters=n_iters,
            num_task_samples=num_task_samples,
            learning_rate=learning_rate,
            num_tasks=num_tasks,
            num_meta_task_samples=num_meta_task_samples,
            device=device
        )
        torch.save(model, f'maml - {sampler.task_name}.pt')

    # Plotting
    plt.title(f'MAML - {sampler.task_name}, K={num_task_samples}')
    # Pre-update
    x = torch.linspace(-5, 5, 50).to(device).view(-1, 1)  # x in [-5, 5]
    y = model(x).detach().view(-1)  # calc y = f(x), detach to allow conversion to numpy
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='lightgreen', linestyle='--', linewidth=2.2, label='pre-update')
    # New task
    task = sample_task()
    ground_truth_y = task.fn(x)
    plt.plot(x.data.cpu().numpy(), ground_truth_y.data.cpu().numpy(), c='red', label='ground truth')
    # Fine-tuning, ft_num_steps gradient steps
    model = fine_tune(
        model=model, 
        task=task,
        num_steps=ft_num_steps,
        num_task_samples=ft_num_task_samples,
        learning_rate=ft_learning_rate,
        device=device
    )
    # After Fine-tuning
    y = model(x).detach().view(-1)  # calc y = f(x), detach to allow conversion to numpy
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='darkgreen', linestyle='--', 
    linewidth=2.2, label=f'{ft_num_steps} grad step')
    plt.legend()
    if save_fig:
        plt.savefig(f'maml - {sampler.task_name}.png')
    plt.show()