"""
Toy sinusoid experiment for task-coordinate-conditioned meta-initialization.

The task family is:

    y = amplitude * sin(x + phase)

In this toy setting the true task coordinate is known:

    z = [amplitude, phase]

The script compares:

1. maml
   A single learned initialization theta_0 for every task.

2. hyper
   A continuous parameter field theta(z) = theta_base + h(z).

3. anchor
   A smooth piecewise/local parameter field
   theta(z) = theta_base + sum_k w_k(z) * Delta_k,
   where w_k are RBF-like weights around fixed task-space anchors.

This is deliberately standalone so the image-classification MAML code stays
untouched while the interpolation idea is tested in a controlled regression
setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ParamDict = OrderedDict[str, torch.Tensor]


@dataclass
class Range2D:
    amp_min: float
    amp_max: float
    phase_min: float
    phase_max: float

    @property
    def amp_mid(self) -> float:
        return 0.5 * (self.amp_min + self.amp_max)

    @property
    def phase_mid(self) -> float:
        return 0.5 * (self.phase_min + self.phase_max)

    @property
    def amp_half_width(self) -> float:
        return 0.5 * (self.amp_max - self.amp_min)

    @property
    def phase_half_width(self) -> float:
        return 0.5 * (self.phase_max - self.phase_min)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        center = z.new_tensor([self.amp_mid, self.phase_mid])
        scale = z.new_tensor([
            max(self.amp_half_width, 1e-8),
            max(self.phase_half_width, 1e-8),
        ])
        return (z - center) / scale

    def distance_to_box(self, z: torch.Tensor) -> torch.Tensor:
        """Distance from z to this range, measured in normalized coordinates."""
        z_norm = self.normalize(z)
        overflow = (z_norm.abs() - 1.0).clamp_min(0.0)
        return torch.linalg.vector_norm(overflow, dim=-1)


PARAM_SHAPES = OrderedDict(
    [
        ("w1", (40, 1)),
        ("b1", (40,)),
        ("w2", (40, 40)),
        ("b2", (40,)),
        ("w3", (1, 40)),
        ("b3", (1,)),
    ]
)


def set_hidden_dim(hidden_dim: int) -> None:
    PARAM_SHAPES.clear()
    PARAM_SHAPES.update(
        [
            ("w1", (hidden_dim, 1)),
            ("b1", (hidden_dim,)),
            ("w2", (hidden_dim, hidden_dim)),
            ("b2", (hidden_dim,)),
            ("w3", (1, hidden_dim)),
            ("b3", (1,)),
        ]
    )


def num_model_params() -> int:
    return sum(math.prod(shape) for shape in PARAM_SHAPES.values())


def init_weight(shape: Tuple[int, ...]) -> torch.Tensor:
    tensor = torch.empty(*shape)
    if len(shape) >= 2:
        nn.init.xavier_uniform_(tensor)
    else:
        nn.init.zeros_(tensor)
    return tensor


def flatten_params(params: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([params[name].reshape(-1) for name in PARAM_SHAPES.keys()])


def unflatten_params(vector: torch.Tensor) -> ParamDict:
    params: ParamDict = OrderedDict()
    offset = 0
    for name, shape in PARAM_SHAPES.items():
        n = math.prod(shape)
        params[name] = vector[..., offset : offset + n].reshape(*vector.shape[:-1], *shape)
        offset += n
    return params


def mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def population_std(values: List[float]) -> float:
    if len(values) == 0:
        return 0.0
    m = mean(values)
    return math.sqrt(mean([(value - m) ** 2 for value in values]))


def batched_unflatten_params(vectors: torch.Tensor) -> List[ParamDict]:
    return [
        OrderedDict((name, value[i]) for name, value in unflatten_params(vectors).items())
        for i in range(vectors.shape[0])
    ]


def mlp_forward(x: torch.Tensor, params: ParamDict) -> torch.Tensor:
    x = F.linear(x, params["w1"], params["b1"])
    x = torch.relu(x)
    x = F.linear(x, params["w2"], params["b2"])
    x = torch.relu(x)
    x = F.linear(x, params["w3"], params["b3"])
    return x


def clone_for_adaptation(params: ParamDict) -> ParamDict:
    return OrderedDict(
        (name, value.detach().clone().requires_grad_(True))
        for name, value in params.items()
    )


def inner_adapt(
    params: ParamDict,
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    first_order: bool,
) -> ParamDict:
    adapted = OrderedDict((name, value) for name, value in params.items())
    for _ in range(inner_steps):
        loss = F.mse_loss(mlp_forward(x_support, adapted), y_support)
        grads = torch.autograd.grad(
            loss,
            tuple(adapted.values()),
            create_graph=not first_order,
        )
        adapted = OrderedDict(
            (name, value - inner_lr * grad)
            for (name, value), grad in zip(adapted.items(), grads)
        )
    return adapted


class SinusoidTaskSampler:
    def __init__(
        self,
        train_range: Range2D,
        x_min: float,
        x_max: float,
        device: torch.device,
    ) -> None:
        self.train_range = train_range
        self.x_min = x_min
        self.x_max = x_max
        self.device = device

    def sample_z(self, batch_size: int, task_range: Range2D | None = None) -> torch.Tensor:
        task_range = task_range or self.train_range
        amp = torch.empty(batch_size, 1, device=self.device).uniform_(
            task_range.amp_min, task_range.amp_max
        )
        phase = torch.empty(batch_size, 1, device=self.device).uniform_(
            task_range.phase_min, task_range.phase_max
        )
        return torch.cat([amp, phase], dim=-1)

    def sample_xy(self, z: torch.Tensor, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.shape[0]
        x = torch.empty(batch_size, n_points, 1, device=self.device).uniform_(
            self.x_min, self.x_max
        )
        amp = z[:, 0].view(batch_size, 1, 1)
        phase = z[:, 1].view(batch_size, 1, 1)
        y = amp * torch.sin(x + phase)
        return x, y

    def sample_batch(
        self,
        batch_size: int,
        n_support: int,
        n_query: int,
        task_range: Range2D | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.sample_z(batch_size, task_range)
        x_support, y_support = self.sample_xy(z, n_support)
        x_query, y_query = self.sample_xy(z, n_query)
        return z, x_support, y_support, x_query, y_query


class BaseInitializer(nn.Module):
    name = "base"

    def initial_params(self, z: torch.Tensor) -> List[ParamDict]:
        raise NotImplementedError

    def initial_params_for_eval(self, z: torch.Tensor) -> List[ParamDict]:
        with torch.no_grad():
            params = self.initial_params(z)
        return [clone_for_adaptation(p) for p in params]


class MAMLInitializer(BaseInitializer):
    name = "maml"

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.ParameterDict(
            {name: nn.Parameter(init_weight(shape)) for name, shape in PARAM_SHAPES.items()}
        )

    def base_params(self) -> ParamDict:
        return OrderedDict((name, self.base[name]) for name in PARAM_SHAPES.keys())

    def initial_params(self, z: torch.Tensor) -> List[ParamDict]:
        base = self.base_params()
        return [base for _ in range(z.shape[0])]


class HyperInitializer(BaseInitializer):
    name = "hyper"

    def __init__(
        self,
        train_range: Range2D,
        hidden_dim: int,
        delta_scale: float,
    ) -> None:
        super().__init__()
        self.train_range = train_range
        self.delta_scale = delta_scale
        self.base = nn.ParameterDict(
            {name: nn.Parameter(init_weight(shape)) for name, shape in PARAM_SHAPES.items()}
        )
        out_dim = num_model_params()
        self.hyper = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

    def base_vector(self) -> torch.Tensor:
        return flatten_params(OrderedDict((name, self.base[name]) for name in PARAM_SHAPES.keys()))

    def initial_params(self, z: torch.Tensor) -> List[ParamDict]:
        z_norm = self.train_range.normalize(z)
        delta = self.hyper(z_norm)
        vectors = self.base_vector().unsqueeze(0) + self.delta_scale * delta
        return batched_unflatten_params(vectors)


class AnchorInitializer(BaseInitializer):
    name = "anchor"

    def __init__(
        self,
        train_range: Range2D,
        anchor_grid: int,
        temperature: float,
        delta_scale: float,
    ) -> None:
        super().__init__()
        if anchor_grid < 2:
            raise ValueError("anchor_grid must be >= 2")
        self.train_range = train_range
        self.temperature = temperature
        self.delta_scale = delta_scale
        self.base = nn.ParameterDict(
            {name: nn.Parameter(init_weight(shape)) for name, shape in PARAM_SHAPES.items()}
        )
        anchors = self._make_anchors(anchor_grid)
        self.register_buffer("anchors", anchors)
        self.anchor_deltas = nn.Parameter(torch.zeros(anchors.shape[0], num_model_params()))
        nn.init.normal_(self.anchor_deltas, mean=0.0, std=1e-3)

    def _make_anchors(self, anchor_grid: int) -> torch.Tensor:
        amp = torch.linspace(-1.0, 1.0, anchor_grid)
        phase = torch.linspace(-1.0, 1.0, anchor_grid)
        aa, pp = torch.meshgrid(amp, phase, indexing="ij")
        return torch.stack([aa.reshape(-1), pp.reshape(-1)], dim=-1)

    def base_vector(self) -> torch.Tensor:
        return flatten_params(OrderedDict((name, self.base[name]) for name in PARAM_SHAPES.keys()))

    def anchor_weights(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = self.train_range.normalize(z)
        dist_sq = ((z_norm[:, None, :] - self.anchors[None, :, :]) ** 2).sum(dim=-1)
        return torch.softmax(-dist_sq / max(self.temperature, 1e-8), dim=-1)

    def initial_params(self, z: torch.Tensor) -> List[ParamDict]:
        weights = self.anchor_weights(z)
        delta = weights @ self.anchor_deltas
        vectors = self.base_vector().unsqueeze(0) + self.delta_scale * delta
        return batched_unflatten_params(vectors)


def meta_objective(
    learner: BaseInitializer,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    inner_lr: float,
    inner_steps: int,
    first_order: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    z, x_support, y_support, x_query, y_query = batch
    init_params = learner.initial_params(z)
    pre_losses = []
    post_losses = []
    for task_id, params in enumerate(init_params):
        pre_loss = F.mse_loss(mlp_forward(x_query[task_id], params), y_query[task_id])
        adapted = inner_adapt(
            params,
            x_support[task_id],
            y_support[task_id],
            inner_lr,
            inner_steps,
            first_order,
        )
        post_loss = F.mse_loss(mlp_forward(x_query[task_id], adapted), y_query[task_id])
        pre_losses.append(pre_loss)
        post_losses.append(post_loss)

    pre = torch.stack(pre_losses).mean()
    post = torch.stack(post_losses).mean()
    return post, {"pre_mse": float(pre.detach().cpu()), "post_mse": float(post.detach().cpu())}


def train_one_model(
    model_name: str,
    learner: BaseInitializer,
    sampler: SinusoidTaskSampler,
    args: argparse.Namespace,
    output_dir: str,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.outer_lr)
    history_path = os.path.join(output_dir, f"{model_name}_train.csv")
    best_loss = float("inf")
    start_time = time.time()

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "meta_loss", "pre_mse", "post_mse", "elapsed_sec"],
        )
        writer.writeheader()

        for step in range(1, args.meta_steps + 1):
            batch = sampler.sample_batch(
                args.meta_batch_size,
                args.n_support,
                args.n_query,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = meta_objective(
                learner,
                batch,
                args.inner_lr,
                args.inner_steps,
                args.first_order,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), args.grad_clip)
            optimizer.step()
            best_loss = min(best_loss, metrics["post_mse"])

            if step == 1 or step % args.log_every == 0 or step == args.meta_steps:
                row = {
                    "step": step,
                    "meta_loss": metrics["post_mse"],
                    "pre_mse": metrics["pre_mse"],
                    "post_mse": metrics["post_mse"],
                    "elapsed_sec": time.time() - start_time,
                }
                writer.writerow(row)
                print(
                    f"[{model_name}] step {step:05d} "
                    f"pre={metrics['pre_mse']:.5f} "
                    f"post={metrics['post_mse']:.5f}"
                )

    return {"best_train_post_mse": best_loss, "train_time_sec": time.time() - start_time}


def evaluate_random_split(
    learner: BaseInitializer,
    sampler: SinusoidTaskSampler,
    task_range: Range2D,
    args: argparse.Namespace,
) -> Dict[str, float]:
    pre_losses = []
    post_losses = []
    for _ in range(args.eval_batches):
        z, x_support, y_support, x_query, y_query = sampler.sample_batch(
            args.eval_batch_size,
            args.n_support,
            args.n_query,
            task_range,
        )
        init_params = learner.initial_params_for_eval(z)
        for task_id, params in enumerate(init_params):
            with torch.no_grad():
                pre_loss = F.mse_loss(mlp_forward(x_query[task_id], params), y_query[task_id])
            adapted = inner_adapt(
                params,
                x_support[task_id],
                y_support[task_id],
                args.inner_lr,
                args.inner_steps,
                first_order=True,
            )
            with torch.no_grad():
                post_loss = F.mse_loss(mlp_forward(x_query[task_id], adapted), y_query[task_id])
            pre_losses.append(float(pre_loss.cpu()))
            post_losses.append(float(post_loss.cpu()))

    return {
        "pre_mse": mean(pre_losses),
        "post_mse": mean(post_losses),
        "post_mse_std": population_std(post_losses),
    }


def evaluate_grid(
    model_name: str,
    learner: BaseInitializer,
    sampler: SinusoidTaskSampler,
    eval_range: Range2D,
    args: argparse.Namespace,
    output_dir: str,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    amp_values = torch.linspace(
        eval_range.amp_min, eval_range.amp_max, args.grid_size, device=sampler.device
    )
    phase_values = torch.linspace(
        eval_range.phase_min, eval_range.phase_max, args.grid_size, device=sampler.device
    )

    for amp in amp_values:
        for phase in phase_values:
            z = torch.stack([amp, phase]).view(1, 2)
            pre_rep = []
            post_rep = []
            for _ in range(args.grid_repeats):
                x_support, y_support = sampler.sample_xy(z, args.n_support)
                x_query, y_query = sampler.sample_xy(z, args.n_query)
                params = learner.initial_params_for_eval(z)[0]
                with torch.no_grad():
                    pre_loss = F.mse_loss(mlp_forward(x_query[0], params), y_query[0])
                adapted = inner_adapt(
                    params,
                    x_support[0],
                    y_support[0],
                    args.inner_lr,
                    args.inner_steps,
                    first_order=True,
                )
                with torch.no_grad():
                    post_loss = F.mse_loss(mlp_forward(x_query[0], adapted), y_query[0])
                pre_rep.append(float(pre_loss.cpu()))
                post_rep.append(float(post_loss.cpu()))

            distance_to_train = sampler.train_range.distance_to_box(z).item()
            rows.append(
                {
                    "model": model_name,
                    "amplitude": float(amp.cpu()),
                    "phase": float(phase.cpu()),
                    "distance_to_train_box": float(distance_to_train),
                    "pre_mse": mean(pre_rep),
                    "post_mse": mean(post_rep),
                }
            )

    path = os.path.join(output_dir, f"{model_name}_grid.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "amplitude",
                "phase",
                "distance_to_train_box",
                "pre_mse",
                "post_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    save_heatmap(rows, model_name, args.grid_size, output_dir)
    return rows


def save_heatmap(
    rows: List[Dict[str, float]],
    model_name: str,
    grid_size: int,
    output_dir: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[{model_name}] heatmap skipped: matplotlib unavailable ({exc})")
        return

    flat_post = [row["post_mse"] for row in rows]
    post = [
        flat_post[row_start : row_start + grid_size]
        for row_start in range(0, len(flat_post), grid_size)
    ]
    post_t = [list(row) for row in zip(*post)]
    amps = sorted({row["amplitude"] for row in rows})
    phases = sorted({row["phase"] for row in rows})

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=140)
    image = ax.imshow(
        post_t,
        origin="lower",
        aspect="auto",
        extent=[min(amps), max(amps), min(phases), max(phases)],
        cmap="viridis",
    )
    ax.set_title(f"{model_name}: post-adaptation MSE")
    ax.set_xlabel("amplitude")
    ax.set_ylabel("phase")
    fig.colorbar(image, ax=ax, label="MSE")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{model_name}_post_mse_heatmap.png"))
    plt.close(fig)


def make_output_dir(args: argparse.Namespace) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_part = args.model.replace(",", "-")
    output_dir = os.path.join(args.output_dir, f"{stamp}_{model_part}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def parse_range(
    amp_min: float,
    amp_max: float,
    phase_min: float,
    phase_max: float,
) -> Range2D:
    return Range2D(amp_min, amp_max, phase_min, phase_max)


def make_learner(
    model_name: str,
    train_range: Range2D,
    args: argparse.Namespace,
    device: torch.device,
) -> BaseInitializer:
    if model_name == "maml":
        learner: BaseInitializer = MAMLInitializer()
    elif model_name == "hyper":
        learner = HyperInitializer(
            train_range,
            hidden_dim=args.hyper_hidden_dim,
            delta_scale=args.delta_scale,
        )
    elif model_name == "anchor":
        learner = AnchorInitializer(
            train_range,
            anchor_grid=args.anchor_grid,
            temperature=args.anchor_temperature,
            delta_scale=args.delta_scale,
        )
    else:
        raise ValueError(f"unknown model: {model_name}")
    return learner.to(device)


def selected_models(model_arg: str) -> List[str]:
    if model_arg == "all":
        return ["maml", "hyper", "anchor"]
    return [part.strip() for part in model_arg.split(",") if part.strip()]


def save_summary(
    output_dir: str,
    args: argparse.Namespace,
    train_range: Range2D,
    split_results: Dict[str, Dict[str, Dict[str, float]]],
    train_results: Dict[str, Dict[str, float]],
) -> None:
    summary = {
        "args": vars(args),
        "train_range": asdict(train_range),
        "train_results": train_results,
        "split_results": split_results,
    }
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    table_path = os.path.join(output_dir, "summary.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "split",
                "pre_mse",
                "post_mse",
                "post_mse_std",
                "best_train_post_mse",
                "train_time_sec",
            ],
        )
        writer.writeheader()
        for model_name, splits in split_results.items():
            for split_name, metrics in splits.items():
                row = {
                    "model": model_name,
                    "split": split_name,
                    **metrics,
                    **train_results[model_name],
                }
                writer.writerow(row)


def main(args: argparse.Namespace) -> None:
    if args.hidden_dim != 40:
        set_hidden_dim(args.hidden_dim)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_range = parse_range(
        args.train_amp_min,
        args.train_amp_max,
        args.train_phase_min,
        args.train_phase_max,
    )
    interp_range = train_range
    amp_extra_range = parse_range(
        args.train_amp_max,
        args.extra_amp_max,
        args.train_phase_min,
        args.train_phase_max,
    )
    phase_extra_range = parse_range(
        args.train_amp_min,
        args.train_amp_max,
        args.train_phase_max,
        args.extra_phase_max,
    )
    grid_range = parse_range(
        args.grid_amp_min,
        args.grid_amp_max,
        args.grid_phase_min,
        args.grid_phase_max,
    )

    output_dir = make_output_dir(args)
    sampler = SinusoidTaskSampler(
        train_range=train_range,
        x_min=args.x_min,
        x_max=args.x_max,
        device=device,
    )

    print(f"device: {device}")
    print(f"output: {output_dir}")
    print(f"model params: {num_model_params()}")
    print(f"train task range: {train_range}")

    train_results: Dict[str, Dict[str, float]] = {}
    split_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_name in selected_models(args.model):
        print(f"\n=== training {model_name} ===")
        learner = make_learner(model_name, train_range, args, device)
        train_results[model_name] = train_one_model(
            model_name,
            learner,
            sampler,
            args,
            output_dir,
        )

        print(f"=== evaluating {model_name} ===")
        split_results[model_name] = {
            "interpolation": evaluate_random_split(learner, sampler, interp_range, args),
            "extra_amplitude": evaluate_random_split(learner, sampler, amp_extra_range, args),
            "extra_phase": evaluate_random_split(learner, sampler, phase_extra_range, args),
        }
        for split_name, metrics in split_results[model_name].items():
            print(
                f"[{model_name}] {split_name}: "
                f"pre={metrics['pre_mse']:.5f} "
                f"post={metrics['post_mse']:.5f} "
                f"std={metrics['post_mse_std']:.5f}"
            )
        evaluate_grid(model_name, learner, sampler, grid_range, args, output_dir)

    save_summary(output_dir, args, train_range, split_results, train_results)
    print(f"\nDone. Results written to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sinusoid toy experiment for theta(z) meta-initialization."
    )
    parser.add_argument("--model", default="all", help="maml, hyper, anchor, or comma list")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/sinusoid_theta_field")

    parser.add_argument("--meta-steps", type=int, default=2000)
    parser.add_argument("--meta-batch-size", type=int, default=16)
    parser.add_argument("--n-support", type=int, default=10)
    parser.add_argument("--n-query", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--first-order", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--hidden-dim", type=int, default=40)
    parser.add_argument("--hyper-hidden-dim", type=int, default=128)
    parser.add_argument("--delta-scale", type=float, default=0.1)
    parser.add_argument("--anchor-grid", type=int, default=3)
    parser.add_argument("--anchor-temperature", type=float, default=0.35)

    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--train-amp-min", type=float, default=0.5)
    parser.add_argument("--train-amp-max", type=float, default=4.0)
    parser.add_argument("--train-phase-min", type=float, default=0.0)
    parser.add_argument("--train-phase-max", type=float, default=math.pi)
    parser.add_argument("--extra-amp-max", type=float, default=6.0)
    parser.add_argument("--extra-phase-max", type=float, default=1.5 * math.pi)

    parser.add_argument("--eval-batches", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--grid-repeats", type=int, default=3)
    parser.add_argument("--grid-amp-min", type=float, default=0.5)
    parser.add_argument("--grid-amp-max", type=float, default=6.0)
    parser.add_argument("--grid-phase-min", type=float, default=0.0)
    parser.add_argument("--grid-phase-max", type=float, default=1.5 * math.pi)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
