"""Shared command-line options for the Monte Carlo scripts: Jacobian, precision, isochromats.

The defaults (forward differences, float64, 256 isochromats, 40 iterations) reproduce the
published results exactly. Any other choice gets an output-file suffix so the reference
results are never overwritten.
"""

from __future__ import annotations

import argparse

import torch


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("fit settings")
    g.add_argument("--jacobian", choices=("fd", "implicit"), default="fd",
                   help="forward differences (default, reproduces the reference) or the exact "
                        "implicit Jacobian (10-12x faster)")
    g.add_argument("--float32", action="store_true",
                   help="single precision (requires --jacobian implicit)")
    g.add_argument("--n-iso", type=int, default=256, help="isochromats per voxel (default 256)")
    g.add_argument("--n-iter", type=int, default=40, help="LM iterations (default 40)")


def fit_kwargs(args: argparse.Namespace) -> dict:
    if args.float32 and args.jacobian != "implicit":
        raise SystemExit("--float32 requires --jacobian implicit")
    kw = dict(jacobian=args.jacobian, n_iso=args.n_iso, n_iter=args.n_iter)
    if args.float32:
        kw["dtype"] = torch.float32
    return kw


def suffix(args: argparse.Namespace) -> str:
    """'' for the reference settings, otherwise e.g. '_implicit_f32_iso128_it15'."""
    parts = []
    if args.jacobian != "fd":
        parts.append(args.jacobian)
    if args.float32:
        parts.append("f32")
    if args.n_iso != 256:
        parts.append(f"iso{args.n_iso}")
    if args.n_iter != 40:
        parts.append(f"it{args.n_iter}")
    return "".join("_" + p for p in parts)


def cost_tolerance(args: argparse.Namespace) -> float:
    """Relative tolerance for 'final cost exceeds the cost at the true parameters'."""
    return 1e-5 if args.float32 else 1e-9
