from pathlib import Path

import torch
import torch.nn as nn


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def extract_model_state(checkpoint, state_key="model_state_dict"):
    if isinstance(checkpoint, dict):
        if state_key in checkpoint:
            return checkpoint[state_key]
        if state_key != "model_state_dict" and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
    return checkpoint


def load_checkpoint_object(checkpoint_or_path, map_location=None):
    if isinstance(checkpoint_or_path, (str, Path)):
        return torch.load(checkpoint_or_path, map_location=map_location)
    return checkpoint_or_path


def load_model_state(model, checkpoint_or_path, map_location=None, state_key="model_state_dict"):
    checkpoint = load_checkpoint_object(checkpoint_or_path, map_location=map_location)
    state_dict = extract_model_state(checkpoint, state_key=state_key)

    try:
        unwrap_model(model).load_state_dict(state_dict)
        return checkpoint
    except RuntimeError:
        pass

    if any(key.startswith("module.") for key in state_dict.keys()):
        stripped_state_dict = {
            key[len("module."):]: value for key, value in state_dict.items()
        }
        unwrap_model(model).load_state_dict(stripped_state_dict)
        return checkpoint

    if isinstance(model, nn.DataParallel):
        wrapped_state_dict = {
            f"module.{key}": value for key, value in state_dict.items()
        }
        model.load_state_dict(wrapped_state_dict)
        return checkpoint

    if (
        isinstance(checkpoint, dict)
        and state_key == "model_state_dict"
        and "backbone_state_dict" in checkpoint
    ):
        return load_model_state(
            model,
            checkpoint,
            map_location=map_location,
            state_key="backbone_state_dict",
        )

    raise RuntimeError("Could not load checkpoint into model state")


def build_checkpoint_payload(
    model,
    optimizer=None,
    epoch=None,
    best_metric=None,
    args_dict=None,
    extra=None,
):
    payload = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if args_dict is not None:
        payload["args"] = args_dict
    if extra:
        payload.update(extra)
    return payload
