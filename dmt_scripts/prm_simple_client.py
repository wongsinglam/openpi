from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    """Simple client to exercise pi05 + PRM server with synthetic Dobot observations."""

    host: str = "127.0.0.1"
    port: int = 17777
    prompt: str = "tidy the desk"
    num_steps: int = 20

    # PRM-related hyperparameters (mirrors PrmClientConfig).
    prm_num_action_candidates: int = 3
    prm_expand_candidate_actions: int = 0
    prm_gaussian_noise_scale: float = 0.10
    prm_use_direction_guidance: bool = False
    prm_direction_cone_deg: float = 90.0


def _random_dobot_observation(args: Args) -> dict[str, Any]:
    """Construct a random Dobot-style observation for pi05_dobot_* policies."""
    state = np.random.uniform(-1.0, 1.0, size=(14,)).astype(np.float32)

    def _rand_image() -> np.ndarray:
        return np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

    obs: dict[str, Any] = {
        "observation/state": state,
        "observation/top_image": _rand_image(),
        "observation/left_wrist_image": _rand_image(),
        "observation/right_wrist_image": _rand_image(),
        "prompt": args.prompt,
        "prm_config": {
            "num_action_candidates": int(args.prm_num_action_candidates),
            "expand_candidate_actions": int(args.prm_expand_candidate_actions),
            "gaussian_noise_scale": float(args.prm_gaussian_noise_scale),
            "use_direction_guidance": bool(args.prm_use_direction_guidance),
            "direction_cone_deg": float(args.prm_direction_cone_deg),
        },
    }
    return obs


def main(args: Args) -> None:
    client = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info("Server metadata: %s", client.get_server_metadata())

    server_infer_ms: list[float] = []
    policy_infer_ms: list[float] = []

    for step in range(args.num_steps):
        obs = _random_dobot_observation(args)
        result = client.infer(obs)
        actions = np.asarray(result["actions"])
        server_timing = result["server_timing"]
        policy_timing = result["policy_timing"]

        if isinstance(server_timing, dict) and "infer_ms" in server_timing:
            server_infer_ms.append(float(server_timing["infer_ms"]))
        if isinstance(policy_timing, dict) and "infer_ms" in policy_timing:
            policy_infer_ms.append(float(policy_timing["infer_ms"]))

        # logging.info(
        #     "Step %d: actions_shape=%s server_timing=%s policy_timing=%s",
        #     step,
        #     tuple(actions.shape) if actions is not None else None,
        #     server_timing,
        #     policy_timing,
        # )

    def _summarize(label: str, values: list[float]) -> None:
        if not values:
            logging.info("%s: no timing data collected.", label)
            return
        arr = np.asarray(values, dtype=np.float64)
        logging.info(
            "%s infer_ms: mean=%.2f std=%.2f p50=%.2f p95=%.2f min=%.2f max=%.2f (n=%d)",
            label,
            float(arr.mean()),
            float(arr.std()),
            float(np.quantile(arr, 0.5)),
            float(np.quantile(arr, 0.95)),
            float(arr.min()),
            float(arr.max()),
            arr.size,
        )

    _summarize("server_timing", server_infer_ms)
    _summarize("policy_timing", policy_infer_ms)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
