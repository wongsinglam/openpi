import dataclasses
import enum
import logging
import math
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_pi05_prm script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # ===== PRM / test-time scaling 配置 =====
    # 路径指向 robotprm 训练脚本保存的 ckpt（dobot_prm_epoch_*.pt）
    prm: Optional[str] = None
    # PRM 模型所在设备（例如 "cuda", "cuda:0", "cpu"）
    prm_device: str = "cuda"
    # 初始候选动作数量（包含 base action）
    num_action_candidates: int = 1
    # 额外扩展候选动作数量
    expand_candidate_actions: int = 0
    # 高斯噪声尺度（用于在 base action 附近采样）
    gaussian_noise_scale: float = 0.05
    # 是否启用方向引导采样（使用 PRM 的方向头）
    use_direction_guidance: bool = False
    # 方向引导时的锥形采样角度（度）。默认 90 度，相当于保留半球约束。
    direction_cone_deg: float = 90.0


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


class PRMScaledPolicy:
    """Wrap an openpi policy and apply PRM-based test-time scaling on server side.

    - 接收来自客户端的 obs（与原 WebsocketPolicyServer 相同）；
    - 先调用 base_policy.infer(obs) 得到 π0.5 的动作块；
    - 若未配置 PRM 或候选数量<=1，则直接返回原始结果；
    - 否则在第一个动作附近生成若干候选动作，通过 PRM 评分并选择得分最高者，
      用于替换动作块中的第一个动作（其余保持不变）。
    """

    def __init__(self, base_policy: "_policy.Policy", args: Args):
        self.base_policy = base_policy
        self.args = args
        self._prm = None
        if args.prm:
            self._init_prm()

    # ---- PRM 初始化 ----
    def _init_prm(self) -> None:
        """延迟导入 robotprm 与 GR1RewardModel 并加载 ckpt。"""
        import torch
        import clip  # type: ignore

        project_root = Path(__file__).resolve().parents[3]
        robotprm_root = project_root / "robotprm"
        src_root = robotprm_root / "src"
        for p in (robotprm_root, src_root):
            p_str = str(p)
            if p_str not in __import__("sys").path:
                __import__("sys").path.insert(0, p_str)

        from models.gr1_reward_model import GR1RewardModel  # type: ignore

        device = torch.device(self.args.prm_device if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(self.args.prm).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"PRM checkpoint not found: {ckpt_path}")

        # 与 robotprm/scripts/train_direction_real_data.py 中保持一致的结构
        model = GR1RewardModel(
            state_dim=7,
            act_dim=14,
            hidden_size=384,
            sequence_length=10,
            img_feat_dim=768,
            patch_feat_dim=768,
            lang_feat_dim=512,
            clip_backbone="ViT-B/32",
            mae_ckpt=None,
            pretrained_gr1_path=None,
            use_hand_rgb=True,
            direction_dim=14,
            device=str(device),
            rank=0,
        ).to(device)

        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self._prm = {
            "model": model,
            "device": device,
            "clip": clip,
        }
        logging.info("[PRM] Loaded checkpoint from %s", ckpt_path)

    # ---- 观测预处理 ----
    @staticmethod
    def _prep_img_chw_norm(image: np.ndarray) -> "torch.Tensor":
        import torch
        import cv2  # type: ignore

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = tensor.permute(2, 0, 1)  # CHW
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)
        tensor = (tensor - imagenet_mean) / imagenet_std
        return tensor

    @staticmethod
    def _split_dual_arm_state(state14: np.ndarray, threshold: float = 0.0) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        import torch
        import torch.nn.functional as F  # type: ignore

        left7 = state14[:7]
        right7 = state14[7:]

        def _split(state7: np.ndarray) -> tuple["torch.Tensor", "torch.Tensor"]:
            arm6 = torch.tensor(state7[:6], dtype=torch.float32)
            gval = float(state7[6])
            open_flag = 1 if gval > threshold else 0
            gripper2 = F.one_hot(torch.tensor(open_flag, dtype=torch.long), num_classes=2).to(torch.float32)
            return arm6, gripper2

        arm_left, grip_left = _split(left7)
        arm_right, grip_right = _split(right7)
        return arm_left, grip_left, arm_right, grip_right

    def _build_prm_inputs(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将 websocket obs 转换为 PRM 所需格式。"""
        if self._prm is None:
            return None
        import torch

        model = self._prm["model"]
        device = self._prm["device"]
        clip = self._prm["clip"]

        top = obs.get("observation/top_image")
        left = obs.get("observation/left_wrist_image")
        right = obs.get("observation/right_wrist_image")
        state = obs.get("observation/state")
        prompt = obs.get("prompt", "")
        if top is None or state is None:
            return None

        # 确保图像为 uint8 HWC
        top_np = np.asarray(top, dtype=np.uint8)
        left_np = np.asarray(left if left is not None else top_np, dtype=np.uint8)
        right_np = np.asarray(right if right is not None else top_np, dtype=np.uint8)

        rgb_top = self._prep_img_chw_norm(top_np).to(device)       # [3,224,224]
        rgb_left = self._prep_img_chw_norm(left_np).to(device)
        rgb_right = self._prep_img_chw_norm(right_np).to(device)

        rgb_top = rgb_top.unsqueeze(0).unsqueeze(0)     # [1,1,3,224,224]
        rgb_left = rgb_left.unsqueeze(0).unsqueeze(0)
        rgb_right = rgb_right.unsqueeze(0).unsqueeze(0)

        state14 = np.asarray(state, dtype=np.float32).reshape(-1)
        arm_left, grip_left, arm_right, grip_right = self._split_dual_arm_state(state14)
        arm_left = arm_left.view(1, 1, -1).to(device)
        grip_left = grip_left.view(1, 1, -1).to(device)
        arm_right = arm_right.view(1, 1, -1).to(device)
        grip_right = grip_right.view(1, 1, -1).to(device)

        state_dict = {
            "arm_left": arm_left,
            "gripper_left": grip_left,
            "arm_right": arm_right,
            "gripper_right": grip_right,
        }

        with torch.no_grad():
            tokens = clip.tokenize([prompt], truncate=True).to(device)
        attention_mask = torch.ones(1, 1, dtype=torch.long, device=device)

        return {
            "rgb_top": rgb_top,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "state": state_dict,
            "language": tokens,
            "attention_mask": attention_mask,
        }

    # ---- 候选动作采样 + PRM 评分 ----
    def _tts_select_action(self, obs: Dict[str, Any], base_actions: np.ndarray) -> np.ndarray:
        """给定 obs 和 π0.5 的动作块，使用 PRM 进行 test-time scaling，返回修改后的动作块。"""
        if self._prm is None:
            return base_actions
        import torch

        if base_actions.ndim == 1:
            base_actions = base_actions.reshape(1, -1)
        if base_actions.shape[1] != 14:
            return base_actions

        # 允许客户端通过 obs["prm_config"] 动态覆盖部分超参数。
        prm_cfg = obs.get("prm_config") or {}
        num_action_candidates = int(prm_cfg.get("num_action_candidates", self.args.num_action_candidates))
        num_action_candidates = max(1, num_action_candidates)
        expand_candidate_actions = int(prm_cfg.get("expand_candidate_actions", self.args.expand_candidate_actions))
        expand_candidate_actions = max(0, expand_candidate_actions)
        total = num_action_candidates + expand_candidate_actions
        noise_scale = float(prm_cfg.get("gaussian_noise_scale", self.args.gaussian_noise_scale))
        use_direction_guidance = bool(prm_cfg.get("use_direction_guidance", self.args.use_direction_guidance))
        direction_cone_deg = float(prm_cfg.get("direction_cone_deg", self.args.direction_cone_deg))
        direction_cone_rad = math.radians(direction_cone_deg)

        prm_inputs = self._build_prm_inputs(obs)
        if prm_inputs is None:
            return base_actions

        model = self._prm["model"]
        device = self._prm["device"]

        base_action = torch.from_numpy(base_actions[0]).to(device=device, dtype=torch.float32)

        # 生成候选动作
        candidates: List[torch.Tensor] = [base_action]

        # 若启用方向引导，先获取 base_action 的方向预测
        direction_vec = None
        if use_direction_guidance:
            model.pre_encode_dual_hand(
                prm_inputs["rgb_top"],
                prm_inputs["rgb_left"],
                prm_inputs["rgb_right"],
                prm_inputs["state"],
                prm_inputs["language"],
                prm_inputs["attention_mask"],
            )
            base_seq = base_action.view(1, 1, -1)
            _, dir_pred = model.action_encode_with_direction(base_seq)
            direction_vec = dir_pred[0, -1, :]  # [14]
            direction_vec = torch.nn.functional.normalize(direction_vec, p=2, dim=-1)

        def _sample_candidate() -> torch.Tensor:
            # 默认：各向同性高斯噪声，在 base_action 周围采样
            if direction_vec is None:
                noise = torch.randn_like(base_action) * noise_scale
                return base_action + noise

            # 启用方向引导时：在以 direction_vec 为轴的锥形区域内采样噪声。
            # 角度由 direction_cone_deg 控制，默认 90 度（半球）。
            axis = direction_vec / (direction_vec.norm() + 1e-8)  # [D]
            dim = axis.shape[0]
            cos_min = math.cos(direction_cone_rad)
            cos_min = max(min(cos_min, 1.0), -1.0)

            # 方向：在球面帽内做拒绝采样
            batch = 64
            dir_vec = None
            while dir_vec is None:
                v = torch.randn(batch, dim, device=device)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
                dots = (v @ axis.view(dim, 1)).squeeze(1)
                keep = dots >= cos_min
                if keep.any():
                    dir_vec = v[keep][0]
                    break
                batch = min(batch * 2, 8192)

            # 半径：对齐各向同性高斯范数分布 r ≈ ||N(0, σ^2 I)||
            z = torch.randn(dim, device=device)
            r = z.norm() * noise_scale
            noise = r * dir_vec
            return base_action + noise

        for _ in range(total - 1):
            candidates.append(_sample_candidate())

        actions_tensor = torch.stack(candidates, dim=0)  # [N,14]

        # 批量评估所有候选动作的 reward
        B = actions_tensor.shape[0]
        rgb_top = prm_inputs["rgb_top"].repeat(B, 1, 1, 1, 1)
        rgb_left = prm_inputs["rgb_left"].repeat(B, 1, 1, 1, 1)
        rgb_right = prm_inputs["rgb_right"].repeat(B, 1, 1, 1, 1)

        state = {
            "arm_left": prm_inputs["state"]["arm_left"].repeat(B, 1, 1),
            "gripper_left": prm_inputs["state"]["gripper_left"].repeat(B, 1, 1),
            "arm_right": prm_inputs["state"]["arm_right"].repeat(B, 1, 1),
            "gripper_right": prm_inputs["state"]["gripper_right"].repeat(B, 1, 1),
        }
        language = prm_inputs["language"].repeat(B, 1)
        attention_mask = prm_inputs["attention_mask"].repeat(B, 1)

        model.pre_encode_dual_hand(rgb_top, rgb_left, rgb_right, state, language, attention_mask)
        action_seq = actions_tensor.view(B, 1, -1)
        reward_pred, _ = model.action_encode_with_direction(action_seq)
        rewards = reward_pred[:, -1]  # [B]

        best_idx = int(torch.argmax(rewards).item())
        best_action = actions_tensor[best_idx].detach().cpu().numpy()

        logging.info(
            "[PRM-TTS] Selected candidate %d / %d (reward=%.4f, base_reward=%.4f)",
            best_idx,
            total,
            float(rewards[best_idx].item()),
            float(rewards[0].item()),
        )

        new_actions = base_actions.copy()
        new_actions[0] = best_action
        return new_actions

    # ---- BasePolicy 接口 ----
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """兼容 openpi_client.base_policy.BasePolicy 接口。"""
        result = self.base_policy.infer(obs)
        if self._prm is None:
            return result

        actions = result.get("actions")
        if actions is None:
            return result

        try:
            actions_np = np.asarray(actions, dtype=np.float32)
        except Exception:
            return result

        if actions_np.size == 0:
            return result

        scaled_actions = self._tts_select_action(obs, actions_np)
        result["actions"] = scaled_actions
        return result

    @property
    def metadata(self) -> dict:
        return getattr(self.base_policy, "metadata", {})


def main(args: Args) -> None:
    policy = create_policy(args)
    policy = PRMScaledPolicy(policy, args)
    policy_metadata = getattr(policy, "metadata", {})

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
