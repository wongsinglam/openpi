## Checkpoints
/data/dmt/Experiment_data/openpi/openpi-assets/checkpoints/pi05_base/params

## Dataset Addr
/data/dmt/dobot-xtrainer

## wsl_finetune
```
OPENPI_DATA_HOME=/data/wsl/vla/openpi \
HF_LEROBOT_HOME=/data/wsl/vla/dobot_xtrainer \
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_dobot_tidy_desk_full --exp_name=finetune_test --fsdp_devices 1 --batch_size 16 \
--checkpoint_base_dir /data/wsl/vla/openpi/Experiment_data/pi05_test --overwrite
```

## codex version
```
OPENPI_DATA_HOME=/data/wsl/vla/openpi \
HF_LEROBOT_HOME=/data/wsl/vla/dobot_xtrainer \
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run -m openpi_ext.train --use-ext-loader \
pi05_dobot_tidy_desk_full --exp_name=finetune_test --fsdp_devices 1 --batch_size 16 \
--checkpoint_base_dir /data/wsl/vla/openpi/Experiment_data/pi05_test --overwrite
```

# finetune
tidy-desk
```
HTTP_PROXY=http://127.0.0.1:33210 HTTPS_PROXY=http://127.0.0.1:33210 OPENPI_DATA_HOME=/mnt/disk2/surui/data/Experiment_data/openpi WANDB_BASE_URL=https://api.bandw.top HF_LEROBOT_HOME=/data/dmt/dobot-xtrainer CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_dobot_tidy_desk_full --exp_name skip_tidy_full_ft --skip_sidecar_dir /mnt/disk2/surui/BoostingVLA/dobot-xtrainer/skip_sidecars/dobot_tidy_up_the_desk_full/amabs_ee_pose__w16__hff50__hq75__msl3__uh1__kfr5__dhbf10__dhbs0__gfm10__sd10__sb4__got50__bhw2__tz20__tymzero --fsdp_devices 1 --batch_size 16 --checkpoint_base_dir /mnt/disk2/surui/data/Experiment_data/dobot_pi0_model_skip --overwrite
```

pour water
```
HTTP_PROXY=http://127.0.0.1:33210 HTTPS_PROXY=http://127.0.0.1:33210 OPENPI_DATA_HOME=/mnt/disk2/surui/data/Experiment_data/openpi WANDB_BASE_URL=https://api.bandw.top HF_LEROBOT_HOME=/data/dmt/dobot-xtrainer CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_dobot_pour_water_full --exp_name skip_pour-water_full_ft --skip_sidecar_dir /mnt/disk2/surui/BoostingVLA/dobot-xtrainer/skip_sidecars/dobot_pour_water_full/amabs_ee_pose__w16__hff50__hq75__msl3__uh1__kfr5__dhbf10__dhbs0__gfm10__sd10__sb4__got50__bhw2__tz20__tymzero --fsdp_devices 1 --batch_size 16 --checkpoint_base_dir /mnt/disk2/surui/data/Experiment_data/dobot_pi0_model_skip --overwrite
```

# SkiP inference

```
CUDA_VISIBLE_DEVICES=3 uv run scripts/serve_policy.py --port 17777 policy:checkpoint --policy.config pi05_dobot_pour_water_full --policy.dir /mnt/disk2/surui/data/Experiment_data/dobot_pi0_model_skip/pi05_dobot_tidy_desk_full/skip_tidy_full_ft/29999
```

policy.config: [pi05_dobot_tidy_desk_full,pi05_dobot_pour_water_full,pi05_dobot_stack_bowls]