# PRM inference

CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_pi05_prm.py \
  --port 17777 \
  --prm /mnt/disk2/surui/data/Experiment_data/dobot_PRM/dobotPRM_dobot_cook_vegetable_full-dobot_pour_water_full-_bs-128_epochs-50_lr-3.0e-04_noise-0.1_20251117004909/dobot_prm_epoch_50.pt \
  policy:checkpoint \
  --policy.config pi05_dobot_tidy_desk_full \
  --policy.dir /mnt/disk2/surui/data/Experiment_data/dobot_pi0_model/pi05_dobot_tidy_desk_full \
  
  <!-- --num_action_candidates ... --expand_candidate_actions ... --gaussian_noise_scale ... --direction_cone_deg ... -->
  <!-- --use_direction_guidance -->

需要客户端通过 `ssh -L port:127.0.0.1:port your_user@machine1` 开启端口转发


# finetune
tidy-desk
```
HTTP_PROXY=http://127.0.0.1:33210 HTTPS_PROXY=http://127.0.0.1:33210 OPENPI_DATA_HOME=/mnt/disk2/surui/data/Experiment_data/openpi WANDB_BASE_URL=https://api.bandw.top HF_LEROBOT_HOME=/mnt/disk2/surui/data/dobot-xtrainer CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_dobot_tidy_desk_full --exp_name skip_tidy_full_ft --skip_sidecar_dir /mnt/disk2/surui/BoostingVLA/dobot-xtrainer/skip_sidecars/dobot_tidy_up_the_desk_full/amabs_ee_pose__w16__hff50__hq75__msl3__uh1__kfr5__dhbf10__dhbs0__gfm10__sd10__sb4__got50__bhw2__tz20__tymzero --fsdp_devices 1 --batch_size 16 --checkpoint_base_dir /mnt/disk2/surui/data/Experiment_data/dobot_pi0_model_skip --overwrite
```

pour water
```
HTTP_PROXY=http://127.0.0.1:33210 HTTPS_PROXY=http://127.0.0.1:33210 OPENPI_DATA_HOME=/mnt/disk2/surui/data/Experiment_data/openpi WANDB_BASE_URL=https://api.bandw.top HF_LEROBOT_HOME=/mnt/disk2/surui/data/dobot-xtrainer CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_dobot_pour_water_full --exp_name skip_pour-water_full_ft --skip_sidecar_dir /mnt/disk2/surui/BoostingVLA/dobot-xtrainer/skip_sidecars/dobot_pour_water_full/amabs_ee_pose__w16__hff50__hq75__msl3__uh1__kfr5__dhbf10__dhbs0__gfm10__sd10__sb4__got50__bhw2__tz20__tymzero --fsdp_devices 1 --batch_size 16 --checkpoint_base_dir /mnt/disk2/surui/data/Experiment_data/dobot_pi0_model_skip --overwrite
```

# SkiP inference

```
CUDA_VISIBLE_DEVICES=3 uv run scripts/serve_policy.py --port 17777 policy:checkpoint --policy.config pi05_dobot_pour_water_full --policy.dir /mnt/disk2/surui/data/Experiment_data/dobot_pi0_model_skip/pi05_dobot_tidy_desk_full/skip_tidy_full_ft/29999
```

policy.config: [pi05_dobot_tidy_desk_full,pi05_dobot_pour_water_full,pi05_dobot_stack_bowls]