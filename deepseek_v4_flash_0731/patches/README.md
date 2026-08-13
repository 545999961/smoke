These patches capture the local runtime fixes used by the successful
DeepSeek-V4-Flash-0731 training environment. They are applied to the pinned
packages by `../prepare_dpsk_env.sh`.

- `megatron-core.patch`: makes optimizer shards leaf tensors and bounds the
  unfused CSA gather/FP32 workspace by processing rows in chunks.
- `mcore-bridge.patch`: rejects generic Megatron MTP for DeepSeek-V4 DSpark.
- `ms-swift.patch`: keeps the Swift trainer's MTP bookkeeping consistent with
  the DSpark guard.

The patches use installed-package paths and are applied from the virtual
environment's `site-packages` directory with `patch -p1`.
