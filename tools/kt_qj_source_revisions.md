# qj5090 KT benchmark source

- KT training-contract base: `bfe3e57c` from `repo/ktransformers-kt-sft-public-contracts`.
- Gradient norm patch: `ebf3756..fa04bb2` from `PLUTO/components/ktransformers`, stored in `kt_grad_norm.patch`.
- Accelerate: `9a0d243ac911afd2100302eef41a1ff713e5075b`.
- Transformers: `e04dcff79d485d03b33b69efed40d3ee7748a923`.
- LLaMA Factory base: `36b12fd0ddc8a3c1126b4fc4f60e3430b654e6c9`; copied into `/mnt/hurry-bench/src/llamafactory-overlay` with `llamafactory_kt_optimizer_signature.patch` applied. Original `train/sft/trainer.py` SHA256: `89d544e84c04b177bd4be908ade7eb247e39698f822ccbac723152e5fd7ea2ed`; patched SHA256: `6f2dc57e2fcd1fa19226f2304ee06819b3f3354fc976565ebff594a5c3a17a6e`.
- KT wheel: built on qj5090 with native AMD CPU flags, AVX512 BF16 and no AMX; installed only in `/mnt/hurry-bench/py-overlay-contract`.
  Wheel SHA256: `17dc9f4c9be83d1ec753d09590ccaf89dc0f4461c4a9ea3f43031c52cbb2e35c`.
- The existing two-GPU FSDP2 rank-local BF16 clipping check passed: 2/2 tests.

The standalone gradient-norm branch lacks the FSDP2 rank-local parameter APIs required by this Transformers Trainer. The benchmark combines its exact five-file patch with the KT training-contract base above, without changing the original branches.
