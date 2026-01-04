# Step 1 - Baseline reproduction

## Goal
- Reproduce seq-JEPA baseline to use as a control.

## Dataset choices
- Baseline: CIFAR-100 with augmentation actions (paper-aligned transforms).
- Fast sanity: CIFAR-10 rotation sequences (`configs/quick/*`) for quick iteration.
- Structural: 3DIEBench or closest available equivariance benchmark.

## Measurements
- Equivariance metric on z_t.
- Invariance metric or linear probe on z_AGG.
- Leakage test: how well z_AGG does on equivariance and z_t on invariance.
- Performance vs inference sequence length.

## Ablations
- Train length M_train in {1, 2, 4}
- Eval length M_eval in {1, 2, 4, 8} if supported
- With and without action conditioning

## Stop condition
- z_t is more equivariant than z_AGG.
- z_AGG is more invariant than z_t.
