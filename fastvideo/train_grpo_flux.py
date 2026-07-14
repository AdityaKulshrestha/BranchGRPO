#!/usr/bin/env python3
import runpy

if __name__ == "__main__":
    # Keep compatibility with existing command paths while using the new single-GPU trainer.
    runpy.run_module("fastvideo.train_branchgrpo_flux", run_name="__main__")
