# Running on Georgia Tech PACE (free tier)

Everything here uses the **free `embers` QOS** on Phoenix: no charge, but jobs can be
preempted. The goal of the first run is to show that an NVIDIA GPU reproduces exactly the
SHA-256 hashes recorded on the laptop (an Intel Arc 140V GPU and an x86 CPU).

1. **Connect.** Connect to the GT VPN, then
   `ssh <gt-username>@login-phoenix.pace.gatech.edu`. For ICE, the course cluster, use
   `login-ice.pace.gatech.edu`.
2. **Get the code and set up once.**
   ```bash
   git clone https://github.com/sramireddy2/exactattention.git ~/scratch/exactattention
   cd ~/scratch/exactattention
   bash scripts/pace/setup.sh
   ```
3. **Edit the charge account.** Change `#SBATCH -A gts-REPLACE_ME` in
   `scripts/pace/gpu_check.sbatch` to your account; `pace-quota` lists yours. On ICE, remove
   the `-A` and `-q` lines.
4. **Submit and watch.**
   ```bash
   sbatch scripts/pace/gpu_check.sbatch
   squeue -u $USER
   cat exactattn-gpu-*.out
   ```

What success looks like:

- Every `fixture.py check` line prints **MATCH**, and each prints `ALL MATCH`.
- The `standard SDPA` hash printed next to each one differs from the laptop's. Standard
  attention is device-dependent; ExactAttention is not.
- The CUDA test suites pass.

Some details above couldn't be confirmed from the laptop, so check them against PACE's
documentation:

- the GPU type names for `--gres`;
- whether `module load anaconda3` exists on your cluster;
- which charge account your group uses for `embers`.
