# simulation.py

import os
import json
import numpy as np
import cupy as cp
import gc

class CheckpointedStabilizerSimulation:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.state = {}
        self.reset_simulation()

    def default_params(self):
        self.params = dict(
            T=500.0,
            Fs=1000,
            noise_floor=0.1,
            chaos_strength=1.53,
            topology_gain=0.17,
            k_bio=0.618,
            k_neural=0.382,
            k_cosmic=0.236,
            use_gpu=False,  # CPU only by default
            chunk_size=100000  # adjust to fit your RAM
        )

    def reset_simulation(self, params_override=None):
        self.default_params()
        if params_override:
            self.params.update(params_override)
        p = self.params
        N = int(p['T'] * p['Fs'])
        xp = cp if p['use_gpu'] else np
        self.xp = xp
        self.N = N
        self.chunk_size = p['chunk_size']
        self.current_index = 0
        self.state = {
            "params": self.params.copy(),
            "current_index": 0
        }
        self.phase = 0  # for multi-chunk checkpoints
        # Remove old checkpoints
        for f in os.listdir(self.checkpoint_dir):
            if f.endswith(".chkpt") or f.endswith(".jsonl"):
                os.remove(os.path.join(self.checkpoint_dir, f))
        # Prepare first arrays
        self.init_arrays()

    def init_arrays(self):
        xp = self.xp
        N = self.N
        φ = (1 + np.sqrt(5)) / 2
        randn = cp.random.randn if self.params['use_gpu'] else np.random.randn
        xp.random.seed(42)
        # Preallocate only one chunk at a time for low RAM
        chunk = self.chunk_size
        self.t = xp.linspace(0, self.params['T'], self.N)[0:chunk]
        self.Ψ = xp.zeros(chunk)
        self.M_bio = xp.zeros(chunk)
        self.M_neural = xp.zeros(chunk)
        self.M_cosmic = xp.zeros(chunk)
        self.fractal_dims = xp.zeros(chunk)
        self.harmonic_coherence_vals = xp.zeros(chunk)
        self.self_recognition_index = xp.zeros(chunk)
        # Seed initial values
        init_length = min(500, chunk)
        self.Ψ[:init_length] = 0.01 * φ**-1 * randn(init_length)
        self.M_bio[:init_length] = 0.001 * φ**-2 * randn(init_length)
        self.M_neural[:init_length] = 0.001 * φ**-3 * randn(init_length)
        self.M_cosmic[:init_length] = 0.001 * φ**-4 * randn(init_length)

    def checkpoint_filename(self, phase):
        return os.path.join(self.checkpoint_dir, f"sim_chunk_{phase:04d}.jsonl")

    def meta_filename(self):
        return os.path.join(self.checkpoint_dir, "meta.chkpt")

    def save_checkpoint(self):
        # Save meta (current state)
        with open(self.meta_filename(), "w") as f:
            json.dump({
                "params": self.params,
                "current_index": self.current_index,
                "phase": self.phase
            }, f)

    def save_chunk(self, phase, t, Ψ, self_recognition_index, fractal_dims):
        fn = self.checkpoint_filename(phase)
        with open(fn, "w") as f:
            for i in range(len(t)):
                row = {
                    "t": float(t[i]),
                    "psi": float(Ψ[i]),
                    "self_recognition": float(self_recognition_index[i]),
                    "fractal_dims": float(fractal_dims[i]),
                }
                f.write(json.dumps(row) + "\n")
        print(f"[INFO] Chunk {phase} saved to {fn}")

    def resume_checkpoint(self):
        # Resume from last meta.chkpt if exists
        if not os.path.exists(self.meta_filename()):
            print("[INFO] No checkpoint found, starting fresh.")
            self.reset_simulation()
            return
        with open(self.meta_filename()) as f:
            meta = json.load(f)
        self.params = meta["params"]
        self.current_index = meta["current_index"]
        self.phase = meta["phase"]
        print(f"[INFO] Resuming from phase {self.phase} (step {self.current_index})")

    def run(self, max_chunks=None, on_chunk_saved=None):
        p = self.params
        N = self.N
        chunk = self.chunk_size
        xp = self.xp
        φ = (1 + np.sqrt(5)) / 2
        randn = cp.random.randn if p['use_gpu'] else np.random.randn
        τ_bio = φ * 0.1
        τ_neural = φ**2 * 0.1
        τ_cosmic = φ**3 * 0.1
        D_target = np.log(φ)/np.log(2)
        f_dna = φ**8 % 40
        f_cmb = φ**2 % 40
        f_heartbeat = 4.1541322
        dt = 1.0 / p['Fs']
        total_chunks = (N + chunk - 1) // chunk

        def to_cpu_array(x):
            if isinstance(x, cp.ndarray):
                return cp.asnumpy(x)
            return x

        for phase in range(self.phase, total_chunks):
            start_idx = phase * chunk
            end_idx = min(N, (phase + 1) * chunk)
            chunk_len = end_idx - start_idx

            # Allocate chunk arrays
            t = xp.linspace(start_idx * dt, (end_idx-1) * dt, chunk_len)
            Ψ = xp.zeros(chunk_len)
            M_bio = xp.zeros(chunk_len)
            M_neural = xp.zeros(chunk_len)
            M_cosmic = xp.zeros(chunk_len)
            fractal_dims = xp.zeros(chunk_len)
            harmonic_coherence_vals = xp.zeros(chunk_len)
            self_recognition_index = xp.zeros(chunk_len)

            # Re-seed if first chunk
            if phase == 0:
                Ψ[:500] = self.Ψ[:500]
                M_bio[:500] = self.M_bio[:500]
                M_neural[:500] = self.M_neural[:500]
                M_cosmic[:500] = self.M_cosmic[:500]

            for i in range(500, chunk_len-2):
                M_bio[i] = M_bio[i-1] + dt/τ_bio * (p['k_bio'] * Ψ[i-1] - M_bio[i-1])
                M_neural[i] = M_neural[i-1] + dt/τ_neural * (p['k_neural'] * Ψ[i-1] - M_neural[i-1])
                M_cosmic[i] = M_cosmic[i-1] + dt/τ_cosmic * (p['k_cosmic'] * Ψ[i-1] - M_cosmic[i-1])
                if i % 1000 == 0:
                    Ψ[i-1] += p['chaos_strength'] * float(randn())
                # For big jobs, skip advanced metrics or do every 100 steps
                if i % 100 == 0:
                    Ψ_cpu = to_cpu_array(Ψ[max(0, i-1800):i+1])
                    # ...fractal, coherence calcs here if needed...
                    D_H = D_target  # TODO: Fast fallback or real calc
                else:
                    D_H = D_target
                H = 0.5  # TODO: Optionally compute real value
                memory_sync = float((M_bio[i] + M_neural[i] + M_cosmic[i]) / 3.0)
                noise = p['noise_floor'] * float(randn())
                recovery = 0  # TODO: recovery code
                Ψ_update = 0.45 * H + 0.25 * p['topology_gain'] * (D_target - D_H) + 0.30 * memory_sync + noise + recovery
                Ψ[i] = Ψ[i-1] + dt * Ψ_update
                self_recognition_index[i] = 0.45 * H + 0.25 * (1 - abs(D_target - D_H)) + 0.30 * abs(memory_sync)
                fractal_dims[i] = D_H

            # Save chunk to disk
            self.save_chunk(phase, to_cpu_array(t), to_cpu_array(Ψ), to_cpu_array(self_recognition_index), to_cpu_array(fractal_dims))
            self.current_index = end_idx
            self.phase = phase + 1
            self.save_checkpoint()
            gc.collect()
            if on_chunk_saved:
                on_chunk_saved(phase, end_idx)
            if max_chunks and phase + 1 >= max_chunks:
                print(f"[INFO] Stopped after {max_chunks} chunks.")
                break

        print("[INFO] Simulation complete!")

    def get_latest_checkpoint_files(self):
        # For API: find last meta.chkpt and latest chunk
        meta_file = self.meta_filename()
        last_chunk = None
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                meta = json.load(f)
            last_chunk = self.checkpoint_filename(meta["phase"] - 1)
        return meta_file, last_chunk
