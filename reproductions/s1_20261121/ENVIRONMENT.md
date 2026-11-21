# Recorded training environment

The formal S1 seed666/667 `protocol.json` and `complete.json` files both record CUDA training on the same GPU UUID `GPU-8a300eaa-6c96-9b7b-bdbc-ec174c306d3e`.

| Component | Recorded value |
|---|---|
| Provider/runtime | AutoDL Linux container, hostname at audit `autodl-container-c13b4eab82-13ed50f2` |
| OS | Ubuntu 22.04.1 LTS, x86_64, kernel 5.15.0-94-generic |
| GPU | NVIDIA GeForce RTX 4090 D, 24,564 MiB reported by `nvidia-smi`; driver 550.120 |
| CUDA / cuDNN | PyTorch CUDA build 12.1; cuDNN 8902 in training protocol |
| CPU model | Intel Xeon Platinum 8474C; `lscpu` exposed 2 sockets × 48 cores × 2 threads = 192 logical CPUs on host |
| Container CPU quota | cgroup `cpu.max`: 1,500,000 / 100,000 μs, equivalent to 15 CPU cores of quota |
| Container memory limit | cgroup `memory.max`: 85,899,345,920 bytes (80 GiB) |
| Host-visible RAM | `free -h` showed 1.0 TiB; this is **not** the container's allocated RAM |
| Python | 3.10.8, dedicated NNDL `.venv` |
| Core libraries | torch 2.2.1+cu121, torchaudio 2.2.1, transformers 4.51.3, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.6.1, soundfile 0.13.1 |
| Full package inventory | [`ENVIRONMENT_PACKAGES.txt`](ENVIRONMENT_PACKAGES.txt), captured with `pip list --format=freeze` |
| Threads | Training CLI `--threads 4`; DataLoader `num_workers=0` |

The **physical computer manufacturer/model, motherboard and exact AutoDL instance SKU were not exposed** by the container or experiment records. “RTX 4090 D / Xeon Platinum 8474C” describes the observed GPU and host CPU model, not a verified laptop or full server model. CPU/RAM information was audited after training on the same workspace; per-run protocol directly records GPU name/UUID and software accelerator versions. Hardware matching alone does not guarantee bitwise reproduction across drivers or kernels.

S1 seed666 took 2018.42 seconds and seed667 2022.97 seconds end to end, including DEV evaluations. These are observed durations, not a promised runtime on another machine.
