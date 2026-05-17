from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import psutil
import torch


DeviceKind = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class HardwareProfile:
    device: DeviceKind
    cpu_cores: int
    ram_gb: float
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None

    @property
    def is_low_ram(self) -> bool:
        return self.ram_gb < 4.0


def detect_hardware(prefer_gpu: bool = True) -> HardwareProfile:
    cpu_cores = psutil.cpu_count(logical=True) or 1
    ram_gb = psutil.virtual_memory().total / 1e9

    if prefer_gpu and torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return HardwareProfile(
            device="cuda",
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpu_name=props.name,
            gpu_memory_gb=props.total_memory / 1e9,
        )

    return HardwareProfile(device="cpu", cpu_cores=cpu_cores, ram_gb=ram_gb)


def print_hardware_profile(profile: HardwareProfile) -> None:
    print("Hardware profile")
    print(f"  device    : {profile.device}")
    print(f"  cpu cores : {profile.cpu_cores}")
    print(f"  ram       : {profile.ram_gb:.1f} GB")
    if profile.gpu_name:
        print(f"  gpu       : {profile.gpu_name}")
        print(f"  gpu memory: {profile.gpu_memory_gb:.1f} GB")

