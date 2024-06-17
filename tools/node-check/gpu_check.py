from mpi4py import MPI
import torch
import socket


def check_gpu_on_node():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()
    error_devices = []

    print(f"Rank {rank}/{size} on {hostname}: Checking GPUs...", flush=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print(f"Rank {rank} on {hostname}: No GPUs found on this node.", flush=True)
        return

    for i in range(num_gpus):
        try:
            torch.cuda.set_device(i)
            _ = torch.rand(10, device='cuda')  # Perform a simple computation to test the GPU
            print(f"Rank {rank}, Device {i} on {hostname}: {torch.cuda.get_device_name(i)} - OK")
        except RuntimeError as e:
            print(f"Rank {rank}, Device {i} on {hostname}: {torch.cuda.get_device_name(i)} - Error encountered: {e}")
            error_devices.append(i)

    if error_devices:
        print(f"Rank {rank} on {hostname}: Error encountered on GPU devices: {error_devices}", flush=True)
    else:
        print(f"Rank {rank} on {hostname}: No errors encountered on any devices.", flush=True)


if __name__ == "__main__":
    check_gpu_on_node()
