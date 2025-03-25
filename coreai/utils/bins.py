import numpy as np


import numpy as np
import torch


def save_tensors_to_bin(tensors, path):
    """保存张量列表到二进制文件，包含自解释的元信息"""
    with open(path, "wb") as f:
        for tensor in tensors:
            np_array = tensor.numpy()  # 假设tensor是PyTorch张量
            dtype_name = np_array.dtype.name.encode("ascii")  # 获取类型名并编码

            # 写入类型名称长度（2字节）和名称
            f.write(len(dtype_name).to_bytes(2, "little"))
            f.write(dtype_name)

            # 写入维度数（4字节）和各维度大小（每个8字节）
            ndim = len(np_array.shape)
            f.write(ndim.to_bytes(4, "little"))
            for dim in np_array.shape:
                f.write(int(dim).to_bytes(8, "little"))

            # 写入原始数据（确保正确的字节顺序）
            f.write(np_array.tobytes())


def load_tensors_from_bin(path):
    """从二进制文件读取张量列表，自动解析元信息"""
    tensors = []
    with open(path, "rb") as f:
        while True:
            # 读取类型名称
            dtype_len_bytes = f.read(2)
            if not dtype_len_bytes:  # 文件结束
                break
            dtype_len = int.from_bytes(dtype_len_bytes, "little")
            dtype_name = f.read(dtype_len).decode("ascii")
            dtype = np.dtype(dtype_name)

            # 读取形状信息
            ndim = int.from_bytes(f.read(4), "little")
            shape = []
            for _ in range(ndim):
                dim = int.from_bytes(f.read(8), "little")
                shape.append(dim)

            # 计算数据长度并读取
            num_elements = np.prod(shape) if shape else 0
            data = f.read(num_elements * dtype.itemsize)
            np_array = np.frombuffer(data, dtype=dtype).reshape(shape)
            tensors.append(torch.from_numpy(np_array))
    return tensors
