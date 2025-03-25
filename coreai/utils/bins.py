import numpy as np


def save_tensors_to_bin(tensors, path):
    with open(path, "wb") as f:
        for tensor in tensors:
            np_array = tensor.numpy()

            # 写入数据类型标识（3字节）
            dtype_str = np_array.dtype.name
            f.write(dtype_str.ljust(3).encode("ascii"))

            # 写入形状数组长度（4字节，小端）
            ndim = len(np_array.shape)
            f.write(ndim.to_bytes(4, "little"))

            # 写入形状数组（每个维度8字节，小端）
            for dim in np_array.shape:
                f.write(int(dim).to_bytes(8, "little"))

            # 写入原始数据
            f.write(np_array.tobytes())
