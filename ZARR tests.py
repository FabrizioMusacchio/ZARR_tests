# %% IMPORTS
import numpy as np
import zarr
import time
from numcodecs import Blosc
import dask.array as da
import s3fs
# %% CREATE A DUMMY NUMPY ARRAY
np.random.seed(1)
array_2D = np.random.rand(1000,1000)
array_3D = np.random.rand(1000,1000,100)

# save the array as textfiles and as a NumPy file:
np.save("array_2D.npy", array_2D)
np.savetxt("array_2D.txt", array_2D, delimiter=" ")
np.save("array_3D.npy", array_3D)
# np.savetxt("array_3D.txt", array_3D, delimiter=" ")
"""the latter will not work: ValueError: Expected 1D or 2D array, got 3D array instead """
# %% ZARR CONVENIENT SAVE AND LOAD
"""convenient save and load (save right away and load directly into memory:"""
zarr_out_3D_convenient = zarr.save('zarr_out_3D_convenient.zarr', array_3D)
zarr_in_3D_convenient  = zarr.load('zarr_out_3D_convenient.zarr')
print(type(zarr_in_3D_convenient))
# %% ZARR WITH MORE CONTROLS
"""save with more controls and do not load directly into memory:"""
zarr_out_3D = zarr.open('zarr_out_3D.zarr', mode='w', shape=array_3D.shape,
                        chunks=(1000,1000,1), dtype=array_3D.dtype)
zarr_out_3D[:] = array_3D

zarr_in_3D  = zarr.open('zarr_out_3D.zarr')
print(type(zarr_in_3D))
print(type(zarr_in_3D[:]))
print(zarr_in_3D.info)

zarr_out_3D_one_chunk = zarr.open('zarr_out_3D_one_big_chunk.zarr', mode='w', shape=array_3D.shape,
                        chunks=(1000,1000,100), dtype=array_3D.dtype)
zarr_out_3D_one_chunk[:] = array_3D

# alternative:
zarr_out_root = zarr.open('zarr_out_3Ds.zarr', mode='w')
zarr_out_3D   = zarr_out_root.create_dataset("array_3D", data=array_3D)
zarr_out_2D   = zarr_out_root.create_dataset("array_2D", data=array_2D)
print(zarr_out_root.info)
print(type(zarr_out_root))
print(type(zarr_out_root["array_3D"]))
print(type(zarr_out_root["array_2D"]))

print(zarr_out_root["array_3D"].info)
print(zarr_out_root["array_2D"].info)
# %% NUMPY MEMORY MAPPING
numpy_mmap_3D = np.memmap("numpy_out_3D.arr", mode="w+", dtype=array_3D.dtype, shape=array_3D.shape)
numpy_mmap_3D[:] = array_3D
numpy_mmap_3D.flush()

numpy_mmap_3D_read = np.memmap("numpy_out_3D.arr", mode="r", dtype=array_3D.dtype, shape=array_3D.shape)
print(np.array_equal(array_3D, numpy_mmap_3D_read))
# %% ZARR COMPRESSION
compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
zarr_out_3D = zarr.open('zarr_out_3D_compressed.zarr', mode='w', shape=array_3D.shape,
                        chunks=(1000,1000,1), dtype=array_3D.dtype, compressor=compressor)
zarr_out_3D[:] = array_3D
print(zarr_out_3D.info)

array_3D_int = np.random.randint(low=1, high=10, size=(1000,1000,100), dtype="int64")
compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
zarr_out_3D_int = zarr.open('zarr_out_3D_compressed_int.zarr', mode='w', shape=array_3D_int.shape,
                        chunks=(1000,1000,1), dtype=array_3D_int.dtype, compressor=compressor)
zarr_out_3D_int[:] = array_3D_int
print(zarr_out_3D_int.info)
# %% ZARR GROUPS
zarr_out_root = zarr.open('zarr_out_3D.zarr', mode='w')
zarr_out_3D   = zarr_out_root.create_dataset("array_3D", data=array_3D)
zarr_out_2D   = zarr_out_root.create_dataset("array_2D", data=array_2D)
zarr_out_root.create_group("group_1")
zarr_out_root["group_1"].create_dataset("array_1D_1",data=np.arange(1000))
zarr_out_sub_1D_1 = zarr_out_root.create_dataset("group_2/array_1D_2", data=np.arange(100))
zarr_out_sub_1D_2 = zarr_out_root.create_dataset("group_2/array_1D_3", data=np.arange(10))
print(zarr_out_root.tree())

zarr_in_root = zarr.open('zarr_out_3D.zarr', mode='r')
zarr_in_group_1 = zarr_in_root["group_1/array_1D_1"]
print(zarr_in_group_1.info)

zarr_in_group_1 = zarr.open('zarr_out_3D.zarr/group_1/array_1D_1', mode='r')
# %% ZARR ATTRIBUTES
""""""
# assign attributes to the (root) Zarr file:
zarr_out_root.attrs["author"] = "Pixel Tracker"
zarr_out_root.attrs["date"]   = "Oct 13, 2022"
zarr_out_root.attrs["description"]  = "A Zarr file containing some NumPy arrays"

# ...to groups:
zarr_out_root["group_1"].attrs["description"] = "A test sub-group"

# ...and to arrays:
zarr_out_root["group_1/array_1D_1"].attrs["description"] = "A test array"

for key in zarr_out_root.attrs.keys():
    print(f"{key}: {zarr_out_root.attrs[key]}")
print(zarr_out_root["group_1"].attrs["description"])
print(zarr_out_root["group_1/array_1D_1"].attrs["description"])
# %% DISTRIBUTED STORAGE
s3_fs     = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='eu-west-2'))
s3_store  = s3fs.S3Map(root='zarr-demo/store', s3=s3_fs, check=False)
zarr_root = zarr.group(store=s3_store)
z_array   = zarr_root['foo/bar/baz']
print(z_array)
print(z_array.info)
print(z_array[:].tobytes())
# %% END