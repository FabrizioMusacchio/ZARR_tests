# %% IMPORTS
import numpy as np
import zarr
import time
from numcodecs import Blosc
# %% FUNCTIONS
def calc_process_time(t0, verbose=False, leadspaces="", output=False, unit="min"):
    """
        Calculates the processing time/time difference for a given input time and the current time

    Usage:
        Process_t0 = time.time()
        #your process
        calc_process_time(Process_t0, verbose=True, leadspaces="  ")

        :param t0:              starting time stamp
        :param verbose:         verbose? True or False
        :param leadspaces:      pre-fix, e.g., some space in front of any verbose output
        :param output:          provide an output (s. below)? True or False
        :return: dt (optional)  the calculated processing time
    :rtype:
    """
    dt = time.time() - t0
    if verbose:
        if unit=="min":
            print(leadspaces + f'(process time: {round(dt / 60, 2)} min)')
        elif unit=="sec":
            print(leadspaces + f'(process time: {round(dt , 10)} sec)')
    if output:
        return dt
# %% TESTS

# create a NumPy dummy array:
array_2D = np.random.rand(1000,1000)
array_3D = np.random.rand(1000,1000,100)

Process_t0 = time.time()
np.savetxt("array_2D.txt", array_2D, delimiter=" ")
Process_t1 = calc_process_time(Process_t0, verbose=True, unit="min")

Process_t0 = time.time()
np.savetxt("array_3D.txt", array_3D, delimiter=" ")
Process_t1 = calc_process_time(Process_t0, verbose=True,  unit="min")
"""this will not work: ValueError: Expected 1D or 2D array, got 3D array instead """

Process_t0 = time.time()
np.save("array_3D.npy", array_3D)
Process_t1 = calc_process_time(Process_t0, verbose=True, unit="min")

Process_t0 = time.time()
zarr.save('array_3D.zarr', array_3D)
Process_t1 = calc_process_time(Process_t0, verbose=True, unit="min")

Process_t0 = time.time()
array_3D_zarr = zarr.load('array_3D.zarr')
Process_t1 = calc_process_time(Process_t0, verbose=True, unit="min")
print(type(array_3D_zarr))

# %% AMAMON S3

array_380kb = zarr.load("s3://coiled-datasets/synthetic-data/array-random-390KB.zarr")
array_370GB = zarr.load("s3://coiled-datasets/synthetic-data/array-random-370GB.zarr")

# %% END