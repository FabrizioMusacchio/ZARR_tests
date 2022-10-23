# %% IMPORTS
import matplotlib
matplotlib.use('TkAgg')   # ⟵ Required in PyCharm in macOS!!  Qt5Agg or TkAgg
import matplotlib.pyplot as plt
import numpy as np
import zarr
# import ome_zarr.reader
# import ome_zarr as omezarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from ome_zarr.reader import Reader
from skimage import exposure, filters, data
import napari
import plotly
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from skimage import segmentation as seg
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy import ndimage
# %% FUNCTIONS
def plot_projection(array3D, title="dummy 3D stack", projection_method="mean", axis=2):
    """
    Plot function: plots a 2D average intensity z-projection of an input 3D array.
    """
    fig = plt.figure(2, figsize=(5, 5))
    plt.clf()
    if projection_method =="mean":
        plt.imshow(array3D.mean(axis=axis))
        plt.title(title + "\naverage intensity z-projection", fontweight="bold")
    elif projection_method=="max":
        plt.imshow(array3D.max(axis=axis))
        plt.title(title+"\nmaximum intensity z-projection", fontweight="bold")
    plt.xlabel("x-axis", fontweight="bold")
    plt.ylabel("y-axis", fontweight="bold")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.savefig(title+" projected.png", dpi=120)

def enhance(image):
    """
    Function for rescaling the intensity of each input-image layer to enhance visibility:
    """
    vmin, vmax = np.percentile(image, q=(0.5, 99.5))
    image = exposure.rescale_intensity(image, in_range=(vmin, vmax), out_range=np.float32 )
    return image
# %% CREATE DUMMY DATA-SET
""""""
# load 4D image (3D + 2 channels) from skimage samples
array_3D = data.cells3d()
# select the nuclei channel and refine the image depth:
array_3D = array_3D[22:54,1,:,:]
image_shape = array_3D.shape
print(image_shape)

# plot with plotly (into the default browser):
fig = px.imshow(enhance(array_3D), animation_frame=0, binary_string=True, binary_format='jpg')
fig.write_html("cells.html")
plotly.io.show(fig)
# %% SAVE IMAGE AS DEFAULT ZARR FILE
chunks = (1, image_shape[1], image_shape[2])
zarr_out_3D = zarr.open('zarr_3D_image.zarr', mode='w', shape=array_3D.shape,
                        chunks=chunks, dtype=array_3D.dtype)
zarr_out_3D[:] = array_3D

# control read and plot:
zarr_in_3D  = zarr.open('zarr_3D_image.zarr')
fig = px.imshow(enhance(zarr_in_3D[:]), animation_frame=0, binary_string=True, binary_format='jpg')
plotly.io.show(fig)
# %% SAVE IMAGE AS OME-ZARR FILE
""""""
# using default Zarr i/o-syntax:
store = zarr.storage.FSStore("zarr_3D_image.ome.zarr", mode="w", format="FormatV04",
                             dimension_separator="/")
root  = zarr.group(store=store, overwrite=True)
write_image(image=array_3D, group=root, axes="zyx",
                           storage_options=dict(chunks=chunks, overwrite=True))

# using ome-zarr methods:
store = parse_url("zarr_3D_image.ome.zarr", mode="w").store
root  = zarr.group(store=store, overwrite=True)
write_image(image=array_3D, group=root, axes="zyx",
                           storage_options=dict(chunks=chunks))

print(root.info)
print(root.tree())

# add some omero-standard attributes:
root.attrs["omero"] = {
    "channels": [{
        "color": "00FFFF",
        "window": {"start": 0, "end": 20},
        "label": "nuclei",
        "active": True,
    }]
}

# read the OME-ZARR file with default methods:
zarr_in_3D  = zarr.open("zarr_3D_image.ome.zarr")
print(zarr_in_3D.info)
fig = px.imshow(enhance(zarr_in_3D["0"][:]), animation_frame=0,
                binary_string=True, binary_format='jpg')
plotly.io.show(fig)

# read the OME-ZARR file with the ome_zarr io-method:
reader = Reader(parse_url("zarr_3D_image.ome.zarr"))
# nodes may include images, labels etc.:
nodes = list(reader())
# first node will be the image pixel data at full resolution:
image_node = nodes[0]
zarr_in_3D = image_node.data
fig = px.imshow(zarr_in_3D[0][:], animation_frame=0,binary_string=True, binary_format='jpg')
plotly.io.show(fig)
# %% SAVE IMAGE WITH LABELS AS OME-ZARR FILE
store = parse_url("zarr_3D_image.ome.zarr", mode="w").store
root  = zarr.group(store=store, overwrite=True)
write_image(image=array_3D, group=root, axes="zyx",
                           storage_options=dict(chunks=chunks))
root.attrs["omero"] = {
    "channels": [{
        "color": "00FFFF",
        "window": {"start": 0, "end": 20},
        "label": "nuclei",
        "active": True,
    }]
}

# segment the 3D image:
# pre-filter the image stack:
array_3D_filtered = ndimage.median_filter(array_3D, size=7)
array_3D_filtered = filters.gaussian(array_3D_filtered, sigma=2)
# threshold:
threshold = filters.threshold_otsu(array_3D_filtered)
array_3D_threshold = array_3D_filtered > threshold
# segment array_3D_threshold via the watershed method:
distance     = ndi.distance_transform_edt(array_3D_threshold.astype("bool"))
max_coords   = peak_local_max(distance, min_distance=10,labels=array_3D_threshold.astype("bool"))
local_maxima = np.zeros_like(array_3D_threshold, dtype=bool)
local_maxima[tuple(max_coords.T)] = True
markers = ndi.label(local_maxima)[0]
labels  = seg.watershed(-distance, markers, mask=array_3D_threshold.astype("bool"))
plot_projection(labels, title="dummy 3D stack labels", projection_method="max", axis=0)
plot_projection(enhance(array_3D), title="dummy 3D stack", projection_method="mean", axis=0)

# write the labels to "/labels":
labels_grp = root.create_group("labels", overwrite=True)
# the 'labels' .zattrs lists the named labels data
label_name = "watershed"
labels_grp.attrs["labels"] = [label_name]
label_grp = labels_grp.create_group(label_name)
# the 'image-label' attribute is required to be recognized as label:
label_grp.attrs["image-label"] = { }
# alternative:
# label_grp.attrs["image-label"] = {
#     "colors": [
#         {"label-value": 1, "rgba": [255, 0, 0, 255]},
#         {"label-value": 2, "rgba": [0, 255, 0, 255]},
#         {"label-value": 3, "rgba": [255, 255, 0, 255]},
#     ]
# }
write_image(labels, label_grp, axes="zyx")

# read the OME-ZARR file with default methods:
zarr_in_3D  = zarr.open("zarr_3D_image.ome.zarr")
print(zarr_in_3D.info)
fig = px.imshow(enhance(zarr_in_3D["0"][:]), animation_frame=0,
                binary_string=True, binary_format='jpg')
plotly.io.show(fig)

# read the stored labels:
fig = px.imshow(zarr_in_3D["labels/watershed"]["0"][:], animation_frame=0,
                binary_string=True, binary_format='jpg')
plotly.io.show(fig)
plot_projection(zarr_in_3D["labels/watershed"]["0"][:],
                title="dummy 3D stack read labels",
                projection_method="max", axis=0)

print(root.info)
print(root.tree())
# %% WRITING OME-ZARR USING GROUPS
store = zarr.storage.FSStore("zarr_3D_image_groups.ome.zarr", mode="w", format="FormatV04",
                             dimension_separator="/")
root  = zarr.group(store=store, overwrite=True)
root_sub_1 = root.create_group("sub_array_1", overwrite=True)
root_sub_2 = root.create_group("sub_array_2", overwrite=True)
root_sub_3 = root.create_group("sub_array_2/sub_sub_array_1", overwrite=True)
write_image(image=array_3D, group=root, axes="zyx",
                           storage_options=dict(chunks=chunks, overwrite=True))
write_image(image=array_3D, group=root_sub_1, axes="zyx",
                           storage_options=dict(chunks=chunks, overwrite=True))
write_image(image=array_3D, group=root_sub_2, axes="zyx",
                           storage_options=dict(chunks=chunks, overwrite=True))
write_image(image=array_3D, group=root_sub_3, axes="zyx",
                           storage_options=dict(chunks=chunks, overwrite=True))
print(root.tree())

zarr_in_3D  = zarr.open("zarr_3D_image_groups.ome.zarr")
fig = px.imshow(enhance(zarr_in_3D["sub_array_1"]["0"][:]), animation_frame=0,
                binary_string=True, binary_format='jpg')
plotly.io.show(fig)
viewer = napari.view_image(enhance(zarr_in_3D["sub_array_1"]["0"][:]))
# %% READ OME-ZARR AND OPEN WITH NAPARI
zarr_in_3D  = zarr.open("zarr_3D_image.ome.zarr")
viewer = napari.view_image(enhance(zarr_in_3D["0"][:]))
labels_layer = viewer.add_labels(zarr_in_3D["labels/watershed"]["0"][:],
                                 name='watershed')
# labels_layer = viewer.add_labels(labels, name='segmentation')

# read the OME-ZARR file with the ome_zarr io-method:
reader = Reader(parse_url("zarr_3D_image.ome.zarr"))
# nodes may include images, labels etc
nodes = list(reader())
# first node will be the image pixel data
image_node = nodes[0]
zarr_in_3D = image_node.data
viewer = napari.view_image(zarr_in_3D)
labels_layer = viewer.add_labels(zarr_in_3D["labels/watershed"]["0"][:],
                                 name='watershed')
# %% OPEN AND VIEW REMOTE OME-ZARR FILE
path   = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001246.zarr/"
store  = parse_url(path, mode="r").store
reader = Reader(parse_url(path))
nodes  = list(reader())
image_node = nodes[0]
read_data  = image_node.data
print(read_data)
viewer = napari.view_image(read_data[0], channel_axis=0)
# %% END