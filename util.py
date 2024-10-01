
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

import imageio.v2 as imageio
import tempfile
import shutil

from IPython.display import Image

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import pyvista as pv
import nibabel as nib
import numpy as np
from skimage import measure
import plotly.graph_objects as go

from IPython.display import HTML


import nibabel as nib
import numpy as np

import pyvista as pv

def load_ct_scan(file_path):

    if not file_path:
        print("No .nii.gz file found in the folder.")
        return
    # Load the CT scan file
    ct_scan = nib.load(file_path)
    
    # Get the image data as a NumPy array
    ct_data = ct_scan.get_fdata()
    
    # Check if data is 3D
    if len(ct_data.shape) != 3:
        print("Expected a 3D CT scan, but got a different shape.")
        return
    
    return ct_data

def load_mesh(file_path):
    # Load the mesh file using PyVista
    mesh = pv.read(file_path)
    
    return mesh

def visualize_ct_scan(file_path, figsize=(15,5)):
    
    ct_data = load_ct_scan(file_path)

    # Visualize the middle slice in each dimension
    mid_slices = [ct_data.shape[i] // 2 for i in range(3)]

    plt.figure(figsize=figsize)
    
    # Plot sagittal slice (mid slice along the first axis)
    plt.subplot(1, 3, 1)
    plt.imshow(ct_data[mid_slices[0], :, :], cmap="gray")
    plt.title(f'Sagittal Slice (Index {mid_slices[0]})')
    plt.axis('off')

    # Plot coronal slice (mid slice along the second axis)
    plt.subplot(1, 3, 2)
    plt.imshow(ct_data[:, mid_slices[1], :], cmap="gray")
    plt.title(f'Coronal Slice (Index {mid_slices[1]})')
    plt.axis('off')

    # Plot axial slice (mid slice along the third axis)
    plt.subplot(1, 3, 3)
    plt.imshow(ct_data[:, :, mid_slices[2]], cmap="gray")
    plt.title(f'Axial Slice (Index {mid_slices[2]})')
    plt.axis('off')

    plt.show() 


def generate_gif(file_path):

    # Load the CT scan file using nibabel
    ct_data = load_ct_scan(file_path)

    filename = os.path.basename(file_path)

    # Normalize the CT scan data for visualization
    ct_data_normalized = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min()) * 255
    ct_data_normalized = ct_data_normalized.astype(np.uint8)

    # Define the output GIF path
    output_gif_path = os.path.join("gifs", f"{filename}.gif")

    # Create a temporary directory for storing images
    temp_dir = tempfile.mkdtemp()

    # Create a GIF animation
    images = []
    for i in range(ct_data_normalized.shape[2]):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(ct_data_normalized[:, :, i], cmap="gray")
        ax.axis('off')
        plt.title(f"Slice {i + 1} of {ct_data_normalized.shape[2]}")
        
        # Save image to an in-memory file object
        plt.tight_layout()
        image_file = os.path.join(temp_dir, f"{filename}_slice_{i}.png")
        plt.savefig(image_file)
        plt.close()
        
        # Append image to the images list for GIF creation
        images.append(imageio.imread(image_file))

    # Create the GIF
    imageio.mimsave(output_gif_path, images, duration=0.1)

    # Remove the temporary directory and its contents after GIF creation
    shutil.rmtree(temp_dir)

    #print(f"GIF saved at: {output_gif_path}")
    return output_gif_path


def show_gif(file_path, width=500, height=500):
    return Image(filename=file_path, width=width, height=height)


def generate_3d_reconstruction(file_path):

    # Load the segmented mask file
    ct_data = load_ct_scan(file_path)

    filename = os.path.basename(file_path)
    
    if "segmentations" in file_path:
        id = file_path.split('/')[-3]
    else:
        id = None
    
    # Extract the 3D surface using the marching cubes algorithm
    vertices, faces, _, _ = measure.marching_cubes(ct_data, level=0.7)

    # Reformat the faces array to match the expected format for PyVista
    # PyVista expects a flat array where each face is prefixed by the number of points (e.g., 3 for triangles)
    faces_formatted = np.hstack([[3] + list(face) for face in faces])

    # Create a PyVista mesh for visualization
    mesh = pv.PolyData(vertices, faces_formatted)
    
    # save mesh to file
    if id is not None:
        mesh.save(f"./mesh/{id}_{filename}.vtk")
    else:
        mesh.save(f"./mesh/{filename}.vtk")
    
    return mesh

def visualize_mesh(mesh, file_path, smoothing_iter = 50, relaxation_factor = 0.1, mesh_color = '#FFCC99', opacity = 0.7, background_color = "black" ):
    
    filename = os.path.basename(file_path)
    
    if "segmentations" in file_path:
        id = file_path.split('/')[-3]
    else:
        id = file_path.split('/')[-2]
        
    mesh_smooth = mesh.smooth(n_iter=smoothing_iter, relaxation_factor=relaxation_factor)

    # Set up the PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_smooth, 
                        color=mesh_color, 
                        show_edges=False, 
                        opacity=opacity,
                        smooth_shading=True,
                        ambient=0,
                        )
    plotter.add_axes()
    plotter.set_background(background_color)

    # Save the plot as an interactive HTML file using 'pythreejs' backend
    output_filename = f"./3d_reconstruction/{id}_{filename}.html"
    plotter.export_html(output_filename, backend='pythreejs')
    html_content = open(output_filename, 'r').read()

    return html_content
