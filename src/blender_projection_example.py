import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import trimesh
from utils.blender_projection import (
    project_points_blender_style,
    create_depth_image,
    print_blender_camera_info,
    get_blender_default_viewpoints
)
from utils.data import normalize_pc

def load_demo_point_cloud(file_path=None, num_points=10000, y_up=True, normalize=True):
    """Load point cloud using the same logic as data.py

    Args:
        file_path: Path to .npy file with 'xyz' and 'rgb' keys (like data.py), .glb file, or .ply file
        num_points: Number of points to sample
        y_up: Whether to swap Y and Z axes
        normalize: Whether to normalize points
    """
    if file_path:
        try:
            if file_path.endswith('.glb') or file_path.endswith('.gltf'):
                # Load GLB/GLTF file using trimesh (similar to render_single_glb.py)
                print(f"Loading GLB file from {file_path}")
                mesh = trimesh.load(file_path, force='mesh')

                # If it's a scene with multiple meshes, combine them
                if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
                    # Scene with multiple geometries - combine them
                    vertices_list = []
                    colors_list = []
                    for geom in mesh.geometry.values():
                        if hasattr(geom, 'vertices'):
                            vertices_list.append(geom.vertices)
                            # Handle colors/materials
                            if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
                                colors = geom.visual.vertex_colors[:, :3] / 255.0  # Convert to [0,1]
                            else:
                                # Default color if no vertex colors
                                colors = np.ones((len(geom.vertices), 3)) * 0.7
                            colors_list.append(colors)

                    if vertices_list:
                        xyz = np.vstack(vertices_list)
                        rgb = np.vstack(colors_list)
                    else:
                        raise ValueError("No valid geometry found in GLB file")
                else:
                    # Single mesh
                    xyz = mesh.vertices
                    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                        rgb = mesh.visual.vertex_colors[:, :3] / 255.0  # Convert to [0,1]
                    else:
                        # Default color if no vertex colors
                        rgb = np.ones((len(xyz), 3)) * 0.7

                # Normalize and center the mesh (similar to normalize_scene in render_single_glb.py)
                bbox_min = xyz.min(axis=0)
                bbox_max = xyz.max(axis=0)
                scale = 1.0 / np.max(bbox_max - bbox_min)
                xyz = (xyz - (bbox_min + bbox_max) / 2) * scale * 0.8

                # Sample points if we have more than requested
                n = len(xyz)
                if n > num_points:
                    idx = np.random.choice(n, num_points, replace=False)
                    xyz = xyz[idx]
                    rgb = rgb[idx]
                elif n < num_points:
                    # Duplicate points if we have fewer than requested
                    idx = np.random.choice(n, num_points, replace=True)
                    xyz = xyz[idx]
                    rgb = rgb[idx]

                print(f"Loaded {n} vertices from GLB, sampled {len(xyz)} points")

            elif file_path.endswith('.ply'):
                # Load PLY file using Open3D
                print(f"Loading PLY file from {file_path}")
                pcd = o3d.io.read_point_cloud(file_path)

                xyz = np.asarray(pcd.points).astype(np.float32)

                # Handle colors
                if pcd.has_colors():
                    rgb = np.asarray(pcd.colors).astype(np.float32)
                else:
                    # Default color if no colors
                    rgb = np.ones((len(xyz), 3)) * 0.7

                # Normalize and center the point cloud
                bbox_min = xyz.min(axis=0)
                bbox_max = xyz.max(axis=0)
                scale = 1.0 / np.max(bbox_max - bbox_min)
                xyz = (xyz - (bbox_min + bbox_max) / 2) * scale * 0.8

                # Sample points if we have more than requested
                n = len(xyz)
                if n > num_points:
                    idx = np.random.choice(n, num_points, replace=False)
                    xyz = xyz[idx]
                    rgb = rgb[idx]
                elif n < num_points:
                    # Duplicate points if we have fewer than requested
                    idx = np.random.choice(n, num_points, replace=True)
                    xyz = xyz[idx]
                    rgb = rgb[idx]

                print(f"Loaded {n} points from PLY, sampled {len(xyz)} points")

            else:
                # Load .npy file using exact same logic as data.py
                print(f"Loading point cloud from {file_path}")
                data = np.load(file_path, allow_pickle=True).item()

                n = data['xyz'].shape[0]
                idx = np.random.choice(n, num_points, replace=(n < num_points))
                xyz = data['xyz'][idx]
                rgb = data['rgb'][idx]

                print(f"Loaded {n} points, sampled {len(xyz)} points")

        except Exception as e:
            print(f"Failed to load file {file_path}: {e}")
            file_path = None

    if file_path is None:
        print("Demo point cloud not found, creating synthetic object...")
        # Create a simple colored object
        t = np.linspace(0, 4*np.pi, num_points)
        xyz = np.column_stack([
            0.3 * np.sin(t) * np.cos(t/2),
            0.3 * np.cos(t) * np.cos(t/2),
            0.2 * np.sin(t/2)
        ]).astype(np.float32)
        rgb = np.column_stack([
            0.5 + 0.5 * np.sin(t),
            0.5 + 0.5 * np.cos(t),
            0.5 + 0.5 * np.sin(t/2)
        ]).astype(np.float32)

    # Apply same preprocessing as data.py
    if y_up:
        # swap y and z axis (exact same as data.py:88, 1337, etc.)
        xyz[:, [1, 2]] = xyz[:, [2, 1]]

    if normalize:
        xyz = normalize_pc(xyz)

    print(f"Final point cloud: {len(xyz)} points")
    print(f"XYZ range: [{xyz.min():.3f}, {xyz.max():.3f}]")
    print(f"RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")

    return xyz, rgb

def example_exact_blender_matching():
    """Example showing exact matching with Blender rendering setup"""
    print("=== Exact Blender Camera Matching ===")

    # Print camera setup info
    print_blender_camera_info()

    print(f"\nLoading and preprocessing point cloud...")
    # dataset_path = '/data/bingchen/data/objaverse-processed/merged_for_training_final'
    dataset_path = '/data/bingchen/KeypointNet/ShapeNetCore.v2.ply'
    model_cls = '02691156'
    model_id = '1b300fd9ad4050e6301fa0a0663ee996'
    xyz, rgb = load_demo_point_cloud(
        file_path=f'{dataset_path}/{model_cls}/{model_id}.ply',
        y_up=False, normalize=False
    )
    print(f"Point cloud: {len(xyz)} points")

    # Test projection with different viewpoints
    results = []
    for view_idx in [0, 4, 8]:  # Sample a few views
        result = project_points_blender_style(
            xyz, rgb,
            view_index=view_idx,
            camera_distance=1.2,  # Default from render_single_glb.py
            image_size=(512, 512)
        )
        results.append(result)

        print(f"\nView {view_idx}:")
        print(f"  Visible points: {len(result['pixels'])} / {len(xyz)}")
        print(f"  Camera position: ({result['camera_pose'][0,3]:.2f}, {result['camera_pose'][1,3]:.2f}, {result['camera_pose'][2,3]:.2f})")
        print(f"  Phi: {result['phi']:.3f}, Theta: {result['theta']:.3f}")
        print(f"  Depth range: [{result['depths'].min():.3f}, {result['depths'].max():.3f}]")

    return results, xyz, rgb


def visualize_blender_projection(results, xyz, rgb, save_path="blender_projection_demo.png"):
    """Visualize projection results matching Blender output"""
    print(f"\n=== Visualizing Blender-Style Projection ===")

    n_views = len(results)
    fig, axes = plt.subplots(2, n_views, figsize=(4*n_views, 8))
    if n_views == 1:
        axes = axes.reshape(2, 1)

    for i, result in enumerate(results):
        # Create color image
        color_image = np.zeros((512, 512, 3))
        coords = result['image_coords']
        colors = result.get('colors', np.ones((len(coords), 3)) * 0.5)

        for j in range(len(coords)):
            x, y = coords[j, 0], coords[j, 1]
            if 0 <= x < 512 and 0 <= y < 512:
                color_image[y, x] = colors[j]

        # Create depth image (matching Blender depth output)
        depth_image = create_depth_image(result)

        # Plot color image
        axes[0, i].imshow(color_image, origin='upper')
        axes[0, i].set_title(f"View {result['view_index']}: Color\n{len(result['pixels'])} points")
        axes[0, i].set_xlabel("X (pixels)")
        axes[0, i].set_ylabel("Y (pixels)")

        # Plot depth image (like Blender depth output)
        valid_depths = depth_image[depth_image > 0]
        if len(valid_depths) > 0:
            depth_vis = axes[1, i].imshow(depth_image, cmap='gray', vmin=0, vmax=valid_depths.max())
            axes[1, i].set_title(f"Depth (range: {valid_depths.min():.2f}-{valid_depths.max():.2f})")
            fig.colorbar(depth_vis, ax=axes[1, i])
        else:
            axes[1, i].imshow(depth_image, cmap='gray')
            axes[1, i].set_title("Depth (no valid points)")
        axes[1, i].set_xlabel("X (pixels)")
        axes[1, i].set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

    return color_image, depth_image


if __name__ == "__main__":
    print("Blender-Style Point Cloud Projection Examples")
    print("=" * 55)

    # Run examples
    try:
        results, xyz, rgb = example_exact_blender_matching()

        # Visualize selected views
        selected_results = [results[0], results[1], results[2]]  # Views 0, 4, 8
        visualize_blender_projection(selected_results, xyz, rgb)

        print("\n" + "="*55)
        print("SUCCESS: All examples completed!")
        print("Check 'blender_projection_demo.png' for visualizations.")
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()