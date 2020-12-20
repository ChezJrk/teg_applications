import open3d as o3d

if __name__ == '__main__':
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
    o3d.visualization.draw_geometries([mesh_frame])

