import numpy as np
import matplotlib.pyplot as plt
import time
import math
import tkinter as tk     # from tkinter import Tk for Python 3.x
import tkinter.filedialog as fd # askopenfilename
import pandas as pd
import argparse
import open3d as o3d
from decimal import Decimal
import os
# TK_SILENCE_DEPRECATION=1

#get some information from user about what kind of experiment they want to run
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sensitivity", help = "accuracy value",  type = float,default = 2)
parser.add_argument("-d", "--density", help = "point cloud density", type = int,default = 100000)
parser.add_argument("-n", "--name", help = "name of job", type = str,default = 'test')

args = parser.parse_args()
density = args.density
sensitivity_int = args.sensitivity
job_name = args.name

start = time.time()

    
    
"ASK USER TO SELECT PLY FILE"
def fileSelect():
    root = tk.Tk()
    files = fd.askopenfilenames(parent=root, title='Choose a file')
    file_list = list(files)
    return (file_list)

##cannibalized from osteoplane
def loadFiles(file, density):
    mesh = o3d.io.read_triangle_mesh(file)
    "convert mesh to point cloud and optionally visualize"
    #visualize Mesh
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([.5, .5, .5])
    # o3d.visualization.draw_geometries([mesh])
    pcd = mesh.sample_points_uniformly(number_of_points=density)
    return pcd 

def open3DPick(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    indices = vis.get_picked_points()
    pcd_array = np.asarray(pcd.points)
    point_cloud_list = pcd_array.tolist()  
    return indices,pcd_array, point_cloud_list

"Calculate theta and phi and radius valuse from basion (0,0,0) to all points and store in standard format (x,y,z,r,phi,theta)"
def doMath(list_of_coords):
    returnList = []
    for coord in list_of_coords: 
        temp = []
        temp.append(coord[0])
        temp.append(coord[1])
        temp.append(coord[2])
        radius1 = math.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2)
        if radius1 == 0.0:
            continue
        phi_rad = math.atan2(coord[1], coord[0])
        theta_rad = math.acos(coord[2]/radius1)
        last_rad = math.atan2(coord[2],math.sqrt(coord[0]**2 + coord[1]**2))
        phi_deg = math.degrees(phi_rad)
        theta_deg = math.degrees(theta_rad)
        last_deg = math.degrees(last_rad)
        temp.append(radius1)
        temp.append(theta_deg)
        temp.append(phi_deg)
        temp.append(last_deg)
        returnList.append(temp)
    result_array = np.array(returnList)

    return (result_array)

"initial write excel spreadsheet of rounded xyz and theta phi values - not sure best way to store data. Need alkureishi and server guy input"
def writeExcel(toWrite, name):
    df = pd.DataFrame(toWrite)
    df.rename(columns={0: 'X', 1: 'Y' , 2: 'Z' , 3: 'Radius from Basion', 4: 'Phi from Basion',5:'Theta From Basion', 6:'Psi From Basion'}, inplace=True)
    writer = pd.ExcelWriter(name+'.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Results of' + job_name, index=False)
    writer.close()
    # writer.handles = None

"round radius, theta, and phi values to some significant digits"
def forceSigFigs(data,radius, angles):
    for value in data:
        value[3] = np.format_float_positional(value[3], precision = radius, unique = False)
        value[4] = np.format_float_positional(value[4], precision = angles, unique = False)
        value[5] = np.format_float_positional(value[5], precision = angles, unique = False)
        value[6] = np.format_float_positional(value[6], precision = angles, unique = False)
    data = data.tolist()
    return(data)

"write PLY files from final data might be good idea to try and store oriented normals if attempting stl conversion within code"
def writePLY(xyz, name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.io.write_point_cloud(name, pcd)
                     
"retain phi values that are on the user designated  angular interval"
def sortByPhi(n, values):
    sorted = []
    for item in values:
        result = Decimal(str(item[5])) % Decimal(str(n))
        #result_round = round(result, 1)
        if result == 0:
            sorted.append(item)   
    return(sorted)

"retain theta values that are on the user designated angular interval"
def sortByTheta(n, values):
    sorted = []
    for item in values:
        result = Decimal(str(item[4])) % Decimal(str(n))
        #result_round = round(result, 1)
        if result == 0:
            sorted.append(item)   
    return(sorted) 
  
"retain last values that are on the user designated angular interval"
def sortByLast(n, values):
    sorted = []
    for item in values:
        result = Decimal(str(item[6])) % Decimal(str(n))
        #result_round = round(result, 1)
        if result == 0:
            sorted.append(item)   
    return(sorted) 
  
def surfaceModel(big_list):
    # Step 1: Create a dictionary to store the points with the longest radius for each unique barcode
    barcode_to_points = {}
    for point in big_list:
        # for point in point_cloud:
        barcode = tuple(point[4:7])  # Extract the barcode from the point
        radius = point[3]  # Extract the radius from the point
        if barcode not in barcode_to_points:
            barcode_to_points[barcode] = [point]  # Add the point to the dictionary with the barcode as the key
        else:
            longest_radius = max([p[3] for p in barcode_to_points[barcode]])  # Find the longest radius for the current barcode
            if radius > longest_radius:
                barcode_to_points[barcode] = [point]

    
    pointcloud_surface_full = [value[0] for value in barcode_to_points.values()]
    pointcloud_surface = [value[0:3] for value in pointcloud_surface_full]
    return   pointcloud_surface, barcode_to_points

"function that does the actual averaging. Extremely inefficient due to nested loops and should probably be improved eventually"
def compare(dict_list):
    running_average = {}
    dict_list = sorted(dict_list, key=lambda x: len(x), reverse=True)
    running_average_list = []
    print('STARTING AVERAGING PROCESS')
    start_single_skull = time.time()
    for barcode in dict_list[0].keys() & dict_list[1].keys():  # Find the intersection of the barcode keys between the two dictionaries
        points1 = dict_list[0][barcode]
        points2 = dict_list[1][barcode]
        temp = []
        for p1, p2 in zip(points1, points2):
            ave_x = (p1[0] + p2[0])/2
            ave_y = (p1[1] + p2[1])/2
            ave_z = (p1[2] + p2[2])/2
            ave_radius = (p1[3] + p2[3])/2
            running_average[barcode] = [ave_x,ave_y,ave_z]
            temp.append(ave_x)
            temp.append(ave_y)
            temp.append(ave_z)
            temp.append(ave_radius)
            for i in  barcode:
                temp.append(i)
        running_average_list.append(temp)

    if len(dict_list) > 2:
            # update running average with any remaining dictionaries
        for i in range(2, len(dict_list)):
            dict_i = dict_list[i]
            running_average_list = []
            for barcode in running_average.keys() & dict_i.keys():
                points_avg = running_average[barcode]
                points_i = dict_i[barcode]
                points_i = points_i[0][0:3]
                temp = []
                for p_avg, p_i in zip([points_avg], [points_i]):
                    ave_x = (p_avg[0] + p_i[0])/2
                    ave_y = (p_avg[1] + p_i[1])/2
                    ave_z = (p_avg[2] + p_i[2])/2 
                    ave_radius = (p1[3] + p2[3])/2
                    temp.append(ave_x)
                    temp.append(ave_y)
                    temp.append(ave_z)        
                    temp.append(ave_radius)
                    for i in barcode:
                        temp.append(i)
                    running_average[barcode] = [ave_x,ave_y,ave_z]
                running_average_list.append(temp)

            # for barcode in dict_i.keys() - running_average.keys():
            #     running_average[barcode] = dict_i[barcode]

    running_average_arr = np.array(running_average_list)
    end_single_skull = time.time()

    #convert final running average into excel list for analysis
    res = []
    for key, val in running_average.items():
        res.append([key] + val)
 
    print(f'Averaging took {(end_single_skull - start_single_skull)} seconds')
    return running_average_arr, list(running_average.values()), running_average, running_average_list
        
def convertToSTL(pointcloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros( (1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    depth_list = [7,8,9]
    meshes = []
    for i in depth_list:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=i)
        mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
        densities = np.asarray(densities)
        density_colors = plt.get_cmap('plasma')(
            (densities - densities.min()) / (densities.max() - densities.min()))
        density_colors = density_colors[:, :3]
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = mesh.vertices
        density_mesh.triangles = mesh.triangles
        density_mesh.triangle_normals = mesh.triangle_normals
        density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        meshes.append(mesh)
    # print('Final mesh model')
    # o3d.visualization.draw_geometries([mesh])

    return meshes[0], meshes[1], meshes[2], pcd

def calculateVolume(pcd):
    obb = pcd.get_oriented_bounding_box()
    volume = obb.extent[0] * obb.extent[1] * obb.extent[2]
    return (volume)
    
    

density = 1000000
"set these as 0 all the time. If desired can change here, but safe bet to use here"
sig_fig = 0

file_input = fileSelect()

"main loop of progran"
def main_skull(file_input, job_name, sensitivity_int=1, density=1000000, excel=False):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    new_dir_path = 'NORMSCAN_JOBS/'+job_name
    cwd = os.getcwd()
    dir_out = os.path.join(cwd,new_dir_path)
    # check if the new directory exists
    if not os.path.exists(dir_out):
        # if the directory doesn't exist, create it
        os.makedirs(new_dir_path)
    else:
        # if the directory already exists, create a copy with a unique name
        i = 1
        while os.path.exists(dir_out):
            dir_out = dir_out + '_' + str(i)
            i += 1
        os.makedirs(dir_out)
        

    pointclouds = []
    #grab point clouds from stl files 
    for file in file_input:
        print(file, flush=True)
        pointcloud  = loadFiles(file, density) 
        pointclouds.append(pointcloud)
    
    indices = []
    distances = []

    # have user select 4 landmarks on each skull and return landmarks (R porino, L porion, basion, nasion) to determine distance between porions
    for pointcloud in pointclouds:
        index, pcd_array, point_cloud_list = open3DPick(pointcloud)
        while len(index) != 4:
            print('Please select only 4 points on the model!', flush=True)
            index, pcd_array, point_cloud_list = open3DPick(pointcloud)
        print('Point selection for model suceeded', flush=True)
        r_porion = pcd_array[index[0]]
        l_porion = pcd_array[index[1]]
        nasion = pcd_array[index[2]]
        basion = pcd_array[index[3]]
        dist = np.linalg.norm(r_porion-l_porion)
        distances.append(dist)
        indices.append(index)
  
    # o3d.visualization.draw_geometries(pointclouds, zoom=0.4459,front=[0.9288, -0.2951, -0.2242],lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])    

    #sort pointclouds based on largest nasion-basion distances
    data = list(zip(pointclouds, distances, indices))
    sorted_data= sorted(data, key=lambda x: x[1], reverse=True)
    sorted_point_clouds, sorted_distances, sorted_indices = zip(*sorted_data)
    
    # calculate the scalar required for each pointcloud using largest pcd as target
    scalars = []
    for i in range(1, len(sorted_distances)):
       scalar = sorted_distances[0]/sorted_distances[i]
       scalars.append(scalar)
    
    # scale all pointclouds to be same p-p distance as target. also translate to put everything at the origin Append target to new list right away.
    scaled_pointclouds = []
    scaled_pointclouds.append(sorted_point_clouds[0])
    for i in range(1, len(sorted_point_clouds)):
        pcd = sorted_point_clouds[i]
        pcd.scale(scalars[i-1], center = np.asarray(pcd.points)[sorted_indices[i][3]])
        # pcd.translate(np.asarray(sorted_point_clouds[0].points)[sorted_indices[0][3]], relative=False)
        scaled_pointclouds.append(pcd)
        
    # o3d.visualization.draw_geometries(scaled_pointclouds, zoom=0.4459,front=[0.9288, -0.2951, -0.2242],lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])    
    
    # move pointclouds to basion to get them close to one another 
    edited_pointclouds = []
    for idx, pcd in enumerate(scaled_pointclouds):
        basion = np.array(pcd.points[sorted_indices[idx][3]])
        pcd.translate(-basion)
        edited_pointclouds.append(pcd)
        
    # o3d.visualization.draw_geometries(scaled_pointclouds, zoom=0.4459,front=[0.9288, -0.2951, -0.2242],lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])    

    # calculate initial transformatino using selected points and then use ICP to register pointclouds and improve lateral alignment
    #define target as cloud with largest initial distance
    registered_clouds = []
    target = edited_pointclouds[0]
    target.paint_uniform_color([0, 0.651, 0.929])
    registered_clouds.append(target)
    picked_id_target = sorted_indices[0]
    # loop through skulls and do registration. paint to easily plot later 
    for idx, i in enumerate(range(1, len(edited_pointclouds))):
        source = edited_pointclouds[i]
        if idx == 0:
            source.paint_uniform_color([1, 0.706, 0])
        elif idx == 1:
            source.paint_uniform_color([1, 0, 0])
        elif idx == 2:
            source.paint_uniform_color([0, 1, 0])
        elif idx == 3:
            source.paint_uniform_color([1, .3, 0])
        elif idx == 4:
            source.paint_uniform_color([.5, 1, 0])
        elif idx == 5:
            source.paint_uniform_color([1, 0.3, 1])
            
        picked_id_source = sorted_indices[i]
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))
        threshold = 0.03   # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        registered_cloud = source.transform(reg_p2p.transformation)
        registered_clouds.append(registered_cloud)
    
    # o3d.visualization.draw_geometries(registered_clouds, zoom=0.4459,front=[0.9288, -0.2951, -0.2242],lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])    
    
    #translate the pointclouds to the basion again to account for movement during registration 
    final_pointclouds = []
    for idx, pcd in enumerate(registered_clouds):
        basion = np.array(pcd.points[sorted_indices[idx][3]])
        pcd.translate(-basion)
        volume = calculateVolume(pcd)
        print('Volume of skull'+str(idx+1)+ '='+str(round(volume,2)), flush=True)
        final_pointclouds.append(pcd)
    
    print('If you are happy with the registration, click q to continue. \n If you are not, you may need to restart', flush=True)
    # o3d.visualization.draw_geometries(final_pointclouds, zoom=0.4459,front=[0.9288, -0.2951, -0.2242],lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])    

    # send each pointcloud to doMath and surfaceModel function to create barcodes for future averaging
    big_list = []
    print('BEGINNING FINAL ANALYSIS')

    for idx, pcd in enumerate(final_pointclouds):
        print('Starting analysis of model '+str(idx+1), flush=True)
        point_array = np.array(pcd.points)
        point_list = point_array.tolist()
        result = doMath(point_list)
        "convert to array to do rouding before converting back to list. should probably be improved"
        abridged_list = forceSigFigs(result, 9, sig_fig)
        "keep data only with proper theta and phi values. Could maybe wrap into one function"
        phi_edited = sortByPhi(float(sensitivity_int),abridged_list)
        last_edited = sortByLast(float(sensitivity_int),phi_edited)
        final_list = sortByTheta(float(sensitivity_int), last_edited)
        file_name_excel = 'processed_data_input_'+str(idx+1)
        excel_out = os.path.join(dir_out, file_name_excel)
        if excel == True:
            writeExcel(final_list, excel_out)
        surface_array,point_dict = surfaceModel(final_list)
        file_name_pcd = 'processed_cloud_' + str(idx+1) + '.ply'
        out_input_pcd = os.path.join(dir_out, file_name_pcd)
        writePLY(surface_array, out_input_pcd)
        big_list.append(point_dict)

    print('Preparing to average and make STL Files', flush=True)
    average_array, average_list, average_dict, pcd_storage_excel = compare(big_list)
    if excel == True:
        writeExcel(pcd_storage_excel, os.path.join(dir_out, 'FINAL_processed_data'))
    mesh_7, mesh_8, mesh_9, pointcloud_final = convertToSTL(average_list)
    volume_final = calculateVolume(pointcloud_final)
    print('Volume of final skull =' + str(round(volume_final,2)), flush=True)

    # save meshes
    meshes = [mesh_7, mesh_8, mesh_9]
    for idx, mesh in enumerate(meshes):
        if idx == 0:
            filename = 'FINAL_STL_low_res.stl' 
        elif idx == 1:
            filename = 'FINAL_STL_mid_res.stl' 
        elif idx == 2:
            filename = 'FINAL_STL_high_res.stl' 
        out = os.path.join(dir_out, filename)
        o3d.io.write_triangle_mesh(out, mesh)

    # save pointcloud
    out_pcd = os.path.join(dir_out, 'FINAL_pointcloud_out.ply')
    o3d.io.write_point_cloud(out_pcd, pointcloud_final)
    print('Final Pointcloud. Press q to continue and finish analysis',flush=True)
    # o3d.visualization.draw_geometries([pointcloud_final], zoom=0.4459,front=[0.9288, -0.2951, -0.2242],lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])    
    finish = time.time()
    time_tot = finish-start
    print(time_tot, flush=True)

    print('RUN COMPLETE')
    return  mesh_7, mesh_8, mesh_9, pointcloud_final
    # writePLY(average_array, './2_ave_TEST0208.ply')

    
mesh_7, mesh_8, mesh_9, pointcloud = main_skull(file_input, job_name)










