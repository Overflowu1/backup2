import SimpleITK as sitk
import numpy as np

def bool_calculate_kernel(img_cal, length, width, height):
    i = length
    j = width
    k = height
    kernel_data = [img_cal[i - 1, j - 1, k - 1], img_cal[i - 1, j - 1, k], img_cal[i - 1, j - 1, k + 1],
                   img_cal[i - 1, j, k - 1], img_cal[i - 1, j, k], img_cal[i - 1, j, k + 1],
                   img_cal[i - 1, j + 1, k - 1], img_cal[i - 1, j + 1, k], img_cal[i - 1, j + 1, k + 1],

                   img_cal[i, j - 1, k - 1], img_cal[i, j - 1, k], img_cal[i, j - 1, k + 1],
                   img_cal[i, j, k - 1], img_cal[i, j, k + 1],
                   img_cal[i, j + 1, k - 1], img_cal[i, j + 1, k], img_cal[i, j + 1, k + 1],

                   img_cal[i + 1, j - 1, k - 1], img_cal[i + 1, j - 1, k], img_cal[i + 1, j - 1, k + 1],
                   img_cal[i + 1, j, k - 1], img_cal[i + 1, j, k], img_cal[i + 1, j, k + 1],
                   img_cal[i + 1, j + 1, k - 1], img_cal[i + 1, j + 1, k], img_cal[i + 1, j + 1, k + 1]
                   ]
    arr_kernel_data = np.array(kernel_data)
    if (arr_kernel_data > 1).any():
        return 1
    else:
        return 0

def calculate_edge(seg_label_path):
    print('-----------------------------Start edge selected-------------------------')
    img_cal = sitk.ReadImage(seg_label_path)
    img_array = sitk.GetArrayFromImage(img_cal)
    img_shape = img_array.shape
    edge_label_arr = np.zeros(img_shape)

    img_length, img_width, img_height = img_shape
    print(img_length, img_width, img_height)

    for i in range(1, img_length - 1):
        for j in range(1, img_width - 1):
            for k in range(1, img_height - 1):
                if img_array[i, j, k] == 1:
                    if bool_calculate_kernel(img_array, i, j, k) == 1:
                        edge_label_arr[i, j, k] = 1

    edge_label = sitk.GetImageFromArray(edge_label_arr)
    edge_label.CopyInformation(img_cal)
    print('-----------------------------Finished edge selected-------------------------')
    return edge_label

def distance_map(seg_label_path, edge_label):
    background_param = -10

    seg_label = sitk.ReadImage(seg_label_path)
    seg_label_arr = sitk.GetArrayFromImage(seg_label)
    edge_label_arr = sitk.GetArrayFromImage(edge_label)

    distance_map_arr = np.ones(seg_label_arr.shape) * background_param
    edge_indexes_arr = np.argwhere(edge_label_arr == 1)

    if edge_indexes_arr.shape[0] > 50:
        img_length, img_width, img_height = edge_label_arr.shape
        for i in range(1, img_length - 1):
            print('\r', "--------" + "Progress of search:" + str(round(i * 100 / img_length, 2)) + "%" + "--------", end='', flush=True)
            for j in range(1, img_width - 1):
                for k in range(1, img_height - 1):
                    if seg_label_arr[i, j, k] != -1:
                        distance_map_arr[i, j, k] = calculate_voxels_distance(edge_indexes_arr, i, j, k)

        distance_map_matrix = sitk.GetImageFromArray(distance_map_arr)
        distance_map_matrix.CopyInformation(seg_label)
        print('\n-----------------------------Finished distance calculate-------------------------')
    else:
        distance_map_matrix = sitk.Image(seg_label.GetSize(), sitk.sitkFloat32)
        distance_map_matrix.CopyInformation(seg_label)
        distance_map_matrix += 1.0

    return distance_map_matrix

def calculate_voxels_distance(edge_indexes_arr, length, width, height):
    voxel_number = edge_indexes_arr.shape[0]
    voxel_set = np.array(edge_indexes_arr)
    current_set = np.array([length, width, height])
    dis_cur = voxel_set - current_set
    distance_matrix = np.sum(dis_cur ** 2, axis=1)
    distance = np.sqrt(np.min(distance_matrix))
    return distance

def disMap_weight_relu(distance_map):
    distance_map_arr = sitk.GetArrayFromImage(distance_map)
    matrix_relu = np.zeros_like(distance_map_arr)
    distance_map_arr[distance_map_arr != -10] = distance_map_arr[distance_map_arr != -10] / np.max(distance_map_arr) * 10
    matrix_relu[distance_map_arr == -10] = 100
    matrix_relu[distance_map_arr != -10] = distance_map_arr[distance_map_arr != -10] / np.max(distance_map_arr) * 10
    matrix_relu = np.reciprocal(1 + np.exp(matrix_relu - 5)) * 0.8 + 0.2
    return sitk.GetImageFromArray(matrix_relu)

def calculate_disMap(label):
    if np.unique(label).max() > 1:
        seg_label = label[0]
        edge_label = calculate_edge(seg_label)
        distance_map_matrix = distance_map(seg_label, edge_label)
        print('distance_map_matrix.shape:', distance_map_matrix.GetSize())
        distance_map_matrix = disMap_weight_relu(distance_map_matrix)
        distance_map_matrix_new = distance_map_matrix[None].astype(np.float32)
    else:
        distance_map_matrix_new = np.ones_like(label) * 0.2
    return distance_map_matrix_new

if __name__ == '__main__':
    fileName = r'\dataset6_CLINIC_0094.nii.gz'
    write_dir = r'D:\BaiduNetdiskDownload\CTxiugai\ff\mask'

    print('-----------------------------edge selected-------------------------')
    edgeLabel = calculate_edge(write_dir + fileName)
    sitk.WriteImage(edgeLabel, write_dir + 'edge_label.nii.gz')
    print('-----------------------------Finished writing edge-------------------------')

    print('-----------------------------Distance map calculate-------------------------')
    distance_map_nii = distance_map(write_dir + fileName, edgeLabel)
    sitk.WriteImage(distance_map_nii, write_dir + 'distance_map.nii.gz')
    print('-----------------------------Finished writing distance map-------------------------')