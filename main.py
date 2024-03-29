import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import csv
import cv2
import os
import sys
from pathlib import Path


def gray_image(image):
    red, green, blue = image[:,:,0], image[:,:,1],image[:,:,2]
    image = 0.2989*red + 0.5870 * green + 0.1140 * blue
    return image
    

def convolution(img, ker):
    img_h = img.shape[0]
    img_w = img.shape[1]
    ker_h = ker.shape[0]
    ker_w = ker.shape[1]
    fin_h = img_h - ker_h + 1
    fin_w = img_w - ker_w + 1
    conv_img = np.zeros((fin_h, fin_w))
    for i in range(fin_h):
        for j in range(fin_w):
            conv_img[i, j] = np.sum(img[i:i+ker_h, j:j+ker_w] * ker)
    return conv_img


kernel_x = (np.array([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]]))/8
kernel_y = (np.array([[1,-2,-1],
                      [0,0,0],
                      [1,2,1]]))/8
kernel_uni = (np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]]))/9
gaussian = (np.array([[1,2,1],
                      [2,4,2],
                      [1,2,1]]))


def thresholding(image, lower_thres, upper_thres):
    height,width =image.shape
    for i in range(height):
        for j in range(width):
            if i<10 or j<10 or i>height-10 or j>width-10:
                image[i][j]=0
            elif image[i][j]<=lower_thres:
                image[i][j]=255
            elif image[i][j]>=upper_thres:
                image[i][j]=0
            elif image[i-1][j-1] >= upper_thres or  image[i-1][j] >= upper_thres or image[i-1][j+1] >= upper_thres or image[i][j-1] >= upper_thres or image[i][j+1] >= upper_thres or image[i+1][j-1] >= upper_thres or image[i+1][j] >= upper_thres or image[i+1][j+1] >= upper_thres:
                image[i][j]=0
            else:
                image[i][j]=0      
    return image


def resize_image(image):
    image = Image.open(image_path)
    H=600
    img_h = image.height
    img_w = image.width
    x = H/img_h
    img_w = int(img_w * x)
    img_h = int(img_h * x)
    image = image.resize((img_w, img_h))
    image = np.array(image)
    return image


def gardient(sobelx_img,sobely_img):
    H = sobelx_img.shape[0]
    W = sobelx_img.shape[1]
    res_img = np.zeros((H, W))
    dir_mat = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            res_img[i][j] = (sobelx_img[i][j]**2 + sobely_img[i][j]**2)**0.5
            if sobelx_img[i][j]==0:
                dir_mat[i][j] = float('inf')
            else:
                dir_mat[i][j] = sobely_img[i][j]/sobelx_img[i][j]
    dir_mat = np.arctan(dir_mat)
    dir_mat = np.degrees(dir_mat)
    dir_mat = np.where(dir_mat < 0, dir_mat + 180, dir_mat)
    return res_img, dir_mat     

    
def non_max_suppression(grad_img, dir_mat):
    H, W = grad_img.shape
    res_img = np.zeros((H, W))
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            theta = dir_mat[i][j]
            if (theta >= 0 and theta <= 22.5) or (theta > 157.5 and theta <= 180):
                val = max(grad_img[i][j-1], grad_img[i][j+1])
            elif (theta > 22.5 and theta <= 67.5):
                val = max(grad_img[i - 1][j + 1], grad_img[i + 1][j - 1])
            elif (theta > 67.5 and theta <= 112.5):
                val = max(grad_img[i - 1][j], grad_img[i + 1][j])
            elif (theta > 112.5 and theta <= 180):
                val = max(grad_img[i-1][j-1], grad_img[i+1][j+1])
            else:
                val = 0
            if grad_img[i][j] < val:
                grad_img[i][j] = 0
    return grad_img


def dilate_erode(image, ker_h, ker_w, flag):
    x,y=image.shape
    ker = np.ones((ker_h,ker_w))
    new_img= np.zeros((x,y))
    if flag==0:
        for i in range(x-ker_h+1):
            for j in range(y-ker_w+1):
                new_img[i][j] = np.min(image[i:i+ker_h, j:j+ker_w])
    else:
        for i in range(x-ker_h+1):
            for j in range(y-ker_w+1):
                new_img[i][j] = np.max(image[i:i+ker_h, j:j+ker_w])
    return new_img

def horizontal_filter(image,k_h,k_w,flag):
    kernel = np.ones((k_h,k_w))
    h,w = image.shape
    new_img = np.zeros((image.shape))
    if flag==0:
        for i in range(h-k_h+1):
            for j in range(w-k_w+1):
                new_img[i][j]=np.min(image[i:i+k_h,j:j+k_w]*kernel)
    else:
        for i in range(h-k_h+1):
            for j in range(w-k_w+1):
                new_img[i][j]=np.max(image[i:i+k_h,j:j+k_w]*kernel)
    return new_img

def lab_conn_comp(img):
    rows, cols = len(img), len(img[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    high_pixel_counts = []
    stack = []

    def is_legal(row, col):
        return 0 <= row < rows and 0 <= col < cols and not visited[row][col] and img[row][col] == 1

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and img[i][j] == 1:
                stack.append((i, j))
                curr_comp = []

                while stack:
                    row, col = stack.pop()

                    if not visited[row][col] and img[row][col] == 1:
                        visited[row][col] = True
                        curr_comp.append((row, col))
                        n_pixels = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

                        for row_n, col_n in n_pixels:
                            if is_legal(row_n, col_n):
                                stack.append((row_n, col_n))

                components.append(curr_comp)
                high_pixel_counts.append(len(curr_comp))

    return components, high_pixel_counts

def cal_centroids(pix_idx, cluster):
    
    centroids = []
    left_bot_points = []
    left_top_points = []
    
    for i in pix_idx:
        x=0
        y=0
        curr_cluster_len = len(cluster[i])
        
        for j in range(curr_cluster_len):
            point_i = cluster[i][j][0]
            point_j = cluster[i][j][1]
            x += point_i
            y += point_j
        x = int(x/curr_cluster_len)
        y = int(y/curr_cluster_len)
        centroids.append([x,y])
    
        bot_point_i = 0
        top_point_i = 1000
        for j in range(curr_cluster_len):
            point_i = cluster[i][j][0]
            point_j = cluster[i][j][1]
            if point_i > bot_point_i  and point_j < y:
                bot_point_i = point_i
                bot_point_j = point_j
            if point_i < top_point_i  and point_j < y:
                top_point_i = point_i
                top_point_j = point_j
        left_bot_points.append([bot_point_i, bot_point_j])
        left_top_points.append([top_point_i, top_point_j])
       
    return centroids, left_bot_points, left_top_points

 
def pixel_out_fun(pix_idx_out,cluster,new_img):
    for i in pix_idx_out:
        curr_cluster_len = len(cluster[i])
        for j in range(curr_cluster_len):
            new_img[cluster[i][j][0]][cluster[i][j][1]]=0
    return new_img


def comp_calculation(pixel_num,cluster,new_img):
    
    pixel_num = np.array(pixel_num)
    pix_mean = np.mean(pixel_num)
    pix_med = np.median(pixel_num)
    pix_max = np.max(pixel_num)
    pix_min = np.min(pixel_num)
    pix_std = np.std(pixel_num)
    
    pix_idx_out1 = np.where(pixel_num<=(pix_med/2))[0]
    pix_idx_out2 = np.where(pixel_num<=100)[0]
    pix_idx_out = np.concatenate((pix_idx_out1, pix_idx_out2))
    pix_idx_out = np.unique(pix_idx_out)
    new_img = pixel_out_fun(pix_idx_out, cluster, new_img)

    pix_idx = []
    for i in range(len(cluster)):
        if i not in pix_idx_out:
            pix_idx.append(i)

    pixel_num = np.delete(pixel_num,pix_idx_out)
    # print(pixel_num)
    centroids, left_bot_points, left_top_points = cal_centroids(pix_idx,cluster)
    if pixel_num.shape[0]==0:  
        return 0, centroids, left_bot_points, left_top_points, new_img
    # large_pix = pixel_num[pixel_num > pix_med + pix_std]
    return pixel_num.shape[0], centroids, left_bot_points, left_top_points, new_img

    
if int(sys.argv[1])==1:
    img_dir = sys.argv[2]
    image_extensions = ['.png']
    image_files = []

    for filename in os.listdir(img_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)

    image_files = np.sort(image_files)      
    image_data = []
    heading = ['image_name', 'number of sutures','mean inter suture spacing', 'variance of inter suture spacing', 'mean suture angle wrt x-axis', 'variance of suture angle wrt x-axis']
    image_data.append(heading)

    for img in image_files:
        # print('Image:', img)
        image_name = img
        loc = 'img1'
        image_path = img_dir+'/'+image_name
        image=np.array(Image.open(image_path))
        image = resize_image(image_path)
        image_gray = gray_image(image)
        conv_img = convolution(image_gray, gaussian)
        sobelx_img = convolution(conv_img, kernel_x)
        sobely_img = convolution(conv_img, kernel_y)
        grad_img, dir_mat = gardient(sobelx_img,sobely_img)
        max_pix_val = grad_img.max()
        thres_img = thresholding(grad_img,max_pix_val*0.3,max_pix_val*0.4)
        new_img = dilate_erode(thres_img,1,10,0)
        new_img = dilate_erode(new_img,2,2,1)
        new_img = dilate_erode(new_img,2,3,0)
        new_img = dilate_erode(new_img,2,7,1)
        bin_image = (new_img > 0).astype(int)
        cluster, pixel_num = lab_conn_comp(bin_image)
        post_labels, centroids, left_bot_points, left_top_points, new_img = comp_calculation(pixel_num, cluster,new_img)
        # print('Clusters: ',post_labels)
        centroid_img = np.zeros((new_img.shape[0], new_img.shape[1])) 
        centroids = np.array(centroids)
        h_new_img = new_img.shape[0]
        w_new_img = new_img.shape[1]
        h_image = image.shape[0]
        w_image = image.shape[1]
        h_diff = h_image - h_new_img
        w_diff = w_image - w_new_img
        centroid_dist = []

        height_img = image.shape[0]
        weidth_img = image.shape[1]
        angle_sut = []
        line_color = (255, 0, 0) 
        cent_color = (0, 0, 255)
        thickness = 1
        rad = 2
        for k in range(centroids.shape[0]):
            cent_i = centroids[k][0] + h_diff
            cent_j = centroids[k][1] + w_diff
            point_i = int((left_bot_points[k][0] + 2*h_diff + left_top_points[k][0])/2)
            point_j = int((left_bot_points[k][1] + 2*h_diff + left_top_points[k][1])/2)
            if k < centroids.shape[0]-1:
                cent_i1 = centroids[k+1][0] + h_diff
                cent_j1 = centroids[k+1][1] + w_diff
                centroid_dist.append(np.sqrt( ( (cent_i-cent_i1)/height_img)**2 + ( (cent_j-cent_j1)/weidth_img)**2 ) )
                cv2.line(image, (centroids[k][1] + w_diff, centroids[k][0] + h_diff), (centroids[k+1][1] + w_diff, centroids[k+1][0] + h_diff), line_color, thickness)
                cv2.line(new_img, (centroids[k][1], centroids[k][0]), (centroids[k+1][1], centroids[k+1][0]), line_color, thickness)
            cv2.circle(image, (centroids[k][1] + w_diff, centroids[k][0] + h_diff), rad, cent_color, -1)
            cv2.circle(new_img, (centroids[k][1], centroids[k][0]), rad, cent_color, -1)
            angle = np.arctan((point_i-cent_i)/(cent_j-point_j))
            angle = np.degrees(angle)
            angle_sut.append(angle)

        mean_cent_dist = np.round(np.mean(centroid_dist),6)
        var_cent_dist = np.round(np.var(centroid_dist),6)
        mean_sut_angle = np.round(np.mean(angle_sut),2)
        var_sut_angle = np.round(np.var(angle_sut),2)
            
        temp = [image_name,post_labels,mean_cent_dist,var_cent_dist,mean_sut_angle,var_sut_angle]
        image_data.append(temp)

        folder_name = 'output_images'
        folder_path = Path.cwd() / folder_name
        if not os.path.isdir(folder_path):
            folder_path.mkdir(exist_ok=True)

        dir =f'{folder_name}/{image_name}'
        cv2.imwrite(dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # print('Output Images created')

    output_csv = sys.argv[3]
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(image_data)
    # print('Done')

else:
    file_name = sys.argv[2]
    data = pd.read_csv(file_name)
    images_names = []
    file_content = []
    heading = ['img1_path', 'img2_path','output_distance', 'output_angle']
    file_content.append(heading)
    image_metrics = []
    for k in range(0,data.shape[0]):
        # print('Sample No: ',k+1)
        temp=[]
        pic1 = data.at[k, 'img1_path']
        pic2 = data.at[k, 'img2_path']
        temp.append(pic1)
        temp.append(pic2)
        image_data = [pic1, pic2]
        for pic in image_data:
            if pic in images_names:
                continue
            else:
                images_names.append(pic)
                image_path = pic
                image=np.array(Image.open(image_path))
                height_img = image.shape[0]
                image = resize_image(image_path)
                image_gray = gray_image(image)
                conv_img = convolution(image_gray, gaussian)
                sobelx_img = convolution(conv_img, kernel_x)
                sobely_img = convolution(conv_img, kernel_y)
                grad_img, dir_mat = gardient(sobelx_img,sobely_img)
                max_pix_val = grad_img.max()
                thres_img = thresholding(grad_img,max_pix_val*0.3,max_pix_val*0.4)
                new_img = dilate_erode(thres_img,1,10,0)
                new_img = dilate_erode(new_img,2,2,1)
                new_img = dilate_erode(new_img,2,3,0)  
                new_img = dilate_erode(new_img,2,7,1)
                bin_image = (new_img > 0).astype(int)
                cluster, pixel_num = lab_conn_comp(bin_image)
                post_labels, centroids, left_bot_points, left_top_points, new_img = comp_calculation(pixel_num, cluster,new_img)
                centroid_img = np.zeros((new_img.shape[0], new_img.shape[1])) 
                centroids = np.array(centroids)
                h_new_img = new_img.shape[0]
                w_new_img = new_img.shape[1]
                h_image = image.shape[0]
                w_image = image.shape[1]
                h_diff = h_image - h_new_img
                w_diff = w_image - w_new_img
                centroid_dist = []
                centroid_dist = []
                angle_sut = []
                line_color = (255, 0, 0) 
                cent_color = (0, 0, 255)
                thickness = 1
                rad = 2

                for k in range(centroids.shape[0]):
                    cent_i = centroids[k][0] + h_diff
                    cent_j = centroids[k][1] + w_diff
                    point_i = int((left_bot_points[k][0] + 2*h_diff + left_top_points[k][0])/2)
                    point_j = int((left_bot_points[k][1] + 2*h_diff + left_top_points[k][1])/2)
                    if k < centroids.shape[0]-1:
                        cent_i1 = centroids[k+1][0] + h_diff
                        cent_j1 = centroids[k+1][1] + w_diff
                        centroid_dist.append(np.sqrt(((cent_i-cent_i1)**2 + (cent_j-cent_j1)**2))/height_img)
                        cv2.line(image, (centroids[k][1] + w_diff, centroids[k][0] + h_diff), (centroids[k+1][1] + w_diff, centroids[k+1][0] + h_diff), line_color, thickness)
                        cv2.line(new_img, (centroids[k][1], centroids[k][0]), (centroids[k+1][1], centroids[k+1][0]), line_color, thickness)
                    cv2.circle(image, (centroids[k][1] + w_diff, centroids[k][0] + h_diff), rad, cent_color, -1)
                    cv2.circle(new_img, (centroids[k][1], centroids[k][0]), rad, cent_color, -1)
                    angle = np.arctan((point_i-cent_i)/(cent_j-point_j))
                    angle = np.degrees(angle)
                    angle_sut.append(angle)
                mean_cent_dist = np.round(np.mean(centroid_dist),6)
                var_cent_dist = np.round(np.var(centroid_dist),6)
                mean_sut_angle = np.round(np.mean(angle_sut),2)
                var_sut_angle = np.round(np.var(angle_sut),2)
                # print(var_cent_dist, var_sut_angle)
                image_metrics.append([var_cent_dist,var_sut_angle])
        pic1_loc = images_names.index(pic1)
        pic2_loc = images_names.index(pic2)
        if image_metrics[pic1_loc][0] < image_metrics[pic2_loc][0]:
            temp.append(1)
        else:
            temp.append(2)
        if image_metrics[pic1_loc][1] < image_metrics[pic1_loc][1]:
            temp.append(1)
        else:
            temp.append(2)
        file_content.append(temp)
    output_csv = sys.argv[3]
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(file_content)
    # print('Done')
        
        