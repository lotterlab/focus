import os
import sys
import json
import boto3
import pandas as pd

csv_file = sys.argv[1]
session=boto3.Session()
s3=session.client('s3')
input_path=sys.argv[2]
in_l=input_path.split('/')
output_path=sys.argv[3]
out_l=output_path.split('/')

s3.upload_file(csv_file, 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_scores.csv")

# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
s3.upload_file('/root/OncoNet/L_CC.png', 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_L_CC.png")
s3.upload_file('/root/OncoNet/L_MLO.png', 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_L_MLO.png")
s3.upload_file('/root/OncoNet/R_CC.png', 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_R_CC.png")
s3.upload_file('/root/OncoNet/R_MLO.png', 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_R_MLO.png")

# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
s3.upload_file('/root/OncoNet/heatmap_L_CC.png', 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_heatmap_L_CC.png")
s3.upload_file('/root/OncoNet/heatmap_L_MLO.png', 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_heatmap_L_MLO.png")
s3.upload_file('/root/OncoNet/heatmap_R_CC.png', 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_heatmap_R_CC.png")
s3.upload_file('/root/OncoNet/heatmap_R_MLO.png', 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_heatmap_R_MLO.png")

# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
s3.upload_file('/root/OncoNet/heatmap_array.npy', 
               str(out_l[2]), 
               '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_heatmap_array.npy")

# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
# s3.upload_file('/root/OncoNet/heatmap_image_0.png', 
#                str(out_l[2]), 
#                '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_heatmap_image_0.png")
# s3.upload_file('/root/OncoNet/heatmap_image_1.png', 
#                str(out_l[2]), 
#                '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_heatmap_image_1.png")
# s3.upload_file('/root/OncoNet/heatmap_image_2.png', 
#                str(out_l[2]), 
#                '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_heatmap_image_2.png")
# s3.upload_file('/root/OncoNet/heatmap_image_3.png', 
#                str(out_l[2]), 
#                '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_heatmap_image_3.png")

# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
# s3.upload_file('/root/OncoNet/GlobalMaxPool_Output.pt', 
#                str(out_l[2]), 
#                '/'.join(out_l[3:]) + '/' + in_l[-1][:-4] + '/' + in_l[-1][:-4]  + "_GlobalMaxPool_Output.pt")
