# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
import os
import stat
import time
import boto3
import json
import subprocess
import pandas as pd

def within_range(row, top_left, bottom_right):
    receptive_field_top_left = row['Top_Left'].replace('(', '').replace(')', '').split(', ')
    receptive_field_right_bottom = row['Bottom_Right'].replace('(', '').replace(')', '').split(', ')
    
    receptive_field_top_left_x, receptive_field_top_left_y = float(receptive_field_top_left[0]), float(receptive_field_top_left[1])
    receptive_field_right_bottom_x, receptive_field_right_bottom_y = float(receptive_field_right_bottom[0]), float(receptive_field_right_bottom[1])

    top_left_x, top_left_y = top_left[0], top_left[1]
    bottom_right_x, bottom_right_y = bottom_right[0], bottom_right[1]
    
    if ((receptive_field_right_bottom_x < top_left_x) or (receptive_field_right_bottom_y < top_left_y) or 
        (receptive_field_top_left_x > bottom_right_x) or (receptive_field_top_left_y > bottom_right_y)):
        return False
    else:
        return True
    
def map_coordinates(x, y, width, height):
    coordinate_mapping=pd.read_csv('/Users/ramwala/Documents/coordinate_mapping.csv')
    coordinate_mapping=coordinate_mapping.rename(columns={'Bottom_Left':'Top_Left', 'Top_Right':'Bottom_Right'})

    top_left=(x,y)
    bottom_right=(x+width,y+height)
    coordinate_mapping['is_within']=coordinate_mapping.apply(within_range, axis=1, top_left=top_left, bottom_right=bottom_right)
    coordinate_mapping['is_within']=coordinate_mapping.apply(within_range, axis=1, top_left=top_left, bottom_right=bottom_right)
    intervened_coordinates=list(coordinate_mapping[coordinate_mapping['is_within']]['Intervened_Coordinates'])

    return int(intervened_coordinates[0].replace('(', '').replace(')', '').split(', ')[0]), \
        int(intervened_coordinates[-1].replace('(', '').replace(')', '').split(', ')[0]) + 1, \
        int(intervened_coordinates[0].replace('(', '').replace(')', '').split(', ')[1]), \
        int(intervened_coordinates[-1].replace('(', '').replace(')', '').split(', ')[1]) + 1

high_scoring_exams=['70000006010', '70000002017', '70000003325', '70000012069', '70000016872', '70000018175', '70000029359', '70000031923', '70000031930', '70000035833', '70000037112', '70000039592', '70000043838', '70000046327', '70000046627', '70000047616', '70000051715', '70000051868', '70000055474', '70000059140', '70000059889', '70000061532', '70000061859', '70000062129', '70000064022', '70000065372']
print(len(high_scoring_exams))

region_type="Case" 
input_path = "s3://bcsc-standardized-inputs/DCM_Annotation_Dataset"
output_path = "s3://bcsc-scores/Explaining_Mirai/DCM_Occlusion_Maps_256_256"
annotation_path="s3://bcsc-standardized-inputs/UW_Mirai_Annotations"
annotation_l = annotation_path.split('/')

session = boto3.Session()
s3 = session.client('s3')

annotate_study_exam=pd.read_csv("/Users/ramwala/Documents/Complete_Annotation_Dataset_Manifest.csv", dtype=str)

patch_x=256
patch_y=256
stride_x=patch_x//2
stride_y=patch_y//2

for exam in high_scoring_exams:
    exam=str(exam)
    print(exam)
    study=list(annotate_study_exam[annotate_study_exam['Exam ID']==exam]["Study ID"])[0]
    for obj in session.resource('s3').Bucket(annotation_l[2]).objects.filter(Prefix='/'.join(annotation_l[3:])):
        key=obj.key
        if (key.endswith(str(study) + '.json')):
            s3.download_file(str(annotation_l[2]), 
                             '/'.join(annotation_l[3:]) + '/' + study + '.json', 
                             '/Users/ramwala/Downloads/Annotations_Temp/study_annotations.json')
    annotation_json='/Users/ramwala/Downloads/Annotations_Temp/study_annotations.json'
    with open(annotation_json) as annotation_file:
        annotations=json.load(annotation_file)
    for image_name in annotations.keys():
        if str(exam) in image_name:
            if (annotations[image_name]['regions'][0]['region_attributes']['Region'] == region_type):
                laterality=image_name.split('_')[2]
                projection=image_name.split('_')[3].split('.')[0]
                if (laterality=='R' and projection=='CC'):
                    image_index=0
                elif (laterality=='R' and projection=='MLO'):
                    image_index=1
                elif (laterality=='L' and projection=='CC'):
                    image_index=2
                elif (laterality=='L' and projection=='MLO'):   
                    image_index=3
                
                print(laterality, projection)
                input_output_uri=[]
                for x in range (0, 1664, stride_x):
                    for y in range (0, 2048, stride_y):
                        input_output_d = {}
                        InputS3URI = input_path + '/' + str(exam) + '.zip'
                        OutputS3URI = output_path + '/' + str(exam) + '/' + str(laterality) + "_" + str(projection) + '/' + str(x) + "_" + str(y)
                        try:
                            y_l, y_u, x_l, x_u = map_coordinates(x, y, patch_x, patch_y)
                            input_output_d["InputS3URI"]=InputS3URI
                            input_output_d["OutputS3URI"]=OutputS3URI
                            input_output_d["img_index"]=image_index
                            input_output_d["y_l"]=y_l
                            input_output_d["y_u"]=y_u
                            input_output_d["x_l"]=x_l
                            input_output_d["x_u"]=x_u
                            input_output_uri.append(input_output_d)
                        except:
                            continue
                d={}
                d["Mirai_agc_gpu.URIs"]=input_output_uri
                os.chdir(os.getcwd())
                with open('/Users/ramwala/Documents/AIExternalValidation_Git/AIExternalValidation/miniwdl_workflows/Mirai_XAI_workflow/Mirai_XAI_workflow.inputs.json', 'w') as json_file:
                    json.dump(d, json_file)
                os.chdir('/Users/ramwala/Documents/AIExternalValidation_Git/AIExternalValidation/miniwdl_workflows')
                os.chmod('run_xai_mirai_miniwdl_workflow.sh', stat.S_IRWXU|stat.S_IRWXG|stat.S_IRWXO)
                print(subprocess.call('./run_xai_mirai_miniwdl_workflow.sh'))
                input_output_uri=[]
                time.sleep(8*60) # Sleep for 8 minutes    