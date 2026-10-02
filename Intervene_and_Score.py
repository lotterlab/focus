# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
import os
import stat
import math
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
    
def map_coordinates(image_name, area):
    coordinate_mapping=pd.read_csv('/Users/ramwala/Documents/coordinate_mapping.csv')
    coordinate_mapping=coordinate_mapping.rename(columns={'Bottom_Left':'Top_Left', 'Top_Right':'Bottom_Right'})

    name,x,y,width,height=annotations[image_name]['regions'][0]['shape_attributes'].values()
    
    # Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
    scale_factor=math.sqrt(area)
    # new_width=width*scale_factor
    # new_height=height*scale_factor

    # To modify the size of ground truth bounding boxes
    new_width=512
    new_height=512

    new_x=max(0, x+(width/2)-(new_width/2))
    new_y=max(0, y+(height/2)-(new_height/2))

    if (new_x+new_width > 1664):
        new_x=1664-new_width
    new_x=max(0, new_x)

    if (new_y+new_height > 2048):
        new_y=2048-new_height
    new_y=max(0, new_y)

    # To keep original ground truth bounding boxes
    # new_x=x
    # new_y=y
    # new_width=width
    # new_height=height

    top_left=(new_x,new_y)
    bottom_right=(new_x+new_width,new_y+new_height)
    coordinate_mapping['is_within']=coordinate_mapping.apply(within_range, axis=1, top_left=top_left, bottom_right=bottom_right)
    coordinate_mapping['is_within']=coordinate_mapping.apply(within_range, axis=1, top_left=top_left, bottom_right=bottom_right)
    intervened_coordinates=list(coordinate_mapping[coordinate_mapping['is_within']]['Intervened_Coordinates'])

    return int(intervened_coordinates[0].replace('(', '').replace(')', '').split(', ')[0]), \
        int(intervened_coordinates[-1].replace('(', '').replace(')', '').split(', ')[0]) + 1, \
        int(intervened_coordinates[0].replace('(', '').replace(')', '').split(', ')[1]), \
        int(intervened_coordinates[-1].replace('(', '').replace(')', '').split(', ')[1]) + 1

# high_scoring_exams=['70000004343', '70000000200', '70000001185', '70000006010', '70000001603', '70000001739', '70000001912', '70000002017', '70000002315', '70000003021', '70000003325', '70000003637', '70000010981', '70000011935', '70000012069', '70000015015', '70000015169', '70000016872', '70000017118', '70000018175', '70000018421', '70000019997', '70000020098', '70000020171', '70000024715', '70000026882', '70000028509', '70000028720', '70000028808', '70000029359', '70000030374', '70000030794', '70000031923', '70000031930', '70000033354', '70000034428', '70000034831', '70000035833', '70000036031', '70000036408', '70000037112', '70000037272', '70000038308', '70000039592', '70000042340', '70000043838', '70000044079', '70000045211', '70000045850', '70000046327', '70000046627', '70000047616', '70000051715', '70000051868', '70000053377', '70000054484', '70000055474', '70000056108', '70000056941', '70000059140', '70000059711', '70000059889', '70000061532', '70000061859', '70000062129', '70000062512', '70000064022', '70000064278', '70000064473', '70000065372', '70000065838', '70000066157', '70000068085', '70000069040', '70000069430', '70000071783', '70000072505', '70000072666', '70000072857']
# high_scoring_exams=['70000004343', '70000000200', '70000001185', '70000006010', '70000001603', '70000001739', '70000001912', '70000002017', '70000002315', '70000003021', '70000003325', '70000003637', '70000010981', '70000011935', '70000012069', '70000015015', '70000015169', '70000016872', '70000017118', '70000018175', '70000018421', '70000019997', '70000020098', '70000020171', '70000024715', '70000026882', '70000028509', '70000028720', '70000028808', '70000029359', '70000030374', '70000030794', '70000031923', '70000031930', '70000033354', '70000034428', '70000034831', '70000035833', '70000036031', '70000036408', '70000037112', '70000037272', '70000038308', '70000039592', '70000042340', '70000043838', '70000044079', '70000045211', '70000045850', '70000046327', '70000046627', '70000047616', '70000051715', '70000051868', '70000053377', '70000054484', '70000055474', '70000056108', '70000056941', '70000059140', '70000059711', '70000059889', '70000061532', '70000061859', '70000062129', '70000062512', '70000064022', '70000064278', '70000064473', '70000065372', '70000065838', '70000066157', '70000068085', '70000069040', '70000069430', '70000071783', '70000072505', '70000072666', '70000072857', '70000000299', '70000002234', '70000008951', '70000003969', '70000004127', '70000009850', '70000013753', '70000015993', '70000018470', '70000019895', '70000020016', '70000024702', '70000026618', '70000027291', '70000031763', '70000033076', '70000035249', '70000036558', '70000038582', '70000041139', '70000045096', '70000046081', '70000047726', '70000049642', '70000045462', '70000050139', '70000059434', '70000064251', '70000067322', '70000071403']
# print(len(high_scoring_exams))
# all_exams=['70000045850', '70000007836', '70000015993', '70000031763', '70000000087', '70000051868', '70000021096', '70000036408', '70000008117', '70000053855', '70000072847', '70000018470', '70000003969', '70000012538', '70000026882', '70000044062', '70000028613', '70000013887', '70000059711', '70000013493', '70000027291', '70000003963', '70000019895', '70000031923', '70000002017', '70000016872', '70000002419', '70000032577', '70000046627', '70000014144', '70000062129', '70000029359', '70000017832', '70000033354', '70000052935', '70000032064', '70000050139', '70000062866', '70000030374', '70000015770', '70000000975', '70000048571', '70000005075', '70000023293', '70000041276', '70000018421', '70000001911', '70000037990', '70000024038', '70000007414', '70000065838', '70000064278', '70000010981', '70000038308', '70000000299', '70000049642', '70000013753', '70000033076', '70000031490', '70000005947', '70000069040', '70000008951', '70000036558', '70000050284', '70000034258', '70000018703', '70000004482', '70000027863', '70000013254', '70000064251', '70000027170', '70000012654', '70000047009', '70000030754', '70000011141', '70000062512', '70000017118', '70000002315', '70000019172', '70000061532', '70000003985', '70000042280', '70000055474', '70000008143', '70000039301', '70000023985', '70000021138', '70000052065', '70000036014', '70000006475', '70000024521', '70000067720', '70000050712', '70000005788', '70000038582', '70000014784', '70000053274', '70000068085', '70000037272', '70000003637', '70000028509', '70000021558', '70000004756', '70000039516', '70000072140', '70000030319', '70000059981', '70000009196', '70000031930', '70000001912', '70000016174', '70000046327', '70000059140', '70000012069', '70000027999', '70000043391', '70000019997', '70000001214', '70000034831', '70000015667', '70000061859', '70000030939', '70000045303', '70000001739', '70000035833', '70000018426', '70000065372', '70000028808', '70000015169', '70000045211', '70000054484', '70000041342', '70000072505', '70000011935', '70000071783', '70000053377', '70000014967', '70000043939', '70000028444', '70000060850', '70000069430', '70000000200', '70000030794', '70000045462', '70000015015', '70000022225', '70000006967', '70000051715', '70000066157', '70000005960', '70000051513', '70000036263', '70000020462', '70000066020', '70000004281', '70000035249', '70000018399', '70000021213', '70000042397', '70000005868', '70000023271', '70000009216', '70000018175', '70000034428', '70000037112', '70000003325', '70000009850', '70000026618', '70000059434', '70000041139', '70000013559', '70000028517', '70000047616', '70000003559', '70000016843', '70000031927', '70000035949', '70000060820', '70000010064', '70000019174', '70000006010', '70000036031', '70000014172', '70000028720', '70000000428', '70000020098', '70000004343', '70000061783', '70000027912', '70000043854', '70000013065', '70000024715', '70000009043', '70000038422', '70000024702', '70000046081', '70000030632', '70000072220', '70000014771', '70000001863', '70000022593', '70000028682', '70000044079', '70000007629', '70000006114', '70000020171', '70000005652', '70000056108', '70000021209', '70000020016', '70000004127', '70000065863', '70000047726', '70000002234', '70000071403', '70000001185', '70000042340', '70000023987', '70000059889', '70000005655', '70000043838', '70000015585', '70000060065', '70000029471', '70000045235', '70000033011', '70000016262', '70000003021', '70000054241', '70000071797', '70000049654', '70000072857', '70000000326', '70000014871', '70000031360', '70000001603', '70000064473', '70000017054', '70000072666', '70000056941', '70000000529', '70000045096', '70000064022', '70000001402', '70000039592', '70000021710', '70000029684', '70000016091', '70000067322', '70000005284', '70000065750', '70000051784', '70000028782']
all_exams=['70000051513']
print(len(all_exams))

session = boto3.Session()
s3 = session.client('s3')
area=1 # in percentage (original: 1*A, 50% increase: 1.5*A, 100% increase: 2*A, 200% increase: 3*A)
region_type="Control" 
input_path = "s3://bcsc-standardized-inputs/DCM_Annotation_Dataset"
output_path = "s3://bcsc-scores/Explaining_Mirai/All_DCM_512_Control_Intervened_Scores"
annotation_path="s3://bcsc-standardized-inputs/UW_Mirai_Annotations"
annotation_l = annotation_path.split('/')

annotate_study_exam=pd.read_csv("/Users/ramwala/Documents/Complete_Annotation_Dataset_Manifest.csv", dtype=str)

input_output_uri=[]
for exam in all_exams:
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
    try:
        cc_present=False
        mlo_present=False 
        input_output_d={}
        input_output_d["InputS3URI"]=input_path + "/" + str(exam) + ".zip" 
        input_output_d["OutputS3URI"]=output_path
        for image_name in annotations.keys():
            if str(exam) in image_name:
                if (annotations[image_name]['regions'][0]['region_attributes']['Region'] == region_type):
                    laterality=image_name.split('_')[2]
                    projection=image_name.split('_')[3].split('.')[0]
                    if (laterality=='R' and projection=='CC'):
                        cc_present=True
                        y_cc_l, y_cc_u, x_cc_l, x_cc_u = map_coordinates(image_name, area)
                        input_output_d["CC_index"]=0
                        input_output_d["y_cc_l"]=y_cc_l
                        input_output_d["y_cc_u"]=y_cc_u
                        input_output_d["x_cc_l"]=x_cc_l
                        input_output_d["x_cc_u"]=x_cc_u
                    elif (laterality=='R' and projection=='MLO'):
                        mlo_present=True
                        y_mlo_l, y_mlo_u, x_mlo_l, x_mlo_u = map_coordinates(image_name, area)
                        input_output_d["MLO_index"]=1
                        input_output_d["y_mlo_l"]=y_mlo_l
                        input_output_d["y_mlo_u"]=y_mlo_u
                        input_output_d["x_mlo_l"]=x_mlo_l
                        input_output_d["x_mlo_u"]=x_mlo_u 
                    elif (laterality=='L' and projection=='CC'):
                        cc_present=True
                        y_cc_l, y_cc_u, x_cc_l, x_cc_u = map_coordinates(image_name, area)
                        input_output_d["CC_index"]=2
                        input_output_d["y_cc_l"]=y_cc_l
                        input_output_d["y_cc_u"]=y_cc_u
                        input_output_d["x_cc_l"]=x_cc_l
                        input_output_d["x_cc_u"]=x_cc_u
                    elif (laterality=='L' and projection=='MLO'): 
                        mlo_present=True
                        y_mlo_l, y_mlo_u, x_mlo_l, x_mlo_u = map_coordinates(image_name, area)
                        input_output_d["MLO_index"]=3
                        input_output_d["y_mlo_l"]=y_mlo_l
                        input_output_d["y_mlo_u"]=y_mlo_u
                        input_output_d["x_mlo_l"]=x_mlo_l
                        input_output_d["x_mlo_u"]=x_mlo_u 

        if (cc_present and mlo_present):
            input_output_uri.append(input_output_d)
    except:
        print("Unprocessed exam:", exam)
    
d={}
d["Mirai_agc_gpu.URIs"]=input_output_uri
os.chdir(os.getcwd())
with open('/Users/ramwala/Documents/AIExternalValidation_Git/AIExternalValidation/miniwdl_workflows/Mirai_XAI_workflow/Mirai_XAI_workflow.inputs.json', 'w') as json_file:
    json.dump(d, json_file)
os.chdir('/Users/ramwala/Documents/AIExternalValidation_Git/AIExternalValidation/miniwdl_workflows')
os.chmod('run_xai_mirai_miniwdl_workflow.sh', stat.S_IRWXU|stat.S_IRWXG|stat.S_IRWXO)
print(subprocess.call('./run_xai_mirai_miniwdl_workflow.sh'))