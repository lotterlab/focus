# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
import os
import stat
import time
import json
import boto3
import subprocess
import pandas as pd

input_path = "s3://bcsc-standardized-inputs/DCM_Annotation_Dataset"
output_path = "s3://bcsc-scores/Explaining_Mirai/DCM_Original_Scores_Heatmaps"

session = boto3.Session()
s3 = session.resource('s3')

input_l = input_path.split('/')
in_bucket = s3.Bucket(input_l[2])
prefix='/'.join(input_l[3:]) 

count = 0
input_output_uri=[]
for obj in in_bucket.objects.filter(Prefix=prefix):
    key=obj.key
    if (key.endswith('.zip')):
        exam=key.split('/')[-1]
        count += 1
        input_output_d={}
        input_output_d["InputS3URI"]=input_path + "/" + str(exam)
        input_output_d["OutputS3URI"] = output_path
        input_output_uri.append(input_output_d)
        if (count==32):
            d={}
            d["Mirai_agc_gpu.URIs"]=input_output_uri
            os.chdir(os.getcwd())
            with open('/Users/ramwala/Documents/AIExternalValidation_Git/AIExternalValidation/miniwdl_workflows/Mirai_XAI_workflow/Mirai_XAI_workflow.inputs.json', 'w') as json_file:
                json.dump(d, json_file)
            os.chdir('/Users/ramwala/Documents/AIExternalValidation_Git/AIExternalValidation/miniwdl_workflows')
            os.chmod('run_xai_mirai_miniwdl_workflow.sh', stat.S_IRWXU|stat.S_IRWXG|stat.S_IRWXO)
            print(subprocess.call('./run_xai_mirai_miniwdl_workflow.sh'))
            count = 0
            input_output_uri=[]
            time.sleep(4*60) # Sleep for 4 minutes


# Run for unprocessed exams
input_path = "s3://bcsc-standardized-inputs/DCM_Annotation_Dataset"
output_path = "s3://bcsc-scores/Explaining_Mirai/DCM_Original_Scores_Heatmaps"

session = boto3.Session()
s3 = session.resource('s3')
input_l = input_path.split('/')
in_bucket = s3.Bucket(input_l[2])
prefix='/'.join(input_l[3:]) 

input_exams=[]
for obj in in_bucket.objects.filter(Prefix=prefix):
    key=obj.key
    if (key.endswith('.zip')):
        exam=key.split('/')[-1]
        input_exams.append(exam.split('.')[0])

session = boto3.Session()
s3 = session.resource('s3')
output_l = output_path.split('/')
out_bucket = s3.Bucket(output_l[2])
prefix='/'.join(output_l[3:]) 
outputs=set()
for obj in out_bucket.objects.filter(Prefix=prefix):
    key=obj.key
    outputs.add(key.split('/')[2])
unprocessed_exams=[exam for exam in input_exams if exam not in outputs]
print(len(unprocessed_exams))

session = boto3.Session()
s3 = session.resource('s3')

input_l = input_path.split('/')
in_bucket = s3.Bucket(input_l[2])
prefix='/'.join(input_l[3:]) 

input_output_uri=[]
for obj in in_bucket.objects.filter(Prefix=prefix):
    key=obj.key
    if (key.endswith('.zip')):
        exam=key.split('/')[-1]
        if exam.split('.')[0] in unprocessed_exams:
            input_output_d={}
            input_output_d["InputS3URI"]=input_path + "/" + str(exam)
            input_output_d["OutputS3URI"] = output_path
            input_output_uri.append(input_output_d)
d={}
d["Mirai_agc_gpu.URIs"]=input_output_uri
os.chdir(os.getcwd())
with open('/Users/ramwala/Documents/AIExternalValidation_Git/AIExternalValidation/miniwdl_workflows/Mirai_XAI_workflow/Mirai_XAI_workflow.inputs.json', 'w') as json_file:
    json.dump(d, json_file)
os.chdir('/Users/ramwala/Documents/AIExternalValidation_Git/AIExternalValidation/miniwdl_workflows')
os.chmod('run_xai_mirai_miniwdl_workflow.sh', stat.S_IRWXU|stat.S_IRWXG|stat.S_IRWXO)
print(subprocess.call('./run_xai_mirai_miniwdl_workflow.sh'))