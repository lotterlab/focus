# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
import os
import sys
import csv
import numpy as np
from PIL import Image

exam_path=str(sys.argv[1])
print(exam_path)

complete_metadata=[]
for exam_name in os.listdir(exam_path):
    for image in os.listdir(os.path.join(exam_path, exam_name)):
        if (image.endswith('.png')):
            patient_id = exam_name
            exam_id = exam_name
            laterality = image.split("_")[0]
            view = image.split("_")[1].split(".")[0]
            image_metadata=[patient_id, exam_id, laterality, view, 
                            os.path.join(exam_path, exam_name, image), 0, 0, "test"]
            complete_metadata.append(image_metadata) 

# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
with open('/root/metadata.csv', 'w', newline='') as f:
    writer=csv.writer(f)
    field = ["patient_id","exam_id","laterality","view","file_path","years_to_cancer","years_to_last_followup","split_group"]
    writer.writerow(field)
    for image_metadata in complete_metadata:
        writer.writerow(image_metadata)