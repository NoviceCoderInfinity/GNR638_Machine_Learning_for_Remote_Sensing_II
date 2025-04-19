# this file generates a submission file for a Kaggle competition
import base64
import cv2
from tqdm import tqdm  # <- this is the correct way
import os
import pandas as pd  # You also use pd but didn't import pandas

output_dir = './outputs'
test_hr_dir = './outputs'
output_csv_file='submission.csv'
def encode_image_to_base64(image_path):
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

encoded_images = []
for file_name in tqdm(os.listdir(output_dir)):
    file_path = os.path.join(test_hr_dir, file_name)
    encoded_image = encode_image_to_base64(file_path)
    encoded_images.append(pd.DataFrame({'id': file_name, 'Encoded_Image': encoded_image}, index = [0]))

# Create a DataFrame
df_encoded = pd.concat(encoded_images).reset_index(drop=True)
df_encoded.to_csv(output_csv_file, index=False)
