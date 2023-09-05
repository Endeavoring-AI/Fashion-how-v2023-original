import pandas as pd
from skimage import io, transform, color
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os
from tqdm import tqdm


df_ = pd.read_csv('./Dataset/info_etri20_emotion_train.csv')
base_path='./Dataset/Train/'

print(df_.image_name.size)

existing_data = pd.read_csv("./output_pd.csv")


#############################################################################
#                                                                           #
#  본인의 학습 이미지 증강 데이터 코드를 적용해서 list로 이미지를 반환해주세요   #
#                                                                           #
#############################################################################
def augmentation(orig_img):

    aug_list = []

    orig_img = Image.fromarray(orig_img)

    randomhorizontalflip = transforms.RandomHorizontalFlip(p=0.5)
    randomhorizontalflip_img = randomhorizontalflip(orig_img)

    randomrotation = transforms.RandomRotation(degrees=20)
    randomrotation_img = randomrotation(orig_img)

    blur = transforms.GaussianBlur(kernel_size=19, sigma=(0.1, 2.0))
    blur_img = blur(orig_img)

    aug_list.append(orig_img)
    aug_list.append(randomhorizontalflip_img)
    aug_list.append(randomrotation_img)
    aug_list.append(blur_img)

    return aug_list


new_data = []
cnt = 0

# train.csv의 행의 개수만큼 반복
for i in tqdm(range(0, df_.image_name.size-1), leave=True, desc="total"):
    cnt += 1
    cnt_ = 0
    sampl = df_.iloc[i]
    image = io.imread(base_path + sampl['image_name'])
    if image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    if image.shape == (1, 1, 3):
        image = np.squeeze(image)
    return_image = augmentation(image)

    first_check = False

    # 학습데이터 증강으로 만들어진 이미지 리스트 반환
    # 반환 받은 이미지 리스트로 반복문 실행
    # 이미지 리스트에 순번을 붙여서 증강 이미지 저장
    for j in return_image:

        cnt_ += 1
        file_name = sampl['image_name'][:-4] + "-{}".format(cnt) + "-{}".format(cnt_) + ".jpg"
        dir_name = file_name.split('/')
        dir_name[0]
        image_name = f"./test_data/{sampl['image_name'][:-4]}-{cnt}-{cnt_}.jpg"
        
        if first_check == False:
            file_name = sampl['image_name']
            image_name = f"./test_data/{sampl['image_name']}"
            first_check = True
        
        if not os.path.exists("/Users/DoD/Desktop/jiho/meow/sub-task1/test_data/" + dir_name[0]):
            os.makedirs("/Users/DoD/Desktop/jiho/meow/sub-task1/test_data/" + dir_name[0])
        
        j_ = np.array(j)
        cv2.imwrite(image_name, j_)
        new_row = {"image_name" : file_name,
                    "BBox_xmin" : sampl['BBox_xmin'],
                    "BBox_ymin" : sampl['BBox_ymin'],
                    "BBox_xmax" : sampl['BBox_xmax'],
                    "BBox_ymax" : sampl['BBox_ymax'],
                    "Daily" : sampl['Daily'],
                    "Gender" : sampl['Gender'],
                    "Embellishment" : sampl['Embellishment']}
        new_data.append(new_row)

new_df = pd.DataFrame(new_data)

#existing_data 는 테스트용으로 임시로 만들어 놓음. 실제로 원본에 붙여넣으려면 상위에 있는 df_를 existing_data에 넣어주면 된다.
combined_df = pd.concat([existing_data, new_df], ignore_index=True)

#최종적으로 csv 파일이 저장되며 추후 저장이 완료되면 _csv 를 .csv로 바꿔주면 된다.
combined_df.to_csv("combined_data_csv", index=False)