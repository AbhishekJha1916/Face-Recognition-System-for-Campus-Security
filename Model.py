import os
import cv2
import numpy as np
from PIL import Image

names = [] 
path = []

for users in os.listdir("dataset"):
    names.append(users)

for name in names:
    for image in os.listdir("dataset/{}".format(name)):
        path_string = os.path.join("dataset/{}".format(name), image)
        path.append(path_string)


faces = []
ids = []

for img_path in path:
    image = Image.open(img_path).convert("L")

    imgNp = np.array(image, "uint8")

    #Accessing the id from the dataset
    id = int(img_path.split('\\')[1].split("_")[0])
    # id1 = id.split("/")[1] 
    print(id)
#     # for i in range(len(img_path)):
#     #     print(img_path)

    faces.append(imgNp)
    ids.append(id)

ids = np.array(ids)

print("Created faces and names Numpy Arrays")
print("Initializing the Classifier")

trainer = cv2.face.LBPHFaceRecognizer_create()
trainer.train(faces, ids)
trainer.write("training.yml")

print("Training Done")