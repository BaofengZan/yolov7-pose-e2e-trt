import os
image_path = r'D:/LearningCodes/GithubRepo/yolov7-pose-e2e-trt/test_multi/images/train2017/'#修改为自己的路径
file = open('./train2017.txt', 'w')#修改为自己的路径

for filename in os.listdir(image_path):
 if(filename.endswith('.jpg')):
    print(filename)
    file.write(image_path+filename)
    file.write('\n')
    
