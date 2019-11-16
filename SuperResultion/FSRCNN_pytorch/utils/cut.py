"""
    获取图片中某一部分
"""
# 导入库
import cv2
def cut(image_path,start =[80,380],end = [100,400],type = 1,line =1):

    image = cv2.imread(image_path)
    print(image.shape)

    start_x = start[0]
    start_y = start[1]

    end_x = end[0]
    end_y = end[1]
    if type == 2:
        start_x = start_x * 4
        start_y = start_y * 4
        end_x = end_x * 4
        end_y = end_y * 4

    small = image[start_x:end_x,start_y:end_y]
    cv2.imwrite("show2.jpg",small)

    cv2.rectangle(image,(start_y,start_x),(end_y,end_x),(0,0,255),line)
    cv2.imwrite("show.jpg",image)

if __name__ == '__main__':
    cut("../11_003.jpg",type=1,line=1)

