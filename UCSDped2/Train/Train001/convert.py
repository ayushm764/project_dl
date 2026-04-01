import cv2
import os

image_folder = '.'

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".tif")])

frame = cv2.imread(images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()

print("Video created: output.mp4")