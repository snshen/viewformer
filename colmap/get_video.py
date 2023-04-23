from pathlib import Path
from pytube import YouTube
import cv2

imgpath = Path('custom/images')

url = "https://youtu.be/kacyaEXqVhs"
yt = YouTube(url)
yt = yt.streams.filter(res="360p").first()
yt.download()

cap = cv2.VideoCapture("557 Marksbury Road Pickering Open House Video Tour.mp4")
frame_num= 0
while (cap.isOpened()):
    ret, frame = cap.read()

    if frame_num % 10 == 0:
        target = str(imgpath / f'{frame_num}.jpg')
        cv2.imwrite(target, frame)

    frame_num+=1

    if frame_num > 60*30:
        break

cap.release()
print("done")