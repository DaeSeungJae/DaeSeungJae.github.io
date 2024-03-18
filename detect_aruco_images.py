'''
cmd 명령어: python detect_aruco_images.py --image tags/DICT_5X5_100_id_24.png --type DICT_5X5_100
            python detect_aruco_images.py --image Images/test_image_1.png --type DICT_5X5_100
도움: python detect_aruco_images.py --help
'''
# 오류 고치기 완료
import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys

# 명령행 인수 처리
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")  # 이미지 경로
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")  # 탐지할 아르코마커 태그 유형 지정
ap.add_argument("-s", "--size", type=int, default=200, help="Size of the ArUCo tag")  # 아르코마커 태그의 크기 지정
args = vars(ap.parse_args())



print("Loading image...")
image = cv2.imread(args["image"])  # 입력 이미지 앍기
# 입력 이미지 크기 조정
h,w,_ = image.shape
width=600
height = int(width*(h/w))
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


# 지정된 아르코마커 태그 유형이 opencv에서 지원되는지 확인
if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

# 아르코마커 딕셔너리 로드, 파라미터 설정 후 마커 감지
print("Detecting '{}' tags....".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

# 감지된 마커를 화면에 표시
detected_markers = aruco_display(corners, ids, rejected, image)
cv2.imshow("Image", detected_markers)

# 탐지된 아르코 마커가 표시된 이미지 저장
# cv2.imwrite("저장할 파일의 경로와 이름", 저장할 이미지 데이터)

cv2.waitKey(0)