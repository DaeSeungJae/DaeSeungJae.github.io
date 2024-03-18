'''
cmd 명렁어: python detect_aruco_video.py --type DICT_5X5_100 --camera True
            python detect_aruco_video.py --type DICT_5X5_100 --camera False --video test_video.mp4
도움: python detect_aruco_video.py --help
'''
# 오류 고치기 성공
import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys

# 명령행 인수 처리
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--camera", required=True, help="Set to True if using webcam")  # 웹캠 사용 여부 지정
ap.add_argument("-v", "--video", help="Path to the video file")  # 비디오 파일 경로 지정
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")  # 탐지할 아르코마커 태그의 유형 지정
ap.add_argument("-s", "--size", type=int, default=200, help="Size of the ArUCo tag")  # 아르코마커 태그의 크기 지정
args = vars(ap.parse_args())

# 웹캠에서 비디오 캡쳐하고 2초 대기
if args["camera"].lower() == "true":
	video = cv2.VideoCapture(0)
	time.sleep(2.0)
	
    # 비디오 파일 로드
else:
	if args["video"] is None:
		print("[Error] Video file location is not provided")
		sys.exit(1)

	video = cv2.VideoCapture(args["video"])
# 사용 가능한 아르코마커 태그 유형인지 확인
if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)
# 아르코마커 딕셔너리를 로드하고, 파라미터 설정한 후 마커 감지
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()

while True:
	ret, frame = video.read()  # 프레임 읽
	
	if ret is False:  # 프레임이 제대로 읽혔는지 나타내는 부울 값
		break


	h, w, _ = frame.shape  # 읽은 프레임의 높이와 너비

	width=1000  # 폭
	height = int(width*(h/w))  # 비율 유지하면서 새로운 높이 계산
	frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)  # 새로 계산된 폭과 높이로 재조정
	corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)  # 재조정된 프레임에서 아르코마커 탐지

    # 감지된 마커 화면에 표시
	detected_markers = aruco_display(corners, ids, rejected, frame)
	cv2.imshow("Image", detected_markers)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
	    break

cv2.destroyAllWindows()
video.release()