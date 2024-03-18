'''
cmd 명령어: python calibration.py --dir calibration_checkerboard/ --width 7 --height 10       (단위는 m)
도움: python calibration.py --help
'''

import numpy as np
import cv2
import os
import argparse

# 카메라 보정, 주어진 디렉토리에서 체커보드 이미지를 로드하고 처리하여 보정 행렬 및 왜곡 계수 반환
def calibrate(dirpath, square_size, width, height, visualize=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # 알고리즘에서 사용할 종료 조건 설정
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 체커보드 패턴의 실제 3D 좌표 준비
    objp = np.zeros((height*width, 3), np.float32)  # 3D 객체 포인트 배열을 초기화하고 설정, 체스보드의 각 격자점의 실제좌표
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []  # 체커보드 이미지에서 감지된 내부 코너의 3D좌표 저장 배열
    imgpoints = []  #                  ''                    2D좌표 저장 배열

    images = os.listdir(dirpath)  # 디렉토리에서 이미지 파일 로드

    for fname in images:
        img = cv2.imread(os.path.join(dirpath, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 이미지 로드한 후 그레이스케일로 변환

        # 체커보드의 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)  # 체커보드 코너 찾기

        # 코너를 세밀하게 조정하고 objpoints 와 imgpoints 배열에 추가
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
        
        # visualize 플래그가 True 일 때 체커보드 코너를 이미지에 그리기
        if visualize:
            cv2.imshow('img',img)
            cv2.waitKey(0)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  # 카메라 보정을 수행하고, 보정 행렬과 왜곡 계수 반환

    return [ret, mtx, dist, rvecs, tvecs]


if __name__ == '__main__':  # 명령행 인수를 처리하고 calibrate 함수를 호출하여 보정 수행
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Path to folder containing checkerboard images for calibration")  # --dir: 체커보드 이미지가 있는 디렉토리 경로 지정
    ap.add_argument("-w", "--width", type=int, help="Width of checkerboard (default=9)",  default=9)  # --width: 체커보드의 가로 내부 코너의 개수 지정
    ap.add_argument("-t", "--height", type=int, help="Height of checkerboard (default=6)", default=6)  # --height: 체커보드의 세로 내부 코너의 개수 지정
    ap.add_argument("-s", "--square_size", type=float, default=1, help="Length of one edge (in metres)")  # square_size: 체커보드 한 칸의 한 변의 길이(m)
    ap.add_argument("-v", "--visualize", type=str, default="False", help="To visualize each checkerboard image")  # visualize: 이미지 처리과정을 시각화할지 여부를 결정
    args = vars(ap.parse_args())
    
    dirpath = args['dir']
    # 2.4 cm == 0.024 m
    # square_size = 0.024
    square_size = args['square_size']

    width = args['width']
    height = args['height']

    if args["visualize"].lower() == "true":
        visualize = True
    else:
        visualize = False

    ret, mtx, dist, rvecs, tvecs = calibrate(dirpath, square_size, visualize=visualize, width=width, height=height)

    print(mtx)
    print(dist)

    np.save("calibration_matrix", mtx)  # 파일로 저장
    np.save("distortion_coefficients", dist)