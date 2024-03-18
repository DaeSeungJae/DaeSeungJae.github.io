'''
cmd 명령어: python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
도움: python pose_estimation.py --help
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

def draw_axis(frame, rvec, tvec, camera_matrix, distortion_coefficients, axis_length=0.05):  # 이미지, 회전벡터, 변위벡터, 내부 파라미터 행렬, 왜곡 계수, 축의 길이
    axis_points_3d = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])  # 3차원
    axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, distortion_coefficients)  # 2차원

    axis_points_2d = np.array(axis_points_2d, dtype=np.int32)  # 좌표값을 정수형으로 변환
    
    # 선 그리기
    frame = cv2.line(frame, tuple(axis_points_2d[0].ravel()), tuple(axis_points_2d[1].ravel()), (0, 0, 255), 1)  # x축 (red)
    frame = cv2.line(frame, tuple(axis_points_2d[0].ravel()), tuple(axis_points_2d[2].ravel()), (0, 255, 0), 1)  # y축 (green)
    frame = cv2.line(frame, tuple(axis_points_2d[0].ravel()), tuple(axis_points_2d[3].ravel()), (255, 0, 0), 1)  # z축 (blue)

    return frame

# 입력된 프레임에서 아르코마커의 위치와 방향을 추정하고 이를 시각화하여 프레임에 그림
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) > 0:
        for i, marker_id in enumerate(ids):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)

            if rvec is not None and tvec is not None:
                # Draw axis
                frame = draw_axis(frame, rvec, tvec, matrix_coefficients, distortion_coefficients, axis_length=0.05)

                # Draw marker ID
                frame = cv2.putText(frame, str(marker_id), tuple(corners[i][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw marker outline
                frame = cv2.polylines(frame, [np.int32(corners[i])], True, (0, 255, 0), 2)

    return frame   

if __name__ == '__main__':

    ap = argparse.ArgumentParser()  # 명령행 인수 처리
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")  # 옵션 이름, 옵션 이름, 필수 인수, 값 설명 제공
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())  # 명령형 인수를 파싱하고 결과를 딕셔너리로 변환하여 args에 저장

    
    if ARUCO_DICT.get(args["type"], None) is None:  # 사용자가 지정한 아르코마커 유형이 지원되는지 확인
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    # 카메라 행렬과 왜곡 계수 로드
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:  # 웹캠에서 프레임을 연속적으로 읽고 처리
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)  # 아르코마커 추정하고 시각화된 결과 반환

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()