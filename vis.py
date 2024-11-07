import cv2
import numpy as np

# 파일 경로
image_path = "LC_GG_AP25_38714091_196_2021.png"
label_path = "LC_GG_AP25_38714091_196_2021.txt"

# 클래스별 색상 설정 (BGR 형식)
class_colors = {
    0: (0, 255, 0),    # 건물 - Green
    1: (0, 0, 255),    # 주차장 - Red
    2: (255, 0, 0),    # 도로 - Blue
    3: (0, 255, 255),  # 가로수 - Yellow
    4: (255, 255, 0),  # 논 - Cyan
    5: (255, 0, 255),  # 비닐하우스 - Magenta
    6: (192, 192, 192), # 밭 - Silver
    7: (128, 0, 128),  # 활엽수림 - Purple
    8: (0, 128, 128),  # 침엽수림 - Teal
    9: (128, 128, 0),  # 나지 - Olive
    10: (255, 165, 0), # 수역 - Orange
    11: (0, 255, 255), # 비대상지 - Light Yellow
}

# 이미지 로드
image = cv2.imread(image_path)
img_height, img_width = image.shape[:2]

# 디버그: 이미지 크기 출력
print(f"이미지 크기: {img_width}x{img_height}")

# 다각형 좌표가 이미지 내에 있는지 확인하는 함수
def is_within_image_bounds(points, img_width, img_height):
    for x, y in points:
        if x < 0 or x >= img_width or y < 0 or y >= img_height:
            return False
    return True

# 라벨 파일 읽기
with open(label_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])

        # 클래스별 색상 가져오기 (기본값은 흰색)
        color = class_colors.get(class_id, (255, 255, 255))  # 기본값: 흰색
        
        # 다각형 좌표 추출
        coords = list(map(float, parts[1:]))
        points = []
        for i in range(0, len(coords), 2):
            # 좌표를 이미지 크기에 맞게 변환
            x = int(coords[i] * img_width)
            y = int(coords[i + 1] * img_height)
            
            # 디버그: 변환된 좌표 출력
            print(f"정규화된 좌표: ({coords[i]}, {coords[i + 1]}) -> 변환된 좌표: ({x}, {y})")
            
            points.append((x, y))

        # 다각형이 이미지 내에 있는지 확인
        if is_within_image_bounds(points, img_width, img_height):
            # 다각형 그리기
            points = np.array([points], dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        else:
            print(f"경고: 클래스 {class_id}의 폴리곤이 이미지 경계를 벗어나 그리지 않았습니다.")

# 결과 이미지 저장
output_image_path = "output_polygon_overlay.png"
cv2.imwrite(output_image_path, image)
print(f"폴리곤 오버레이 결과 저장 완료: {output_image_path}")

