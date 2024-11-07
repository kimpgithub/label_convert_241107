import json
import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import shutil

class YOLOSegmentConverter:
    def __init__(self):
        # 원본 클래스 코드와 새로운 YOLO 인덱스 매핑
        self.class_mapping = {
            10: 0,  # 건물
            20: 1,  # 주차장
            30: 2,  # 도로
            40: 3,  # 가로수
            50: 4,  # 논
            55: 5,  # 비닐하우스
            60: 6,  # 밭
            71: 7,  # 활엽수림
            75: 8,  # 침엽수림
            80: 9,  # 나지
            95: 10 # 수역
            # 100: 11 # 비대상지
        }

    def normalize_coordinates(self, coords, img_width, img_height, origin_x, origin_y):
        """좌표를 이미지 크기에 맞게 정규화"""
        normalized = []
        for x, y in coords:
            rel_x = x - origin_x
            rel_y = origin_y - y  # y축 반전
    
            norm_x = max(0.0, min(1.0, rel_x / img_width))
            norm_y = max(0.0, min(1.0, rel_y / img_height))
    
            normalized.extend([norm_x, norm_y])
    
        return normalized

    def adjust_for_scaling(self, coords):
        """1/16 사이즈 면적을 1/4 크기로 조정"""
        adjusted_coords = []
        for i in range(0, len(coords), 2):
            adjusted_x = coords[i] * 4
            adjusted_y = coords[i + 1] * 4
            adjusted_coords.extend([adjusted_x, adjusted_y])
        return adjusted_coords

    def is_within_image_bounds(self, coords, img_width, img_height):
        """좌표가 이미지 내에 있는지 확인"""
        for i in range(0, len(coords), 2):
            x = coords[i]
            y = coords[i + 1]
            if x < 0 or x > img_width or y < 0 or y > img_height:
                return False
        return True

    def convert_to_yolo_format(self, meta_json_path, annotation_json_path, output_path):
        """JSON 파일들을 YOLO format으로 변환"""
        with open(meta_json_path, "r", encoding="utf-8") as meta_file:
            meta_data = json.load(meta_file)[0]
            img_width = meta_data['img_width']
            img_height = meta_data['img_height']
            
            coordinates = meta_data['coordinates'].split(',')
            origin_x = float(coordinates[0])
            origin_y = float(coordinates[1])

        with open(annotation_json_path, "r", encoding="utf-8") as ann_file:
            annotation_data = json.load(ann_file)
        
        yolo_lines = []
        
        for feature in annotation_data['features']:
            ann_cd = feature['properties']['ANN_CD']
            
            if ann_cd not in self.class_mapping:
                continue  # 비대상지 혹은 매핑되지 않은 클래스는 무시
            
            yolo_class_id = self.class_mapping[ann_cd]
            coords = feature['geometry']['coordinates'][0]
            
            normalized_coords = self.normalize_coordinates(coords, img_width, img_height, origin_x, origin_y)
            adjusted_coords = self.adjust_for_scaling(normalized_coords)
            
            if not self.is_within_image_bounds(adjusted_coords, img_width, img_height):
                continue
            
            coords_str = ' '.join([f'{x:.6f}' for x in adjusted_coords])
            yolo_line = f'{yolo_class_id} {coords_str}'
            yolo_lines.append(yolo_line)
        
        output_file = Path(output_path)
        output_file.write_text('\n'.join(yolo_lines))
        
        return len(yolo_lines)

    def process_dataset(self, image_folder, label_folder, meta_folder, output_image_train, output_image_val, output_label_train, output_label_val, split_ratio=0.8):
        image_files = list(Path(image_folder).glob("*.tif"))
        random.shuffle(image_files)
        split_index = int(len(image_files) * split_ratio)
        
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]
        
        for dataset, output_image_dir, output_label_dir in [(train_files, output_image_train, output_label_train), (val_files, output_image_val, output_label_val)]:
            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)
            
            for image_path in dataset:
                # 이미지 변환 및 저장
                image = Image.open(image_path)
                output_image_path = Path(output_image_dir) / (image_path.stem + ".png")
                image.convert("RGB").save(output_image_path)
                
                # 관련 JSON 파일 경로 설정
                annotation_json_path = Path(label_folder) / (image_path.stem + ".json")
                meta_json_path = Path(meta_folder) / (image_path.stem + "_META.json")
                
                # YOLO 라벨 파일 생성
                output_label_path = Path(output_label_dir) / (image_path.stem + ".txt")
                if meta_json_path.exists() and annotation_json_path.exists():
                    self.convert_to_yolo_format(meta_json_path, annotation_json_path, output_label_path)

def main():
    # 폴더 경로
    image_folder = 'TS_AP25_512'
    label_folder = 'TS_AP25_512_Json'
    meta_folder = 'TS_AP25_512_META'
    
    # 출력 경로
    output_image_train = 'aihub_dataset/images/train'
    output_image_val = 'aihub_dataset/images/val'
    output_label_train = 'aihub_dataset/labels/train'
    output_label_val = 'aihub_dataset/labels/val'
    
    # 변환기 초기화
    converter = YOLOSegmentConverter()
    
    # 데이터셋 처리
    converter.process_dataset(
        image_folder,
        label_folder,
        meta_folder,
        output_image_train,
        output_image_val,
        output_label_train,
        output_label_val
    )
    
    print("Dataset processing completed.")

if __name__ == '__main__':
    main()
