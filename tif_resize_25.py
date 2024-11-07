import rasterio
import torch
import torch.nn.functional as F
import numpy as np
from torch import amp

def resize_tile(tile, new_height, new_width, device):
    """단일 타일을 리사이즈합니다."""
    with amp.autocast('cuda'):
        tensor = torch.from_numpy(tile).float().to(device)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        
        resized = F.interpolate(
            tensor,
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=True
        )
        
        result = resized.squeeze().cpu().numpy()  # squeeze() 추가
        del tensor, resized
        torch.cuda.empty_cache()
        return result

def create_blending_mask(height, width, overlap):
    """블렌딩을 위한 가중치 마스크를 생성합니다."""
    mask = np.ones((height, width), dtype=np.float32)
    
    # 오버랩 영역에 대한 그라데이션 생성
    for i in range(overlap):
        weight = i / overlap
        mask[:, i] = weight
        mask[:, -(i+1)] = weight
        mask[i, :] = weight
        mask[-(i+1), :] = weight
    
    return mask

def resize_tiff_pytorch(input_path, output_path, target_pixel_size_cm=25, tile_size=5000, overlap=200):
    """
    PyTorch GPU를 사용하여 TIFF 파일을 오버랩된 타일 단위로 처리하여 리사이즈합니다.
    
    Parameters:
    input_path (str): 입력 TIFF 파일 경로
    output_path (str): 출력 TIFF 파일 경로
    target_pixel_size_cm (float): 목표 픽셀 크기 (cm 단위, 기본값: 25cm)
    tile_size (int): 처리할 타일의 크기 (기본값: 5000)
    overlap (int): 타일 간 오버랩 크기 (기본값: 200)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with rasterio.open(input_path) as dataset:
        # 기존 픽셀 크기 (3.8 cm)와 목표 픽셀 크기 (25 cm) 기준으로 scale_factor 계산
        original_pixel_size_cm = 3.8
        scale_factor = original_pixel_size_cm / target_pixel_size_cm
        new_width = int(dataset.width * scale_factor)
        new_height = int(dataset.height * scale_factor)
        print(f"Original size: {dataset.width}x{dataset.height}")
        print(f"New size: {new_width}x{new_height}")
        
        # 메타데이터 설정
        meta = dataset.meta.copy()
        new_transform = rasterio.transform.from_bounds(
            *dataset.bounds, new_width, new_height
        )
        meta.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })

        # 결과를 저장할 배열 초기화
        with rasterio.open(output_path, 'w', **meta) as dest:
            for band_idx in range(dataset.count):
                print(f"Processing band {band_idx + 1}/{dataset.count}")
                
                # 결과와 가중치 누적을 위한 배열
                resized_band = np.zeros((new_height, new_width), dtype=np.float32)
                weight_sum = np.zeros((new_height, new_width), dtype=np.float32)
                
                # 타일 단위로 처리
                for y in range(0, dataset.height, tile_size - overlap):
                    for x in range(0, dataset.width, tile_size - overlap):
                        # 현재 타일의 크기 계산 (오버랩 포함)
                        current_tile_size_y = min(tile_size, dataset.height - y + overlap)
                        current_tile_size_x = min(tile_size, dataset.width - x + overlap)
                        
                        # 타일의 실제 범위 계산
                        y_end = min(y + current_tile_size_y, dataset.height)
                        x_end = min(x + current_tile_size_x, dataset.width)
                        
                        print(f"Processing tile at ({x}, {y}) to ({x_end}, {y_end})")
                        
                        # 타일 읽기
                        tile = dataset.read(
                            band_idx + 1,
                            window=((y, y_end), (x, x_end))
                        )
                        
                        # 현재 타일의 새로운 크기 계산
                        current_new_tile_size_y = int((y_end - y) * scale_factor)
                        current_new_tile_size_x = int((x_end - x) * scale_factor)
                        
                        # 타일 리사이즈
                        resized_tile = resize_tile(
                            tile, 
                            current_new_tile_size_y, 
                            current_new_tile_size_x,
                            device
                        )
                        
                        # 블렌딩 마스크 생성
                        mask = create_blending_mask(
                            current_new_tile_size_y,
                            current_new_tile_size_x,
                            int(overlap * scale_factor)
                        )
                        
                        # 결과 배열에 블렌딩하여 추가
                        new_y = int(y * scale_factor)
                        new_x = int(x * scale_factor)
                        new_y_end = new_y + current_new_tile_size_y
                        new_x_end = new_x + current_new_tile_size_x
                        
                        print(f"Resized tile shape: {resized_tile.shape}")
                        print(f"Mask shape: {mask.shape}")
                        print(f"Target shape: {new_y_end-new_y}x{new_x_end-new_x}")
                        
                        resized_band[new_y:new_y_end, new_x:new_x_end] += resized_tile * mask
                        weight_sum[new_y:new_y_end, new_x:new_x_end] += mask
                
                # 가중치로 정규화
                weight_sum = np.maximum(weight_sum, 1e-6)  # 0으로 나누기 방지
                resized_band = resized_band / weight_sum
                
                # 밴드 저장
                dest.write(resized_band.astype(meta['dtype']), band_idx + 1)
                print(f"Band {band_idx + 1} completed")

if __name__ == "__main__":
    input_file = "gmsh_up.tif"
    output_file = "gmsh_up_resized_25cm.tif"
    
    resize_tiff_pytorch(input_file, output_file, target_pixel_size_cm=25)
    print(f"GPU-accelerated resizing completed. New file saved as: {output_file}")
