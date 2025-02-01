import os
import cv2
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from utils.except_dir import cust_listdir
def get_video_metadata(video_path, category, benchmark):
    """Extract metadata from a video file."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    # Extract metadata
    video_name = os.path.basename(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = f"{frame_width}x{frame_height}"
    duration_seconds = frame_count / fps if fps > 0 else 0
    aspect_ratio = round(frame_width / frame_height, 2) if frame_height > 0 else 0
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    file_format = os.path.splitext(video_name)[1].lower()
    cap.release()
    
    return {
        "video_name": video_name,
        "resolution": resolution,
        "video_duration": f"{duration_seconds // 60:.0f}:{duration_seconds % 60:.0f}",
        "category": category,
        "benchmark": benchmark,
        "duration_seconds": duration_seconds,
        "total_frames": frame_count,
        "file_format": file_format,
        "file_size_mb": round(file_size, 2),
        "aspect_ratio": aspect_ratio,
        "fps": fps
    }

def process_videos_in_directory(root_dir):
    """Process all videos in the given directory structure."""
    video_metadata_list = []
    
    # 벤치마크 폴더들을 순회
    for benchmark in cust_listdir(root_dir):
        benchmark_path = os.path.join(root_dir, benchmark)
        if not os.path.isdir(benchmark_path):
            continue
        
        # dataset 폴더 경로
        dataset_path = os.path.join(benchmark_path, "dataset")
        if not os.path.isdir(dataset_path):
            continue
            
        # dataset 폴더 안의 카테고리 폴더들을 순회
        for category in cust_listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path):
                continue
                
            # 각 카테고리 폴더 안의 비디오 파일들을 처리
            for file in cust_listdir(category_path):
                file_path = os.path.join(category_path, file)
                
                if file_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', 'MOV')):
                    metadata = get_video_metadata(file_path, category, benchmark)
                    if metadata:
                        video_metadata_list.append(metadata)
    # df = pd.DataFrame(video_metadata_list)
    # df.to_csv('sample.csv', index=False)
    return pd.DataFrame(video_metadata_list)

