import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import numpy as np
logging.basicConfig(level=logging.INFO)

class BenchChecker:
    def __init__(self, base_path: str):
        """Initialize BenchChecker with base assets path.
        
        Args:
            base_path (str): Base path to assets directory containing benchmark folders
        """
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        
    def check_benchmark_exists(self, benchmark_name: str) -> bool:
        """Check if benchmark folder exists."""
        benchmark_path = self.base_path / benchmark_name
        exists = benchmark_path.exists() and benchmark_path.is_dir()
        if exists:
            self.logger.info(f"Found benchmark directory: {benchmark_name}")
        else:
            self.logger.error(f"Benchmark directory not found: {benchmark_name}")
        return exists
        
    def get_video_list(self, benchmark_name: str) -> List[str]:
        """Get list of videos from benchmark's dataset directory. Return empty list if no videos found."""
        dataset_path = self.base_path / benchmark_name / "dataset"
        videos = []
        
        if not dataset_path.exists():
            self.logger.info(f"Dataset directory exists but no videos found for {benchmark_name}")
            return videos  # 빈 리스트 반환
            
        # Recursively find all .mp4 files
        for category in dataset_path.glob("*"):
            if category.is_dir():
                for video_file in category.glob("*.mp4"):
                    videos.append(video_file.stem)
                        
        self.logger.info(f"Found {len(videos)} videos in {benchmark_name} dataset")
        return videos
        
    def check_model_exists(self, benchmark_name: str, model_name: str) -> bool:
        """Check if model directory exists in benchmark's models directory."""
        model_path = self.base_path / benchmark_name / "models" / model_name
        exists = model_path.exists() and model_path.is_dir()
        if exists:
            self.logger.info(f"Found model directory: {model_name}")
        else:
            self.logger.error(f"Model directory not found: {model_name}")
        return exists
        
    def check_cfg_files(self, benchmark_name: str, model_name: str, cfg_prompt: str) -> Tuple[bool, bool]:
        """Check if CFG files/directories exist in both benchmark and model directories."""
        # Check benchmark CFG json
        benchmark_cfg = self.base_path / benchmark_name / "CFG" / f"{cfg_prompt}.json"
        benchmark_cfg_exists = benchmark_cfg.exists() and benchmark_cfg.is_file()
        
        # Check model CFG directory
        model_cfg = self.base_path / benchmark_name / "models" / model_name / "CFG" / cfg_prompt
        model_cfg_exists = model_cfg.exists() and model_cfg.is_dir()
        
        if benchmark_cfg_exists:
            self.logger.info(f"Found benchmark CFG file: {cfg_prompt}.json")
        else:
            self.logger.error(f"Benchmark CFG file not found: {cfg_prompt}.json")
            
        if model_cfg_exists:
            self.logger.info(f"Found model CFG directory: {cfg_prompt}")
        else:
            self.logger.error(f"Model CFG directory not found: {cfg_prompt}")
            
        return benchmark_cfg_exists, model_cfg_exists
    def check_vector_files(self, benchmark_name: str, model_name: str, video_list: List[str]) -> bool:
        """Check if video vectors match with dataset."""
        vector_path = self.base_path / benchmark_name / "models" / model_name / "vector" / "video"
        
        # 비디오가 없는 경우는 무조건 False
        if not video_list:
            self.logger.error("No videos found in dataset - cannot proceed")
            return False
        
        # 벡터 디렉토리가 있는지 확인
        if not vector_path.exists():
            self.logger.error("Vector directory doesn't exist")
            return False
                
        # 벡터 파일 리스트 가져오기
        # vector_files = [f.stem for f in vector_path.glob("*.npy")]
        vector_files = [f.stem for f in vector_path.rglob("*.npy")]
        
        missing_vectors = set(video_list) - set(vector_files)
        extra_vectors = set(vector_files) - set(video_list)
        
        if missing_vectors:
            self.logger.error(f"Missing vectors for videos: {missing_vectors}")
            return False
        if extra_vectors:
            self.logger.error(f"Extra vectors found: {extra_vectors}")
            return False
                
        self.logger.info(f"Vector status: videos={len(video_list)}, vectors={len(vector_files)}")
        return len(video_list) == len(vector_files)
    
    def check_metrics_file(self, benchmark_name: str, model_name: str, cfg_prompt: str) -> bool:
        """Check if overall_metrics.json exists in the model's CFG/metrics directory."""
        metrics_path = self.base_path / benchmark_name / "models" / model_name / "CFG" / cfg_prompt / "metric" / "overall_metrics.json"
        exists = metrics_path.exists() and metrics_path.is_file()
        
        if exists:
            self.logger.info(f"Found overall metrics file for {model_name}")
        else:
            self.logger.error(f"Overall metrics file not found for {model_name}")
        return exists
    
    def check_benchmark(self, benchmark_name: str, model_name: str, cfg_prompt: str) -> Dict[str, bool]:
        """
        Perform all benchmark checks and return status.
        """
        status = {
            'benchmark_exists': False,
            'model_exists': False,
            'cfg_files_exist': False,
            'vectors_match': False,
            'metrics_exist': False
        }
        
        # Check benchmark directory
        status['benchmark_exists'] = self.check_benchmark_exists(benchmark_name)
        if not status['benchmark_exists']:
            return status
                
        # Get video list
        video_list = self.get_video_list(benchmark_name)
        
        # Check model directory
        status['model_exists'] = self.check_model_exists(benchmark_name, model_name)
        if not status['model_exists']:
            return status
                
        # Check CFG files
        benchmark_cfg, model_cfg = self.check_cfg_files(benchmark_name, model_name, cfg_prompt)
        status['cfg_files_exist'] = benchmark_cfg and model_cfg
        if not status['cfg_files_exist']:
            return status

        # Check vectors
        status['vectors_match'] = self.check_vector_files(benchmark_name, model_name, video_list)
        
        # Check metrics file (only if vectors match)
        if status['vectors_match']:
            status['metrics_exist'] = self.check_metrics_file(benchmark_name, model_name, cfg_prompt)
            
        return status

    def get_benchmark_status(self, check_status: Dict[str, bool]) -> str:
        """Determine which execution path to take based on check results."""
        basic_checks = ['benchmark_exists', 'model_exists', 'cfg_files_exist']
        if not all(check_status[check] for check in basic_checks):
            return "cannot_execute"
        if check_status['vectors_match'] and check_status['metrics_exist']:
            return "all_passed"
        elif not check_status['vectors_match']:
            return "no_vectors"
        else:  # vectors exist but no metrics
            return "no_metrics"

# Example usage
if __name__ == "__main__":
    
    bench_checker = BenchChecker("assets")
    status = bench_checker.check_benchmark(
        benchmark_name="huggingface_benchmarks_dataset",
        model_name="MSRVTT",
        cfg_prompt="topk"
    )
    
    execution_path = bench_checker.get_benchmark_status(status)
    print(f"Checks completed. Execution path: {execution_path}")
    print(f"Status: {status}")