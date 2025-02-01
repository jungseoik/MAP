from pia_bench.checker.bench_checker import BenchChecker
from pia_bench.checker.sheet_checker import SheetChecker
from pia_bench.event_alarm import EventDetector
from pia_bench.metric import MetricsEvaluator
from sheet_manager.sheet_crud.sheet_crud import SheetManager
from pia_bench.bench import PiaBenchMark
from dotenv import load_dotenv
from typing import Optional, List , Dict
import os
load_dotenv()
import numpy as np
from typing import Dict, Tuple
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from sheet_manager.sheet_checker.sheet_check import SheetChecker
from sheet_manager.sheet_crud.sheet_crud import SheetManager
from pia_bench.checker.bench_checker import BenchChecker

logging.basicConfig(level=logging.INFO)

from enviroments.config import BASE_BENCH_PATH

@dataclass
class PipelineConfig:
    """파이프라인 설정을 위한 데이터 클래스"""
    model_name: str
    benchmark_name: str
    cfg_target_path: str
    base_path: str = BASE_BENCH_PATH

class BenchmarkPipelineStatus:
    """파이프라인 상태 및 결과 관리"""
    def __init__(self):
        self.sheet_status: Tuple[bool, bool] = (False, False)  # (model_added, benchmark_exists)
        self.bench_status: Dict[str, bool] = {}
        self.bench_result: str = ""
        self.current_stage: str = "not_started"
        
    def is_success(self) -> bool:
        """전체 파이프라인 성공 여부"""
        return (not self.sheet_status[0]  # 모델이 이미 존재하고
                and self.sheet_status[1]  # 벤치마크가 존재하고
                and self.bench_result == "all_passed")  # 벤치마크 체크도 통과
                
    def __str__(self) -> str:
        return (f"Current Stage: {self.current_stage}\n"
                f"Sheet Status: {self.sheet_status}\n"
                f"Bench Status: {self.bench_status}\n"
                f"Bench Result: {self.bench_result}")

class BenchmarkPipeline:
    """벤치마크 실행을 위한 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.status = BenchmarkPipelineStatus()
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.cfg_prompt = os.path.splitext(os.path.basename(self.config.cfg_target_path))[0]
        
        # Initialize checkers
        self.sheet_manager = SheetManager()
        self.sheet_checker = SheetChecker(self.sheet_manager)
        self.bench_checker = BenchChecker(self.config.base_path)

        self.bench_result_dict = None
        
    def run(self) -> BenchmarkPipelineStatus:
        """전체 파이프라인 실행"""
        try:
            self.status.current_stage = "sheet_check"
            proceed = self._check_sheet()
            
            if not proceed:
                self.status.current_stage = "completed_no_action_needed"
                self.logger.info("벤치마크가 이미 존재하여 추가 작업이 필요하지 않습니다.")
                return self.status
                
            self.status.current_stage = "bench_check"
            if not self._check_bench():
                return self.status
                
            self.status.current_stage = "execution"
            self._execute_based_on_status()
            
            self.status.current_stage = "completed"
            return self.status
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 에러 발생: {str(e)}")
            self.status.current_stage = "error"
            return self.status
            
    def _check_sheet(self) -> bool:
        """구글 시트 상태 체크"""
        self.logger.info("시트 상태 체크 시작")
        model_added, benchmark_exists = self.sheet_checker.check_model_and_benchmark(
            self.config.model_name,
            self.config.benchmark_name
        )
        self.status.sheet_status = (model_added, benchmark_exists)
        
        if model_added:
            self.logger.info("새로운 모델이 추가되었습니다")
        if not benchmark_exists:
            self.logger.info("벤치마크 측정이 필요합니다")
            return True  # 벤치마크 측정이 필요한 경우만 다음 단계로 진행
            
        self.logger.info("이미 벤치마크가 존재합니다. 파이프라인을 종료합니다.")
        return False  # 벤치마크가 이미 있으면 여기서 중단
        
    def _check_bench(self) -> bool:
        """로컬 벤치마크 환경 체크"""
        self.logger.info("벤치마크 환경 체크 시작")
        self.status.bench_status = self.bench_checker.check_benchmark(
            self.config.benchmark_name,
            self.config.model_name,
            self.cfg_prompt
        )
        self.status.bench_result = self.bench_checker.get_benchmark_status(
            self.status.bench_status
        )
        
        # no bench 상태 벤치를 돌린적이 없음 폴더구조도 없음
        if self.status.bench_result == "no bench":
            self.logger.error("벤치마크 실행에 필요한 기본 폴더구조가 없습니다.")
            return True
            
        return True  # 그 외의 경우만 다음 단계로 진행
        
    def _execute_based_on_status(self):
        """상태에 따른 실행 로직"""
        if self.status.bench_result == "all_passed":
            self._execute_full_pipeline()
        elif self.status.bench_result == "no_vectors":
            self._execute_vector_generation()
        elif self.status.bench_result == "no_metrics":
            self._execute_metrics_generation()
        else:
            self._execute_vector_generation()
            self.logger.warning("폴더구조가 없습니다")
            
    def _execute_full_pipeline(self):
        """모든 조건이 충족된 경우의 실행 로직"""
        self.logger.info("전체 파이프라인 실행 중...")
        pia_benchmark = PiaBenchMark(
                                benchmark_path  = f"{BASE_BENCH_PATH}/{self.config.benchmark_name}" ,
                                model_name=self.config.model_name, 
                                cfg_target_path= self.config.cfg_target_path , 
                                token=self.access_token )
        pia_benchmark.preprocess_structure()
        print("Categories identified:", pia_benchmark.categories)
        metric = MetricsEvaluator(pred_dir=pia_benchmark.alram_path, 
                        label_dir=pia_benchmark.dataset_path, 
                        save_dir=pia_benchmark.metric_path)
        
        self.bench_result_dict = metric.evaluate()

    def _execute_vector_generation(self):
        """벡터 생성이 필요한 경우의 실행 로직"""
        self.logger.info("벡터 생성 중...")
        # 구현 필요

        pia_benchmark = PiaBenchMark(
                                benchmark_path  = f"{BASE_BENCH_PATH}/{self.config.benchmark_name}" ,
                                model_name=self.config.model_name, 
                                cfg_target_path= self.config.cfg_target_path , 
                                token=self.access_token )
        pia_benchmark.preprocess_structure()
        pia_benchmark.preprocess_label_to_csv()  
        print("Categories identified:", pia_benchmark.categories)

        pia_benchmark.extract_visual_vector()

        detector = EventDetector(config_path=self.config.cfg_target_path, 
                                 model_name=self.config.model_name , 
                                 token=pia_benchmark.token)
        detector.process_and_save_predictions(pia_benchmark.vector_video_path, 
                                            pia_benchmark.dataset_path, 
                                            pia_benchmark.alram_path)
        metric = MetricsEvaluator(pred_dir=pia_benchmark.alram_path, 
                                label_dir=pia_benchmark.dataset_path, 
                                save_dir=pia_benchmark.metric_path)
        
        self.bench_result_dict = metric.evaluate()

        
    def _execute_metrics_generation(self):
        """메트릭 생성이 필요한 경우의 실행 로직"""
        self.logger.info("메트릭 생성 중...")
        # 구현 필요
        pia_benchmark = PiaBenchMark(
                                benchmark_path  = f"{BASE_BENCH_PATH}/{self.config.benchmark_name}" ,
                                model_name=self.config.model_name, 
                                cfg_target_path= self.config.cfg_target_path , 
                                token=self.access_token )
        pia_benchmark.preprocess_structure()
        pia_benchmark.preprocess_label_to_csv()  
        print("Categories identified:", pia_benchmark.categories)

        detector = EventDetector(config_path=self.config.cfg_target_path, 
                                 model_name=self.config.model_name , 
                                 token=pia_benchmark.token)
        detector.process_and_save_predictions(pia_benchmark.vector_video_path, 
                                            pia_benchmark.dataset_path, 
                                            pia_benchmark.alram_path)
        metric = MetricsEvaluator(pred_dir=pia_benchmark.alram_path, 
                                label_dir=pia_benchmark.dataset_path, 
                                save_dir=pia_benchmark.metric_path)
        
        self.bench_result_dict = metric.evaluate()


if __name__ == "__main__":
    # 파이프라인 설정
    config = PipelineConfig(
        model_name="T2V_CLIP4CLIP_MSRVTT",
        benchmark_name="PIA",
        cfg_target_path="topk.json",
        base_path=f"{BASE_BENCH_PATH}"
    )
    
    # 파이프라인 실행
    pipeline = BenchmarkPipeline(config)
    result = pipeline.run()
    
    print(f"\n파이프라인 실행 결과:")
    print(str(result))