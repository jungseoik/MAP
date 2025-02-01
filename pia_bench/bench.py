import os
import shutil
from devmacs_core.devmacs_core import DevMACSCore
import json
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
from utils.except_dir import cust_listdir
from utils.parser import load_config
from utils.logger import custom_logger

logger = custom_logger(__name__)

DATA_SET = "dataset"
CFG = "CFG"
VECTOR = "vector"
TEXT = "text"
VIDEO = "video"
EXECPT = ["@eaDir", "README.md"]
ALRAM = "alarm"
METRIC = "metric"
MSRVTT = "MSRVTT"
MODEL = "models"

class PiaBenchMark:
    def __init__(self, benchmark_path :str, cfg_target_path : str = None , model_name : str = MSRVTT , token:str =None):
        """
        PIA 벤치마크 시스템을 구축 위한 클래스입니다.
        데이터셋 폴더구조, 벡터 추출, 구조 생성 등의 기능을 제공합니다.
        
        Attributes:
            benchmark_path (str): 벤치마크 기본 경로
            cfg_target_path (str): 설정 파일 경로
            model_name (str): 사용할 모델 이름
            token (str): 인증 토큰
            categories (List[str]): 처리할 카테고리 목록
        """
        self.benchmark_path = benchmark_path
        self.token = token
        self.model_name = model_name
        self.devmacs_core = None
        self.cfg_target_path = cfg_target_path
        self.cfg_name = Path(cfg_target_path).stem
        self.cfg_dict = load_config(self.cfg_target_path)

        self.dataset_path = os.path.join(benchmark_path, DATA_SET)
        self.cfg_path = os.path.join(benchmark_path , CFG)

        self.model_path = os.path.join(self.benchmark_path , MODEL)
        self.model_name_path = os.path.join(self.model_path ,self.model_name)
        self.model_name_cfg_path = os.path.join(self.model_name_path , CFG)
        self.model_name_cfg_name_path = os.path.join(self.model_name_cfg_path , self.cfg_name)
        self.alram_path = os.path.join(self.model_name_cfg_name_path , ALRAM)
        self.metric_path = os.path.join(self.model_name_cfg_name_path , METRIC)

        self.vector_path = os.path.join(self.model_name_path , VECTOR)
        self.vector_text_path = os.path.join(self.vector_path , TEXT)
        self.vector_video_path = os.path.join(self.vector_path , VIDEO)

        self.categories = []

    def _create_frame_labels(self, label_data: Dict, total_frames: int) -> pd.DataFrame:
        """
        프레임 기반의 레이블 데이터프레임을 생성합니다.
        
        Args:
            label_data (Dict): 레이블 정보가 담긴 딕셔너리
            total_frames (int): 총 프레임 수
            
        Returns:
            pd.DataFrame: 프레임별 레이블이 저장된 데이터프레임
            
        Note:
            반환되는 데이터프레임은 각 프레임별로 카테고리의 존재 여부를 0과 1로 표시합니다.
        """
        colmuns = ['frame'] + sorted(self.categories)
        df = pd.DataFrame(0, index=range(total_frames), columns=colmuns)
        df['frame'] = range(total_frames)
        
        for clip_info in label_data['clips'].values():
            category = clip_info['category']
            if category in self.categories:  # 해당 카테고리가 목록에 있는 경우만 처리
                start_frame, end_frame = clip_info['timestamp']
                df.loc[start_frame:end_frame, category] = 1
                
        return df

    def preprocess_label_to_csv(self):
        """
        데이터셋의 모든 JSON 레이블 파일을 프레임 기반 CSV 파일로 변환합니다.
        
        Raises:
            ValueError: JSON 파일을 찾을 수 없는 경우 발생
            
        Note:
            - 각 카테고리 폴더 내의 JSON 파일을 처리합니다.
            - 이미 CSV로 변환된 파일은 건너뜁니다.
        """
        json_files = []
        csv_files = []
        
        # categories가 비어있는 경우에만 채우도록 수정
        if not self.categories:
            for cate in cust_listdir(self.dataset_path):
                if os.path.isdir(os.path.join(self.dataset_path, cate)):
                    self.categories.append(cate)

        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            category_jsons = [os.path.join(category, f) for f in cust_listdir(category_path) if f.endswith('.json')]
            json_files.extend(category_jsons)
            category_csvs = [os.path.join(category, f) for f in cust_listdir(category_path) if f.endswith('.csv')]
            csv_files.extend(category_csvs)

        if not json_files:
            logger.error("No JSON files found in any category directory")
            raise ValueError("No JSON files found in any category directory")
        
        if len(json_files) == len(csv_files):
            logger.info("All JSON files have already been processed to CSV. No further processing needed.")
            return

        for json_file in json_files:
            json_path = os.path.join(self.dataset_path, json_file)
            video_name = os.path.splitext(json_file)[0] 
            
            label_info = load_config(json_path)
            video_info = label_info['video_info']
            total_frames = video_info['total_frame']
            
            df = self._create_frame_labels( label_info, total_frames)
            
            output_path = os.path.join(self.dataset_path, f"{video_name}.csv")
            df.to_csv(output_path , index=False)
        logger.info("Complete !")

    def preprocess_structure(self):
        """
        벤치마크 시스템에 필요한 디렉토리 구조를 생성합니다.
        
        생성되는 구조:
            - dataset/: 데이터셋 저장
            - cfg/: 설정 파일 저장
            - vector/: 추출된 벡터 저장
            - alarm/: 알람 관련 파일 저장
            - metric/: 평가 지표 저장
            
        Note:
            기존 카테고리 구조가 있다면 유지하고, 없다면 새로 생성합니다.
        """
        logger.info("Starting directory structure preprocessing...")
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.cfg_path, exist_ok=True)
        os.makedirs(self.vector_text_path, exist_ok=True)
        os.makedirs(self.vector_video_path, exist_ok=True)
        os.makedirs(self.alram_path, exist_ok=True)
        os.makedirs(self.metric_path, exist_ok=True)
        os.makedirs(self.model_name_cfg_name_path , exist_ok=True)


        # dataset 폴더가 이미 존재하고 그 안에 카테고리 폴더들이 있는지 확인
        if os.path.exists(self.dataset_path) and any(os.path.isdir(os.path.join(self.dataset_path, d)) for d in cust_listdir(self.dataset_path)):
            # 이미 구성된 구조라면, dataset 폴더에서 카테고리들을 가져옴
            self.categories = [d for d in cust_listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        else:
            # 처음 실행되는 경우, 기존 로직대로 진행
            for item in cust_listdir(self.benchmark_path):
                item_path = os.path.join(self.benchmark_path, item)
                
                if item.startswith("@") or item in [METRIC ,"README.md",MODEL,  CFG, DATA_SET, VECTOR, ALRAM] or not os.path.isdir(item_path):
                    continue
                target_path = os.path.join(self.dataset_path, item)
                if not os.path.exists(target_path):
                    shutil.move(item_path, target_path)
                    self.categories.append(item)
                
        for category in self.categories:
            category_path = os.path.join(self.vector_video_path, category)
            os.makedirs(category_path, exist_ok=True)

        logger.info("Folder preprocessing completed.")
    
    def extract_visual_vector(self):
        """
        데이터셋에서 시각적 특징 벡터를 추출합니다.
        
        Note:
            - Hugging Face 모델을 사용하여 특징을 추출합니다.
            - 추출된 벡터는 vector/video/ 경로에 저장됩니다.
            
        Requires:
            DevMACSCore가 초기화되어 있어야 합니다.
        """
        logger.info(f"Starting visual vector extraction using model: {self.model_name}")
        try:
            self.devmacs_core = DevMACSCore.from_huggingface(token=self.token, repo_id=f"PIA-SPACE-LAB/{self.model_name}")
            self.devmacs_core.save_visual_results(
                vid_dir = self.dataset_path,
                result_dir = self.vector_video_path
            )
        except Exception as e:
            logger.error(f"Error during vector extraction: {str(e)}")
            raise

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    access_token = os.getenv("ACCESS_TOKEN")
    model_name = "T2V_CLIP4CLIP_MSRVTT"

    benchmark_path = "/home/jungseoik/data/Abnormal_situation_leader_board/assets/PIA"
    cfg_target_path= "/home/jungseoik/data/Abnormal_situation_leader_board/assets/PIA/CFG/topk.json"

    pia_benchmark = PiaBenchMark(benchmark_path ,model_name=model_name, cfg_target_path= cfg_target_path , token=access_token )
    pia_benchmark.preprocess_structure()
    pia_benchmark.preprocess_label_to_csv()
    print("Categories identified:", pia_benchmark.categories)