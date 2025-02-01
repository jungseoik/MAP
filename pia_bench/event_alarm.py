import os
import numpy as np
import torch
from typing import Dict, List, Tuple
from devmacs_core.devmacs_core import DevMACSCore
# from devmacs_core.devmacs_core_copy import DevMACSCore
from devmacs_core.utils.common.cal import loose_similarity
from utils.parser import load_config, PromptManager
import json
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from utils.except_dir import cust_listdir
from utils.logger import custom_logger

logger = custom_logger(__name__)

class EventDetector:
    def __init__(self, config_path: str , model_name:str = None, token:str = None):
        self.config = load_config(config_path)
        self.macs = DevMACSCore.from_huggingface(token=token, repo_id=f"PIA-SPACE-LAB/{model_name}")
        # self.macs = DevMACSCore(model_type="clip4clip_web")

        self.prompt_manager = PromptManager(config_path)
        self.sentences = self.prompt_manager.sentences
        self.text_vectors = self.macs.get_text_vector(self.sentences)
        
    def process_and_save_predictions(self, vector_base_dir: str, label_base_dir: str, save_base_dir: str):
        """비디오 벡터를 처리하고 결과를 CSV로 저장"""

        # 전체 비디오 파일 수 계산
        total_videos = sum(len([f for f in cust_listdir(os.path.join(vector_base_dir, d)) 
                                if f.endswith('.npy')]) 
                            for d in cust_listdir(vector_base_dir) 
                            if os.path.isdir(os.path.join(vector_base_dir, d)))
        pbar = tqdm(total=total_videos, desc="Processing videos")
        
        for category in cust_listdir(vector_base_dir):
            category_path = os.path.join(vector_base_dir, category)
            if not os.path.isdir(category_path):
                continue
            
            # 저장 디렉토리 생성
            save_category_dir = os.path.join(save_base_dir, category)
            os.makedirs(save_category_dir, exist_ok=True)
            
            for file in cust_listdir(category_path):
                if file.endswith('.npy'):
                    video_name = os.path.splitext(file)[0]
                    vector_path = os.path.join(category_path, file)
                    
                    # 라벨 파일 읽기
                    label_path = os.path.join(label_base_dir, category, f"{video_name}.json")
                    with open(label_path, 'r') as f:
                        label_data = json.load(f)
                        total_frames = label_data['video_info']['total_frame']
                    
                    # 예측 결과 생성 및 저장
                    self._process_and_save_single_video(
                        vector_path=vector_path,
                        total_frames=total_frames,
                        save_path=os.path.join(save_category_dir, f"{video_name}.csv")
                    )
                    pbar.update(1)
        pbar.close()

    def _process_and_save_single_video(self, vector_path: str, total_frames: int, save_path: str):
        """단일 비디오 처리 및 저장"""
        # 기본 예측 수행
        sparse_predictions = self._process_single_vector(vector_path)
        
        # 데이터프레임으로 변환 및 확장
        df = self._expand_predictions(sparse_predictions, total_frames)
        
        # CSV로 저장
        df.to_csv(save_path, index=False)

    def _process_single_vector(self, vector_path: str) -> Dict:
        """기존 예측 로직"""
        video_vector = np.load(vector_path)
        processed_vectors = []
        frame_interval = 15
        
        for vector in video_vector:
            v = vector.squeeze(0)  # numpy array
            v = torch.from_numpy(v).unsqueeze(0).cuda()  # torch tensor로 변환 후 GPU로
            processed_vectors.append(v)
        
        frame_results = {}
        for vector_idx, v in enumerate(processed_vectors):
            actual_frame = vector_idx * frame_interval
            sim_scores = loose_similarity(
                sequence_output=self.text_vectors.cuda(),
                visual_output=v.unsqueeze(1)
            )
            frame_results[actual_frame] = self._calculate_alarms(sim_scores)
            
        return frame_results

    def _expand_predictions(self, sparse_predictions: Dict, total_frames: int) -> pd.DataFrame:
        """예측을 전체 프레임으로 확장"""
        # 카테고리 목록 추출 (첫 번째 프레임의 알람 결과에서)
        first_frame = list(sparse_predictions.keys())[0]
        categories = list(sparse_predictions[first_frame].keys())
        
        # 전체 프레임 생성
        df = pd.DataFrame({'frame': range(total_frames)})
        
        # 각 카테고리에 대한 알람 값 초기화
        for category in categories:
            df[category] = 0
        
        # 예측값 채우기
        frame_keys = sorted(sparse_predictions.keys())
        for i in range(len(frame_keys)):
            current_frame = frame_keys[i]
            next_frame = frame_keys[i + 1] if i + 1 < len(frame_keys) else total_frames
            
            # 각 카테고리의 알람 값 설정
            for category in categories:
                alarm_value = sparse_predictions[current_frame][category]['alarm']
                df.loc[current_frame:next_frame-1, category] = alarm_value
        
        return df


    def _calculate_alarms(self, sim_scores: torch.Tensor) -> Dict:
        """유사도 점수를 기반으로 각 이벤트의 알람 상태 계산"""
        # 로거 설정
        log_filename = f"alarm_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            filename=log_filename,
            level=logging.ERROR,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        
        event_alarms = {}
        
        for event_config in self.config['PROMPT_CFG']:
            event = event_config['event']
            top_k = event_config['top_candidates']
            threshold = event_config['alert_threshold']
            
            # logger.info(f"\nProcessing event: {event}")
            # logger.info(f"Top K: {top_k}, Threshold: {threshold}")
            
            event_prompts = self._get_event_prompts(event)

            # logger.debug(f"\nEvent Prompts Debug for {event}:")
            # logger.debug(f"Indices: {event_prompts['indices']}")
            # logger.debug(f"Types: {event_prompts['types']}")
            # logger.debug(f"\nSim Scores Debug:")
            # logger.debug(f"Shape: {sim_scores.shape}")
            # logger.debug(f"Raw scores: {sim_scores}")

            # event_scores = sim_scores[event_prompts['indices']]
            event_scores = sim_scores[event_prompts['indices']].squeeze(-1)  # shape 변경
        
            # logger.debug(f"Event scores shape: {event_scores.shape}")
            # logger.debug(f"Event scores: {event_scores}")
            # 각 프롬프트와 점수 출력
            # logger.info("\nDEBUG VALUES:")
            # logger.info(f"event_scores: {event_scores}")
            # logger.info(f"indices: {event_prompts['indices']}")
            # logger.info(f"types: {event_prompts['types']}")

            # logger.info("\nAll prompts and scores:")
            # for idx, (score, prompt_type) in enumerate(zip(event_scores, event_prompts['types'])):
            #     logger.info(f"Type: {prompt_type}, Score: {score.item():.4f}")
            
            top_k_values, top_k_indices = torch.topk(event_scores, min(top_k, len(event_scores)))
        
            # logger.info(f"top_k_values: {top_k_values}")
            # logger.info(f"top_k_indices (raw): {top_k_indices}")
            # Top K 결과 출력
            # logger.info(f"\nTop {top_k} selections:")
            for idx, (value, index) in enumerate(zip(top_k_values, top_k_indices)):
                # indices[index]가 아닌 index를 직접 사용
                prompt_type = event_prompts['types'][index]  # 수정된 부분
                # logger.info(f"DEBUG: index={index}, types={event_prompts['types']}, selected_type={prompt_type}")
                # logger.info(f"Rank {idx+1}: Type: {prompt_type}, Score: {value.item():.4f}")

            abnormal_count = sum(1 for idx in top_k_indices 
                    if event_prompts['types'][idx] == 'abnormal')  # 수정된 부분
            # for idx, (value, orig_idx) in enumerate(zip(top_k_values, top_k_indices)):
            #     prompt_type = event_prompts['types'][orig_idx.item()]
            #     logger.info(f"Rank {idx+1}: Type: {prompt_type}, Score: {value.item():.4f}")
            
            # abnormal_count = sum(1 for idx in top_k_indices 
            #                 if event_prompts['types'][idx.item()] == 'abnormal')
            
            # 알람 결정 과정 출력
            # logger.info(f"\nAbnormal count: {abnormal_count}")
            alarm_result = 1 if abnormal_count >= threshold else 0
            # logger.info(f"Final alarm decision: {alarm_result}")
            # logger.info("-" * 50)
            
            event_alarms[event] = {
                'alarm': alarm_result,
                'scores': top_k_values.tolist(),
                'top_k_types': [event_prompts['types'][idx.item()] for idx in top_k_indices]
            }
        
        # 로거 종료
        logging.shutdown()
                
        return event_alarms

    def _get_event_prompts(self, event: str) -> Dict:
        indices = []
        types = []
        current_idx = 0
        
        for event_config in self.config['PROMPT_CFG']:
            if event_config['event'] == event:
                for status in ['normal', 'abnormal']:
                    for _ in range(len(event_config['prompts'][status])):
                        indices.append(current_idx)
                        types.append(status)
                        current_idx += 1
                        
        return {'indices': indices, 'types': types}


