import json
from typing import Dict, List, Tuple

def load_config(config_path: str) -> Dict:
    """
    JSON 설정 파일을 읽어서 딕셔너리로 반환합니다.
    
    Args:
        config_path (str): JSON 설정 파일의 경로
        
    Returns:
        Dict: 설정 정보가 담긴 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class PromptManager:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.sentences, self.index_mapping = self._extract_all_sentences_with_index()
        self.reverse_mapping = self._create_reverse_mapping()
    
    def _extract_all_sentences_with_index(self) -> Tuple[List[str], Dict]:
        """모든 sentence와 인덱스 매핑 추출"""
        sentences = []
        index_mapping = {}
        
        for event_idx, event_config in enumerate(self.config.get('PROMPT_CFG', [])):
            prompts = event_config.get('prompts', {})
            for status in ['normal', 'abnormal']:
                for prompt_idx, prompt in enumerate(prompts.get(status, [])):
                    sentence = prompt.get('sentence', '')
                    sentences.append(sentence)
                    index_mapping[(event_idx, status, prompt_idx)] = sentence
        
        return sentences, index_mapping
    
    def _create_reverse_mapping(self) -> Dict:
        """sentence -> indices 역방향 매핑 생성"""
        reverse_map = {}
        for indices, sent in self.index_mapping.items():
            if sent not in reverse_map:
                reverse_map[sent] = []
            reverse_map[sent].append(indices)
        return reverse_map
    
    def get_sentence_indices(self, sentence: str) -> List[Tuple[int, str, int]]:
        """특정 sentence의 모든 인덱스 위치 반환"""
        return self.reverse_mapping.get(sentence, [])
    
    def get_details_by_sentence(self, sentence: str) -> List[Dict]:
        """sentence로 모든 관련 상세 정보 찾아 반환"""
        indices = self.get_sentence_indices(sentence)
        return [self.get_details_by_index(*idx) for idx in indices]
    
    def get_details_by_index(self, event_idx: int, status: str, prompt_idx: int) -> Dict:
        """인덱스로 상세 정보 찾아 반환"""
        event_config = self.config['PROMPT_CFG'][event_idx]
        prompt = event_config['prompts'][status][prompt_idx]
        
        return {
            'event': event_config['event'],
            'status': status,
            'sentence': prompt['sentence'],
            'top_candidates': event_config['top_candidates'],
            'alert_threshold': event_config['alert_threshold'],
            'event_idx': event_idx,
            'prompt_idx': prompt_idx
        }
    
    def get_all_sentences(self) -> List[str]:
        """모든 sentence 리스트 반환"""
        return self.sentences