import os
from typing import List
import enviroments.config as config

def cust_listdir(directory: str) -> List[str]:
    """
    os.listdir와 유사하게 작동하지만 config에 정의된 폴더/파일들을 제외하고 목록을 반환합니다.
    
    Args:
        directory (str): 탐색할 디렉토리 경로
        
    Returns:
        List[str]: config의 EXCLUDE_DIRS에 정의된 폴더/파일들을 제외한 목록
    """
    return [item for item in os.listdir(directory) if item not in config.EXCLUDE_DIRS]
