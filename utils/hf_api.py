from huggingface_hub import HfApi
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelInfo:
    """모델 정보를 저장하는 데이터 클래스"""
    model_id: str
    last_modified: Any
    downloads: int
    private: bool
    attributes: Dict[str, Any]

class HuggingFaceInfoManager:
    def __init__(self, access_token: Optional[str] = None, organization: str = "PIA-SPACE-LAB"):
        """
        HuggingFace API 관리자 클래스 초기화
        
        Args:
            access_token (str, optional): HuggingFace 액세스 토큰
            organization (str): 조직 이름 (기본값: "PIA-SPACE-LAB")
        
        Raises:
            ValueError: access_token이 None일 경우 발생
        """
        if access_token is None:
            raise ValueError("액세스 토큰은 필수 입력값입니다. HuggingFace에서 발급받은 토큰을 입력해주세요.")
        
        self.api = HfApi()
        self.access_token = access_token
        self.organization = organization
        
        # API 호출 결과를 바로 처리하여 저장
        api_models = self.api.list_models(author=self.organization, use_auth_token=self.access_token)
        self._stored_models = []
        self._model_infos = []
        
        # 모든 모델 정보를 미리 처리하여 저장
        for model in api_models:
            # 기본 정보 저장
            model_attrs = {}
            for attr in dir(model):
                if not attr.startswith("_"):
                    model_attrs[attr] = getattr(model, attr)
            
            # ModelInfo 객체 생성 및 저장
            model_info = ModelInfo(
                model_id=model.modelId,
                last_modified=model.lastModified,
                downloads=model.downloads,
                private=model.private,
                attributes=model_attrs
            )
            self._model_infos.append(model_info)
            self._stored_models.append(model)

    def get_model_info(self) -> List[Dict[str, Any]]:
        """모든 모델의 정보를 반환"""
        return [
            {
                'model_id': info.model_id,
                'last_modified': info.last_modified,
                'downloads': info.downloads,
                'private': info.private,
                **info.attributes
            }
            for info in self._model_infos
        ]

    def get_model_ids(self) -> List[str]:
        """모든 모델의 ID 리스트 반환"""
        return [info.model_id for info in self._model_infos]
    
    def get_private_models(self) -> List[Dict[str, Any]]:
        """비공개 모델 정보 반환"""
        return [
            {
                'model_id': info.model_id,
                'last_modified': info.last_modified,
                'downloads': info.downloads,
                'private': info.private,
                **info.attributes
            }
            for info in self._model_infos if info.private
        ]
    
    def get_public_models(self) -> List[Dict[str, Any]]:
        """공개 모델 정보 반환"""
        return [
            {
                'model_id': info.model_id,
                'last_modified': info.last_modified,
                'downloads': info.downloads,
                'private': info.private,
                **info.attributes
            }
            for info in self._model_infos if not info.private
        ]

    def refresh_models(self) -> None:
        """모델 정보 새로고침 (새로운 API 호출 수행)"""
        # 클래스 재초기화
        self.__init__(self.access_token, self.organization)