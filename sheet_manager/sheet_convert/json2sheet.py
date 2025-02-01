import json
from sheet_manager.sheet_crud.sheet_crud import SheetManager
import json
from typing import Optional, Dict

def update_benchmark_json(
    model_name: str, 
    benchmark_data: dict, 
    worksheet_name: str = "metric",
    target_column: str = "benchmark"  # 타겟 칼럼 파라미터 추가
):
    """
    특정 모델의 벤치마크 데이터를 JSON 형태로 업데이트합니다.
    
    Args:
        model_name (str): 업데이트할 모델 이름
        benchmark_data (dict): 업데이트할 벤치마크 데이터 딕셔너리
        worksheet_name (str): 작업할 워크시트 이름 (기본값: "metric")
        target_column (str): 업데이트할 타겟 칼럼 이름 (기본값: "benchmark")
    """
    sheet_manager = SheetManager(worksheet_name=worksheet_name)
    
    # 딕셔너리를 JSON 문자열로 변환
    json_str = json.dumps(benchmark_data, ensure_ascii=False)
    
    # 모델명을 기준으로 지정된 칼럼 업데이트
    row = sheet_manager.update_cell_by_condition(
        condition_column="Model name",  # 모델명이 있는 칼럼
        condition_value=model_name,     # 찾을 모델명
        target_column=target_column,    # 업데이트할 타겟 칼럼
        target_value=json_str          # 업데이트할 JSON 값
    )
    
    if row:
        print(f"Successfully updated {target_column} data for model: {model_name}")
    else:
        print(f"Model {model_name} not found in the sheet")



def get_benchmark_dict(
    model_name: str,
    worksheet_name: str = "metric",
    target_column: str = "benchmark",
    save_path: Optional[str] = None
) -> Dict:
    """
    시트에서 특정 모델의 벤치마크 JSON 데이터를 가져와 딕셔너리로 변환합니다.
    
    Args:
        model_name (str): 가져올 모델 이름
        worksheet_name (str): 작업할 워크시트 이름 (기본값: "metric")
        target_column (str): 데이터를 가져올 칼럼 이름 (기본값: "benchmark")
        save_path (str, optional): 딕셔너리를 저장할 JSON 파일 경로
        
    Returns:
        Dict: 벤치마크 데이터 딕셔너리. 데이터가 없거나 JSON 파싱 실패시 빈 딕셔너리 반환
    """
    sheet_manager = SheetManager(worksheet_name=worksheet_name)
    
    try:
        # 모든 데이터 가져오기
        data = sheet_manager.sheet.get_all_records()
        
        # 해당 모델 찾기
        target_row = next(
            (row for row in data if row.get("Model name") == model_name),
            None
        )
        
        if not target_row:
            print(f"Model {model_name} not found in the sheet")
            return {}
            
        # 타겟 칼럼의 JSON 문자열 가져오기
        json_str = target_row.get(target_column)
        
        if not json_str:
            print(f"No data found in {target_column} for model: {model_name}")
            return {}
            
        # JSON 문자열을 딕셔너리로 변환
        result_dict = json.loads(json_str)
        
        # 결과 저장 (save_path가 제공된 경우)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved dictionary to: {save_path}")
            
        return result_dict
        
    except json.JSONDecodeError:
        print(f"Failed to parse JSON data for model: {model_name}")
        return {}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {}

def str2json(json_str):
    """
    문자열을 JSON 객체로 변환합니다.
    
    Args:
        json_str (str): JSON 형식의 문자열
    
    Returns:
        dict: 파싱된 JSON 객체, 실패시 None
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None