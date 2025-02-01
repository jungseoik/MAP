import os
import json
from dotenv import load_dotenv
load_dotenv()

def get_json_from_env_var(env_var_name):
    """
    환경 변수에서 JSON 데이터를 가져와 딕셔너리로 변환하는 함수.
    :param env_var_name: 환경 변수 이름
    :return: 딕셔너리 형태의 JSON 데이터
    """
    json_string = os.getenv(env_var_name)
    if not json_string:
        raise EnvironmentError(f"환경 변수 '{env_var_name}'가 설정되지 않았습니다.")

    try:
        # 줄바꿈(\n)을 이스케이프 문자(\\n)로 변환
        json_string = json_string.replace("\n", "\\n")
        
        # JSON 문자열을 딕셔너리로 변환
        json_data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 변환 실패: {e}")

    return json_data



def json_to_env_var(json_file_path, env_var_name="JSON_ENV_VAR"):
    """
    주어진 JSON 파일의 데이터를 환경 변수 형태로 변환하여 출력하는 함수.

    :param json_file_path: JSON 파일 경로
    :param env_var_name: 환경 변수 이름 (기본값: JSON_ENV_VAR)
    :return: None
    """
    try:
        # JSON 파일 읽기
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)

        # JSON 데이터를 문자열로 변환
        json_string = json.dumps(json_data)

        # 환경 변수 형태로 출력
        env_variable = f'{env_var_name}={json_string}'
        print("\n환경 변수로 사용할 수 있는 출력값:\n")
        print(env_variable)
        print("\n위 값을 .env 파일에 복사하여 붙여넣으세요.")
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {json_file_path}")
    except json.JSONDecodeError:
        print(f"유효한 JSON 파일이 아닙니다: {json_file_path}")

