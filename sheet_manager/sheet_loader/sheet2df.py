import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from dotenv import load_dotenv
import os
import json
import pandas as pd
from enviroments.convert import get_json_from_env_var
load_dotenv()

def sheet2df(sheet_name:str = "model"):
    """
    Reads data from a specified Google Spreadsheet and converts it into a Pandas DataFrame.

    Steps:
    1. Authenticate using a service account JSON key.
    2. Open the spreadsheet by its URL.
    3. Select the worksheet to read.
    4. Convert the worksheet data to a Pandas DataFrame.
    5. Clean up the DataFrame:
        - Rename columns using the first row of data.
        - Drop the first row after renaming columns.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the cleaned data from the spreadsheet.

    Note:
    - The following variables must be configured before using this function:
      - `json_key_path`: Path to the service account JSON key file.
      - `spreadsheet_url`: URL of the Google Spreadsheet.
      - `sheet_name`: Name of the worksheet to load.

    Dependencies:
    - pandas
    - gspread
    - oauth2client
    """
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    json_key_dict =get_json_from_env_var("GOOGLE_CREDENTIALS")
    credential = ServiceAccountCredentials.from_json_keyfile_dict(json_key_dict, scope)
    gc = gspread.authorize(credential)
    
    spreadsheet_url = os.getenv("SPREADSHEET_URL") 
    doc = gc.open_by_url(spreadsheet_url)
    sheet = doc.worksheet(sheet_name)
    
    # Convert to DataFrame
    df = pd.DataFrame(sheet.get_all_values())
    # Clean DataFrame
    df.rename(columns=df.iloc[0], inplace=True)
    df.drop(df.index[0], inplace=True)
    
    return df


def add_scaled_columns(df: pd.DataFrame, origin_col_name: str) -> pd.DataFrame:
    """
    특정 칼럼의 모든 값을 float로 변환한 후 100배 한 다음, 새로운 칼럼을 추가하는 함수.

    :param df: 원본 pandas DataFrame
    :param origin_col_name: 변환할 칼럼 이름 (숫자 값이 들어있는 칼럼)
    :return: 새로운 칼럼이 추가된 DataFrame
    """
    # df[origin_col_name] = df[origin_col_name].astype(float)
    df[origin_col_name] = pd.to_numeric(df[origin_col_name], errors='coerce').fillna(0.0)
    
    new_col_name = f"{origin_col_name}*100"  # 새로운 칼럼명 생성
    df[new_col_name] = df[origin_col_name] * 100  # 값 변환 후 새로운 칼럼 추가
    
    return df