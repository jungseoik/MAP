import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
from enviroments.convert import get_json_from_env_var

load_dotenv()

def push_model_names_to_sheet(spreadsheet_url, sheet_name, access_token, organization):
    """
    Fetches model names from Hugging Face and updates a Google Sheet with the names, links, and HTML links.

    Args:
        json_key_path (str): Path to the Google service account JSON key file.
        spreadsheet_url (str): URL of the Google Spreadsheet.
        sheet_name (str): Name of the sheet to update.
        access_token (str): Hugging Face access token.
        organization (str): Organization name on Hugging Face.
    """
    # Authorize Google Sheets API
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    json_key_dict =get_json_from_env_var("GOOGLE_CREDENTIALS")
    credential = ServiceAccountCredentials.from_json_keyfile_dict(json_key_dict, scope)
    gc = gspread.authorize(credential)

    # Open the Google Spreadsheet
    doc = gc.open_by_url(spreadsheet_url)
    sheet = doc.worksheet(sheet_name)

    # Fetch existing data from the sheet
    existing_data = pd.DataFrame(sheet.get_all_records())

    # Fetch models from Hugging Face
    api = HfApi()
    models = api.list_models(author=organization, use_auth_token=access_token)

    # Extract model names, links, and HTML links
    model_details = [{
        "Model name": model.modelId.split("/")[1],
        "Model link": f"https://huggingface.co/{model.modelId}",
        "Model": f"<a target=\"_blank\" href=\"https://huggingface.co/{model.modelId}\" style=\"color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;\">{model.modelId}</a>"
    } for model in models]

    new_data_df = pd.DataFrame(model_details)

    # Check for duplicates and update only new model names
    if "Model name" in existing_data.columns:
        existing_model_names = existing_data["Model name"].tolist()
    else:
        existing_model_names = []

    new_data_df = new_data_df[~new_data_df["Model name"].isin(existing_model_names)]

    if not new_data_df.empty:
        # Append new model names, links, and HTML links to the existing data
        updated_data = pd.concat([existing_data, new_data_df], ignore_index=True)

        # Push updated data back to the sheet
        updated_data = updated_data.replace([float('inf'), float('-inf')], None)  # Infinity 값을 None으로 변환
        updated_data = updated_data.fillna('')  # NaN 값을 빈 문자열로 변환
        sheet.update([updated_data.columns.values.tolist()] + updated_data.values.tolist())
        print("New model names, links, and HTML links successfully added to Google Sheet.")
    else:
        print("No new model names to add.")

# Example usage
if __name__ == "__main__":
    spreadsheet_url = os.getenv("SPREADSHEET_URL") 
    access_token = os.getenv("ACCESS_TOKEN")
    sheet_name = "시트1"
    organization = "PIA-SPACE-LAB"

    push_model_names_to_sheet(spreadsheet_url, sheet_name, access_token, organization)
