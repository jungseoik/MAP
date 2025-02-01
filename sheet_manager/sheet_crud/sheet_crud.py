import os
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from dotenv import load_dotenv
from enviroments.convert import get_json_from_env_var
from typing import Optional, List

load_dotenv(override=True)

class SheetManager:
    def __init__(self, spreadsheet_url: Optional[str] = None,
                 worksheet_name: str = "flag",
                 column_name: str = "huggingface_id"):
        """
        Initialize SheetManager with Google Sheets credentials and connection.
        
        Args:
            spreadsheet_url (str, optional): URL of the Google Spreadsheet. 
                                           If None, takes from environment variable.
            worksheet_name (str): Name of the worksheet to operate on.
                                Defaults to "flag".
            column_name (str): Name of the column to operate on.
                             Defaults to "huggingface_id".
        """
        self.spreadsheet_url = spreadsheet_url or os.getenv("SPREADSHEET_URL")
        if not self.spreadsheet_url:
            raise ValueError("Spreadsheet URL not provided and not found in environment variables")

        self.worksheet_name = worksheet_name
        self.column_name = column_name

        # Initialize credentials and client
        self._init_google_client()
        
        # Initialize sheet connection
        self.doc = None
        self.sheet = None
        self.col_index = None
        self._connect_to_sheet(validate_column=True)

    def _init_google_client(self):
        """Initialize Google Sheets client with credentials."""
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        json_key_dict = get_json_from_env_var("GOOGLE_CREDENTIALS")
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(json_key_dict, scope)
        self.client = gspread.authorize(credentials)
    
    def _connect_to_sheet(self, validate_column: bool = True):
        """
        Connect to the specified Google Sheet and initialize necessary attributes.
        
        Args:
            validate_column (bool): Whether to validate the column name exists
        """
        try:
            self.doc = self.client.open_by_url(self.spreadsheet_url)
            
            # Try to get the worksheet
            try:
                self.sheet = self.doc.worksheet(self.worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                raise ValueError(f"Worksheet '{self.worksheet_name}' not found in spreadsheet")
            
            # Get headers
            self.headers = self.sheet.row_values(1)
            
            # Validate column only if requested
            if validate_column:
                try:
                    self.col_index = self.headers.index(self.column_name) + 1
                except ValueError:
                    # If column not found, use first available column
                    if self.headers:
                        self.column_name = self.headers[0]
                        self.col_index = 1
                        print(f"Column '{self.column_name}' not found. Using first available column: '{self.headers[0]}'")
                    else:
                        raise ValueError("No columns found in worksheet")
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ConnectionError(f"Failed to connect to sheet: {str(e)}")

    def change_worksheet(self, worksheet_name: str, column_name: Optional[str] = None):
        """
        Change the current worksheet and optionally the column.
        
        Args:
            worksheet_name (str): Name of the worksheet to switch to
            column_name (str, optional): Name of the column to switch to
        """
        old_worksheet = self.worksheet_name
        old_column = self.column_name
        
        try:
            self.worksheet_name = worksheet_name
            if column_name:
                self.column_name = column_name
            
            # First connect without column validation
            self._connect_to_sheet(validate_column=False)
            
            # Then validate the column if specified
            if column_name:
                self.change_column(column_name)
            else:
                # Validate existing column in new worksheet
                try:
                    self.col_index = self.headers.index(self.column_name) + 1
                except ValueError:
                    # If column not found, use first available column
                    if self.headers:
                        self.column_name = self.headers[0]
                        self.col_index = 1
                        print(f"Column '{old_column}' not found in new worksheet. Using first available column: '{self.headers[0]}'")
                    else:
                        raise ValueError("No columns found in worksheet")
            
            print(f"Successfully switched to worksheet: {worksheet_name}, using column: {self.column_name}")
            
        except Exception as e:
            # Restore previous state on error
            self.worksheet_name = old_worksheet
            self.column_name = old_column
            self._connect_to_sheet()
            raise e

    def change_column(self, column_name: str):
        """
        Change the target column.
        
        Args:
            column_name (str): Name of the column to switch to
        """
        if not self.headers:
            self.headers = self.sheet.row_values(1)
            
        try:
            self.col_index = self.headers.index(column_name) + 1
            self.column_name = column_name
            print(f"Successfully switched to column: {column_name}")
        except ValueError:
            raise ValueError(f"Column '{column_name}' not found in worksheet. Available columns: {', '.join(self.headers)}")

    def get_available_worksheets(self) -> List[str]:
        """Get list of all available worksheets in the spreadsheet."""
        return [worksheet.title for worksheet in self.doc.worksheets()]

    def get_available_columns(self) -> List[str]:
        """Get list of all available columns in the current worksheet."""
        return self.headers if self.headers else self.sheet.row_values(1)

    def _reconnect_if_needed(self):
        """Reconnect to the sheet if the connection is lost."""
        try:
            self.sheet.row_values(1)
        except (gspread.exceptions.APIError, AttributeError):
            self._init_google_client()
            self._connect_to_sheet()

    def _fetch_column_data(self) -> List[str]:
        """Fetch all data from the huggingface_id column."""
        values = self.sheet.col_values(self.col_index)
        return values[1:]  # Exclude header

    def _update_sheet(self, data: List[str]):
        """Update the entire column with new data."""
        try:
            # Prepare the range for update (excluding header)
            start_cell = gspread.utils.rowcol_to_a1(2, self.col_index)  # Start from row 2
            end_cell = gspread.utils.rowcol_to_a1(len(data) + 2, self.col_index)
            range_name = f"{start_cell}:{end_cell}"
            
            # Convert data to 2D array format required by gspread
            cells = [[value] for value in data]
            
            # Update the range
            self.sheet.update(range_name, cells)
        except Exception as e:
            print(f"Error updating sheet: {str(e)}")
            raise

    def push(self, text: str) -> int:
        """
        Push a text value to the next empty cell in the huggingface_id column.
        
        Args:
            text (str): Text to push to the sheet
            
        Returns:
            int: The row number where the text was pushed
        """
        try:
            self._reconnect_if_needed()
            
            # Get all values in the huggingface_id column
            column_values = self.sheet.col_values(self.col_index)
            
            # Find the next empty row
            next_row = None
            for i in range(1, len(column_values)):
                if not column_values[i].strip():
                    next_row = i + 1
                    break
            
            # If no empty row found, append to the end
            if next_row is None:
                next_row = len(column_values) + 1

            # Update the cell
            self.sheet.update_cell(next_row, self.col_index, text)
            print(f"Successfully pushed value: {text} to row {next_row}")
            return next_row
            
        except Exception as e:
            print(f"Error pushing to sheet: {str(e)}")
            raise

    def pop(self) -> Optional[str]:
        """Remove and return the most recent value."""
        try:
            self._reconnect_if_needed()
            data = self._fetch_column_data()
            
            if not data or not data[0].strip():
                return None
            
            value = data.pop(0)  # Remove first value
            data.append("")  # Add empty string at the end to maintain sheet size
            
            self._update_sheet(data)
            print(f"Successfully popped value: {value}")
            return value
            
        except Exception as e:
            print(f"Error popping from sheet: {str(e)}")
            raise

    def delete(self, value: str) -> List[int]:
        """Delete all occurrences of a value."""
        try:
            self._reconnect_if_needed()
            data = self._fetch_column_data()
            
            # Find all indices before deletion
            indices = [i + 1 for i, v in enumerate(data) if v.strip() == value.strip()]
            if not indices:
                print(f"Value '{value}' not found in sheet")
                return []
            
            # Remove matching values and add empty strings at the end
            data = [v for v in data if v.strip() != value.strip()]
            data.extend([""] * len(indices))  # Add empty strings to maintain sheet size
            
            self._update_sheet(data)
            print(f"Successfully deleted value '{value}' from rows: {indices}")
            return indices
            
        except Exception as e:
            print(f"Error deleting from sheet: {str(e)}")
            raise
        
    def update_cell_by_condition(self, condition_column: str, condition_value: str, target_column: str, target_value: str) -> Optional[int]:
        """
        Update the value of a cell based on a condition in another column.
        
        Args:
            condition_column (str): The column to check the condition on.
            condition_value (str): The value to match in the condition column.
            target_column (str): The column where the value should be updated.
            target_value (str): The new value to set in the target column.
        
        Returns:
            Optional[int]: The row number where the value was updated, or None if no matching row was found.
        """
        try:
            self._reconnect_if_needed()
            
            # Get all column headers
            headers = self.sheet.row_values(1)
            
            # Find the indices for the condition and target columns
            try:
                condition_col_index = headers.index(condition_column) + 1
            except ValueError:
                raise ValueError(f"조건 칼럼 '{condition_column}'이(가) 없습니다.")

            try:
                target_col_index = headers.index(target_column) + 1
            except ValueError:
                raise ValueError(f"목표 칼럼 '{target_column}'이(가) 없습니다.")

            # Get all rows of data
            data = self.sheet.get_all_records()

            # Find the row that matches the condition
            for i, row in enumerate(data):
                if row.get(condition_column) == condition_value:
                    # Update the target column in the matching row
                    row_number = i + 2  # Row index starts at 2 (1 is header)
                    self.sheet.update_cell(row_number, target_col_index, target_value)
                    print(f"Updated row {row_number}: Set {target_column} to '{target_value}' where {condition_column} is '{condition_value}'")
                    return row_number
            
            print(f"조건에 맞는 행을 찾을 수 없습니다: {condition_column} = '{condition_value}'")
            return None

        except Exception as e:
            print(f"Error updating cell by condition: {str(e)}")
            raise

    def get_all_values(self) -> List[str]:
        """Get all values from the huggingface_id column."""
        self._reconnect_if_needed()
        return [v for v in self._fetch_column_data() if v.strip()]

# Example usage
if __name__ == "__main__":
    # Initialize sheet manager
    sheet_manager = SheetManager()
    
    # # Push some test values
    # sheet_manager.push("test-model-1")
    # sheet_manager.push("test-model-2")
    # sheet_manager.push("test-model-3")
    
    # print("Initial values:", sheet_manager.get_all_values())
    
    # # Pop the most recent value
    # popped = sheet_manager.pop()
    # print(f"Popped value: {popped}")
    # print("After pop:", sheet_manager.get_all_values())
    
    # # Delete a specific value
    # deleted_rows = sheet_manager.delete("test-model-2")
    # print(f"Deleted from rows: {deleted_rows}")
    # print("After delete:", sheet_manager.get_all_values())

    row_updated = sheet_manager.update_cell_by_condition(
        condition_column="model", 
        condition_value="msr", 
        target_column="pia", 
        target_value="new_value"
    )

