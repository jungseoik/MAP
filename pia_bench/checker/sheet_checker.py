from typing import List, Dict, Optional, Set, Tuple
import logging
import gspread
from dotenv import load_dotenv
from typing import Optional, List
from sheet_manager.sheet_crud.sheet_crud import SheetManager

load_dotenv()
class SheetChecker:
    def __init__(self, sheet_manager):
        """Initialize SheetChecker with a sheet manager instance."""
        self.sheet_manager = sheet_manager
        self.bench_sheet_manager = None
        self.logger = logging.getLogger(__name__)
        self._init_bench_sheet()

    def _init_bench_sheet(self):
        """Initialize sheet manager for the model sheet."""
        self.bench_sheet_manager = type(self.sheet_manager)(
            spreadsheet_url=self.sheet_manager.spreadsheet_url,
            worksheet_name="model",
            column_name="Model name"
        )
    def add_benchmark_column(self, column_name: str):
        """Add a new benchmark column to the sheet."""
        try:
            # Get current headers
            headers = self.bench_sheet_manager.get_available_columns()
            
            # If column already exists, return
            if column_name in headers:
                return
            
            # Add new column header
            new_col_index = len(headers) + 1
            cell = gspread.utils.rowcol_to_a1(1, new_col_index)
            # Update with 2D array format
            self.bench_sheet_manager.sheet.update(cell, [[column_name]])  # 값을 2D 배열로 변경
            self.logger.info(f"Added new benchmark column: {column_name}")
            
            # Update headers in bench_sheet_manager
            self.bench_sheet_manager._connect_to_sheet(validate_column=False)
            
        except Exception as e:
            self.logger.error(f"Error adding benchmark column {column_name}: {str(e)}")
            raise
    def validate_benchmark_columns(self, benchmark_columns: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate benchmark columns and add missing ones.
        
        Args:
            benchmark_columns: List of benchmark column names to validate
                
        Returns:
            Tuple[List[str], List[str]]: (valid columns, invalid columns)
        """
        available_columns = self.bench_sheet_manager.get_available_columns()
        valid_columns = []
        invalid_columns = []
        
        for col in benchmark_columns:
            if col in available_columns:
                valid_columns.append(col)
            else:
                try:
                    self.add_benchmark_column(col)
                    valid_columns.append(col)
                    self.logger.info(f"Added new benchmark column: {col}")
                except Exception as e:
                    invalid_columns.append(col)
                    self.logger.error(f"Failed to add benchmark column '{col}': {str(e)}")
        
        return valid_columns, invalid_columns

    def check_model_and_benchmarks(self, model_name: str, benchmark_columns: List[str]) -> Dict[str, List[str]]:
        """
        Check model existence and which benchmarks need to be filled.
        
        Args:
            model_name: Name of the model to check
            benchmark_columns: List of benchmark column names to check
            
        Returns:
            Dict with keys:
                'status': 'model_not_found' or 'model_exists'
                'empty_benchmarks': List of benchmark columns that need to be filled
                'filled_benchmarks': List of benchmark columns that are already filled
                'invalid_benchmarks': List of benchmark columns that don't exist
        """
        result = {
            'status': '',
            'empty_benchmarks': [],
            'filled_benchmarks': [],
            'invalid_benchmarks': []
        }

        # First check if model exists
        exists = self.check_model_exists(model_name)
        if not exists:
            result['status'] = 'model_not_found'
            return result

        result['status'] = 'model_exists'
        
        # Validate benchmark columns
        valid_columns, invalid_columns = self.validate_benchmark_columns(benchmark_columns)
        result['invalid_benchmarks'] = invalid_columns
        
        if not valid_columns:
            return result

        # Check which valid benchmarks are empty
        self.bench_sheet_manager.change_column("Model name")
        all_values = self.bench_sheet_manager.get_all_values()
        row_index = all_values.index(model_name) + 2

        for column in valid_columns:
            try:
                self.bench_sheet_manager.change_column(column)
                value = self.bench_sheet_manager.sheet.cell(row_index, self.bench_sheet_manager.col_index).value
                if not value or not value.strip():
                    result['empty_benchmarks'].append(column)
                else:
                    result['filled_benchmarks'].append(column)
            except Exception as e:
                self.logger.error(f"Error checking column {column}: {str(e)}")
                result['empty_benchmarks'].append(column)

        return result

    def update_model_info(self, model_name: str, model_info: Dict[str, str]):
        """Update basic model information columns."""
        try:
            for column_name, value in model_info.items():
                self.bench_sheet_manager.change_column(column_name)
                self.bench_sheet_manager.push(value)
            self.logger.info(f"Successfully added new model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error updating model info: {str(e)}")
            raise

    def update_benchmarks(self, model_name: str, benchmark_values: Dict[str, str]):
        """
        Update benchmark values.
        
        Args:
            model_name: Name of the model
            benchmark_values: Dictionary of benchmark column names and their values
        """
        try:
            self.bench_sheet_manager.change_column("Model name")
            all_values = self.bench_sheet_manager.get_all_values()
            row_index = all_values.index(model_name) + 2

            for column, value in benchmark_values.items():
                self.bench_sheet_manager.change_column(column)
                self.bench_sheet_manager.sheet.update_cell(row_index, self.bench_sheet_manager.col_index, value)
                self.logger.info(f"Updated benchmark {column} for model {model_name}")

        except Exception as e:
            self.logger.error(f"Error updating benchmarks: {str(e)}")
            raise

    def check_model_exists(self, model_name: str) -> bool:
        """Check if model exists in the sheet."""
        try:
            self.bench_sheet_manager.change_column("Model name")
            values = self.bench_sheet_manager.get_all_values()
            return model_name in values
        except Exception as e:
            self.logger.error(f"Error checking model existence: {str(e)}")
            return False


def process_model_benchmarks(
    model_name: str,
    bench_checker: SheetChecker,
    model_info_func,
    benchmark_processor_func: callable,
    benchmark_columns: List[str],
    cfg_prompt: str
) -> None:
    """
    Process model benchmarks according to the specified workflow.
    
    Args:
        model_name: Name of the model to process
        bench_checker: SheetChecker instance
        model_info_func: Function that returns model info (name, link, etc.)
        benchmark_processor_func: Function that processes empty benchmarks and returns values
        benchmark_columns: List of benchmark columns to check
    """
    try:
        # Check model and benchmarks
        check_result = bench_checker.check_model_and_benchmarks(model_name, benchmark_columns)
        
        # Handle invalid benchmark columns
        if check_result['invalid_benchmarks']:
            bench_checker.logger.warning(
                f"Skipping invalid benchmark columns: {', '.join(check_result['invalid_benchmarks'])}"
            )

        # If model doesn't exist, add it
        if check_result['status'] == 'model_not_found':
            model_info = model_info_func(model_name)
            bench_checker.update_model_info(model_name, model_info)
            bench_checker.logger.info(f"Added new model: {model_name}")
            # Recheck benchmarks after adding model
            check_result = bench_checker.check_model_and_benchmarks(model_name, benchmark_columns)

        # Log filled benchmarks
        if check_result['filled_benchmarks']:
            bench_checker.logger.info(
                f"Skipping filled benchmark columns: {', '.join(check_result['filled_benchmarks'])}"
            )

        # Process empty benchmarks
        if check_result['empty_benchmarks']:
            bench_checker.logger.info(
                f"Processing empty benchmark columns: {', '.join(check_result['empty_benchmarks'])}"
            )
            # Get benchmark values from processor function
            benchmark_values = benchmark_processor_func(
                model_name, 
                check_result['empty_benchmarks'],
                cfg_prompt
            )
            # Update benchmarks
            bench_checker.update_benchmarks(model_name, benchmark_values)
        else:
            bench_checker.logger.info("No empty benchmark columns to process")
        
    except Exception as e:
        bench_checker.logger.error(f"Error processing model {model_name}: {str(e)}")
        raise

def get_model_info(model_name: str) -> Dict[str, str]:
    return {
        "Model name": model_name,
        "Model link": f"https://huggingface.co/PIA-SPACE-LAB/{model_name}",
        "Model": f'<a target="_blank" href="https://huggingface.co/PIA-SPACE-LAB/{model_name}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'
        
    }

def process_benchmarks(
    model_name: str,
    empty_benchmarks: List[str],
    cfg_prompt: str
) -> Dict[str, str]:
    """
    Measure benchmark scores for given model with specific configuration.
    
    Args:
        model_name: Name of the model to evaluate
        empty_benchmarks: List of benchmarks to measure
        cfg_prompt: Prompt configuration for evaluation
        
    Returns:
        Dict[str, str]: Dictionary mapping benchmark names to their scores
    """
    result = {}
    for benchmark in empty_benchmarks:
        # 실제 벤치마크 측정 수행
        # score = measure_benchmark(model_name, benchmark, cfg_prompt)
        if benchmark == "COCO":
            score = 0.5
        elif benchmark == "ImageNet":
            score = 15.0
        result[benchmark] = str(score)
    return result
# Example usage
if __name__ == "__main__":
    
    sheet_manager = SheetManager()
    bench_checker = SheetChecker(sheet_manager)
    
    process_model_benchmarks(
        "test-model",
        bench_checker,
        get_model_info,
        process_benchmarks,
        ["COCO", "ImageNet"],
        "cfg_prompt_value"
    )