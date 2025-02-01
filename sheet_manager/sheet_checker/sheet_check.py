from typing import Dict, Tuple
import logging
import gspread
from sheet_manager.sheet_crud.sheet_crud import SheetManager

class SheetChecker:
    def __init__(self, sheet_manager: SheetManager):
        """SheetChecker 초기화"""
        self.sheet_manager = sheet_manager
        self.bench_sheet_manager = None
        self.logger = logging.getLogger(__name__)
        self._init_bench_sheet()

    def _init_bench_sheet(self):
        """model 시트용 시트 매니저 초기화"""
        self.bench_sheet_manager = type(self.sheet_manager)(
            spreadsheet_url=self.sheet_manager.spreadsheet_url,
            worksheet_name="model",
            column_name="Model name"
        )

    def add_benchmark_column(self, column_name: str):
        """새로운 벤치마크 컬럼 추가"""
        try:
            headers = self.bench_sheet_manager.get_available_columns()
            
            if column_name in headers:
                return
            
            new_col_index = len(headers) + 1
            cell = gspread.utils.rowcol_to_a1(1, new_col_index)
            self.bench_sheet_manager.sheet.update(cell, [[column_name]])
            

           # 관련 컬럼 추가 (벤치마크이름*100)
            next_col_index = new_col_index + 1
            next_cell = gspread.utils.rowcol_to_a1(1, next_col_index)
            self.bench_sheet_manager.sheet.update(next_cell, [[f"{column_name}*100"]])
            
            self.logger.info(f"새로운 벤치마크 컬럼들 추가됨: {column_name}, {column_name}*100")
            # 컬럼 추가 후 시트 매니저 재연결
            self.bench_sheet_manager._connect_to_sheet(validate_column=False)
            
        except Exception as e:
            self.logger.error(f"벤치마크 컬럼 {column_name} 추가 중 오류 발생: {str(e)}")
            raise

    def check_model_and_benchmark(self, model_name: str, benchmark_name: str) -> Tuple[bool, bool]:
        """
        모델 존재 여부와 벤치마크 상태를 확인하고, 필요한 경우 모델 정보를 추가
        
        Args:
            model_name: 확인할 모델 이름
            benchmark_name: 확인할 벤치마크 이름
            
        Returns:
            Tuple[bool, bool]: (모델이 새로 추가되었는지 여부, 벤치마크가 이미 존재하는지 여부)
        """
        try:
            # 모델 존재 여부 확인
            model_exists = self._check_model_exists(model_name)
            model_added = False

            # 모델이 없으면 추가
            if not model_exists:
                self._add_new_model(model_name)
                model_added = True
                self.logger.info(f"새로운 모델 추가됨: {model_name}")

            # 벤치마크 컬럼이 없으면 추가
            available_columns = self.bench_sheet_manager.get_available_columns()
            if benchmark_name not in available_columns:
                self.add_benchmark_column(benchmark_name)
                self.logger.info(f"새로운 벤치마크 컬럼 추가됨: {benchmark_name}")

            # 벤치마크 상태 확인
            benchmark_exists = self._check_benchmark_exists(model_name, benchmark_name)
            
            return model_added, benchmark_exists

        except Exception as e:
            self.logger.error(f"모델/벤치마크 확인 중 오류 발생: {str(e)}")
            raise

    def _check_model_exists(self, model_name: str) -> bool:
        """모델 존재 여부 확인"""
        try:
            self.bench_sheet_manager.change_column("Model name")
            values = self.bench_sheet_manager.get_all_values()
            return model_name in values
        except Exception as e:
            self.logger.error(f"모델 존재 여부 확인 중 오류 발생: {str(e)}")
            raise

    def _add_new_model(self, model_name: str):
        """새로운 모델 정보 추가"""
        try:
            model_info = {
                "Model name": model_name,
                "Model link": f"https://huggingface.co/PIA-SPACE-LAB/{model_name}",
                "Model": f'<a target="_blank" href="https://huggingface.co/PIA-SPACE-LAB/{model_name}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'
            }
            
            for column_name, value in model_info.items():
                self.bench_sheet_manager.change_column(column_name)
                self.bench_sheet_manager.push(value)
                
        except Exception as e:
            self.logger.error(f"모델 정보 추가 중 오류 발생: {str(e)}")
            raise

    def _check_benchmark_exists(self, model_name: str, benchmark_name: str) -> bool:
        """벤치마크 값 존재 여부 확인"""
        try:
            # 해당 모델의 벤치마크 값 확인
            self.bench_sheet_manager.change_column("Model name")
            all_values = self.bench_sheet_manager.get_all_values()
            row_index = all_values.index(model_name) + 2

            self.bench_sheet_manager.change_column(benchmark_name)
            value = self.bench_sheet_manager.sheet.cell(row_index, self.bench_sheet_manager.col_index).value
            
            return bool(value and value.strip())
            
        except Exception as e:
            self.logger.error(f"벤치마크 존재 여부 확인 중 오류 발생: {str(e)}")
            raise

# 사용 예시
if __name__ == "__main__":
    sheet_manager = SheetManager()
    checker = SheetChecker(sheet_manager)
    
    model_added, benchmark_exists = checker.check_model_and_benchmark(
        model_name="test-model",
        benchmark_name="COCO"
    )
    
    print(f"Model added: {model_added}")
    print(f"Benchmark exists: {benchmark_exists}")