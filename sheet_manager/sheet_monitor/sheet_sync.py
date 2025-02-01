import threading
import time
from typing import Optional, Callable
import logging

class SheetMonitor:
    def __init__(self, sheet_manager, check_interval: float = 1.0):
        """
        Initialize SheetMonitor with a sheet manager instance.
        """
        self.sheet_manager = sheet_manager
        self.check_interval = check_interval
        
        # Threading control
        self.monitor_thread = None
        self.is_running = threading.Event()
        self.pause_monitoring = threading.Event()
        self.monitor_paused = threading.Event()
        
        # Queue status
        self.has_data = threading.Event()
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.logger.warning("Monitoring thread is already running")
            return

        self.is_running.set()
        self.pause_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Started monitoring thread")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.is_running.clear()
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Stopped monitoring thread")

    def pause(self):
        """Pause the monitoring."""
        self.pause_monitoring.set()
        self.monitor_paused.wait()
        self.logger.info("Monitoring paused")

    def resume(self):
        """Resume the monitoring."""
        self.pause_monitoring.clear()
        self.monitor_paused.clear()
        # 즉시 체크 수행
        self.logger.info("Monitoring resumed, checking for new data...")
        values = self.sheet_manager.get_all_values()
        if values:
            self.has_data.set()
            self.logger.info(f"Found data after resume: {values}")


    def _monitor_loop(self):
        """Main monitoring loop that checks for data in sheet."""
        while self.is_running.is_set():
            if self.pause_monitoring.is_set():
                self.monitor_paused.set()
                self.pause_monitoring.wait()
                self.monitor_paused.clear()
                # continue

            try:
                # Check if there's any data in the sheet
                values = self.sheet_manager.get_all_values()
                self.logger.info(f"Monitoring: Current column={self.sheet_manager.column_name}, "
                            f"Values found={len(values)}, "
                            f"Has data={self.has_data.is_set()}")
                
                if values:  # If there's any non-empty value
                    self.has_data.set()
                    self.logger.info(f"Data detected: {values}")
                else:
                    self.has_data.clear()
                    self.logger.info("No data in sheet, waiting...")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.check_interval)

class MainLoop:
    def __init__(self, sheet_manager, sheet_monitor, callback_function: Callable = None):
        """
        Initialize MainLoop with sheet manager and monitor instances.
        """
        self.sheet_manager = sheet_manager
        self.monitor = sheet_monitor
        self.callback = callback_function
        self.is_running = threading.Event()
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the main processing loop."""
        self.is_running.set()
        self.monitor.start_monitoring()
        self._main_loop()

    def stop(self):
        """Stop the main processing loop."""
        self.is_running.clear()
        self.monitor.stop_monitoring()

    def process_new_value(self):
        """Process values by calling pop function for multiple columns and custom callback."""
        try:
            # Store original column
            original_column = self.sheet_manager.column_name
            
            # Pop from huggingface_id column
            model_id = self.sheet_manager.pop()
            
            if model_id:
                # Pop from benchmark_name column
                self.sheet_manager.change_column("benchmark_name")
                benchmark_name = self.sheet_manager.pop()
                
                # Pop from prompt_cfg_name column
                self.sheet_manager.change_column("prompt_cfg_name")
                prompt_cfg_name = self.sheet_manager.pop()
                
                # Return to original column
                self.sheet_manager.change_column(original_column)
                
                self.logger.info(f"Processed values - model_id: {model_id}, "
                            f"benchmark_name: {benchmark_name}, "
                            f"prompt_cfg_name: {prompt_cfg_name}")
                
                if self.callback:
                    # Pass all three values to callback
                    self.callback(model_id, benchmark_name, prompt_cfg_name)
                    
                return model_id, benchmark_name, prompt_cfg_name
                
        except Exception as e:
            self.logger.error(f"Error processing values: {str(e)}")
            # Return to original column in case of error
            try:
                self.sheet_manager.change_column(original_column)
            except:
                pass
            return None

    def _main_loop(self):
        """Main processing loop."""
        while self.is_running.is_set():
            # Wait for data to be available
            if self.monitor.has_data.wait(timeout=1.0):
                # Pause monitoring
                self.monitor.pause()
                
                # Process the value
                self.process_new_value()
                
                # Check if there's still data in the sheet
                values = self.sheet_manager.get_all_values()
                self.logger.info(f"After processing: Current column={self.sheet_manager.column_name}, "
                            f"Values remaining={len(values)}")
                
                if not values:
                    self.monitor.has_data.clear()
                    self.logger.info("All data processed, clearing has_data flag")
                else:
                    self.logger.info(f"Remaining data: {values}")
                
                # Resume monitoring
                self.monitor.resume()
## TODO
# API 분당 호출 문제로 만약에 참조하다가 실패할 경우 대기했다가 다시 시도하게끔 설계


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from sheet_manager.sheet_crud.sheet_crud import SheetManager
    from pia_bench.pipe_line.piepline import PiaBenchMark
    def my_custom_function(huggingface_id, benchmark_name, prompt_cfg_name):
        piabenchmark = PiaBenchMark(huggingface_id, benchmark_name, prompt_cfg_name)
        piabenchmark.bench_start()

    # Initialize components
    sheet_manager = SheetManager()
    monitor = SheetMonitor(sheet_manager, check_interval=10.0)
    main_loop = MainLoop(sheet_manager, monitor, callback_function=my_custom_function)

    try:
        main_loop.start()
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        main_loop.stop()