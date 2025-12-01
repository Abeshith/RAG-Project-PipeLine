import sys
import traceback
from typing import Optional


class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(error_message)
        self.error_message = error_message
        
        if error_detail:
            _, _, exc_tb = error_detail
            self.error_message = self._get_detailed_error_message(error_message, exc_tb)
    
    def _get_detailed_error_message(self, error_message: str, exc_tb) -> str:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        return f"Error in [{file_name}] at line [{line_number}]: {error_message}"
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.error_message}')"

