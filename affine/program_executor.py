import sys
import subprocess
import tempfile
import os
from typing import Tuple
from contextlib import contextmanager
from threading import Lock
from .val_config import DEFAULT_PROGRAM_EXECUTION_TIMEOUT

class ProgramExecutor:
    """Handles execution of Python programs with input data"""
    
    def __init__(self):
        self.temp_files = []
        self.temp_files_lock = Lock()
    
    @contextmanager
    def create_temp_file(self, content: str, suffix: str = '.py'):
        """Context manager for creating and cleaning up temporary files"""
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_file = f.name
            
            with self.temp_files_lock:
                self.temp_files.append(temp_file)
            
            yield temp_file
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file)
                    with self.temp_files_lock:
                        if temp_file in self.temp_files:
                            self.temp_files.remove(temp_file)
                except Exception:
                    pass
    
    def execute_program(self, program: str, input_data: str, timeout: int = DEFAULT_PROGRAM_EXECUTION_TIMEOUT) -> Tuple[str, str]:
        """Execute program with input data and return output and error"""
        try:
            # Clean up program code
            program = self._clean_program_code(program)
            
            # Use context manager for temp file
            with self.create_temp_file(program, '.py') as prog_file_path:
                process = subprocess.run(
                    [sys.executable, prog_file_path],
                    input=input_data,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    encoding='utf-8'
                )
                
                return process.stdout, process.stderr
                
        except subprocess.TimeoutExpired:
            return "", "Program execution timed out"
        except Exception as e:
            return "", f"Execution error: {str(e)}"
    
    def _clean_program_code(self, program: str) -> str:
        """Clean up program code by removing markdown formatting"""
        if program.startswith('```python'):
            program = program.replace('```python', '').replace('```', '').strip()
        elif program.startswith('```'):
            program = program.replace('```', '').strip()
        return program
    
    def cleanup(self):
        """Clean up temporary files"""
        with self.temp_files_lock:
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception:
                    pass
            self.temp_files.clear() 