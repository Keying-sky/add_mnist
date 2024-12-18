from pathlib import Path

class SavePath:
    """Configuration class for project paths"""
    def __init__(self):
        self.ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT / 'data'
        self.MODEL_DIR = self.ROOT / 'model'
        self.RESULTS_DIR = self.ROOT / 'result'
        
        for dir in [self.DATA_DIR, self.MODEL_DIR, self.RESULTS_DIR]:
            dir.mkdir(parents=True, exist_ok=True)
