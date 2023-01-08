from typing import Any, List 

import numpy as np
from cog import BasePredictor, Input, Path, BaseModel
from stable_whisper import load_model, stabilize_timestamps
import whisper

class Predictor(BasePredictor):
    def setup(self):
        self.models = {}
        pass

    def get_model(self, model: str) -> Any: 
        if not model in whisper.available_models():
            raise Exception("Model %s not found")
        elif not model in self.models:
            self.models[model] = load_model(model) # from stable-ts
        return self.models[model]

    def predict(
        self,
        audio: Path = Input(description="Audio to transcribe"),
        model: str = Input(description="Whisper model to use", choices=whisper.available_models()),
        stabilize: bool = Input(description="Stabilize timestamps", default=True),
    ) -> Any:
        _whisper = self.get_model(model)
        result = _whisper.transcribe(str(audio)) # segments, transcription, detected_language
        
        if stabilize: result['segments'] = stabilize_timestamps(result, top_focus=True)
        
        return result