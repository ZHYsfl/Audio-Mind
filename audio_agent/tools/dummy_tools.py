"""
Dummy tool implementations for testing and development.

These tools return mock results without real processing.
"""

from audio_agent.core.schemas import ToolSpec, ToolCallRequest, ToolResult
from audio_agent.tools.base import BaseTool


class DummyASRTool(BaseTool):
    """
    Dummy ASR (Automatic Speech Recognition) tool.
    
    Returns mock transcription results.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="dummy_asr",
            description="Transcribe speech from audio to text",
            input_schema={
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string", "description": "Path to audio file"},
                },
                "required": ["audio_path"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "transcript": {"type": "string"},
                    "language": {"type": "string"},
                    "word_count": {"type": "integer"},
                },
            },
            tags=["speech", "transcription", "asr"],
        )
    
    def invoke(self, request: ToolCallRequest) -> ToolResult:
        """Return mock transcription."""
        self.validate_request(request)
        
        audio_path = request.args.get("audio_path", "unknown")
        
        # Mock transcription
        transcript = (
            "[DummyASR] Hello, this is a mock transcription. "
            "The audio appears to contain speech about the topic in question. "
            "Several key points were mentioned including relevant details."
        )
        
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            output={
                "transcript": transcript,
                "language": "en",
                "word_count": len(transcript.split()),
                "audio_path": audio_path,
            },
        )


class DummyAudioEventDetectorTool(BaseTool):
    """
    Dummy audio event detection tool.
    
    Returns mock detected audio events.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="dummy_audio_event_detector",
            description="Detect non-speech audio events (music, effects, environment)",
            input_schema={
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string", "description": "Path to audio file"},
                },
                "required": ["audio_path"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "events": {"type": "array", "items": {"type": "object"}},
                    "event_count": {"type": "integer"},
                },
            },
            tags=["audio", "events", "detection"],
        )
    
    def invoke(self, request: ToolCallRequest) -> ToolResult:
        """Return mock detected events."""
        self.validate_request(request)
        
        audio_path = request.args.get("audio_path", "unknown")
        
        # Mock events
        events = [
            {"event_type": "music", "start_time": 0.0, "end_time": 5.0, "confidence": 0.85},
            {"event_type": "applause", "start_time": 5.0, "end_time": 7.0, "confidence": 0.72},
            {"event_type": "background_noise", "start_time": 0.0, "end_time": 10.0, "confidence": 0.90},
        ]
        
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            output={
                "events": events,
                "event_count": len(events),
                "audio_path": audio_path,
            },
        )
