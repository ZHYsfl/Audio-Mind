"""
Main entry point for the audio agent framework.

Provides convenience functions for running the agent.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

from audio_agent.core.state import AgentState, create_initial_state
from audio_agent.core.constants import AgentStatus
from audio_agent.core.schemas import FinalAnswer, AudioItem, AudioOutput
from audio_agent.core.logging import setup_logger, set_debug_mode, log_info
from audio_agent.config.settings import AgentConfig
from audio_agent.graph.builder import build_graph
from audio_agent.frontend.base import BaseFrontend
from audio_agent.planner.base import BasePlanner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.fusion.base import BaseEvidenceFuser
from audio_agent.log import RunLogger


class AudioAgent:
    """
    Main audio agent class.
    
    Encapsulates the LangGraph workflow and provides a clean interface
    for running the agent on audio queries.
    """
    
    def __init__(
        self,
        frontend: BaseFrontend,
        planner: BasePlanner,
        registry: ToolRegistry,
        fuser: BaseEvidenceFuser,
        config: AgentConfig | None = None,
    ) -> None:
        """
        Initialize the audio agent.
        
        Args:
            frontend: Frontend for initial audio processing
            planner: Planner for decision making
            registry: Tool registry with available tools
            fuser: Evidence fuser for tool results
            config: Optional configuration
        """
        if frontend is None:
            raise ValueError("frontend cannot be None")
        if planner is None:
            raise ValueError("planner cannot be None")
        if registry is None:
            raise ValueError("registry cannot be None")
        if fuser is None:
            raise ValueError("fuser cannot be None")
        
        self.frontend = frontend
        self.planner = planner
        self.registry = registry
        self.fuser = fuser
        self.config = config or AgentConfig()
        # Config is the single source of truth for direct-answer mode: push it onto
        # frontends that support the mode so frontend behavior and graph wiring agree.
        # Frontends without a settable direct_answer are QoP-only and keep the QoP node.
        if isinstance(getattr(type(frontend), "direct_answer", None), property):
            frontend.direct_answer = self.config.frontend_direct_answer
        self._temp_dir: str | None = None
        
        # Set up logging
        setup_logger()
        if self.config.debug:
            set_debug_mode(True)
        
        # Build the graph
        self._graph = build_graph(frontend, planner, registry, fuser, config=self.config)
        
        # Set up run logger
        self._run_logger: RunLogger | None = None
        if self.config.enable_run_logging:
            self._run_logger = RunLogger(log_dir=self.config.log_dir)
    
    def _setup_temp_dir(self, audio_paths: list[str]) -> tuple[str, list[AudioItem]]:
        """
        Create temp directory and copy original audio(s).
        
        Creates a temp directory at {temp_dir_base}/agent_{timestamp}_{random}/
        and copies the original audio(s) as audio_0, audio_1, etc.
        
        Args:
            audio_paths: List of paths to the original audio files
            
        Returns:
            Tuple of (temp_dir_path, audio_list)
        """
        import random
        import string
        
        # Generate temp directory name (use absolute path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        temp_dir_name = f"agent_{timestamp}_{random_suffix}"
        temp_dir = os.path.abspath(os.path.join(self.config.temp_dir_base, temp_dir_name))
        
        # Create directory
        os.makedirs(temp_dir, exist_ok=True)
        log_info("temp_dir_created", {"path": temp_dir})
        
        # Copy all original audios
        audio_list: list[AudioItem] = []
        for i, audio_path in enumerate(audio_paths):
            original_ext = Path(audio_path).suffix or ".wav"
            dest_path = os.path.join(temp_dir, f"audio_{i}{original_ext}")
            shutil.copy2(audio_path, dest_path)
            log_info("audio_copied", {"source": audio_path, "dest": dest_path})
            
            # Create audio item with absolute path
            audio_item = AudioItem(
                audio_id=f"audio_{i}",
                path=dest_path,
                source="original",
                description=f"input audio {i}",
            )
            audio_list.append(audio_item)
        
        self._temp_dir = temp_dir
        return temp_dir, audio_list
    
    def cleanup(self, temp_dir: str | None = None) -> None:
        """
        Clean up temporary files and directory.
        
        Removes the temp directory if it exists and cleanup is enabled.
        
        Args:
            temp_dir: Specific temp directory to clean up. If None, cleans up
                     the last temp directory stored on this instance.
        """
        target_dir = temp_dir or self._temp_dir
        if target_dir and os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
                log_info("temp_dir_cleaned", {"path": target_dir})
                if target_dir == self._temp_dir:
                    self._temp_dir = None
            except Exception as e:
                log_info("temp_dir_cleanup_failed", {"path": target_dir, "error": str(e)})
    
    def _setup_output_dir(self) -> str:
        """
        Create output directory if it doesn't exist.
        
        Returns:
            Absolute path to the output directory
        """
        output_dir = os.path.abspath(self.config.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        log_info("output_dir_ready", {"path": output_dir})
        return output_dir
    
    def _copy_output_audio(self, final_answer: FinalAnswer) -> AudioOutput | None:
        """
        Copy output audio to output_dir if configured.
        
        Args:
            final_answer: The final answer containing output_audio
            
        Returns:
            Updated AudioOutput with the new path, or None if no copy was made
        """
        if not self.config.copy_output_to_dir:
            return final_answer.output_audio
        
        if not final_answer.output_audio:
            return None
        
        src_path = final_answer.output_audio.path
        if not os.path.exists(src_path):
            log_info("output_audio_source_not_found", {"path": src_path})
            return final_answer.output_audio
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(src_path).stem
        ext = Path(src_path).suffix
        filename = f"{original_name}_{timestamp}{ext}"
        dest_path = os.path.join(self.config.output_dir, filename)
        
        try:
            shutil.copy2(src_path, dest_path)
            log_info("output_audio_copied", {"source": src_path, "dest": dest_path})
            
            # Return updated AudioOutput with new path
            return AudioOutput(
                audio_id=final_answer.output_audio.audio_id,
                path=dest_path,
                description=final_answer.output_audio.description,
                metadata={
                    **final_answer.output_audio.metadata,
                    "original_path": src_path,
                    "copied_to_output": True,
                }
            )
        except Exception as e:
            log_info("output_audio_copy_failed", {"source": src_path, "error": str(e)})
            return final_answer.output_audio
    
    def run(
        self,
        question: str,
        audio_paths: list[str],
        max_steps: int | None = None,
    ) -> AgentState:
        """
        Run the agent on an audio query (synchronous).
        
        Args:
            question: User question about the audio
            audio_paths: List of paths to audio files (one or more)
            max_steps: Override default max_steps
        
        Returns:
            Final agent state with answer or error
        """
        import asyncio
        # Use asyncio.run to execute the async version
        return asyncio.run(self.arun(question, audio_paths, max_steps))
    
    async def arun(
        self,
        question: str,
        audio_paths: list[str],
        max_steps: int | None = None,
        run_log_name: str | None = None,
    ) -> AgentState:
        """
        Run the agent on an audio query (asynchronous).
        
        Required when using MCP tools or async tool execution.
        
        Args:
            question: User question about the audio
            audio_paths: List of paths to audio files (one or more)
            max_steps: Override default max_steps
        
        Returns:
            Final agent state with answer or error
        """
        effective_max_steps = max_steps if max_steps is not None else self.config.max_steps
        
        # Setup directories
        temp_dir, audio_list = self._setup_temp_dir(audio_paths)
        self._setup_output_dir()
        
        initial_state = create_initial_state(
            question=question,
            audio_paths=audio_paths,
            max_steps=effective_max_steps,
            temp_dir=temp_dir,
            audio_list=audio_list,
            config=self.config.model_dump(),
        )
        
        try:
            # Execute the graph asynchronously.
            # Each tool round is ~3 node hops (planner -> tool -> fusion), plus the
            # frontend/plan prelude and the summarize/final/format/answer tail. Give the
            # recursion limit enough headroom for max_steps rounds so legitimate long runs
            # are not aborted by LangGraph's low default (25), while still bounding any
            # unforeseen loop.
            recursion_limit = effective_max_steps * 5 + 25
            final_state = await self._graph.ainvoke(
                initial_state, config={"recursion_limit": recursion_limit}
            )
            
            # Copy output audio if present
            final_answer = final_state.get("final_answer")
            if final_answer and final_answer.output_audio:
                updated_audio = self._copy_output_audio(final_answer)
                if updated_audio and updated_audio.path != final_answer.output_audio.path:
                    # Update the final_state with the new output_audio path
                    final_answer.output_audio = updated_audio
            
            # Log the run
            if self._run_logger:
                log_path = self._run_logger.log_run(final_state, custom_name=run_log_name)
                if log_path:
                    log_info("run_logged", {"log_file": log_path})
            
            return final_state
        finally:
            # Cleanup if enabled — pass the specific temp_dir so concurrent
            # runs don't accidentally delete each other's directories.
            if self.config.cleanup_temp_on_exit:
                self.cleanup(temp_dir)
    
    def get_answer(self, state: AgentState) -> FinalAnswer | None:
        """Extract the final answer from a completed state."""
        return state.get("final_answer")
    
    def get_status(self, state: AgentState) -> AgentStatus:
        """Extract the status from a state."""
        return state.get("status", AgentStatus.RUNNING)
    
    def is_successful(self, state: AgentState) -> bool:
        """Check if the agent completed successfully with an answer."""
        return state.get("status") == AgentStatus.ANSWERED


def create_dummy_agent(config: AgentConfig | None = None) -> AudioAgent:
    """
    Create an audio agent with dummy components for testing.
    
    Args:
        config: Optional configuration
    
    Returns:
        AudioAgent instance with dummy components
    """
    from audio_agent.frontend.dummy_frontend import DummyFrontend
    from audio_agent.planner.dummy_planner import DummyPlanner
    from audio_agent.tools.dummy_tools import DummyASRTool, DummyAudioEventDetectorTool
    from audio_agent.fusion.default_fuser import DefaultEvidenceFuser
    
    if config is None:
        config = AgentConfig(planner_tool_scope="all")

    frontend = DummyFrontend()
    planner = DummyPlanner()
    registry = ToolRegistry()
    registry.register(DummyASRTool())
    registry.register(DummyAudioEventDetectorTool())
    fuser = DefaultEvidenceFuser()
    
    return AudioAgent(
        frontend=frontend,
        planner=planner,
        registry=registry,
        fuser=fuser,
        config=config,
    )


def create_openai_planner(
    model: str = "qwen3.5-plus",
    api_key: str | None = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    enable_thinking: bool = False,
    **kwargs,
) -> "OpenAICompatiblePlanner":
    """
    Create an OpenAI-compatible API planner.
    
    Works with qwen3.5-plus, kimi-k2.5, OpenAI models, and any other
    OpenAI-compatible API endpoint.
    
    Args:
        model: Model name (e.g., "qwen3.5-plus", "kimi-k2.5", "gpt-4")
        api_key: API key. If None, reads from DASHSCOPE_API_KEY or OPENAI_API_KEY env var.
        base_url: API base URL
        enable_thinking: Enable thinking mode for models that support it (qwen3.5-plus)
        **kwargs: Additional arguments passed to OpenAICompatiblePlanner
    
    Returns:
        Configured OpenAICompatiblePlanner instance
    
    Example:
        # qwen3.5-plus with thinking
        planner = create_openai_planner(
            model="qwen3.5-plus",
            api_key="sk-xxx",
            enable_thinking=True,
        )
        
        # kimi-k2.5
        planner = create_openai_planner(
            model="kimi-k2.5",
            api_key="sk-xxx",
        )
        
        # Using environment variable for API key
        import os
        os.environ["DASHSCOPE_API_KEY"] = "sk-xxx"
        planner = create_openai_planner(model="qwen3.5-plus")
    """
    from audio_agent.planner.openai_compatible_planner import OpenAICompatiblePlanner
    
    return OpenAICompatiblePlanner(
        model=model,
        api_key=api_key,
        base_url=base_url,
        enable_thinking=enable_thinking,
        **kwargs,
    )
