# Mantus Building Manual: Phase 5-6
## Self-Correction, Multimodal Integration, and Continuous Learning

**Author:** Manus AI  
**Version:** 1.0  
**Date:** October 22, 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Phase 5: Self-Correction and Error Recovery](#phase-5-self-correction-and-error-recovery)
3. [Phase 6: Multimodal Integration and Continuous Learning](#phase-6-multimodal-integration-and-continuous-learning)
4. [References](#references)

---

## Introduction

Phases 5 and 6 elevate Mantus from a capable agent into an intelligent, self-improving system. These phases implement mechanisms for detecting and correcting errors, processing multiple modalities (text, images, audio), and continuously learning from interactions.

### Key Capabilities Enabled

**Self-Correction:** Mantus can detect failures, analyze root causes, and autonomously correct course.  
**Multimodal Understanding:** Mantus can process and reason about text, images, and audio simultaneously.  
**Continuous Learning:** Mantus improves over time through feedback loops and reinforcement learning.

---

## Phase 5: Self-Correction and Error Recovery

### 5.1 Error Detection and Analysis

Create `core/error_handler.py` for comprehensive error management:

```python
"""
Error Handler: Detection, analysis, and recovery from errors
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import traceback

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ErrorCategory(Enum):
    """Categories of errors"""
    TOOL_EXECUTION = "tool_execution"
    LLM_INFERENCE = "llm_inference"
    MEMORY_ACCESS = "memory_access"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    EXTERNAL_API = "external_api"
    UNKNOWN = "unknown"


@dataclass
class ErrorReport:
    """Detailed error report"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    timestamp: float
    traceback: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None


class ErrorDetector:
    """Detects and analyzes errors"""

    def __init__(self):
        self.logger = logging.getLogger("ErrorDetector")
        self.error_history: List[ErrorReport] = []

    def detect_and_report(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> ErrorReport:
        """Detect and create an error report"""
        import time
        
        # Determine severity based on error type
        severity = self._determine_severity(error)
        
        # Create error report
        report = ErrorReport(
            error_id=f"err_{int(time.time() * 1000)}",
            category=category,
            severity=severity,
            message=str(error),
            context=context,
            timestamp=time.time(),
            traceback=traceback.format_exc()
        )
        
        self.error_history.append(report)
        self.logger.error(f"Error detected: {report.error_id} - {report.message}")
        
        return report

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity"""
        if isinstance(error, (TimeoutError, ConnectionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, ValueError):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, Exception):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def get_error_patterns(self) -> Dict[ErrorCategory, int]:
        """Analyze error patterns"""
        patterns = {}
        for report in self.error_history:
            category = report.category
            patterns[category] = patterns.get(category, 0) + 1
        return patterns


class ErrorRecoveryStrategy:
    """Strategy for recovering from errors"""

    def __init__(self, name: str, handler: Callable):
        self.name = name
        self.handler = handler
        self.success_rate = 0.0
        self.attempts = 0
        self.successes = 0

    async def apply(self, error_report: ErrorReport) -> bool:
        """Apply recovery strategy"""
        try:
            self.attempts += 1
            result = await self.handler(error_report)
            if result:
                self.successes += 1
                self.success_rate = self.successes / self.attempts
            return result
        except Exception as e:
            logger.error(f"Recovery strategy '{self.name}' failed: {e}")
            return False


class ErrorRecoveryManager:
    """Manages error recovery strategies"""

    def __init__(self):
        self.logger = logging.getLogger("ErrorRecoveryManager")
        self.strategies: Dict[ErrorCategory, List[ErrorRecoveryStrategy]] = {}

    def register_strategy(
        self,
        category: ErrorCategory,
        strategy: ErrorRecoveryStrategy
    ) -> None:
        """Register a recovery strategy"""
        if category not in self.strategies:
            self.strategies[category] = []
        self.strategies[category].append(strategy)
        self.logger.info(f"Registered recovery strategy '{strategy.name}' for {category.value}")

    async def recover(self, error_report: ErrorReport) -> bool:
        """Attempt to recover from error"""
        strategies = self.strategies.get(error_report.category, [])
        
        # Sort by success rate
        strategies = sorted(strategies, key=lambda s: s.success_rate, reverse=True)
        
        for strategy in strategies:
            self.logger.info(f"Attempting recovery with strategy: {strategy.name}")
            if await strategy.apply(error_report):
                error_report.recovery_attempted = True
                error_report.recovery_successful = True
                error_report.recovery_method = strategy.name
                return True
        
        return False


# Built-in recovery strategies

async def retry_with_backoff(error_report: ErrorReport) -> bool:
    """Retry the failed operation with exponential backoff"""
    # This would be implemented based on the specific error context
    return False


async def fallback_to_alternative_tool(error_report: ErrorReport) -> bool:
    """Fall back to an alternative tool"""
    # This would be implemented based on available alternatives
    return False


async def simplify_and_retry(error_report: ErrorReport) -> bool:
    """Simplify the task and retry"""
    # This would be implemented based on task decomposition
    return False
```

### 5.2 Self-Correction Loop

Create `core/self_corrector.py` for autonomous error correction:

```python
"""
Self-Corrector: Autonomous error detection and correction
"""

from typing import Optional, Dict, Any
import logging

from core.llm_component import BaseLLM
from core.error_handler import ErrorReport, ErrorDetector, ErrorRecoveryManager

logger = logging.getLogger(__name__)


class SelfCorrector:
    """Implements self-correction mechanisms"""

    def __init__(
        self,
        llm: BaseLLM,
        error_detector: ErrorDetector,
        recovery_manager: ErrorRecoveryManager
    ):
        self.llm = llm
        self.error_detector = error_detector
        self.recovery_manager = recovery_manager
        self.logger = logging.getLogger("SelfCorrector")
        self.correction_history: Dict[str, Any] = {}

    async def analyze_and_correct(
        self,
        error_report: ErrorReport,
        original_task: str
    ) -> Optional[str]:
        """
        Analyze an error and attempt correction
        """
        # Step 1: Analyze the error with LLM
        analysis_prompt = f"""
        An error occurred while executing the following task:
        
        Task: {original_task}
        
        Error Category: {error_report.category.value}
        Error Message: {error_report.message}
        Error Context: {error_report.context}
        
        Please analyze:
        1. What went wrong?
        2. Why did it happen?
        3. What should be done differently?
        4. What is the corrected approach?
        """

        system_prompt = """You are an expert at analyzing and correcting errors in AI agent operations.
        Provide a detailed analysis and a corrected approach."""

        analysis = self.llm.generate(
            prompt=analysis_prompt,
            system_prompt=system_prompt
        )

        self.logger.info(f"Error analysis for {error_report.error_id}:\n{analysis}")

        # Step 2: Attempt automated recovery
        recovery_success = await self.recovery_manager.recover(error_report)

        # Step 3: If automated recovery fails, propose manual intervention
        if not recovery_success:
            self.logger.warning(f"Automated recovery failed for {error_report.error_id}")
            return f"Error analysis:\n{analysis}\n\nAutomated recovery failed. Manual intervention required."

        # Step 4: Verify correction
        verification_prompt = f"""
        The following error was corrected:
        
        Original Error: {error_report.message}
        Recovery Method: {error_report.recovery_method}
        
        Verify that the correction is appropriate and complete.
        """

        verification = self.llm.generate(
            prompt=verification_prompt,
            system_prompt="Verify the error correction."
        )

        self.correction_history[error_report.error_id] = {
            "analysis": analysis,
            "recovery_method": error_report.recovery_method,
            "verification": verification,
            "success": recovery_success
        }

        return f"Error corrected using {error_report.recovery_method}"

    def get_correction_insights(self) -> Dict[str, Any]:
        """Get insights from correction history"""
        total_corrections = len(self.correction_history)
        successful = sum(
            1 for c in self.correction_history.values()
            if c.get("success", False)
        )
        
        return {
            "total_corrections": total_corrections,
            "successful_corrections": successful,
            "success_rate": successful / total_corrections if total_corrections > 0 else 0.0,
            "error_patterns": self.error_detector.get_error_patterns()
        }
```

---

## Phase 6: Multimodal Integration and Continuous Learning

### 6.1 Multimodal Processing

Create `core/multimodal_processor.py` for handling multiple modalities:

```python
"""
Multimodal Processor: Handle text, images, and audio
"""

from typing import Union, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging
import base64
from pathlib import Path

logger = logging.getLogger(__name__)


class Modality(Enum):
    """Supported modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MediaInput:
    """Represents a media input"""
    modality: Modality
    content: Union[str, bytes]
    metadata: Dict[str, Any]
    format: Optional[str] = None


class ImageProcessor:
    """Process image inputs"""

    @staticmethod
    async def encode_image(image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    async def analyze_image(image_path: str, llm) -> str:
        """Analyze an image using vision capabilities"""
        # This would integrate with a vision model
        # For now, return placeholder
        return f"Image analysis for {image_path}"

    @staticmethod
    async def extract_text_from_image(image_path: str) -> str:
        """Extract text from image (OCR)"""
        # This would integrate with OCR capabilities
        return f"Text extracted from {image_path}"


class AudioProcessor:
    """Process audio inputs"""

    @staticmethod
    async def transcribe_audio(audio_path: str) -> str:
        """Transcribe audio to text"""
        # This would integrate with speech-to-text
        return f"Transcription of {audio_path}"

    @staticmethod
    async def analyze_audio(audio_path: str) -> Dict[str, Any]:
        """Analyze audio for sentiment, emotion, etc."""
        # This would integrate with audio analysis
        return {
            "duration": 0,
            "language": "unknown",
            "sentiment": "neutral"
        }


class MultimodalProcessor:
    """Process multiple modalities together"""

    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger("MultimodalProcessor")
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()

    async def process(self, inputs: List[MediaInput]) -> str:
        """Process multiple modality inputs"""
        processed_content = []

        for media_input in inputs:
            if media_input.modality == Modality.TEXT:
                processed_content.append(media_input.content)
            
            elif media_input.modality == Modality.IMAGE:
                # Process image
                image_analysis = await self.image_processor.analyze_image(
                    media_input.content, self.llm
                )
                processed_content.append(f"[Image Analysis]: {image_analysis}")
            
            elif media_input.modality == Modality.AUDIO:
                # Process audio
                transcription = await self.audio_processor.transcribe_audio(
                    media_input.content
                )
                processed_content.append(f"[Audio Transcription]: {transcription}")

        # Combine all processed content
        combined_content = "\n".join(processed_content)

        # Generate response considering all modalities
        prompt = f"""
        Process the following multimodal input and provide a comprehensive response:
        
        {combined_content}
        """

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert at processing and reasoning about multimodal information."
        )

        return response
```

### 6.2 Continuous Learning and RLHF

Create `core/continuous_learning.py` for learning from interactions:

```python
"""
Continuous Learning: Learn from interactions and feedback
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """A single interaction for learning"""
    user_input: str
    agent_response: str
    feedback: Optional[float] = None  # -1 to 1 scale
    feedback_text: Optional[str] = None
    timestamp: float = 0.0
    success: bool = False


class FeedbackCollector:
    """Collects feedback on agent performance"""

    def __init__(self):
        self.logger = logging.getLogger("FeedbackCollector")
        self.interactions: List[Interaction] = []

    def collect_feedback(
        self,
        user_input: str,
        agent_response: str,
        feedback: float,
        feedback_text: Optional[str] = None
    ) -> None:
        """Collect feedback on an interaction"""
        import time
        
        interaction = Interaction(
            user_input=user_input,
            agent_response=agent_response,
            feedback=feedback,
            feedback_text=feedback_text,
            timestamp=time.time(),
            success=feedback > 0
        )
        
        self.interactions.append(interaction)
        self.logger.info(f"Feedback collected: score={feedback}")

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback"""
        if not self.interactions:
            return {"total_interactions": 0}

        total = len(self.interactions)
        positive = sum(1 for i in self.interactions if i.success)
        average_score = sum(i.feedback for i in self.interactions if i.feedback) / total

        return {
            "total_interactions": total,
            "positive_feedback": positive,
            "negative_feedback": total - positive,
            "average_score": average_score,
            "success_rate": positive / total
        }


class LearningModel:
    """Model for continuous learning"""

    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger("LearningModel")
        self.feedback_collector = FeedbackCollector()
        self.learned_patterns: Dict[str, str] = {}

    async def learn_from_feedback(self, interaction: Interaction) -> None:
        """Learn from a single interaction"""
        if interaction.feedback is None:
            return

        # Extract patterns from successful interactions
        if interaction.success:
            pattern_prompt = f"""
            Analyze this successful interaction and extract the key pattern:
            
            User Input: {interaction.user_input}
            Agent Response: {interaction.agent_response}
            Feedback: {interaction.feedback_text or 'Positive'}
            
            What pattern or principle made this response successful?
            """

            pattern = self.llm.generate(
                prompt=pattern_prompt,
                system_prompt="Extract the key successful pattern from this interaction."
            )

            # Store learned pattern
            pattern_key = f"pattern_{len(self.learned_patterns)}"
            self.learned_patterns[pattern_key] = pattern
            self.logger.info(f"Learned pattern: {pattern_key}")

    async def apply_learned_patterns(self, context: str) -> str:
        """Apply learned patterns to new context"""
        if not self.learned_patterns:
            return ""

        patterns_text = "\n".join(
            f"- {pattern}" for pattern in self.learned_patterns.values()
        )

        prompt = f"""
        Apply the following learned patterns to this context:
        
        Learned Patterns:
        {patterns_text}
        
        Current Context: {context}
        
        How should these patterns guide the response?
        """

        guidance = self.llm.generate(
            prompt=prompt,
            system_prompt="Apply learned patterns to provide guidance."
        )

        return guidance

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get metrics on learning progress"""
        return {
            "patterns_learned": len(self.learned_patterns),
            "feedback_summary": self.feedback_collector.get_feedback_summary(),
            "learning_efficiency": len(self.learned_patterns) / max(
                self.feedback_collector.get_feedback_summary().get("total_interactions", 1), 1
            )
        }
```

### 6.3 Integration with Main Agent Loop

Create `core/advanced_agent.py` to integrate all Phase 5-6 capabilities:

```python
"""
Advanced Agent: Integrates self-correction, multimodal processing, and continuous learning
"""

from typing import Optional, List, Dict, Any
import logging

from core.llm_component import BaseLLM
from core.tool_orchestrator import ToolOrchestrator
from core.memory import MemoryManager
from core.error_handler import ErrorDetector, ErrorRecoveryManager, ErrorCategory
from core.self_corrector import SelfCorrector
from core.multimodal_processor import MultimodalProcessor, MediaInput
from core.continuous_learning import LearningModel, FeedbackCollector

logger = logging.getLogger(__name__)


class AdvancedMantusAgent:
    """Advanced Mantus agent with self-correction, multimodal, and learning capabilities"""

    def __init__(
        self,
        llm: BaseLLM,
        tool_orchestrator: ToolOrchestrator,
        memory_manager: MemoryManager,
        session_id: str
    ):
        self.llm = llm
        self.tool_orchestrator = tool_orchestrator
        self.memory_manager = memory_manager
        self.session_id = session_id
        self.logger = logging.getLogger("AdvancedMantusAgent")

        # Initialize advanced components
        self.error_detector = ErrorDetector()
        self.recovery_manager = ErrorRecoveryManager()
        self.self_corrector = SelfCorrector(llm, self.error_detector, self.recovery_manager)
        self.multimodal_processor = MultimodalProcessor(llm)
        self.learning_model = LearningModel(llm)

    async def process_request(
        self,
        user_input: str,
        media_inputs: Optional[List[MediaInput]] = None,
        collect_feedback: bool = True
    ) -> str:
        """
        Process a user request with full advanced capabilities
        """
        try:
            # Step 1: Process multimodal inputs if provided
            if media_inputs:
                multimodal_response = await self.multimodal_processor.process(media_inputs)
                combined_input = f"{user_input}\n\n[Multimodal Analysis]:\n{multimodal_response}"
            else:
                combined_input = user_input

            # Step 2: Store in episodic memory
            await self.memory_manager.store_conversation("user", combined_input)

            # Step 3: Apply learned patterns
            learned_guidance = await self.learning_model.apply_learned_patterns(combined_input)

            # Step 4: Execute with tools
            response = await self.tool_orchestrator.execute_with_tools(
                user_prompt=combined_input,
                system_prompt=f"Use the following learned patterns:\n{learned_guidance}"
            )

            # Step 5: Store response in memory
            await self.memory_manager.store_conversation("assistant", response)

            # Step 6: Collect feedback if enabled
            if collect_feedback:
                # In production, this would be collected from the user
                self.learning_model.feedback_collector.collect_feedback(
                    user_input=combined_input,
                    agent_response=response,
                    feedback=0.5  # Placeholder
                )

            return response

        except Exception as e:
            # Error handling with self-correction
            error_report = self.error_detector.detect_and_report(
                error=e,
                category=ErrorCategory.UNKNOWN,
                context={"user_input": user_input}
            )

            correction_result = await self.self_corrector.analyze_and_correct(
                error_report=error_report,
                original_task=user_input
            )

            return correction_result

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent metrics"""
        return {
            "learning_metrics": self.learning_model.get_learning_metrics(),
            "error_insights": self.self_corrector.get_correction_insights(),
            "execution_history": self.tool_orchestrator.get_execution_history()
        }
```

---

## References

[1] Srinivasan, A. (2025). "Building Production-Ready AI Agents: A Full-Stack Blueprint for Reliability and Scalability." AI with Aish. Retrieved from https://aishwaryasrinivasan.substack.com/p/building-production-ready-ai-agents

[2] Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." arXiv preprint arXiv:2203.02155. Retrieved from https://arxiv.org/abs/2203.02155

[3] OpenAI. (2024). "Vision Capabilities." OpenAI API Documentation. Retrieved from https://platform.openai.com/docs/guides/vision

[4] Anthropic. (2024). "Multimodal Capabilities." Anthropic Documentation. Retrieved from https://docs.anthropic.com/claude/reference/vision

---

**End of Phase 5-6 Manual**

*Next: Phase 7-8 will cover Testing, Deployment, Monitoring, and Operations.*

