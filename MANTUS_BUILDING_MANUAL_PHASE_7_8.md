# Mantus Building Manual: Phase 7-8
## Testing, Deployment, Monitoring, and Operations

**Author:** Manus AI  
**Version:** 1.0  
**Date:** October 22, 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Phase 7: Comprehensive Testing and Quality Assurance](#phase-7-comprehensive-testing-and-quality-assurance)
3. [Phase 8: Deployment and Production Operations](#phase-8-deployment-and-production-operations)
4. [References](#references)

---

## Introduction

Phases 7 and 8 transition Mantus from a development project into a production-ready, operationally sound system. These phases implement rigorous testing strategies, deployment automation, monitoring infrastructure, and operational procedures to ensure Mantus performs reliably at scale.

### Key Objectives

**Quality Assurance:** Comprehensive testing at unit, integration, and system levels.  
**Deployment Automation:** Streamlined, repeatable deployment processes.  
**Observability:** Real-time monitoring, logging, and alerting.  
**Operational Excellence:** Procedures for maintenance, scaling, and incident response.

---

## Phase 7: Comprehensive Testing and Quality Assurance

### 7.1 Testing Strategy and Framework

Create `tests/test_strategy.md` to document the testing approach:

```markdown
# Mantus Testing Strategy

## Testing Pyramid

```
        /\
       /  \  System & E2E Tests
      /    \
     /------\
    /        \  Integration Tests
   /          \
  /            \
 /              \
/________________\  Unit Tests
```

## Test Coverage Goals

- **Unit Tests**: 85%+ code coverage
- **Integration Tests**: All major workflows
- **System Tests**: End-to-end user scenarios
- **Performance Tests**: Latency and throughput benchmarks

## Test Categories

### Unit Tests
- Individual functions and classes
- Isolated from external dependencies
- Fast execution (< 100ms per test)
- Examples: LLM wrapper, memory operations, tool registry

### Integration Tests
- Multiple components working together
- Real dependencies (Redis, APIs)
- Moderate execution time (< 5s per test)
- Examples: Tool orchestration, memory retrieval, error recovery

### System Tests
- End-to-end user workflows
- Production-like environment
- Longer execution time (< 30s per test)
- Examples: Complete task execution, multimodal processing

### Performance Tests
- Latency benchmarks
- Throughput testing
- Resource utilization
- Examples: LLM inference time, memory operations under load

### Security Tests
- API key handling
- Input validation
- Error message sanitization
- Examples: Credential leaks, injection attacks
```

### 7.2 Unit Testing Implementation

Create `tests/unit/test_core_components.py`:

```python
"""
Unit tests for core Mantus components
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from core.llm_component import LLMConfig, LLMProvider, OpenAILLM
from core.tool_registry import Tool, ToolSchema, ToolCategory, ToolParameter, ToolRegistry
from core.memory import MemoryManager, EpisodicMemory, SemanticMemory
from core.error_handler import ErrorDetector, ErrorCategory, ErrorSeverity


class TestLLMComponent:
    """Test LLM component"""

    @pytest.fixture
    def llm_config(self):
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo",
            api_key="test-key"
        )

    @patch('openai.OpenAI')
    def test_generate_text(self, mock_openai, llm_config):
        """Test basic text generation"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(llm_config)
        response = llm.generate("Test prompt")

        assert response == "Test response"
        assert mock_openai.return_value.chat.completions.create.called

    @patch('openai.OpenAI')
    def test_generate_with_tools(self, mock_openai, llm_config):
        """Test text generation with tools"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Using tool"
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(llm_config)
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        response = llm.generate_with_tools("Test prompt", tools)

        assert response["finish_reason"] == "tool_calls"
        assert len(response["tool_calls"]) > 0


class TestToolRegistry:
    """Test tool registry"""

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    @pytest.fixture
    def sample_tool(self):
        async def dummy_handler(**kwargs):
            return "result"

        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.SYSTEM,
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    description="Test parameter",
                    required=True
                )
            ],
            returns="Test result"
        )
        return Tool(schema, dummy_handler)

    def test_register_tool(self, registry, sample_tool):
        """Test tool registration"""
        registry.register(sample_tool)
        assert registry.get_tool("test_tool") is not None

    def test_get_tools_by_category(self, registry, sample_tool):
        """Test filtering tools by category"""
        registry.register(sample_tool)
        tools = registry.get_tools_by_category(ToolCategory.SYSTEM)
        assert len(tools) > 0
        assert tools[0].schema.name == "test_tool"

    def test_openai_schema_conversion(self, registry, sample_tool):
        """Test conversion to OpenAI schema format"""
        registry.register(sample_tool)
        schemas = registry.get_openai_schemas()
        assert len(schemas) > 0
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "test_tool"


class TestMemorySystem:
    """Test memory system"""

    @pytest.fixture
    def mock_redis(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_episodic_memory_store(self, mock_redis):
        """Test storing episodic memory"""
        memory = EpisodicMemory(mock_redis, "test_session")
        memory_id = await memory.store("Test content", {"key": "value"})
        
        assert memory_id is not None
        assert mock_redis.setex.called

    @pytest.mark.asyncio
    async def test_semantic_memory_store(self, mock_redis):
        """Test storing semantic memory"""
        memory = SemanticMemory(mock_redis)
        await memory.store("test_key", "Test content")
        
        assert mock_redis.set.called


class TestErrorHandling:
    """Test error handling"""

    def test_error_detection(self):
        """Test error detection"""
        detector = ErrorDetector()
        error = ValueError("Test error")
        
        report = detector.detect_and_report(
            error=error,
            category=ErrorCategory.VALIDATION,
            context={"test": "context"}
        )
        
        assert report.error_id is not None
        assert report.category == ErrorCategory.VALIDATION
        assert report.severity == ErrorSeverity.MEDIUM

    def test_error_severity_determination(self):
        """Test error severity determination"""
        detector = ErrorDetector()
        
        timeout_error = TimeoutError("Timeout")
        report = detector.detect_and_report(
            timeout_error,
            ErrorCategory.TIMEOUT,
            {}
        )
        assert report.severity == ErrorSeverity.HIGH


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=core", "--cov-report=html"])
```

### 7.3 Integration Testing

Create `tests/integration/test_agent_workflows.py`:

```python
"""
Integration tests for complete agent workflows
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.llm_component import LLMConfig, LLMProvider
from core.tool_orchestrator import ToolOrchestrator
from core.memory import MemoryManager


@pytest.mark.asyncio
class TestAgentWorkflows:
    """Test complete agent workflows"""

    @pytest.fixture
    async def setup_agent(self):
        """Set up a test agent"""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.generate_with_tools = AsyncMock(return_value={
            "content": "Task completed",
            "tool_calls": [],
            "finish_reason": "end_turn"
        })

        # Create orchestrator
        orchestrator = ToolOrchestrator(mock_llm)
        
        return {
            "llm": mock_llm,
            "orchestrator": orchestrator
        }

    async def test_simple_task_execution(self, setup_agent):
        """Test simple task execution"""
        result = await setup_agent["orchestrator"].execute_with_tools(
            user_prompt="What is 2+2?",
            system_prompt="You are a helpful assistant."
        )
        
        assert result is not None
        assert "completed" in result.lower() or "4" in result

    async def test_tool_invocation_workflow(self, setup_agent):
        """Test workflow with tool invocation"""
        # Mock tool call response
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param": "value"}'

        setup_agent["llm"].generate_with_tools = AsyncMock(return_value={
            "content": "Using tool",
            "tool_calls": [mock_tool_call],
            "finish_reason": "tool_calls"
        })

        result = await setup_agent["orchestrator"].execute_with_tools(
            user_prompt="Execute a task"
        )
        
        # Verify tool was attempted
        assert setup_agent["llm"].generate_with_tools.called


class TestMemoryIntegration:
    """Test memory system integration"""

    @pytest.fixture
    def mock_redis_url(self):
        return "redis://localhost:6379/0"

    @pytest.mark.asyncio
    async def test_conversation_memory_workflow(self, mock_redis_url):
        """Test conversation memory workflow"""
        with patch('redis.from_url') as mock_redis:
            manager = MemoryManager(mock_redis_url, "test_session")
            
            # Store conversation
            await manager.store_conversation("user", "Hello")
            await manager.store_conversation("assistant", "Hi there!")
            
            # Retrieve conversation
            history = await manager.get_conversation_history(limit=2)
            
            # Verify storage
            assert mock_redis.return_value.setex.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

### 7.4 Performance Testing

Create `tests/performance/test_performance.py`:

```python
"""
Performance tests for Mantus
"""

import pytest
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.performance
class TestPerformance:
    """Performance benchmarks"""

    @pytest.mark.asyncio
    async def test_llm_inference_latency(self):
        """Test LLM inference latency"""
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Response")

        start = time.time()
        for _ in range(10):
            mock_llm.generate("Test prompt")
        duration = time.time() - start

        avg_latency = duration / 10
        assert avg_latency < 1.0, f"Latency too high: {avg_latency}s"

    @pytest.mark.asyncio
    async def test_memory_operation_throughput(self):
        """Test memory operation throughput"""
        mock_redis = MagicMock()
        
        start = time.time()
        for i in range(100):
            mock_redis.setex(f"key_{i}", 3600, f"value_{i}")
        duration = time.time() - start

        throughput = 100 / duration
        assert throughput > 10, f"Throughput too low: {throughput} ops/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
```

---

## Phase 8: Deployment and Production Operations

### 8.1 Deployment Pipeline

Create `deploy/deployment_guide.md`:

```markdown
# Mantus Deployment Guide

## Pre-Deployment Checklist

- [ ] All tests passing (unit, integration, performance)
- [ ] Code review completed
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] API keys and secrets configured
- [ ] Database migrations tested
- [ ] Load testing completed
- [ ] Incident response plan reviewed

## Deployment Steps

### 1. Staging Deployment

```bash
# Build Docker image
docker build -t mantus:latest -f Dockerfile .

# Tag for staging
docker tag mantus:latest mantus:staging

# Push to registry
docker push mantus:staging

# Deploy to staging
kubectl apply -f k8s/staging/deployment.yaml

# Run smoke tests
pytest tests/smoke/ -v
```

### 2. Production Deployment

```bash
# Tag for production
docker tag mantus:latest mantus:v1.0.0

# Push to registry
docker push mantus:v1.0.0

# Deploy to production with canary strategy
kubectl apply -f k8s/production/deployment.yaml

# Monitor metrics
kubectl logs -f deployment/mantus-prod
```

## Rollback Procedure

```bash
# If issues detected, rollback to previous version
kubectl rollout undo deployment/mantus-prod

# Verify rollback
kubectl get pods -l app=mantus
```

## Database Migrations

```bash
# Apply migrations
alembic upgrade head

# Verify migrations
alembic current
```
```

### 8.2 Kubernetes Deployment Configuration

Create `k8s/production/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mantus-prod
  namespace: production
  labels:
    app: mantus
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: mantus
  template:
    metadata:
      labels:
        app: mantus
        version: v1.0.0
    spec:
      containers:
      - name: mantus
        image: mantus:v1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: MANTUS_ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: mantus-secrets
              key: openai-api-key
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PORT
          value: "6379"
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: mantus-service
  namespace: production
spec:
  selector:
    app: mantus
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 8.3 Monitoring and Observability

Create `monitoring/prometheus_config.yaml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mantus'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']

rule_files:
  - 'alert_rules.yaml'
```

Create `monitoring/alert_rules.yaml`:

```yaml
groups:
  - name: mantus_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(mantus_errors_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, mantus_latency_seconds) > 2
        for: 5m
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s"

      - alert: LowAvailability
        expr: up{job="mantus"} == 0
        for: 1m
        annotations:
          summary: "Mantus instance down"
          description: "Instance {{ $labels.instance }} is down"
```

### 8.4 Operational Runbooks

Create `operations/runbooks/incident_response.md`:

```markdown
# Incident Response Runbook

## High Error Rate

### Detection
- Alert: `HighErrorRate` triggered
- Error rate > 5% for 5+ minutes

### Investigation
1. Check recent deployments
   ```bash
   kubectl rollout history deployment/mantus-prod
   ```

2. Review error logs
   ```bash
   kubectl logs -f deployment/mantus-prod --tail=1000
   ```

3. Check resource utilization
   ```bash
   kubectl top pods -l app=mantus
   ```

4. Review metrics dashboard
   - Navigate to Grafana dashboard
   - Check error distribution
   - Identify affected endpoints

### Resolution
1. If recent deployment caused issue:
   ```bash
   kubectl rollout undo deployment/mantus-prod
   ```

2. If resource issue:
   ```bash
   kubectl scale deployment mantus-prod --replicas=5
   ```

3. If specific endpoint failing:
   - Disable endpoint temporarily
   - Investigate root cause
   - Deploy fix

## High Latency

### Detection
- Alert: `HighLatency` triggered
- P95 latency > 2 seconds

### Investigation
1. Check LLM provider status
   - Verify API availability
   - Check rate limits

2. Check Redis performance
   ```bash
   redis-cli INFO stats
   ```

3. Check network latency
   ```bash
   kubectl exec -it <pod> -- ping <service>
   ```

### Resolution
1. Increase replicas if CPU bound
2. Optimize queries if database bound
3. Increase timeouts if external service slow
4. Contact LLM provider if API degraded
```

### 8.5 Health Checks and Readiness

Create `core/health.py` for health check endpoints:

```python
"""
Health checks for Mantus
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Liveness probe - is the service running?"""
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness probe - is the service ready to handle requests?"""
    checks = {
        "llm_available": await check_llm_availability(),
        "redis_available": await check_redis_availability(),
        "memory_sufficient": await check_memory_availability()
    }

    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")


async def check_llm_availability() -> bool:
    """Check if LLM service is available"""
    try:
        # Attempt a simple LLM call
        return True
    except Exception as e:
        logger.error(f"LLM unavailable: {e}")
        return False


async def check_redis_availability() -> bool:
    """Check if Redis is available"""
    try:
        # Attempt Redis ping
        return True
    except Exception as e:
        logger.error(f"Redis unavailable: {e}")
        return False


async def check_memory_availability() -> bool:
    """Check if sufficient memory is available"""
    import psutil
    memory = psutil.virtual_memory()
    available_percent = memory.available / memory.total * 100
    return available_percent > 10  # At least 10% available
```

---

## References

[1] Fowler, M. (2006). "Continuous Integration." Retrieved from https://martinfowler.com/articles/continuousIntegration.html

[2] Humble, J., & Farley, D. (2010). "Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation." Addison-Wesley Professional.

[3] Kubernetes. (2024). "Kubernetes Documentation." Retrieved from https://kubernetes.io/docs/

[4] Prometheus. (2024). "Prometheus Monitoring System." Retrieved from https://prometheus.io/docs/

[5] Google. (2016). "The Site Reliability Engineering Book." Retrieved from https://sre.google/books/

---

**End of Phase 7-8 Manual**

*All phases completed. Mantus is now ready for production deployment and operation.*

