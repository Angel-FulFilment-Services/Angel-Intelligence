# Testing Guide

Comprehensive testing procedures for Angel Intelligence.

## Test Categories

1. [Unit Tests](#unit-tests)
2. [Integration Tests](#integration-tests)
3. [End-to-End Tests](#end-to-end-tests)
4. [Performance Tests](#performance-tests)
5. [Manual Testing](#manual-testing)

---

## Unit Tests

### Running Unit Tests

```bash
# Activate virtual environment
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pii_detector.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_pii_detector.py     # PII detection tests
├── test_transcriber.py      # Transcription tests
├── test_analyzer.py         # Analysis tests
├── test_api.py              # API endpoint tests
├── test_database.py         # Database model tests
└── fixtures/
    ├── sample_audio.wav     # Test audio file
    └── sample_transcript.json
```

### Writing Tests

```python
# tests/test_pii_detector.py
import pytest
from src.services import PIIDetector

class TestPIIDetector:
    """Tests for PII detection service."""
    
    @pytest.fixture
    def detector(self):
        """Create a PII detector instance."""
        return PIIDetector()
    
    def test_detect_ni_number(self, detector):
        """Test National Insurance number detection."""
        text = "My NI number is AB123456C"
        result = detector.detect(text)
        
        assert result["pii_count"] == 1
        assert result["pii_types"] == ["national_insurance_number"]
        assert result["pii_detected"][0]["original"] == "AB123456C"
    
    def test_detect_postcode(self, detector):
        """Test UK postcode detection."""
        text = "I live at SW1A 1AA"
        result = detector.detect(text)
        
        assert "postcode" in result["pii_types"]
    
    def test_no_pii_clean_text(self, detector):
        """Test that clean text has no PII detected."""
        text = "Hello, how can I help you today?"
        result = detector.detect(text)
        
        assert result["pii_count"] == 0
        assert result["redacted_text"] == text
```

---

## Integration Tests

### Database Integration

```python
# tests/test_database_integration.py
import pytest
from src.database import CallRecording, get_db_connection

class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Clean up test data before and after."""
        db = get_db_connection()
        db.execute("DELETE FROM ai_call_recordings WHERE apex_id LIKE 'TEST-%'")
        yield
        db.execute("DELETE FROM ai_call_recordings WHERE apex_id LIKE 'TEST-%'")
    
    def test_create_recording(self):
        """Test creating a call recording."""
        recording = CallRecording(
            apex_id="TEST-001",
            call_date="2026-01-17",
        )
        record_id = recording.save()
        
        assert record_id > 0
        
        # Verify it was saved
        loaded = CallRecording.get_by_id(record_id)
        assert loaded is not None
        assert loaded.apex_id == "TEST-001"
    
    def test_status_transitions(self):
        """Test processing status transitions."""
        recording = CallRecording(apex_id="TEST-002", call_date="2026-01-17")
        recording.save()
        
        recording.mark_processing()
        assert recording.processing_status == "processing"
        
        recording.mark_completed()
        assert recording.processing_status == "completed"
```

### API Integration

```python
# tests/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.config import get_settings

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Authentication headers."""
    settings = get_settings()
    return {"Authorization": f"Bearer {settings.api_auth_token}"}

class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    def test_health_no_auth(self, client):
        """Health endpoint should not require auth."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_submit_recording(self, client, auth_headers):
        """Test submitting a recording."""
        response = client.post(
            "/recordings/submit",
            headers=auth_headers,
            json={
                "apex_id": "TEST-API-001",
                "call_date": "2026-01-17"
            }
        )
        assert response.status_code == 201
        assert "id" in response.json()
    
    def test_auth_required(self, client):
        """Test that auth is required for protected endpoints."""
        response = client.get("/recordings/pending")
        assert response.status_code == 401
```

---

## End-to-End Tests

### Full Pipeline Test

```python
# tests/test_e2e_pipeline.py
import pytest
import os
import time
from src.database import CallRecording
from src.worker.processor import CallProcessor

class TestFullPipeline:
    """End-to-end pipeline tests."""
    
    @pytest.fixture
    def sample_audio(self):
        """Path to sample audio file."""
        return "tests/fixtures/sample_audio.wav"
    
    @pytest.fixture
    def processor(self):
        """Create processor with mock mode."""
        os.environ["USE_MOCK_MODELS"] = "true"
        return CallProcessor()
    
    def test_full_processing_pipeline(self, processor, sample_audio):
        """Test complete processing from audio to analysis."""
        # Create recording
        recording = CallRecording(
            apex_id="E2E-TEST-001",
            call_date="2026-01-17",
            r2_path=sample_audio,  # Use local file
        )
        recording.save()
        
        # Process
        result = processor.process(recording)
        
        assert result is True
        
        # Reload and check
        recording = CallRecording.get_by_id(recording.id)
        assert recording.processing_status == "completed"
        
        # Check transcription exists
        from src.database import CallTranscription
        transcription = CallTranscription.get_by_recording_id(recording.id)
        assert transcription is not None
        assert len(transcription.full_transcript) > 0
        
        # Check analysis exists
        from src.database import CallAnalysis
        analysis = CallAnalysis.get_by_recording_id(recording.id)
        assert analysis is not None
        assert analysis.sentiment_score is not None
```

---

## Performance Tests

### Throughput Testing

```python
# tests/test_performance.py
import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    """Performance and load tests."""
    
    def test_transcription_speed(self):
        """Test transcription meets performance targets."""
        from src.services import TranscriptionService
        
        service = TranscriptionService()
        audio_path = "tests/fixtures/60_second_audio.wav"
        
        start = time.time()
        result = service.transcribe(audio_path)
        elapsed = time.time() - start
        
        # Should process 60s audio in under 30s (real-time factor < 0.5)
        assert elapsed < 30, f"Transcription took {elapsed:.1f}s (too slow)"
    
    def test_analysis_speed(self):
        """Test analysis meets performance targets."""
        from src.services import AnalysisService
        
        service = AnalysisService()
        transcript = {"full_transcript": "Sample transcript..." * 100}
        
        start = time.time()
        result = service.analyse("tests/fixtures/sample.wav", transcript, 1)
        elapsed = time.time() - start
        
        # Analysis should complete in under 60s
        assert elapsed < 60, f"Analysis took {elapsed:.1f}s (too slow)"
    
    def test_api_response_time(self, client, auth_headers):
        """Test API response times."""
        times = []
        
        for _ in range(10):
            start = time.time()
            response = client.get("/health")
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        assert avg_time < 0.1, f"Average response time {avg_time:.3f}s (too slow)"
```

---

## Manual Testing

### Pre-Deployment Checklist

#### 1. Environment Setup
- [ ] `.env` file configured correctly
- [ ] Database connection works: `python -c "from src.database import get_db_connection; get_db_connection()"`
- [ ] GPU detected (if applicable): `python -c "import torch; print(torch.cuda.is_available())"`

#### 2. API Health
- [ ] Health endpoint responds: `curl http://localhost:8000/health`
- [ ] Models loaded in health response
- [ ] Swagger UI accessible: `http://localhost:8000/docs`

#### 3. Authentication
- [ ] Protected endpoints require token
- [ ] Invalid token returns 401
- [ ] Valid token allows access

#### 4. Recording Submission
- [ ] Submit recording via API
- [ ] Recording appears in database
- [ ] Status is "pending"

#### 5. Worker Processing
- [ ] Worker starts without errors
- [ ] Worker picks up pending recording
- [ ] Processing completes successfully
- [ ] Status changes to "completed"

#### 6. Transcription
- [ ] Full transcript generated
- [ ] Segments have timestamps
- [ ] Word-level timestamps present
- [ ] Speaker labels assigned

#### 7. PII Detection
- [ ] NI numbers detected and redacted
- [ ] Postcodes detected and redacted
- [ ] Phone numbers detected and redacted
- [ ] Redacted transcript generated

#### 8. Analysis
- [ ] Summary in British English
- [ ] Sentiment score in range (-10 to +10)
- [ ] Quality score in range (0-100)
- [ ] Topics matched from config
- [ ] Performance scores generated

#### 9. Error Handling
- [ ] Missing recording handles gracefully
- [ ] Invalid audio format returns error
- [ ] Network errors trigger retry

---

### Test Audio Files

Create test audio files with known content:

#### Sample 1: Clean Call (sample_clean.wav)
- Duration: 60 seconds
- Content: Agent greeting, donation request, successful outcome
- Expected: High quality score, positive sentiment

#### Sample 2: PII Call (sample_pii.wav)
- Duration: 45 seconds
- Content: Contains NI number, postcode, phone number
- Expected: 3 PII items detected

#### Sample 3: Poor Quality (sample_noisy.wav)
- Duration: 30 seconds
- Content: Background noise, interruptions
- Expected: Lower quality score, audio quality = "fair" or "poor"

---

### cURL Test Commands

```bash
# Set token
TOKEN="your-64-char-token"

# Health check
curl http://localhost:8000/health | jq

# Submit recording
curl -X POST http://localhost:8000/recordings/submit \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"apex_id": "TEST-001", "call_date": "2026-01-17"}' | jq

# Check status
curl http://localhost:8000/recordings/1/status \
  -H "Authorization: Bearer $TOKEN" | jq

# Get transcription
curl http://localhost:8000/recordings/1/transcription \
  -H "Authorization: Bearer $TOKEN" | jq

# Get analysis
curl http://localhost:8000/recordings/1/analysis \
  -H "Authorization: Bearer $TOKEN" | jq

# List pending
curl http://localhost:8000/recordings/pending \
  -H "Authorization: Bearer $TOKEN" | jq

# List failed
curl http://localhost:8000/recordings/failed \
  -H "Authorization: Bearer $TOKEN" | jq

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What was the average quality score today?"}' | jq
```

---

### Mock Mode Testing

Enable mock mode for testing without GPU:

```bash
export USE_MOCK_MODELS=true
python -m src.worker.worker
```

Mock mode returns deterministic test data:
- Transcription: Sample transcript with speaker labels
- Analysis: Quality score 85, sentiment 7.5, positive outcome
- PII: No PII detected (clean mock data)

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      mysql:
        image: mysql:8.0
        env:
          MYSQL_ROOT_PASSWORD: test
          MYSQL_DATABASE: ai_test
        ports:
          - 3306:3306
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      env:
        AI_DB_HOST: localhost
        AI_DB_DATABASE: ai_test
        AI_DB_USERNAME: root
        AI_DB_PASSWORD: test
        USE_MOCK_MODELS: true
      run: pytest tests/ -v --cov=src
```

---

## Test Data Management

### Creating Test Fixtures

```bash
# Create test database
mysql -e "CREATE DATABASE ai_test"

# Load schema
mysql ai_test < schema.sql

# Load test data
mysql ai_test < tests/fixtures/test_data.sql
```

### Cleaning Up

```bash
# Reset test database
mysql ai_test < tests/fixtures/reset.sql
```
