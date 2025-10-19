import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from utils.data_fetcher import AQIDataFetcher
        from utils.models import EnhancedAQIPredictor
        from utils.visualization import AQIVisualizer
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_data_fetcher():
    """Test data fetcher initialization"""
    from utils.data_fetcher import AQIDataFetcher
    fetcher = AQIDataFetcher()
    assert fetcher is not None

def test_predictor():
    """Test predictor initialization"""
    from utils.models import EnhancedAQIPredictor
    predictor = EnhancedAQIPredictor()
    assert predictor is not None

if __name__ == "__main__":
    pytest.main()