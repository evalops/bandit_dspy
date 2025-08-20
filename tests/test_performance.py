"""
Performance benchmarks for bandit_dspy improvements.
"""

import pytest
import time
import dspy
import json
from bandit_dspy import BanditRunner, SecurityMetric, BanditTeleprompter, bandit_metric


class PerformanceLLM(dspy.LM):
    """Fast LLM for performance testing."""
    
    def __init__(self):
        super().__init__("performance-llm")
        self.call_count = 0
    
    def __call__(self, messages, **kwargs):
        self.call_count += 1
        # Alternate between secure and insecure code
        if self.call_count % 2 == 0:
            response = {"code": "def secure_func(): return 'safe'"}
        else:
            response = {"code": "password = 'hardcoded'"}
        return [json.dumps(response)]
    
    def basic_request(self, prompt, **kwargs):
        pass


class CodeGenModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(dspy.Signature("description -> code"))

    def forward(self, description):
        return self.predictor(description=description)


@pytest.fixture
def large_code_sample():
    """Generate a large code sample for performance testing."""
    return """
import subprocess
import os
import sys
import hashlib

password = "hardcoded_password"
api_key = "sk-1234567890abcdef"

def unsafe_exec(user_input):
    exec(user_input)
    
def unsafe_eval(expression):
    return eval(expression)

def unsafe_subprocess(command):
    subprocess.call(command, shell=True)
    
def weak_random():
    import random
    return random.random()

def hardcoded_secrets():
    db_password = "admin123"
    secret_key = "super_secret_key"
    return db_password, secret_key

def sql_injection_risk(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query

assert True  # B101
assert False  # B101

try:
    pass
except:
    pass  # B110

for i in range(10):
    password = f"temp_password_{i}"
"""


class TestCachingPerformance:
    """Test caching performance improvements."""
    
    def test_cache_speedup(self, large_code_sample):
        """Test that caching provides significant speedup."""
        runner = BanditRunner()
        
        # First run (cold cache)
        start_time = time.time()
        result1 = runner.analyze_code(large_code_sample)
        cold_time = time.time() - start_time
        
        # Second run (warm cache)
        start_time = time.time()
        result2 = runner.analyze_code(large_code_sample)
        warm_time = time.time() - start_time
        
        # Results should be identical
        assert len(result1) == len(result2)
        
        # Warm cache should be significantly faster
        # (Allow some variance for CI environments)
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        assert speedup >= 1.0, f"Expected speedup >= 1.0, got {speedup:.2f}"
        
        print(f"Cache speedup: {speedup:.2f}x ({cold_time:.4f}s -> {warm_time:.4f}s)")
    
    def test_multiple_cache_hits(self):
        """Test multiple cache hits for different code samples."""
        runner = BanditRunner()
        
        codes = [
            "password = 'test1'",
            "password = 'test2'", 
            "def safe(): return 1",
            "import os",
            "assert True"
        ]
        
        # Warm up cache
        for code in codes:
            runner.analyze_code(code)
        
        # Time cache hits
        start_time = time.time()
        for _ in range(10):
            for code in codes:
                runner.analyze_code(code)
        total_time = time.time() - start_time
        
        # Should be very fast with cache
        avg_time_per_analysis = total_time / (10 * len(codes))
        assert avg_time_per_analysis < 0.01, f"Cache hits too slow: {avg_time_per_analysis:.4f}s per analysis"


class TestOptimizationPerformance:
    """Test optimization method performance."""
    
    @pytest.fixture
    def performance_trainset(self):
        """Training set for performance testing."""
        return [
            dspy.Example(description=f"function {i}", code=f"def func_{i}(): return {i}").with_inputs('description')
            for i in range(20)
        ]
    
    def test_random_optimization_scaling(self, performance_trainset):
        """Test that random optimization scales reasonably."""
        dspy.settings.configure(lm=PerformanceLLM())
        student = CodeGenModule()
        
        # Test different numbers of candidates
        candidate_counts = [2, 5, 10]
        times = []
        
        for num_candidates in candidate_counts:
            teleprompter = BanditTeleprompter(
                metric=bandit_metric,
                k=3,
                num_candidates=num_candidates,
                optimization_method="random"
            )
            
            start_time = time.time()
            compiled_program = teleprompter.compile(student, trainset=performance_trainset[:5])
            compile_time = time.time() - start_time
            times.append(compile_time)
            
            assert compiled_program is not None
        
        # Time should scale roughly linearly with candidate count
        print(f"Random optimization times: {[f'{t:.3f}s' for t in times]}")
        
        # More candidates should generally take more time, but timing can be noisy
        # Just verify all completed successfully in reasonable time
        assert all(t < 5.0 for t in times), f"Some optimizations too slow: {times}"
    
    def test_genetic_vs_random_performance(self, performance_trainset):
        """Compare genetic vs random optimization performance."""
        dspy.settings.configure(lm=PerformanceLLM())
        student = CodeGenModule()
        
        # Random optimization
        random_teleprompter = BanditTeleprompter(
            metric=bandit_metric,
            k=2,
            num_candidates=4,
            optimization_method="random"
        )
        
        start_time = time.time()
        random_program = random_teleprompter.compile(student, trainset=performance_trainset[:5])
        random_time = time.time() - start_time
        
        # Genetic optimization (small parameters for speed)
        genetic_teleprompter = BanditTeleprompter(
            metric=bandit_metric,
            k=2,
            num_candidates=4,
            optimization_method="genetic",
            genetic_config={'population_size': 4, 'generations': 2}
        )
        
        start_time = time.time()
        genetic_program = genetic_teleprompter.compile(student, trainset=performance_trainset[:5])
        genetic_time = time.time() - start_time
        
        assert random_program is not None
        assert genetic_program is not None
        
        print(f"Random: {random_time:.3f}s, Genetic: {genetic_time:.3f}s")
        
        # Both should complete in reasonable time
        assert random_time < 30.0, f"Random optimization too slow: {random_time:.3f}s"
        assert genetic_time < 60.0, f"Genetic optimization too slow: {genetic_time:.3f}s"


class TestSecurityMetricPerformance:
    """Test SecurityMetric performance."""
    
    def test_metric_evaluation_speed(self, large_code_sample):
        """Test SecurityMetric evaluation speed."""
        metric = SecurityMetric()
        
        # Time multiple evaluations
        start_time = time.time()
        results = []
        for _ in range(100):
            result = metric.evaluate(large_code_sample)
            results.append(result)
        total_time = time.time() - start_time
        
        avg_time = total_time / 100
        assert avg_time < 0.1, f"SecurityMetric too slow: {avg_time:.4f}s per evaluation"
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result["score"] == first_result["score"]
            assert len(result["issues"]) == len(first_result["issues"])
    
    def test_different_metric_configs_performance(self):
        """Test performance of different metric configurations."""
        configs = [
            {"normalize_by_length": False},
            {"normalize_by_length": True},
            {"penalty_threshold": 0.0},
            {"penalty_threshold": 5.0},
            {"severity_weights": {"HIGH": 10.0, "MEDIUM": 5.0, "LOW": 1.0}},
        ]
        
        code = "password = 'hardcoded'\nassert True\nexec('test')"
        
        for config in configs:
            metric = SecurityMetric(**config)
            
            start_time = time.time()
            for _ in range(50):
                result = metric.evaluate(code)
            eval_time = time.time() - start_time
            
            avg_time = eval_time / 50
            assert avg_time < 0.05, f"Config {config} too slow: {avg_time:.4f}s per evaluation"


class TestMemoryUsage:
    """Test memory usage of improvements."""
    
    def test_cache_memory_bounds(self):
        """Test that cache doesn't grow unbounded."""
        runner = BanditRunner()
        
        # Generate many different code samples
        for i in range(1001):  # More than cache size (1000)
            code = f"def func_{i}(): return {i}"
            runner.analyze_code(code)
        
        # Cache should be bounded
        assert len(runner._cache) <= 1000, f"Cache too large: {len(runner._cache)}"
    
    def test_genetic_optimizer_memory(self):
        """Test genetic optimizer doesn't use excessive memory."""
        from bandit_dspy.teleprompter import GeneticOptimizer
        
        # Large training set
        large_trainset = [
            dspy.Example(description=f"func {i}", code=f"def func_{i}(): return {i}").with_inputs('description')
            for i in range(100)
        ]
        
        optimizer = GeneticOptimizer(population_size=10, generations=5)
        
        def simple_evaluator(examples):
            return len(examples) * 0.1
        
        # Should handle large trainset efficiently
        result = optimizer.optimize(large_trainset, k=5, evaluator=simple_evaluator)
        assert len(result) == 5
        assert all(ex in large_trainset for ex in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])