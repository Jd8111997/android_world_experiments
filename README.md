# Android World LLM Agent Evaluation Framework

A comprehensive framework for evaluating LLM agents on mobile automation tasks using the Android World benchmark.

## Project Structure

```
android-world-evaluation/
├── main.py                 # Main runner script
├── README.md              # This file
├── report.md             # Evaluation findings (sample)
│
├── src/                   # Core framework code
│   ├── __init__.py
│   ├── framework.py       # Main Android World framework
│   ├── agent.py          # Enhanced Android agent with LLM integration
│   ├── llm_client.py     # LLM client for multiple providers
│   ├── prompts.py        # Prompt template system
│   ├── evaluation.py     # Evaluation framework for Tasks 2 & 3
│   ├── report_generator.py # Markdown report generation
│   └── utils.py          # Utility functions
│
├── prompts/              # Prompt templates and examples
│   ├── few_shot_examples.json    # Curated examples
│   ├── template_configs.yaml     # Template configurations
│
└── results/              # Evaluation outputs (created at runtime)
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd android-world-evaluation

# Install dependencies
pip install -r requirements.txt

# Install Android World
pip install android-world
```

### 2. Setup Environment

```bash
# Set up Android emulator
# Follow Android World setup instructions

# Set API keys (choose one)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### 3. Run Evaluation

```bash
# Run all tasks
python main.py

# Or run specific tasks
python -c "
from src.framework import AndroidWorldFramework
framework = AndroidWorldFramework()
framework.setup_environment()
# Your evaluation code here
"
```

## Tasks Completed

### ✅ Task 1: Setup & Agent Framework Scaffold
- [x] Environment loading and episode exploration
- [x] Basic agent loop (goal + observation → LLM → action)
- [x] Configurable prompt templates (6 variants)
- [x] Full episode execution with proper error handling

### ✅ Task 2: Prompting & Evaluation Strategy
- [x] Few-shot prompting with curated examples
- [x] Self-reflection capabilities with progress analysis
- [x] Multi-episode evaluation (3+ episodes)
- [x] Action comparison and step accuracy metrics
- [x] Per-episode logs with detailed tracking

### ✅ Task 3: Benchmarking & Report
- [x] Benchmark evaluation (10+ episodes)
- [x] Performance metrics and failure analysis
- [x] LLM and template comparisons
- [x] Comprehensive Markdown report generation
- [x] Recommendations for improvement

## Key Features

### 🤖 Enhanced Agent Design
- **T3A-Inspired Architecture:** Based on Android World's proven T3A agent
- **Index-Based Actions:** Eliminates hallucination with precise UI element targeting
- **JSON Action Format:** Structured actions like `{"action_type": "click", "index": 1}`
- **Multi-LLM Support:** OpenAI GPT-4 and Anthropic Claude integration

### 📝 Advanced Prompting
- **Six Template Types:** Basic, detailed, few-shot, chain-of-thought, self-reflection, combined
- **Curated Examples:** 8 high-quality few-shot examples across different task categories
- **Error-Informed Design:** Prompts designed to avoid common failure patterns
- **Dynamic Context:** Action history integration for reflection templates

### 📊 Comprehensive Evaluation
- **Multiple Metrics:** Episode success, step accuracy, action success rate, error analysis
- **Detailed Logging:** Full episode traces with prompts, responses, and outcomes
- **Comparative Analysis:** LLM provider and template performance comparison
- **Failure Analysis:** Systematic categorization of error types and patterns

### 📈 Professional Reporting
- **Automated Report Generation:** Comprehensive Markdown reports with analysis
- **Visual Performance Tables:** Clear comparison of different approaches
- **Actionable Recommendations:** Specific suggestions for improvement
- **Example Episodes:** Illustrative cases showing success and failure patterns

## Usage Examples

### Basic Agent Evaluation
```python
from src.framework import AndroidWorldFramework

# Initialize framework
framework = AndroidWorldFramework()
framework.setup_environment()

# Run single episode
task, task_name, task_type, params = framework.load_episode()
result = framework.evaluator.run_single_episode_evaluation(
    task_name=task_name,
    llm_provider="openai",
    template_name="few_shot",
    max_steps=15
)
```

### Multi-Episode Evaluation
```python
# Test multiple configurations
results = framework.evaluator.run_multi_episode_evaluation(
    num_episodes=5,
    llm_providers=["openai", "anthropic"],
    template_names=["basic", "few_shot", "self_reflection"],
    max_steps=15
)
```

### Benchmark Evaluation
```python
# Run comprehensive benchmark
benchmark_results = framework.evaluator.run_benchmark_evaluation(
    num_episodes=10
)
```

## Configuration

### Template Configuration (`prompts/template_configs.yaml`)
```yaml
templates:
  few_shot:
    description: "Template with curated examples"
    max_tokens: 400
    temperature: 0.1
    use_examples: true
    num_examples: 3
```

### Few-Shot Examples (`prompts/few_shot_examples.json`)
```json
{
  "examples": [
    {
      "goal": "Turn on Wi-Fi",
      "action": {"action_type": "click", "index": 0},
      "reasoning": "Click Wi-Fi option to access settings"
    }
  ]
}
```

## Results Structure

### Episode Logs (`results/episodes/`)
```json
{
  "task_name": "wifi_enablement",
  "goal": "Turn on Wi-Fi",
  "success": true,
  "steps_taken": 4,
  "episode_history": [...],
  "error_types": [],
  "template_name": "few_shot"
}
```

### Benchmark Results (`results/benchmarks/`)
```json
{
  "benchmark_summary": {
    "total_episodes": 10,
    "episode_success_rate": 0.6,
    "avg_step_accuracy": 0.75,
    "template_performance": {...}
  }
}
```

## Contributing

1. **Adding New Templates:** Extend `src/prompts.py` with new template methods
2. **Adding Examples:** Update `prompts/few_shot_examples.json` with new scenarios
3. **Custom Metrics:** Extend `src/evaluation.py` with additional evaluation metrics
4. **Report Customization:** Modify `src/report_generator.py` for custom report formats

## Dependencies

- `android-world`: Core Android automation framework
- `openai`: OpenAI API client
- `anthropic`: Anthropic API client
- `pyyaml`: Configuration file parsing
- `pathlib`: Path manipulation
- Standard library: `json`, `logging`, `datetime`, `statistics`
