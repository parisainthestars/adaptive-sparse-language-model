# Metrics Summary

This repository was trained on the included synthetic benchmark.

## Final metrics

### Train
- Loss: 0.0167
- Token accuracy: 1.0000
- Sequence exact match: 1.0000
- Prompt compression ratio: 0.8096
- Mean adaptive measurement budget: 11.7854
- Support F1: 0.9977
- Support drift: 0.2738
- Active support fraction: 0.5000

### Validation
- Loss: 0.0177
- Token accuracy: 1.0000
- Sequence exact match: 1.0000
- Prompt compression ratio: 0.8182
- Mean adaptive measurement budget: 11.7636
- Support F1: 0.9971
- Support drift: 0.2718
- Active support fraction: 0.5000

### Test
- Loss: 0.0181
- Token accuracy: 1.0000
- Sequence exact match: 1.0000
- Prompt compression ratio: 0.8133
- Mean adaptive measurement budget: 11.7566
- Support F1: 0.9990
- Support drift: 0.2724
- Active support fraction: 0.5000

## Caveat

These numbers are from a compact research prototype on a synthetic benchmark included with the repo. They validate the implementation path and instrumentation, but they should not be interpreted as production-scale LLM results.
