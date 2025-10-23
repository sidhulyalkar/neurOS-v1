# Changelog

All notable changes to NeuroFM-X will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-23

### Added
- Initial release of NeuroFM-X foundation model
- Mamba/SSM backbone for efficient sequence modeling
- Perceiver-IO fusion for multi-modal integration
- PopT (Population Transformer) for population-level aggregation
- Multi-task heads (encoder, decoder, contrastive, forecast)
- YAML-based configuration system
- Docker deployment infrastructure
- AWS cloud training support
- Comprehensive documentation and guides
- Progressive training strategy (4 → 200+ sessions)
- Multi-modal tokenizers (Binned, Calcium, LFP, Miniscope)
- Transfer learning adapters (Unit-ID, Session Stitcher)
- Monitoring and benchmarking tools
- TensorBoard and WandB integration

### Fixed
- Tokenizer dimension mismatch (384 vs 128 units)
- Encoder head shape mismatch for reconstruction
- Training speed optimization for RTX 3070 Ti
- Batch size configuration for efficient GPU utilization

### Documentation
- README with quick start guide
- Scaling strategy for progressive training
- Training guide with detailed instructions
- Optimal training plan for multi-dataset training
- Docker and AWS deployment guides

---

## Future Releases

### [0.2.0] - Planned
- Multi-GPU distributed training
- Additional modality tokenizers (ECoG, EEG, Utah arrays)
- Enhanced contrastive learning
- Cross-species pre-training
- Production serving infrastructure
- Comprehensive unit tests

### [0.3.0] - Planned
- Real-time inference optimization
- Model compression and quantization
- Additional task heads (RL, attention decoding)
- Interactive visualization tools
- API documentation
- Tutorial notebooks
