"""
SageMaker training job launcher.

This module demonstrates how to configure and launch a distributed
training job on Amazon SageMaker for the Constellation brain state
model.  The `launch_training` function accepts a job name and
optional configuration file specifying hyperparameters, input data
locations and the Docker image containing your training code.  The
function builds a ``sagemaker.estimator.Estimator`` object with
SMDistributed options and submits the job.

Note: Running this function requires AWS credentials configured in your
environment and the ``sagemaker`` Python package.  When running
outside of AWS the function will log the configuration without
executing the job.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

try:
    import boto3
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore
    Estimator = None  # type: ignore
    TrainingInput = None  # type: ignore

logger = logging.getLogger(__name__)


def load_config(config_path: str | None) -> Dict[str, any]:
    """Load job configuration from a JSON or YAML file.

    The configuration can specify hyperparameters, input channels,
    instance counts, instance types and the training image.  If no
    configuration file is provided default values are used.
    """
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {config_path} does not exist")
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text())
    else:
        return json.loads(path.read_text())


def launch_training(job_name: str, config_path: Optional[str] = None) -> None:
    """Launch a SageMaker training job.

    Parameters
    ----------
    job_name:
        A unique name for the training job.
    config_path:
        Optional path to a JSON or YAML configuration file specifying
        hyperparameters and resource parameters.  See below for the
        expected keys.

    The configuration may contain the following keys:
    - ``image_uri``: Docker image containing your training code.
    - ``role``: IAM role ARN for SageMaker.
    - ``instance_type``: EC2 instance type (e.g. ``ml.p3.16xlarge``).
    - ``instance_count``: Number of instances for distributed training.
    - ``hyperparameters``: Dict of hyperparameters passed to your script.
    - ``input_data``: Dict mapping channel names to S3 URIs.
    - ``s3_output_path``: S3 path for model artifacts.
    """
    config = load_config(config_path)

    if boto3 is None or Estimator is None:
        logger.info("sagemaker package not available; printing job configuration")
        logger.info(json.dumps(config, indent=2))
        return

    # Set defaults
    image_uri = config.get("image_uri", "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.1.0-gpu-py39")
    role = config.get("role", "arn:aws:iam::123456789012:role/SageMakerExecution")
    instance_type = config.get("instance_type", "ml.p3.16xlarge")
    instance_count = config.get("instance_count", 1)
    hyperparameters = config.get("hyperparameters", {})
    input_data = config.get("input_data", {})
    s3_output_path = config.get("s3_output_path", f"s3://constellation/models/{job_name}")

    # Build estimator
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        hyperparameters=hyperparameters,
        output_path=s3_output_path,
        use_spot_instances=True,
        max_run=12 * 60 * 60,
        max_wait=14 * 60 * 60,
    )

    # Convert input channels to TrainingInput
    inputs = {k: TrainingInput(v) for k, v in input_data.items()}

    logger.info("Launching SageMaker training job %s", job_name)
    estimator.fit(inputs=inputs, job_name=job_name)
    logger.info("Job submitted")


__all__ = ["launch_training", "load_config"]