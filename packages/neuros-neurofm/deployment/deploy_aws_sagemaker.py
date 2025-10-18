"""
Deploy NeuroFM-X to AWS SageMaker.

This script creates a SageMaker endpoint for real-time inference.
"""

import argparse
import json
import time
from pathlib import Path

try:
    import boto3
    from sagemaker import get_execution_role
    from sagemaker.pytorch import PyTorchModel
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def deploy_to_sagemaker(
    model_path: str,
    role: str,
    instance_type: str = 'ml.p3.2xlarge',
    endpoint_name: str = 'neurofm-x-endpoint',
    framework_version: str = '2.3',
    py_version: str = 'py310',
):
    """Deploy model to SageMaker endpoint.

    Parameters
    ----------
    model_path : str
        Path to TorchScript model file.
    role : str
        SageMaker execution role ARN.
    instance_type : str, optional
        Instance type for endpoint.
    endpoint_name : str, optional
        Name for the endpoint.
    framework_version : str, optional
        PyTorch version.
    py_version : str, optional
        Python version.
    """
    if not BOTO3_AVAILABLE:
        raise ImportError(
            "boto3 and sagemaker are required. "
            "Install with: pip install boto3 sagemaker"
        )

    print(f"Deploying NeuroFM-X to SageMaker endpoint: {endpoint_name}")
    print(f"Model: {model_path}")
    print(f"Instance: {instance_type}")

    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=model_path,
        role=role,
        framework_version=framework_version,
        py_version=py_version,
        entry_point='inference.py',  # Custom inference script
        source_dir='sagemaker',
    )

    # Deploy to endpoint
    print("Deploying model (this may take several minutes)...")
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    print(f"✓ Endpoint deployed: {endpoint_name}")
    print(f"  Endpoint ARN: {predictor.endpoint}")

    return predictor


def test_endpoint(endpoint_name: str, test_data_path: str = None):
    """Test SageMaker endpoint.

    Parameters
    ----------
    endpoint_name : str
        Name of the endpoint.
    test_data_path : str, optional
        Path to test data. If None, uses random data.
    """
    import numpy as np

    client = boto3.client('sagemaker-runtime')

    # Load or generate test data
    if test_data_path:
        test_data = np.load(test_data_path)
    else:
        test_data = np.random.randn(1, 96, 100).astype(np.float32)

    print("Testing endpoint with sample data...")

    # Invoke endpoint
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps({'data': test_data.tolist()}),
    )

    # Parse response
    result = json.loads(response['Body'].read().decode())

    print(f"✓ Prediction received: {result}")
    print(f"  Latency: {response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-invoked-production-variant', 'N/A')}")

    return result


def delete_endpoint(endpoint_name: str):
    """Delete SageMaker endpoint.

    Parameters
    ----------
    endpoint_name : str
        Name of the endpoint to delete.
    """
    client = boto3.client('sagemaker')

    print(f"Deleting endpoint: {endpoint_name}")

    try:
        # Delete endpoint
        client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✓ Endpoint {endpoint_name} deleted")

        # Delete endpoint config
        client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"✓ Endpoint config {endpoint_name} deleted")

    except Exception as e:
        print(f"Error deleting endpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="Deploy NeuroFM-X to AWS SageMaker")
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to TorchScript model file (local or S3)',
    )
    parser.add_argument(
        '--role',
        type=str,
        default=None,
        help='SageMaker execution role ARN',
    )
    parser.add_argument(
        '--instance-type',
        type=str,
        default='ml.p3.2xlarge',
        help='Instance type for endpoint',
    )
    parser.add_argument(
        '--endpoint-name',
        type=str,
        default='neurofm-x-endpoint',
        help='Name for the endpoint',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test endpoint after deployment',
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete existing endpoint',
    )

    args = parser.parse_args()

    # Delete endpoint if requested
    if args.delete:
        delete_endpoint(args.endpoint_name)
        return

    # Get execution role
    if args.role is None:
        try:
            role = get_execution_role()
        except Exception:
            raise ValueError(
                "Could not get SageMaker execution role. "
                "Please specify --role or run from SageMaker notebook."
            )
    else:
        role = args.role

    # Deploy
    predictor = deploy_to_sagemaker(
        model_path=args.model_path,
        role=role,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )

    # Test if requested
    if args.test:
        test_endpoint(args.endpoint_name)

    print("\nDeployment complete!")
    print(f"Endpoint: {args.endpoint_name}")
    print(f"To test: python deploy_aws_sagemaker.py --endpoint-name {args.endpoint_name} --test")
    print(f"To delete: python deploy_aws_sagemaker.py --endpoint-name {args.endpoint_name} --delete")


if __name__ == '__main__':
    main()
