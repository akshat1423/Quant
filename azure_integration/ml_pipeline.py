"""
Azure ML Pipeline Configuration

Provides Azure ML integration for training and deploying algorithmic trading models.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AzureMLPipeline:
    """
    Azure ML pipeline for training and deploying quantitative finance models.
    
    This class provides a framework for integrating with Azure ML services
    for scalable model training and deployment.
    """
    
    def __init__(self, workspace_name: str, subscription_id: str, 
                 resource_group: str, region: str = "East US"):
        """
        Initialize Azure ML pipeline.
        
        Args:
            workspace_name: Name of Azure ML workspace
            subscription_id: Azure subscription ID
            resource_group: Azure resource group name
            region: Azure region for deployment
        """
        self.workspace_name = workspace_name
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.region = region
        
        # Pipeline configuration
        self.pipeline_config = {
            "workspace": {
                "name": workspace_name,
                "subscription_id": subscription_id,
                "resource_group": resource_group,
                "region": region
            },
            "compute": {
                "training_cluster": "quant-training-cluster",
                "inference_cluster": "quant-inference-cluster",
                "vm_size": "Standard_D4s_v3",
                "max_nodes": 10,
                "min_nodes": 0
            },
            "environments": {
                "quant_env": {
                    "name": "quant-trading-env",
                    "python_version": "3.9",
                    "conda_dependencies": self._get_conda_dependencies()
                }
            }
        }
        
        logger.info(f"Initialized Azure ML pipeline for workspace: {workspace_name}")
    
    def _get_conda_dependencies(self) -> Dict[str, Any]:
        """Get Conda environment dependencies for Azure ML."""
        return {
            "name": "quant-trading",
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                "python=3.9",
                "pip",
                {
                    "pip": [
                        "numpy>=1.21.0",
                        "scipy>=1.7.0",
                        "pandas>=1.3.0",
                        "scikit-learn>=1.0.0",
                        "tensorflow>=2.8.0",
                        "torch>=1.10.0",
                        "stable-baselines3>=1.6.0",
                        "quantlib-python>=1.26",
                        "azureml-core>=1.44.0",
                        "azureml-train-core>=1.44.0",
                        "azureml-mlflow>=1.44.0",
                        "joblib>=1.1.0",
                        "matplotlib>=3.5.0",
                        "plotly>=5.0.0"
                    ]
                }
            ]
        }
    
    def create_training_pipeline(self, model_type: str, data_path: str,
                               hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create training pipeline configuration for specified model.
        
        Args:
            model_type: Type of model to train ('actor_critic', 'epsilon_greedy', 'don')
            data_path: Path to training data in Azure storage
            hyperparameters: Model hyperparameters
            
        Returns:
            Pipeline configuration dictionary
        """
        logger.info(f"Creating training pipeline for {model_type}")
        
        pipeline_config = {
            "name": f"quant-training-{model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "description": f"Training pipeline for {model_type} algorithmic trading model",
            "compute_target": self.pipeline_config["compute"]["training_cluster"],
            "environment": self.pipeline_config["environments"]["quant_env"]["name"],
            "source_directory": "./",
            "entry_script": f"train_{model_type}.py",
            "arguments": [
                "--data-path", data_path,
                "--model-type", model_type,
                "--hyperparameters", json.dumps(hyperparameters),
                "--output-path", "${{outputs.model_output}}"
            ],
            "inputs": {
                "training_data": {
                    "type": "uri_folder",
                    "path": data_path
                }
            },
            "outputs": {
                "model_output": {
                    "type": "uri_folder",
                    "mode": "rw_mount"
                },
                "metrics_output": {
                    "type": "uri_folder", 
                    "mode": "rw_mount"
                }
            },
            "tags": {
                "model_type": model_type,
                "framework": "tensorflow" if model_type in ["actor_critic", "don"] else "pytorch",
                "project": "algorithmic_trading"
            }
        }
        
        return pipeline_config
    
    def create_option_pricing_pipeline(self, pricing_method: str, 
                                     market_data_path: str,
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create option pricing pipeline configuration.
        
        Args:
            pricing_method: Pricing method ('stochastic_mesh', 'longstaff_schwartz')
            market_data_path: Path to market data
            parameters: Pricing parameters
            
        Returns:
            Pipeline configuration dictionary
        """
        logger.info(f"Creating option pricing pipeline for {pricing_method}")
        
        pipeline_config = {
            "name": f"option-pricing-{pricing_method}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "description": f"Option pricing pipeline using {pricing_method} method",
            "compute_target": self.pipeline_config["compute"]["training_cluster"],
            "environment": self.pipeline_config["environments"]["quant_env"]["name"],
            "source_directory": "./",
            "entry_script": "price_options.py",
            "arguments": [
                "--pricing-method", pricing_method,
                "--market-data-path", market_data_path,
                "--parameters", json.dumps(parameters),
                "--output-path", "${{outputs.pricing_output}}"
            ],
            "inputs": {
                "market_data": {
                    "type": "uri_folder",
                    "path": market_data_path
                }
            },
            "outputs": {
                "pricing_output": {
                    "type": "uri_folder",
                    "mode": "rw_mount"
                }
            },
            "tags": {
                "pricing_method": pricing_method,
                "project": "option_pricing"
            }
        }
        
        return pipeline_config
    
    def create_model_registry_config(self, model_name: str, model_version: str,
                                   model_type: str, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Create model registry configuration.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_type: Type of model
            performance_metrics: Model performance metrics
            
        Returns:
            Model registry configuration
        """
        registry_config = {
            "name": model_name,
            "version": model_version,
            "description": f"{model_type} model for algorithmic trading",
            "tags": {
                "model_type": model_type,
                "framework": "tensorflow" if model_type in ["actor_critic", "don"] else "pytorch",
                "project": "algorithmic_trading",
                "created_date": datetime.now().isoformat()
            },
            "properties": {
                "sharpe_ratio": str(performance_metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown": str(performance_metrics.get("max_drawdown", 0.0)),
                "annual_return": str(performance_metrics.get("annual_return", 0.0)),
                "win_rate": str(performance_metrics.get("win_rate", 0.0))
            },
            "flavors": {
                "python_function": {
                    "env": self.pipeline_config["environments"]["quant_env"]["name"],
                    "python_version": "3.9"
                }
            }
        }
        
        return registry_config
    
    def create_deployment_config(self, model_name: str, endpoint_name: str,
                               instance_type: str = "Standard_F4s_v2",
                               min_instances: int = 1, max_instances: int = 5) -> Dict[str, Any]:
        """
        Create deployment configuration for model endpoint.
        
        Args:
            model_name: Name of registered model
            endpoint_name: Name of deployment endpoint
            instance_type: Azure VM instance type
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            
        Returns:
            Deployment configuration
        """
        deployment_config = {
            "endpoint": {
                "name": endpoint_name,
                "description": f"Real-time endpoint for {model_name}",
                "auth_mode": "key",
                "tags": {
                    "model": model_name,
                    "project": "algorithmic_trading"
                }
            },
            "deployment": {
                "name": f"{endpoint_name}-deployment",
                "model": model_name,
                "environment": self.pipeline_config["environments"]["quant_env"]["name"],
                "code_configuration": {
                    "code": "./deployment",
                    "scoring_script": "score.py"
                },
                "instance_type": instance_type,
                "instance_count": min_instances,
                "scaling": {
                    "type": "auto",
                    "min_instances": min_instances,
                    "max_instances": max_instances,
                    "target_utilization_percentage": 70,
                    "polling_interval": 300,
                    "delay": 60
                },
                "request_settings": {
                    "request_timeout_ms": 90000,
                    "max_concurrent_requests_per_instance": 1,
                    "max_queue_wait_ms": 500
                },
                "liveness_probe": {
                    "failure_threshold": 30,
                    "success_threshold": 1,
                    "timeout": 2,
                    "period": 10,
                    "initial_delay": 10
                },
                "readiness_probe": {
                    "failure_threshold": 10,
                    "success_threshold": 1,
                    "timeout": 10,
                    "period": 10,
                    "initial_delay": 10
                }
            }
        }
        
        return deployment_config
    
    def create_batch_endpoint_config(self, model_name: str, endpoint_name: str,
                                   compute_cluster: str = None) -> Dict[str, Any]:
        """
        Create batch endpoint configuration for large-scale inference.
        
        Args:
            model_name: Name of registered model
            endpoint_name: Name of batch endpoint
            compute_cluster: Compute cluster for batch processing
            
        Returns:
            Batch endpoint configuration
        """
        if compute_cluster is None:
            compute_cluster = self.pipeline_config["compute"]["inference_cluster"]
        
        batch_config = {
            "endpoint": {
                "name": endpoint_name,
                "description": f"Batch endpoint for {model_name}",
                "tags": {
                    "model": model_name,
                    "project": "algorithmic_trading",
                    "type": "batch"
                }
            },
            "deployment": {
                "name": f"{endpoint_name}-batch-deployment",
                "model": model_name,
                "environment": self.pipeline_config["environments"]["quant_env"]["name"],
                "code_configuration": {
                    "code": "./deployment",
                    "scoring_script": "batch_score.py"
                },
                "compute": compute_cluster,
                "instance_count": 2,
                "max_concurrency_per_instance": 2,
                "mini_batch_size": 10,
                "retry_settings": {
                    "max_retries": 3,
                    "timeout": 300
                },
                "error_threshold": 10,
                "logging_level": "info"
            }
        }
        
        return batch_config
    
    def create_monitoring_config(self, endpoint_name: str, 
                               model_name: str) -> Dict[str, Any]:
        """
        Create monitoring configuration for deployed models.
        
        Args:
            endpoint_name: Name of endpoint to monitor
            model_name: Name of model being monitored
            
        Returns:
            Monitoring configuration
        """
        monitoring_config = {
            "model_monitor": {
                "name": f"{model_name}-monitor",
                "description": f"Data drift and performance monitoring for {model_name}",
                "endpoint_name": endpoint_name,
                "monitoring_type": "model_data_drift",
                "monitoring_frequency": "Daily",
                "alert_email": ["admin@company.com"],
                "metrics": [
                    "data_drift_coefficient",
                    "prediction_drift_coefficient",
                    "feature_attribution_drift_coefficient"
                ],
                "thresholds": {
                    "data_drift_coefficient": 0.3,
                    "prediction_drift_coefficient": 0.3,
                    "feature_attribution_drift_coefficient": 0.3
                }
            },
            "performance_monitor": {
                "name": f"{model_name}-performance-monitor",
                "description": f"Performance monitoring for {model_name}",
                "endpoint_name": endpoint_name,
                "monitoring_type": "model_performance",
                "monitoring_frequency": "Hourly",
                "metrics": [
                    "requests_per_minute",
                    "average_response_time",
                    "error_rate",
                    "cpu_utilization",
                    "memory_utilization"
                ],
                "thresholds": {
                    "average_response_time": 5000,  # milliseconds
                    "error_rate": 0.05,  # 5%
                    "cpu_utilization": 80,  # 80%
                    "memory_utilization": 80  # 80%
                }
            }
        }
        
        return monitoring_config
    
    def generate_pipeline_yaml(self, pipeline_config: Dict[str, Any], 
                             output_path: str = "azure_pipeline.yml") -> str:
        """
        Generate Azure ML pipeline YAML configuration.
        
        Args:
            pipeline_config: Pipeline configuration dictionary
            output_path: Output file path for YAML
            
        Returns:
            Generated YAML content
        """
        yaml_content = f"""
# Azure ML Pipeline Configuration
# Generated on: {datetime.now().isoformat()}

$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline

display_name: {pipeline_config.get('name', 'quant-pipeline')}
description: {pipeline_config.get('description', 'Quantitative Finance Pipeline')}

settings:
  default_compute: {pipeline_config.get('compute_target', 'cpu-cluster')}
  default_datastore: azureml:workspaceblobstore

inputs:
  training_data:
    type: uri_folder
    path: {pipeline_config.get('inputs', {}).get('training_data', {}).get('path', 'azureml:sample-data:1')}

outputs:
  model_output:
    type: uri_folder
    mode: rw_mount

jobs:
  train_job:
    type: command
    inputs:
      training_data: $${{{{parent.inputs.training_data}}}}
    outputs:
      model_output: $${{{{parent.outputs.model_output}}}}
    code: {pipeline_config.get('source_directory', './')}
    environment: {pipeline_config.get('environment', 'azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:33')}
    command: >-
      python {pipeline_config.get('entry_script', 'train.py')}
      {' '.join(pipeline_config.get('arguments', []))}

tags:
  project: algorithmic_trading
  environment: production
"""
        
        # Write YAML to file
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Generated pipeline YAML: {output_path}")
        return yaml_content
    
    def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Get status of running pipeline (placeholder implementation).
        
        Args:
            pipeline_name: Name of pipeline to check
            
        Returns:
            Pipeline status information
        """
        # This would integrate with Azure ML SDK in production
        return {
            "name": pipeline_name,
            "status": "Running",
            "created_time": datetime.now().isoformat(),
            "duration": "00:15:30",
            "progress": 75,
            "message": "Training in progress..."
        }
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """
        Get Azure ML workspace information.
        
        Returns:
            Workspace information dictionary
        """
        return {
            "workspace_name": self.workspace_name,
            "subscription_id": self.subscription_id,
            "resource_group": self.resource_group,
            "region": self.region,
            "compute_targets": [
                self.pipeline_config["compute"]["training_cluster"],
                self.pipeline_config["compute"]["inference_cluster"]
            ],
            "environments": list(self.pipeline_config["environments"].keys()),
            "status": "Active"
        }