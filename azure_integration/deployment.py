"""
Azure Deployment Module

Provides deployment utilities for Azure App Services and related cloud infrastructure.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AzureDeployment:
    """
    Azure deployment utilities for the algorithmic trading platform.
    
    Handles deployment to Azure App Services, container instances,
    and other Azure cloud infrastructure.
    """
    
    def __init__(self, subscription_id: str, resource_group: str, 
                 app_service_plan: str, location: str = "East US"):
        """
        Initialize Azure deployment manager.
        
        Args:
            subscription_id: Azure subscription ID
            resource_group: Resource group name
            app_service_plan: App Service plan name
            location: Azure region
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.app_service_plan = app_service_plan
        self.location = location
        
        logger.info(f"Initialized Azure deployment for resource group: {resource_group}")
    
    def create_streamlit_app_config(self, app_name: str, 
                                  pricing_tier: str = "B1") -> Dict[str, Any]:
        """
        Create configuration for Streamlit dashboard deployment to Azure App Services.
        
        Args:
            app_name: Name of the App Service
            pricing_tier: Pricing tier (B1, S1, P1V2, etc.)
            
        Returns:
            App Service configuration
        """
        config = {
            "app_service": {
                "name": app_name,
                "resource_group": self.resource_group,
                "location": self.location,
                "service_plan": {
                    "name": self.app_service_plan,
                    "pricing_tier": pricing_tier,
                    "os": "Linux",
                    "reserved": True
                },
                "runtime": {
                    "language": "Python",
                    "version": "3.9"
                },
                "app_settings": {
                    "SCM_DO_BUILD_DURING_DEPLOYMENT": "true",
                    "ENABLE_ORYX_BUILD": "true",
                    "PRE_BUILD_SCRIPT_PATH": "prebuild.sh",
                    "POST_BUILD_SCRIPT_PATH": "postbuild.sh",
                    "WEBSITE_RUN_FROM_PACKAGE": "1"
                },
                "startup_command": "python -m streamlit run dashboard/streamlit_app.py --server.port 8000 --server.address 0.0.0.0"
            },
            "application_insights": {
                "name": f"{app_name}-insights",
                "application_type": "web",
                "retention_in_days": 90,
                "sampling_percentage": 100
            },
            "deployment": {
                "source_control": {
                    "repo_url": "https://github.com/your-org/Quant",
                    "branch": "main",
                    "is_manual_integration": False,
                    "is_mercurial": False
                },
                "docker": {
                    "enabled": False
                }
            }
        }
        
        return config
    
    def create_container_app_config(self, app_name: str, 
                                  container_image: str) -> Dict[str, Any]:
        """
        Create configuration for Azure Container Apps deployment.
        
        Args:
            app_name: Name of the container app
            container_image: Docker container image
            
        Returns:
            Container Apps configuration
        """
        config = {
            "container_app": {
                "name": app_name,
                "resource_group": self.resource_group,
                "location": self.location,
                "managed_environment": f"{app_name}-env",
                "container": {
                    "image": container_image,
                    "name": "quant-dashboard",
                    "resources": {
                        "cpu": "0.5",
                        "memory": "1Gi"
                    },
                    "env": [
                        {
                            "name": "STREAMLIT_SERVER_PORT",
                            "value": "8000"
                        },
                        {
                            "name": "STREAMLIT_SERVER_ADDRESS",
                            "value": "0.0.0.0"
                        }
                    ]
                },
                "ingress": {
                    "external": True,
                    "target_port": 8000,
                    "transport": "http"
                },
                "scaling": {
                    "min_replicas": 1,
                    "max_replicas": 10,
                    "rules": [
                        {
                            "name": "http-requests",
                            "http": {
                                "concurrent_requests": 30
                            }
                        }
                    ]
                }
            },
            "log_analytics": {
                "name": f"{app_name}-logs",
                "retention_in_days": 30,
                "sku": "PerGB2018"
            }
        }
        
        return config
    
    def create_function_app_config(self, function_name: str,
                                 storage_account: str) -> Dict[str, Any]:
        """
        Create configuration for Azure Functions deployment.
        
        Args:
            function_name: Name of the function app
            storage_account: Storage account name
            
        Returns:
            Function App configuration
        """
        config = {
            "function_app": {
                "name": function_name,
                "resource_group": self.resource_group,
                "location": self.location,
                "service_plan": self.app_service_plan,
                "storage_account": storage_account,
                "runtime": {
                    "language": "Python",
                    "version": "3.9"
                },
                "app_settings": {
                    "FUNCTIONS_WORKER_RUNTIME": "python",
                    "FUNCTIONS_EXTENSION_VERSION": "~4",
                    "AzureWebJobsStorage": f"DefaultEndpointsProtocol=https;AccountName={storage_account};AccountKey=<key>;EndpointSuffix=core.windows.net",
                    "WEBSITE_CONTENTAZUREFILECONNECTIONSTRING": f"DefaultEndpointsProtocol=https;AccountName={storage_account};AccountKey=<key>;EndpointSuffix=core.windows.net",
                    "WEBSITE_CONTENTSHARE": function_name.lower()
                },
                "functions": [
                    {
                        "name": "OptionPricingTrigger",
                        "type": "httpTrigger",
                        "auth_level": "function",
                        "methods": ["GET", "POST"],
                        "route": "pricing/{method}"
                    },
                    {
                        "name": "TradingSignalTrigger", 
                        "type": "timerTrigger",
                        "schedule": "0 */5 * * * *",  # Every 5 minutes
                        "run_on_startup": False
                    },
                    {
                        "name": "ModelTrainingTrigger",
                        "type": "httpTrigger",
                        "auth_level": "admin",
                        "methods": ["POST"],
                        "route": "train/{model_type}"
                    }
                ]
            }
        }
        
        return config
    
    def create_api_management_config(self, api_name: str,
                                   backend_url: str) -> Dict[str, Any]:
        """
        Create API Management configuration for the trading platform.
        
        Args:
            api_name: Name of the API
            backend_url: Backend service URL
            
        Returns:
            API Management configuration
        """
        config = {
            "api_management": {
                "name": f"{api_name}-apim",
                "resource_group": self.resource_group,
                "location": self.location,
                "sku": {
                    "name": "Developer",
                    "capacity": 1
                },
                "publisher_email": "admin@company.com",
                "publisher_name": "Quant Trading Platform",
                "apis": [
                    {
                        "name": "option-pricing-api",
                        "display_name": "Option Pricing API",
                        "description": "API for option pricing calculations",
                        "service_url": f"{backend_url}/api/pricing",
                        "path": "pricing",
                        "protocols": ["https"],
                        "operations": [
                            {
                                "name": "price-american-put",
                                "display_name": "Price American Put Option",
                                "method": "POST",
                                "url_template": "/american-put",
                                "description": "Calculate American put option price"
                            },
                            {
                                "name": "price-european-put",
                                "display_name": "Price European Put Option", 
                                "method": "POST",
                                "url_template": "/european-put",
                                "description": "Calculate European put option price"
                            }
                        ]
                    },
                    {
                        "name": "trading-signals-api",
                        "display_name": "Trading Signals API",
                        "description": "API for trading signal generation",
                        "service_url": f"{backend_url}/api/signals",
                        "path": "signals",
                        "protocols": ["https"],
                        "operations": [
                            {
                                "name": "get-signals",
                                "display_name": "Get Trading Signals",
                                "method": "GET",
                                "url_template": "/{symbol}",
                                "description": "Get trading signals for a symbol"
                            },
                            {
                                "name": "backtest-strategy",
                                "display_name": "Backtest Trading Strategy",
                                "method": "POST", 
                                "url_template": "/backtest",
                                "description": "Backtest a trading strategy"
                            }
                        ]
                    }
                ],
                "policies": {
                    "cors": {
                        "allowed_origins": ["*"],
                        "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
                        "allowed_headers": ["*"]
                    },
                    "rate_limiting": {
                        "calls": 1000,
                        "renewal_period": 3600,
                        "counter_key": "@(context.Request.IpAddress)"
                    },
                    "authentication": {
                        "type": "subscription_key",
                        "header": "Ocp-Apim-Subscription-Key"
                    }
                }
            }
        }
        
        return config
    
    def create_storage_config(self, storage_account_name: str) -> Dict[str, Any]:
        """
        Create Azure Storage configuration for data and model storage.
        
        Args:
            storage_account_name: Name of storage account
            
        Returns:
            Storage configuration
        """
        config = {
            "storage_account": {
                "name": storage_account_name,
                "resource_group": self.resource_group,
                "location": self.location,
                "sku": "Standard_LRS",
                "kind": "StorageV2",
                "access_tier": "Hot",
                "containers": [
                    {
                        "name": "market-data",
                        "access_level": "private",
                        "description": "Container for market data files"
                    },
                    {
                        "name": "models",
                        "access_level": "private", 
                        "description": "Container for trained ML models"
                    },
                    {
                        "name": "results",
                        "access_level": "private",
                        "description": "Container for analysis results"
                    },
                    {
                        "name": "logs",
                        "access_level": "private",
                        "description": "Container for application logs"
                    }
                ],
                "file_shares": [
                    {
                        "name": "config",
                        "quota": 5,
                        "description": "File share for configuration files"
                    }
                ]
            },
            "blob_lifecycle": {
                "rules": [
                    {
                        "name": "move-to-cool",
                        "enabled": True,
                        "filters": {
                            "blob_types": ["blockBlob"],
                            "prefix_match": ["market-data/"]
                        },
                        "actions": {
                            "base_blob": {
                                "tier_to_cool": {
                                    "days_after_modification_greater_than": 30
                                },
                                "tier_to_archive": {
                                    "days_after_modification_greater_than": 90
                                }
                            }
                        }
                    }
                ]
            }
        }
        
        return config
    
    def create_key_vault_config(self, vault_name: str) -> Dict[str, Any]:
        """
        Create Azure Key Vault configuration for secrets management.
        
        Args:
            vault_name: Name of the key vault
            
        Returns:
            Key Vault configuration
        """
        config = {
            "key_vault": {
                "name": vault_name,
                "resource_group": self.resource_group,
                "location": self.location,
                "sku": "standard",
                "enabled_for_deployment": True,
                "enabled_for_disk_encryption": True,
                "enabled_for_template_deployment": True,
                "soft_delete_retention_days": 90,
                "purge_protection_enabled": True,
                "access_policies": [
                    {
                        "tenant_id": "<tenant-id>",
                        "object_id": "<application-object-id>",
                        "permissions": {
                            "keys": ["get", "list", "update", "create", "import", "delete"],
                            "secrets": ["get", "list", "set", "delete"],
                            "certificates": ["get", "list", "update", "create", "import", "delete"]
                        }
                    }
                ],
                "secrets": [
                    {
                        "name": "storage-connection-string",
                        "description": "Connection string for Azure Storage"
                    },
                    {
                        "name": "ml-workspace-key",
                        "description": "Azure ML workspace authentication key"
                    },
                    {
                        "name": "api-keys",
                        "description": "External API keys for data sources"
                    },
                    {
                        "name": "database-connection",
                        "description": "Database connection string"
                    }
                ]
            }
        }
        
        return config
    
    def create_monitoring_config(self, app_name: str) -> Dict[str, Any]:
        """
        Create monitoring and alerting configuration.
        
        Args:
            app_name: Name of the application
            
        Returns:
            Monitoring configuration
        """
        config = {
            "log_analytics": {
                "name": f"{app_name}-logs",
                "resource_group": self.resource_group,
                "location": self.location,
                "sku": "PerGB2018",
                "retention_in_days": 30
            },
            "application_insights": {
                "name": f"{app_name}-insights",
                "resource_group": self.resource_group,
                "location": self.location,
                "application_type": "web",
                "retention_in_days": 90
            },
            "alerts": [
                {
                    "name": "high-response-time",
                    "description": "Alert when average response time exceeds threshold",
                    "severity": 2,
                    "criteria": {
                        "metric_name": "requests/duration",
                        "operator": "GreaterThan",
                        "threshold": 5000,
                        "time_aggregation": "Average"
                    },
                    "evaluation_frequency": "PT5M",
                    "window_size": "PT15M"
                },
                {
                    "name": "high-error-rate",
                    "description": "Alert when error rate exceeds threshold",
                    "severity": 1,
                    "criteria": {
                        "metric_name": "requests/failed",
                        "operator": "GreaterThan",
                        "threshold": 10,
                        "time_aggregation": "Count"
                    },
                    "evaluation_frequency": "PT1M",
                    "window_size": "PT5M"
                },
                {
                    "name": "cpu-utilization",
                    "description": "Alert when CPU utilization is high",
                    "severity": 2,
                    "criteria": {
                        "metric_name": "performanceCounters/processorCpuPercentage",
                        "operator": "GreaterThan",
                        "threshold": 80,
                        "time_aggregation": "Average"
                    },
                    "evaluation_frequency": "PT5M",
                    "window_size": "PT15M"
                }
            ],
            "action_groups": [
                {
                    "name": "quant-alerts",
                    "short_name": "quant",
                    "email_receivers": [
                        {
                            "name": "admin",
                            "email_address": "admin@company.com"
                        }
                    ],
                    "webhook_receivers": [
                        {
                            "name": "slack-webhook",
                            "service_uri": "https://hooks.slack.com/services/..."
                        }
                    ]
                }
            ]
        }
        
        return config
    
    def generate_arm_template(self, configs: List[Dict[str, Any]], 
                            template_path: str = "deployment_template.json") -> str:
        """
        Generate ARM template for Azure resource deployment.
        
        Args:
            configs: List of resource configurations
            template_path: Output path for ARM template
            
        Returns:
            Generated ARM template JSON
        """
        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "metadata": {
                "description": "ARM template for Quant Trading Platform deployment",
                "author": "Quant Trading Team",
                "dateCreated": datetime.now().isoformat()
            },
            "parameters": {
                "environment": {
                    "type": "string",
                    "defaultValue": "dev",
                    "allowedValues": ["dev", "staging", "prod"],
                    "metadata": {
                        "description": "Environment name"
                    }
                },
                "location": {
                    "type": "string",
                    "defaultValue": self.location,
                    "metadata": {
                        "description": "Location for all resources"
                    }
                }
            },
            "variables": {
                "resourcePrefix": "[concat('quant-', parameters('environment'))]",
                "storageAccountName": "[concat(variables('resourcePrefix'), 'storage')]",
                "appServicePlanName": "[concat(variables('resourcePrefix'), '-plan')]"
            },
            "resources": [],
            "outputs": {
                "webAppUrl": {
                    "type": "string",
                    "value": "[concat('https://', variables('resourcePrefix'), '-dashboard.azurewebsites.net')]"
                },
                "storageAccountName": {
                    "type": "string",
                    "value": "[variables('storageAccountName')]"
                }
            }
        }
        
        # Add resources from configs
        for config in configs:
            if "app_service" in config:
                template["resources"].extend(self._generate_app_service_resources(config))
            if "storage_account" in config:
                template["resources"].extend(self._generate_storage_resources(config))
            if "key_vault" in config:
                template["resources"].extend(self._generate_key_vault_resources(config))
        
        # Write template to file
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Generated ARM template: {template_path}")
        return json.dumps(template, indent=2)
    
    def _generate_app_service_resources(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate App Service resources for ARM template."""
        app_service_config = config["app_service"]
        
        resources = [
            {
                "type": "Microsoft.Web/serverfarms",
                "apiVersion": "2021-02-01",
                "name": "[variables('appServicePlanName')]",
                "location": "[parameters('location')]",
                "sku": {
                    "name": app_service_config["service_plan"]["pricing_tier"],
                    "capacity": 1
                },
                "kind": "linux",
                "properties": {
                    "reserved": True
                }
            },
            {
                "type": "Microsoft.Web/sites",
                "apiVersion": "2021-02-01",
                "name": "[concat(variables('resourcePrefix'), '-dashboard')]",
                "location": "[parameters('location')]",
                "dependsOn": [
                    "[resourceId('Microsoft.Web/serverfarms', variables('appServicePlanName'))]"
                ],
                "properties": {
                    "serverFarmId": "[resourceId('Microsoft.Web/serverfarms', variables('appServicePlanName'))]",
                    "siteConfig": {
                        "linuxFxVersion": f"PYTHON|{app_service_config['runtime']['version']}",
                        "appSettings": [
                            {
                                "name": k,
                                "value": v
                            }
                            for k, v in app_service_config["app_settings"].items()
                        ]
                    }
                }
            }
        ]
        
        return resources
    
    def _generate_storage_resources(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Storage Account resources for ARM template."""
        storage_config = config["storage_account"]
        
        return [
            {
                "type": "Microsoft.Storage/storageAccounts",
                "apiVersion": "2021-09-01",
                "name": "[variables('storageAccountName')]",
                "location": "[parameters('location')]",
                "sku": {
                    "name": storage_config["sku"]
                },
                "kind": storage_config["kind"],
                "properties": {
                    "accessTier": storage_config["access_tier"]
                }
            }
        ]
    
    def _generate_key_vault_resources(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Key Vault resources for ARM template."""
        vault_config = config["key_vault"]
        
        return [
            {
                "type": "Microsoft.KeyVault/vaults",
                "apiVersion": "2021-11-01-preview",
                "name": "[concat(variables('resourcePrefix'), '-kv')]",
                "location": "[parameters('location')]",
                "properties": {
                    "sku": {
                        "family": "A",
                        "name": vault_config["sku"]
                    },
                    "tenantId": "[subscription().tenantId]",
                    "enabledForDeployment": vault_config["enabled_for_deployment"],
                    "enabledForDiskEncryption": vault_config["enabled_for_disk_encryption"],
                    "enabledForTemplateDeployment": vault_config["enabled_for_template_deployment"],
                    "enableSoftDelete": True,
                    "softDeleteRetentionInDays": vault_config["soft_delete_retention_days"],
                    "enablePurgeProtection": vault_config["purge_protection_enabled"]
                }
            }
        ]
    
    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """
        Get deployment status (placeholder implementation).
        
        Args:
            deployment_name: Name of deployment to check
            
        Returns:
            Deployment status information
        """
        return {
            "name": deployment_name,
            "status": "Succeeded",
            "timestamp": datetime.now().isoformat(),
            "duration": "00:08:45",
            "resources_deployed": 5,
            "outputs": {
                "webapp_url": f"https://{deployment_name}.azurewebsites.net",
                "api_endpoint": f"https://{deployment_name}-api.azurewebsites.net"
            }
        }