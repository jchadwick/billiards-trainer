"""JSON Schema Generation for API Documentation.

This module provides utilities for generating JSON schemas from Pydantic models
for API documentation, client code generation, and validation. It includes:
- OpenAPI schema generation
- Standalone JSON schema export
- Model documentation generation
- Schema validation utilities

The generated schemas can be used with tools like Swagger UI, Redoc, and various
client generators.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

from . import common, requests, responses, websocket

# =============================================================================
# Schema Generation Classes
# =============================================================================


class EnhancedJsonSchemaGenerator(GenerateJsonSchema):
    """Enhanced JSON schema generator with additional documentation features."""

    def generate_schema(self, schema: Any) -> dict[str, Any]:
        """Generate schema with enhanced documentation."""
        json_schema = super().generate_schema(schema)

        # Add additional metadata
        if hasattr(schema, "__name__"):
            json_schema["title"] = schema.__name__

        # Add module information
        if hasattr(schema, "__module__"):
            json_schema["x-module"] = schema.__module__

        return json_schema

    def field_title_generator(self, field_info, field_name: str) -> str:
        """Generate field titles from field names."""
        # Convert snake_case to Title Case
        return field_name.replace("_", " ").title()

    def generate_field_schema(self, field_info, validation_alias, mode):
        """Generate field schema with enhanced descriptions."""
        schema = super().generate_field_schema(field_info, validation_alias, mode)

        # Add examples if available
        if hasattr(field_info, "examples") and field_info.examples:
            schema["examples"] = field_info.examples

        return schema


# =============================================================================
# Schema Generation Functions
# =============================================================================


def generate_model_schema(
    model_class: type[BaseModel],
    title: Optional[str] = None,
    include_examples: bool = True,
) -> dict[str, Any]:
    """Generate JSON schema for a single Pydantic model."""
    try:
        schema = model_class.model_json_schema(
            schema_generator=EnhancedJsonSchemaGenerator
        )

        if title:
            schema["title"] = title

        # Add model metadata
        schema["x-model-info"] = {
            "module": model_class.__module__,
            "class_name": model_class.__name__,
            "generated_at": str(datetime.now()),
        }

        # Add examples if requested
        if include_examples and hasattr(model_class, "model_config"):
            try:
                # Try to create an example instance
                if hasattr(model_class, "__name__"):
                    f"create_{model_class.__name__.lower()}"
                    # This would need specific factory functions for each model
                    pass
            except Exception:
                pass  # Skip examples if factory doesn't exist

        return schema

    except Exception as e:
        return {
            "error": f"Failed to generate schema for {model_class.__name__}: {str(e)}",
            "type": "object",
        }


def generate_module_schemas(
    module, output_format: str = "json"
) -> dict[str, dict[str, Any]]:
    """Generate schemas for all models in a module."""
    schemas = {}

    for attr_name in dir(module):
        attr = getattr(module, attr_name)

        # Check if it's a Pydantic model class
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseModel)
            and attr is not BaseModel
        ):
            schema = generate_model_schema(attr)
            schemas[attr_name] = schema

    return schemas


def generate_all_api_schemas() -> dict[str, dict[str, Any]]:
    """Generate schemas for all API models."""
    all_schemas = {
        "request_models": generate_module_schemas(requests),
        "response_models": generate_module_schemas(responses),
        "websocket_models": generate_module_schemas(websocket),
        "common_models": generate_module_schemas(common),
    }

    # Add metadata
    all_schemas["metadata"] = {
        "generated_at": str(datetime.now()),
        "version": "1.0.0",
        "description": "Billiards Trainer API Model Schemas",
        "total_models": sum(
            len(schemas)
            for schemas in all_schemas.values()
            if isinstance(schemas, dict)
        ),
    }

    return all_schemas


# =============================================================================
# OpenAPI Schema Generation
# =============================================================================


def generate_openapi_schema(
    title: str = "Billiards Trainer API",
    version: str = "1.0.0",
    description: str = "API for billiards training system",
) -> dict[str, Any]:
    """Generate OpenAPI 3.0 schema for the entire API."""
    # Get all model schemas
    api_schemas = generate_all_api_schemas()

    # Build components section
    components = {"schemas": {}}

    # Add all model schemas to components
    for category, schemas in api_schemas.items():
        if category != "metadata" and isinstance(schemas, dict):
            for model_name, schema in schemas.items():
                components["schemas"][model_name] = schema

    # Create OpenAPI specification
    openapi_spec = {
        "openapi": "3.0.3",
        "info": {
            "title": title,
            "version": version,
            "description": description,
            "contact": {
                "name": "Billiards Trainer API",
                "url": "https://github.com/example/billiards-trainer",
            },
            "license": {"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        },
        "servers": [
            {"url": "http://localhost:8000", "description": "Development server"},
            {
                "url": "https://api.billiards-trainer.example.com",
                "description": "Production server",
            },
        ],
        "paths": generate_openapi_paths(),
        "components": components,
        "tags": [
            {"name": "health", "description": "System health and status"},
            {"name": "config", "description": "Configuration management"},
            {"name": "calibration", "description": "Camera and projector calibration"},
            {"name": "game", "description": "Game state and management"},
            {"name": "websocket", "description": "Real-time WebSocket communication"},
        ],
    }

    return openapi_spec


def generate_openapi_paths() -> dict[str, Any]:
    """Generate OpenAPI paths for the API endpoints."""
    paths = {
        "/api/v1/health": {
            "get": {
                "tags": ["health"],
                "summary": "Health check",
                "description": "Get system health status and metrics",
                "responses": {
                    "200": {
                        "description": "System health information",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/HealthResponse"
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/v1/config": {
            "get": {
                "tags": ["config"],
                "summary": "Get configuration",
                "description": "Retrieve current system configuration",
                "responses": {
                    "200": {
                        "description": "Current configuration",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ConfigResponse"
                                }
                            }
                        },
                    }
                },
            },
            "put": {
                "tags": ["config"],
                "summary": "Update configuration",
                "description": "Update system configuration",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/ConfigUpdateRequest"
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Configuration updated successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ConfigUpdateResponse"
                                }
                            }
                        },
                    },
                    "400": {
                        "description": "Validation error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                },
            },
        },
        "/api/v1/calibration/start": {
            "post": {
                "tags": ["calibration"],
                "summary": "Start calibration",
                "description": "Begin a new calibration session",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/CalibrationStartRequest"
                            }
                        }
                    },
                },
                "responses": {
                    "201": {
                        "description": "Calibration session started",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/CalibrationStartResponse"
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/v1/game/state": {
            "get": {
                "tags": ["game"],
                "summary": "Get game state",
                "description": "Retrieve current game state",
                "responses": {
                    "200": {
                        "description": "Current game state",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/GameStateResponse"
                                }
                            }
                        },
                    }
                },
            }
        },
        "/ws": {
            "get": {
                "tags": ["websocket"],
                "summary": "WebSocket connection",
                "description": "Establish WebSocket connection for real-time data",
                "parameters": [
                    {
                        "name": "token",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Authentication token",
                    }
                ],
                "responses": {
                    "101": {"description": "WebSocket connection established"},
                    "401": {"description": "Authentication failed"},
                },
            }
        },
    }

    return paths


# =============================================================================
# Schema Export Functions
# =============================================================================


def export_schemas_to_files(
    output_dir: str = "schemas", format: str = "json"
) -> dict[str, str]:
    """Export all schemas to separate files."""
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    exported_files = {}

    # Generate all schemas
    all_schemas = generate_all_api_schemas()

    # Export individual model schemas
    for category, schemas in all_schemas.items():
        if category != "metadata" and isinstance(schemas, dict):
            category_dir = output_path / category
            category_dir.mkdir(exist_ok=True)

            for model_name, schema in schemas.items():
                filename = f"{model_name.lower()}.{format}"
                file_path = category_dir / filename

                with open(file_path, "w") as f:
                    if format == "json":
                        json.dump(schema, f, indent=2)
                    else:
                        json.dump(schema, f)

                exported_files[f"{category}/{model_name}"] = str(file_path)

    # Export combined schemas
    combined_file = output_path / f"all_schemas.{format}"
    with open(combined_file, "w") as f:
        json.dump(all_schemas, f, indent=2)
    exported_files["combined"] = str(combined_file)

    # Export OpenAPI schema
    openapi_schema = generate_openapi_schema()
    openapi_file = output_path / f"openapi.{format}"
    with open(openapi_file, "w") as f:
        json.dump(openapi_schema, f, indent=2)
    exported_files["openapi"] = str(openapi_file)

    # Create index file
    index_data = {
        "generated_at": datetime.now().isoformat(),
        "total_files": len(exported_files),
        "files": exported_files,
        "categories": list(all_schemas.keys()),
    }

    index_file = output_path / "index.json"
    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)

    return exported_files


def generate_documentation_markdown(schemas: dict[str, Any]) -> str:
    """Generate Markdown documentation from schemas."""
    markdown = []

    markdown.append("# Billiards Trainer API Models Documentation\n")
    markdown.append(f"Generated at: {datetime.now().isoformat()}\n")

    for category, category_schemas in schemas.items():
        if category == "metadata":
            continue

        markdown.append(f"## {category.replace('_', ' ').title()}\n")

        if isinstance(category_schemas, dict):
            for model_name, schema in category_schemas.items():
                markdown.append(f"### {model_name}\n")

                if "description" in schema:
                    markdown.append(f"{schema['description']}\n")

                # Add properties table
                if "properties" in schema:
                    markdown.append("| Property | Type | Description |")
                    markdown.append("|----------|------|-------------|")

                    for prop_name, prop_schema in schema["properties"].items():
                        prop_type = prop_schema.get("type", "unknown")
                        prop_desc = prop_schema.get("description", "")
                        markdown.append(f"| {prop_name} | {prop_type} | {prop_desc} |")

                    markdown.append("")

                # Add required fields
                if "required" in schema and schema["required"]:
                    markdown.append(
                        f"**Required fields:** {', '.join(schema['required'])}\n"
                    )

                markdown.append("---\n")

    return "\n".join(markdown)


# =============================================================================
# Schema Validation Functions
# =============================================================================


def validate_schema_consistency() -> dict[str, Any]:
    """Validate that all schemas are consistent and well-formed."""
    issues = []
    warnings = []

    try:
        all_schemas = generate_all_api_schemas()

        for category, schemas in all_schemas.items():
            if category == "metadata":
                continue

            if not isinstance(schemas, dict):
                issues.append(f"Category {category} is not a dictionary")
                continue

            for model_name, schema in schemas.items():
                # Check if schema has required top-level fields
                if "type" not in schema:
                    issues.append(f"{category}.{model_name}: Missing 'type' field")

                # Check properties
                if "properties" in schema:
                    for prop_name, prop_schema in schema["properties"].items():
                        if not isinstance(prop_schema, dict):
                            issues.append(
                                f"{category}.{model_name}.{prop_name}: Property schema is not a dictionary"
                            )

                        if "type" not in prop_schema and "$ref" not in prop_schema:
                            warnings.append(
                                f"{category}.{model_name}.{prop_name}: Missing type or $ref"
                            )

                # Check required fields exist in properties
                if "required" in schema and "properties" in schema:
                    for required_field in schema["required"]:
                        if required_field not in schema["properties"]:
                            issues.append(
                                f"{category}.{model_name}: Required field '{required_field}' not in properties"
                            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_schemas": sum(
                len(schemas)
                for schemas in all_schemas.values()
                if isinstance(schemas, dict)
            ),
            "categories": list(all_schemas.keys()),
        }

    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Schema validation failed: {str(e)}"],
            "warnings": [],
            "total_schemas": 0,
            "categories": [],
        }


# =============================================================================
# CLI and Utility Functions
# =============================================================================


def main():
    """Main function for running schema generation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate API schemas")
    parser.add_argument("--output-dir", default="schemas", help="Output directory")
    parser.add_argument(
        "--format", default="json", choices=["json"], help="Output format"
    )
    parser.add_argument("--validate", action="store_true", help="Validate schemas")
    parser.add_argument("--docs", action="store_true", help="Generate documentation")

    args = parser.parse_args()

    if args.validate:
        print("Validating schemas...")
        validation_result = validate_schema_consistency()
        if validation_result["valid"]:
            print("✅ All schemas are valid!")
        else:
            print("❌ Schema validation failed:")
            for issue in validation_result["issues"]:
                print(f"  - {issue}")

        if validation_result["warnings"]:
            print("⚠️ Warnings:")
            for warning in validation_result["warnings"]:
                print(f"  - {warning}")

        return

    print(f"Generating schemas to {args.output_dir}...")
    exported_files = export_schemas_to_files(args.output_dir, args.format)

    print(f"✅ Generated {len(exported_files)} schema files:")
    for name, path in exported_files.items():
        print(f"  - {name}: {path}")

    if args.docs:
        print("Generating documentation...")
        all_schemas = generate_all_api_schemas()
        docs = generate_documentation_markdown(all_schemas)

        docs_file = Path(args.output_dir) / "README.md"
        with open(docs_file, "w") as f:
            f.write(docs)

        print(f"📖 Documentation generated: {docs_file}")


if __name__ == "__main__":
    main()
