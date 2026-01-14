import json

def load_schema(schema_path: str) -> dict:
    with open(schema_path, "r") as f:
        return json.load(f)

def validate_dataframe(df, schema: dict):
    label_col = schema["label_column"]

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    for feature, rules in schema["features"].items():
        if rules.get("required", False) and feature not in df.columns:
            raise ValueError(f"Missing required feature: {feature}")
