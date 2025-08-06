from typing import Callable, Dict
import pandas as pd
from call_model import call_openai, call_gemini



prompt_tmpl = (
    "You are a copywriter for an online store. Using the product attributes, "
    "write an engaging product description (50–90 words).\n\n"
    "Product name: {product_name}\nFeatures: {Product_attribute_list}\nMaterial: {material}\nWarranty: {warranty}\n\n"
    "Description:"
)


csv_path = 'Assignment_02_product_dataset.csv'
df_products = pd.read_csv(csv_path)


def batch_generate(
    sample_df: pd.DataFrame,
    call_model_fn: Callable[[str], Dict[str, object]],
    prompt_template: str = prompt_tmpl,
) -> pd.DataFrame:
    """Generate descriptions and metrics for each row in *sample_df*.

    The model-calling function *must* return a dict with keys:
    - ``text`` (str) – generated description
    - ``latency_ms`` (float | None)
    - ``input_tokens`` (int | None)
    - ``output_tokens`` (int | None)
    """
    if not isinstance(sample_df, pd.DataFrame):
        raise TypeError("sample_df must be a pandas DataFrame")
    if not callable(call_model_fn):
        raise TypeError("call_model_fn must be callable")

    outputs = []
    for _, row in sample_df.iterrows():
        prompt = prompt_template.format(**row.to_dict())
        out = call_model_fn(prompt)
        if not isinstance(out, dict) or 'text' not in out:
            raise ValueError("call_model_fn must return a dict with at least a 'text' field")
        outputs.append(out)

    result_df = sample_df.copy()
    result_df["generated_description"] = [o["text"] for o in outputs]
    result_df["latency_ms"] = [o.get("latency_ms") for o in outputs]
    result_df["input_tokens"] = [o.get("input_tokens") for o in outputs]
    result_df["output_tokens"] = [o.get("output_tokens") for o in outputs]
    return result_df


def add_cost_columns(df, input_price_per_m: float, output_price_per_m: float):
    """Add cost columns based on token counts.
    Args:
        df: DataFrame with `input_tokens` and `output_tokens`.
        input_price_per_m: $ per 1M input tokens.
        output_price_per_m: $ per 1M output tokens.
    Returns: DataFrame with extra `cost_usd` column.
    """
    if 'input_tokens' not in df or 'output_tokens' not in df:
        raise ValueError('Token columns missing; run batch_generate first')
    cost_input = df['input_tokens'] * (input_price_per_m / 1000000)
    cost_output = df['output_tokens'] * (output_price_per_m / 1000000)
    df = df.copy()
    df['cost_usd'] = (cost_input + cost_output).round(4)
    return df

#Update the prices according to the model you used, or leave them at 0 for HF local models
YOUR_MODEL_INPUT_PRICE_PER_M = 0
YOUR_MODEL_OUTPUT_PRICE_PER_M = 0

outputs_df = batch_generate(df_products, call_gemini)  # NOTE: change model function as needed

# Add rating columns (good/ok/bad)
rating_cols = ["fluency", "grammar", "tone", "length", "grounding", "latency", "cost", "final_score"]
for col in rating_cols:
    if col not in outputs_df:
        outputs_df[col] = ""

xlsx_path = "assignment_03_evaluation_sheet.xlsx"

# Add cost columns
outputs_df = add_cost_columns(outputs_df, YOUR_MODEL_INPUT_PRICE_PER_M, YOUR_MODEL_OUTPUT_PRICE_PER_M)

outputs_df.to_excel(xlsx_path, index=False)
print(f"Saved evaluation sheet → {xlsx_path} with {len(outputs_df)} rows")
