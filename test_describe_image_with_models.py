import os
from PIL import Image
from mcp_tools.llm_call.describe_image import describe_image
from mcp_tools.llm_call.initialize_litellm_cache import initialize_litellm_cache

# Initialize LiteLLM cache
initialize_litellm_cache()

# Path to the image
image_path = "/Users/robert/claude_mcp_configs/sample_images/elephant.png"

# Example prompt
prompt = "Describe this image in detail."

# Display model information and test with different models
models_to_test = [
    "vertex_ai/gemini-2.5-pro-preview-05-06",  # Default model
    "gpt-4o"  # Alternative model
]

# Path to service account credentials
credentials_file = "/Users/robert/claude_mcp_configs/vertex_ai_service_account.json"

for model in models_to_test:
    print(f"\n=== Testing describe_image with model: {model} ===")
    
    print("WITHOUT credentials...")
    try:
        response_json = describe_image(
            image_input=image_path,
            prompt=prompt,
            model=model,
            credentials_file=None  # No credentials
        )
        print(f"Response without credentials: {response_json}")
    except Exception as e:
        print(f"Error without credentials: {str(e)}")
    
    print("\nWITH credentials...")
    try:
        response_json = describe_image(
            image_input=image_path,
            prompt=prompt,
            model=model,
            credentials_file=credentials_file
        )
        print(f"Response with credentials: {response_json}")
    except Exception as e:
        print(f"Error with credentials: {str(e)}")