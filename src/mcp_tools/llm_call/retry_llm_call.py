"""
Generic Retry Logic for LLM Calls with Validation.

This module provides a higher-level function `retry_llm_call` that wraps
an underlying LLM call function (like `litellm_call`). It adds a layer of
robustness by:
- Retrying the LLM call up to a specified maximum number of attempts.
- Applying a list of custom validation functions to the LLM response after
  each attempt.
- If validation fails, appending the error messages to the conversation
  history and prompting the LLM to correct its response on the next retry.

This pattern is useful for ensuring the LLM output conforms to specific
structural or content requirements before proceeding.

Relevant Documentation:
- General Retry Pattern Concepts: (Consider adding a link to a relevant article or pattern description if available)
- Project Error Handling Notes: ../../repo_docs/error_handling.md (Placeholder)

Input/Output:
- Input:
    - `llm_call`: The async function to execute (e.g., `litellm_call`).
    - `llm_config`: The configuration dictionary for the `llm_call`.
    - `validation_strategies`: A list of functions, each taking the LLM
      response and returning `True` on success or an error message string
      on failure.
    - `max_retries`: Maximum number of attempts.
- Output: The validated response dictionary from the `llm_call`, or raises
  an exception if validation fails after all retries.
"""
from typing import Callable, Dict, Any, List, Optional, Union, Tuple
import asyncio
from loguru import logger

async def retry_llm_call(
    llm_call: Callable,  # The LLM function to call (e.g., call_litellm_structured)
    # llm_config: QuestionItem,  # Configuration object (QuestionItem) - Replaced by individual args
    model: str,
    messages: List[Dict[str, str]], # Pass the already substituted messages
    temperature: Optional[float],
    max_tokens: Optional[int],
    api_base: Optional[str],
    response_model: Optional[Any], # Pass response_model if needed
    validation_strategies: List[Callable],  # List of validation functions
    max_retries: int = 3,  # Maximum number of retries
) -> Tuple[Dict, int]: # Return response and retry count
    """
    A generic function to handle retries, validation, and iterative improvement for LLM calls.

    Args:
        llm_call (Callable): The LLM function to call.
        llm_config (QuestionItem): Configuration object containing LLM parameters.
        validation_strategies (List[Callable]): List of functions to validate the LLM response.
        max_retries (int): Maximum number of retries.

    Returns:
        Tuple[Dict, int]: A tuple containing the validated LLM response dictionary and the number of retries taken (0-indexed).
    """
    logger.debug(f"Retry Call: Received initial messages: {messages}") # ADDED LOGGING
    retries = 0
    # Initial message list constructed once
    # Initialize messages list
    current_messages = messages # Start with the substituted messages

    while retries < max_retries:
        try:
            # Construct the nested config dict for the underlying litellm_call on each attempt

            llm_params = {
                "model": model,
                "messages": current_messages, # Use the current messages list
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_base": api_base,
                # Add other relevant parameters from QuestionItem if litellm_call uses them
                # e.g., "top_p": llm_config.top_p, "stream": llm_config.stream, etc.
            }
            # If response_model is specified, add it for structured output
            if response_model:
                 # Assuming litellm_call uses 'response_model' key within llm_config
                 # Adjust key if litellm_call expects something different (e.g., 'response_format')
                 llm_params["response_model"] = response_model

            call_config = {"llm_config": llm_params} # Create the nested structure

            logger.debug(f"Attempt {retries + 1}: Calling LLM with config: {call_config}")
            logger.debug(f"Attempt {retries + 1}: Messages being sent: {current_messages}") # ADDED LOGGING (This line might already exist from previous attempt, ensure it's correct)
            response = await llm_call(call_config) # Pass the nested dict
            validation_errors = []

            # Apply all validation strategies
            for validate in validation_strategies:
                validation_result = validate(response)
                if validation_result is not True:
                    validation_errors.append(validation_result)

            # If all validations pass, return the response
            if not validation_errors:
                return response, retries

            # If any validation fails, log the errors and retry
            logger.warning(f"Attempt {retries + 1}: Validation failed: {', '.join(validation_errors)}")
            # Append feedback to the current_messages list for the next retry attempt
            current_messages.append({"role": "assistant", "content": str(response)}) # Add LLM's failed response
            current_messages.append({
                "role": "user",
                "content": f"The previous response failed validation with errors: {', '.join(validation_errors)}. Please correct the response based on the original request and the validation errors.",
            })
        except Exception as e:
            logger.error(f"Attempt {retries + 1} failed: {e}")
        retries += 1

    # Max retries exceeded: Add the failure reason to the messages object
    # Max retries exceeded
    failure_message = f"Max retries ({max_retries}) exceeded. The LLM failed to generate a valid response after validation attempts."
    # Note: We don't modify the original llm_config (QuestionItem) here.
    # The failure is raised, and the calling function (engine.py) handles the error result.
    logger.error(failure_message)
    raise Exception(failure_message)


async def main():
    return


if __name__ == "__main__":
    asyncio.run(main())
