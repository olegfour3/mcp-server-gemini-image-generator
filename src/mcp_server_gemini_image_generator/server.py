import base64
import os
import logging
import sys
import uuid
import asyncio
import re
from io import BytesIO
from typing import Optional, Any, Union, List, Tuple

import PIL.Image
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP

from .prompts import get_image_generation_prompt, get_image_transformation_prompt, get_translate_prompt
from .utils import save_image


# Setup logging
# Default log file in project root or current directory
default_log_path = os.path.join(os.getcwd(), "mcp_server.log")
log_file = os.getenv("MCP_LOG_FILE", default_log_path)
log_file = os.path.abspath(log_file)

# Ensure log directory exists
log_dir = os.path.dirname(log_file)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

# Create file handler with overwrite mode
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Create console handler (stderr)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)

# Setup root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_file}")

# Initialize MCP server
mcp = FastMCP("mcp-server-gemini-image-generator")


# ==================== Gemini API Interaction ====================

async def call_gemini(
    contents: List[Any], 
    model: str = "gemini-2.5-flash-image", 
    config: Optional[types.GenerateContentConfig] = None,
    text_only: bool = False
) -> Union[str, bytes]:
    """Call Gemini API with flexible configuration for different use cases.
    
    Args:
        contents: The content to send to Gemini. list containing text and/or images
        model: The Gemini model to use
        config: Optional configuration for the Gemini API call
        text_only: If True, extract and return only text from the response
        
    Returns:
        If text_only is True: str - The text response from Gemini
        Otherwise: bytes - The binary image data from Gemini
        
    Raises:
        Exception: If there's an error calling the Gemini API
    """
    try:
        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        client = genai.Client(api_key=api_key)
        
        # Normalize deprecated/preview image model names to current ones
        if isinstance(model, str):
            mapping = {
                "gemini-2.0-flash-preview-image-generation": "gemini-2.5-flash-image",
                "gemini-2.5-flash-preview-image": "gemini-2.5-flash-image",
            }
            if model in mapping:
                logger.warning(f"Normalizing deprecated model '{model}' to '{mapping[model]}'")
                model = mapping[model]
            elif "preview" in model:
                logger.warning(f"Normalizing preview model '{model}' to 'gemini-2.5-flash-image'")
                model = "gemini-2.5-flash-image"

        # For image generation, use only gemini-2.5-flash-image (no fallback to text models)
        # Text models don't generate images, only descriptions
        if not text_only:
            # Force gemini-2.5-flash-image for image generation
            fallback_models: List[str] = [model]
        else:
            # For text-only, allow fallbacks
            fallback_models: List[str] = [
                model,
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash-002",
                "gemini-1.5-flash"
            ]

        last_error: Optional[Exception] = None
        response = None
        for candidate_model in fallback_models:
            try:
                # Retry loop for transient errors like 429/503
                max_attempts = 5
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        response = client.models.generate_content(
                            model=candidate_model,
                            contents=contents,
                            config=config
                        )
                        break
                    except Exception as inner_e:
                        err_str = str(inner_e)
                        # Fast-exit if free tier limit is zero for this model
                        if "RESOURCE_EXHAUSTED" in err_str and "limit: 0" in err_str:
                            logger.warning(f"Quota 0 for model {candidate_model}. Skipping to next model.")
                            raise inner_e
                        # Handle retry-after hints
                        if (
                            "RESOURCE_EXHAUSTED" in err_str
                            or "429" in err_str
                            or "rate" in err_str.lower()
                            or "UNAVAILABLE" in err_str
                            or "503" in err_str
                        ):
                            retry_seconds = None
                            m = re.search(r"retry in (\d+)(?:\.\d+)?s", err_str)
                            if m:
                                retry_seconds = int(m.group(1))
                            else:
                                m2 = re.search(r"RetryInfo.*?retryDelay': '([0-9]+)s'", err_str)
                                if m2:
                                    retry_seconds = int(m2.group(1))
                            if retry_seconds is None:
                                retry_seconds = min(2 ** (attempt - 1), 16)
                            if attempt < max_attempts:
                                logger.warning(
                                    f"Transient error on {candidate_model} ({inner_e}), retrying in {retry_seconds}s "
                                    f"(attempt {attempt}/{max_attempts})"
                                )
                                await asyncio.sleep(retry_seconds)
                                continue
                        # If not retryable or attempts exceeded, bubble up to outer except to try next model
                        raise inner_e

                logger.info(f"Response received from Gemini API using model {candidate_model}")
                model = candidate_model
                break
            except Exception as e:
                # Retry on 404 NOT_FOUND or unsupported method
                error_str = str(e)
                if "NOT_FOUND" in error_str or "not found" in error_str or "is not supported" in error_str:
                    if not text_only:
                        # For image generation, don't fallback - only gemini-2.5-flash-image works
                        logger.error(f"Model {candidate_model} not found or not supported for image generation. Only gemini-2.5-flash-image can generate images.")
                        raise ValueError(f"Model {candidate_model} is not available for image generation. Please use gemini-2.5-flash-image (requires paid API access).")
                    logger.warning(f"Model {candidate_model} failed: {error_str}. Trying next fallback model...")
                    last_error = e
                    continue
                if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                    if not text_only:
                        # For image generation, quota error means we can't generate images
                        logger.error(f"Quota exhausted for model {candidate_model}: {error_str}")
                        raise ValueError(f"Quota exhausted for {candidate_model}. Image generation requires paid API access. Free tier limit: 0. Please check your billing: https://ai.dev/usage?tab=rate-limit")
                    logger.error(f"Quota exhausted for model {candidate_model}: {error_str}")
                    last_error = e
                    continue
                # Non-retryable error
                last_error = e
                break

        if response is None:
            raise last_error if last_error else RuntimeError("Failed to obtain response from Gemini API")
        
        logger.info(f"Response received from Gemini API using model {model}")
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response attributes: {[x for x in dir(response) if not x.startswith('_')]}")
        
        # Try to get candidates - check different possible attribute names
        candidates = None
        if hasattr(response, 'candidates'):
            candidates = response.candidates
            logger.info(f"Found candidates: {len(candidates) if candidates else 0}")
        elif hasattr(response, 'candidate'):
            candidates = [response.candidate] if response.candidate else None
            logger.info(f"Found single candidate: {candidates is not None}")
        
        # Log full response structure for debugging
        try:
            logger.info(f"Response repr (first 500 chars): {repr(response)[:500]}")
        except:
            pass
        
        if not candidates:
            # Try to inspect response more carefully
            logger.error(f"Response type: {type(response)}")
            logger.error(f"Response has candidates attr: {hasattr(response, 'candidates')}")
            if hasattr(response, 'candidates'):
                logger.error(f"Candidates value: {response.candidates}")
            logger.error(f"All response attrs: {[x for x in dir(response) if not x.startswith('_')]}")
            raise ValueError("No candidates in Gemini response")
        
        candidate = candidates[0]
        logger.info(f"Candidate type: {type(candidate)}, has content: {hasattr(candidate, 'content')}")
        
        if not hasattr(candidate, 'content') or candidate.content is None:
            logger.error(f"Candidate structure: {[x for x in dir(candidate) if not x.startswith('_')]}")
            raise ValueError("No content in Gemini response candidate")
        
        if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            logger.error(f"Content structure: {[x for x in dir(candidate.content) if not x.startswith('_')]}")
            raise ValueError("No parts in Gemini response content")
        
        logger.info(f"Content has {len(candidate.content.parts)} parts")
        
        # For text-only calls, extract just the text (robustly)
        if text_only:
            for p in candidate.content.parts:
                if p is None:
                    continue
                if hasattr(p, 'text') and p.text:
                    return p.text.strip()
            raise ValueError("No text found in Gemini response")
        
        # Return the image data - exactly as in examples
        logger.info("Scanning parts for inline_data...")
        image_parts = []
        for part in candidate.content.parts:
            logger.info(f"Part type: {type(part)}, has inline_data: {hasattr(part, 'inline_data')}")
            if part.inline_data is not None:
                logger.info(f"Found inline_data! has data: {hasattr(part.inline_data, 'data')}")
                if hasattr(part.inline_data, 'data'):
                    image_parts.append(part.inline_data.data)
                    logger.info(f"Image data type: {type(part.inline_data.data)}, size: {len(part.inline_data.data) if isinstance(part.inline_data.data, bytes) else 'N/A'}")
        
        logger.info(f"Found {len(image_parts)} image parts")
        
        if image_parts:
            # Return first image data (as bytes)
            image_data = image_parts[0]
            if isinstance(image_data, bytes):
                logger.info(f"Returning image data as bytes, size: {len(image_data)}")
                return image_data
            elif isinstance(image_data, str):
                # If it's base64 string, decode it
                logger.info(f"Image data is string, decoding from base64...")
                try:
                    decoded = base64.b64decode(image_data)
                    logger.info(f"Decoded image data, size: {len(decoded)}")
                    return decoded
                except Exception as e:
                    logger.error(f"Failed to decode base64 image data: {str(e)}")
                    raise ValueError(f"Invalid base64 image data: {str(e)}")
            else:
                logger.error(f"Unexpected image data type: {type(image_data)}")
                raise ValueError(f"Unexpected image data type: {type(image_data)}")
        
        # Log detailed structure for debugging if no image found
        logger.error(f"Failed to find image data. Parts count: {len(candidate.content.parts)}")
        for i, part in enumerate(candidate.content.parts):
            logger.error(f"Part {i}: type={type(part)}, has inline_data={hasattr(part, 'inline_data')}")
            if hasattr(part, 'inline_data'):
                logger.error(f"Part {i} inline_data: {part.inline_data}")
                if part.inline_data is not None:
                    logger.error(f"Part {i} inline_data type: {type(part.inline_data)}, attrs: {[x for x in dir(part.inline_data) if not x.startswith('_')]}")
            if hasattr(part, 'text'):
                logger.error(f"Part {i} text: {part.text[:200] if part.text else None}")
            
        raise ValueError("No image data found in Gemini response")

    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise


# ==================== Text Utility Functions ====================

async def convert_prompt_to_filename(prompt: str) -> str:
    """Convert a text prompt into a suitable filename for the generated image using Gemini AI.
    
    Args:
        prompt: The text prompt used to generate the image
        
    Returns:
        A concise, descriptive filename generated based on the prompt
    """
    try:
        # Create a prompt for Gemini to generate a filename
        filename_prompt = f"""
        Based on this image description: "{prompt}"
        
        Generate a short, descriptive file name suitable for the requested image.
        The filename should:
        - Be concise (maximum 5 words)
        - Use underscores between words
        - Not include any file extension
        - Only return the filename, nothing else
        """
        
        # Call Gemini and get the filename
        generated_filename = await call_gemini(filename_prompt, text_only=True)
        logger.info(f"Generated filename: {generated_filename}")
        
        # Return the filename only, without path or extension
        return generated_filename
    
    except Exception as e:
        logger.error(f"Error generating filename with Gemini: {str(e)}")
        # Fallback to a simple filename if Gemini fails
        truncated_text = prompt[:12].strip()
        return f"image_{truncated_text}_{str(uuid.uuid4())[:8]}"


async def translate_prompt(text: str) -> str:
    """Translate and optimize the user's prompt to English for better image generation results.
    
    Args:
        text: The original prompt in any language
        
    Returns:
        English translation of the prompt with preserved intent
    """
    try:
        # Create a prompt for translation with strict intent preservation
        prompt = get_translate_prompt(text)

        # Call Gemini and get the translated prompt
        translated_prompt = await call_gemini(prompt, text_only=True)
        logger.info(f"Original prompt: {text}")
        logger.info(f"Translated prompt: {translated_prompt}")
        
        return translated_prompt
    
    except Exception as e:
        logger.error(f"Error translating prompt: {str(e)}")
        # Return original text if translation fails
        return text


# ==================== Image Processing Functions ====================

async def process_image_with_gemini(
    contents: List[Any], 
    prompt: str, 
    output_path: Optional[str] = None
) -> Tuple[bytes, str]:
    """Process an image request with Gemini and save the result.
    
    Args:
        contents: List containing the prompt and optionally an image (PIL.Image)
        prompt: Original prompt for filename generation
        output_path: Optional path to save the image
        
    Returns:
        Tuple of (image_bytes, saved_image_path)
    """
    # Use generate_content with gemini-2.5-flash-image (as in examples)
    gemini_response = await call_gemini(
        contents,
        model="gemini-2.5-flash-image",
        config=None  # No config needed, model handles image generation
    )
    
    # Generate a filename for the image
    filename = await convert_prompt_to_filename(prompt)
    
    # Save the image and return the path
    saved_image_path = await save_image(gemini_response, filename, output_path)

    return gemini_response, saved_image_path


async def process_image_transform(
    source_image: PIL.Image.Image, 
    optimized_edit_prompt: str, 
    original_edit_prompt: str,
    output_path: Optional[str] = None
) -> Tuple[bytes, str]:
    """Process image transformation with Gemini.
    
    Args:
        source_image: PIL Image object to transform
        optimized_edit_prompt: Optimized text prompt for transformation
        original_edit_prompt: Original user prompt for naming
        output_path: Optional path to save the image
        
    Returns:
        Path to the transformed image file
    """
    # Create prompt for image transformation
    edit_instructions = get_image_transformation_prompt(optimized_edit_prompt)
    
    # Process with Gemini and return the result
    return await process_image_with_gemini(
        [edit_instructions, source_image],
        original_edit_prompt,
        output_path=output_path
    )


async def load_image_from_base64(encoded_image: str) -> Tuple[PIL.Image.Image, str]:
    """Load an image from a base64-encoded string.
    
    Args:
        encoded_image: Base64 encoded image data with header
        
    Returns:
        Tuple containing the PIL Image object and the image format
    """
    if not encoded_image.startswith('data:image/'):
        raise ValueError("Invalid image format. Expected data:image/[format];base64,[data]")
    
    try:
        # Extract the base64 data from the data URL
        image_format, image_data = encoded_image.split(';base64,')
        image_format = image_format.replace('data:', '')  # Get the MIME type e.g., "image/png"
        image_bytes = base64.b64decode(image_data)
        source_image = PIL.Image.open(BytesIO(image_bytes))
        logger.info(f"Successfully loaded image with format: {image_format}")
        return source_image, image_format
    except ValueError as e:
        logger.error(f"Error: Invalid image data format: {str(e)}")
        raise ValueError("Invalid image data format. Image must be in format 'data:image/[format];base64,[data]'")
    except base64.binascii.Error as e:
        logger.error(f"Error: Invalid base64 encoding: {str(e)}")
        raise ValueError("Invalid base64 encoding. Please provide a valid base64 encoded image.")
    except PIL.UnidentifiedImageError:
        logger.error("Error: Could not identify image format")
        raise ValueError("Could not identify image format. Supported formats include PNG, JPEG, GIF, WebP.")
    except Exception as e:
        logger.error(f"Error: Could not load image: {str(e)}")
        raise


# ==================== MCP Tools ====================

@mcp.tool()
async def generate_image_from_text(
    prompt: str,
    output_image_path: Optional[str] = None
) -> Tuple[bytes, str]:
    """Generate an image based on the given text prompt using Google's Gemini model.

    Args:
        prompt: User's text prompt describing the desired image to generate
        output_image_path: Optional path to save the generated image. If not provided, uses default path.
        
    Returns:
        Path to the generated image file using Gemini's image generation capabilities
    """
    try:
        # Translate the prompt to English
        translated_prompt = await translate_prompt(prompt)
        
        # Use prompt directly as in examples (not wrapped in get_image_generation_prompt)
        # Process with Gemini and return the result
        return await process_image_with_gemini(
            [translated_prompt],
            prompt,
            output_path=output_image_path
        )
        
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def transform_image_from_encoded(
    encoded_image: str,
    prompt: str,
    output_image_path: Optional[str] = None
) -> Tuple[bytes, str]:
    """Transform an existing image based on the given text prompt using Google's Gemini model.

    Args:
        encoded_image: Base64 encoded image data with header. Must be in format:
                    "data:image/[format];base64,[data]"
                    Where [format] can be: png, jpeg, jpg, gif, webp, etc.
        prompt: Text prompt describing the desired transformation or modifications
        output_image_path: Optional path to save the transformed image. If not provided, uses default path.
        
    Returns:
        Path to the transformed image file saved on the server
    """
    try:
        logger.info(f"Processing transform_image_from_encoded request with prompt: {prompt}")

        # Load and validate the image
        source_image, _ = await load_image_from_base64(encoded_image)
        
        # Translate the prompt to English
        translated_prompt = await translate_prompt(prompt)
        
        # Process the transformation (используем generate_content маршрут внутри)
        return await process_image_transform(
            source_image,
            translated_prompt,
            prompt,
            output_path=output_image_path
        )
        
    except Exception as e:
        error_msg = f"Error transforming image: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def transform_image_from_file(
    image_file_path: str,
    prompt: str,
    output_image_path: Optional[str] = None
) -> Tuple[bytes, str]:
    """Transform an existing image file based on the given text prompt using Google's Gemini model.

    Args:
        image_file_path: Path to the image file to be transformed
        prompt: Text prompt describing the desired transformation or modifications
        output_image_path: Optional path to save the transformed image. If not provided, uses default path.
        
    Returns:
        Path to the transformed image file saved on the server
    """
    try:
        logger.info(f"Processing transform_image_from_file request with prompt: {prompt}")
        logger.info(f"Image file path: {image_file_path}")

        # Validate file path
        if not os.path.exists(image_file_path):
            raise ValueError(f"Image file not found: {image_file_path}")

        # Translate the prompt to English
        translated_prompt = await translate_prompt(prompt)
            
        # Load the source image directly using PIL
        try:
            source_image = PIL.Image.open(image_file_path)
            logger.info(f"Successfully loaded image from file: {image_file_path}")
        except PIL.UnidentifiedImageError:
            logger.error("Error: Could not identify image format")
            raise ValueError("Could not identify image format. Supported formats include PNG, JPEG, GIF, WebP.")
        except Exception as e:
            logger.error(f"Error: Could not load image: {str(e)}")
            raise 
        
        # Process the transformation (используем generate_content маршрут внутри)
        return await process_image_transform(
            source_image,
            translated_prompt,
            prompt,
            output_path=output_image_path
        )
        
    except Exception as e:
        error_msg = f"Error transforming image: {str(e)}"
        logger.error(error_msg)
        return error_msg


def main():
    logger.info("Starting Gemini Image Generator MCP server...")
    mcp.run(transport="stdio")
    logger.info("Server stopped")

if __name__ == "__main__":
    main()