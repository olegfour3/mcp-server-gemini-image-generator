![fluffy ginger cat](examples/product_mockup.png)


# Gemini Image Generator MCP Server

Generate high-quality images from text prompts using Google's Gemini model through the MCP protocol.

## Overview

This MCP server allows any AI assistant to generate images using Google's Gemini AI model. The server handles prompt engineering, text-to-image conversion, filename generation, and local image storage, making it easy to create and manage AI-generated images through any MCP client.

## Features

- Text-to-image generation using Gemini 2.5 Flash
- Image-to-image transformation based on text prompts
- Support for both file-based and base64-encoded images
- Automatic intelligent filename generation based on prompts
- Automatic translation of non-English prompts
- Local image storage with configurable output path
- Strict text exclusion from generated images
- High-resolution image output
- Direct access to both image data and file path

## Available MCP Tools

The server provides the following MCP tools for AI assistants (uses only Gemini `generate_content` with `gemini-2.5-flash-image`):

### 1. `generate_image_from_text`

Creates a new image from a text prompt description.

```
generate_image_from_text(
  prompt: str,
  output_image_path: Optional[str] = None
) -> Tuple[bytes, str]
```

**Parameters:**
- `prompt`: Text description of the image you want to generate
- `output_image_path` (optional): Absolute directory path to save the image. If omitted, uses `OUTPUT_IMAGE_PATH` or `~/gen_image`.

**Returns:**
- A tuple containing:
  - Raw image data (bytes)
  - Path to the saved image file (str)

This dual return format allows AI assistants to either work with the image data directly or reference the saved file path.

**Examples:**
- "Generate an image of a sunset over mountains"
- "Create a photorealistic picture of a fluffy ginger cat"

Pass a custom output directory (if your MCP client allows specifying arguments):

```
generate_image_from_text(
  prompt="futuristic city at dawn",
  output_image_path="/Users/username/Pictures/ai"
)
```

#### Example Output

This image was generated using the prompt:

```
"A photorealistic picture of a fluffy ginger cat sitting on a wooden floor, looking directly at the camera. Soft, natural light from a window."
```

![fluffy ginger cat](examples/cat.png)

*A photorealistic picture of a fluffy ginger cat sitting on a wooden floor, looking directly at the camera. Soft, natural light from a window.*

### Known Issues

When using this MCP server with Claude Desktop Host:

1. **Performance Issues**: Using `transform_image_from_encoded` may take significantly longer to process compared to other methods. This is due to the overhead of transferring large base64-encoded image data through the MCP protocol.

2. **Path Resolution Problems**: There may be issues with correctly resolving image paths when using Claude Desktop Host. The host application might not properly interpret the returned file paths, making it difficult to access the generated images.

For the best experience, consider using alternative MCP clients or the `transform_image_from_file` method when possible. 

### 2. `transform_image_from_encoded`

Transforms an existing image based on a text prompt using base64-encoded image data.

```
transform_image_from_encoded(
  encoded_image: str,
  prompt: str,
  output_image_path: Optional[str] = None
) -> Tuple[bytes, str]
```

**Parameters:**
- `encoded_image`: Base64 encoded image data with format header (must be in format: "data:image/[format];base64,[data]")
- `prompt`: Text description of how you want to transform the image
- `output_image_path` (optional): Absolute directory path to save the transformed image. If omitted, uses `OUTPUT_IMAGE_PATH` or `~/gen_image`.

**Returns:**
- A tuple containing:
  - Raw transformed image data (bytes)
  - Path to the saved transformed image file (str)

**Example:**
- "Add snow to this landscape"
- "Change the background to a beach"

With custom output directory:

```
transform_image_from_encoded(
  encoded_image="data:image/png;base64,iVBORw0K...",
  prompt="add cinematic teal-orange grading",
  output_image_path="/tmp/ai_out"
)
```

### 3. `transform_image_from_file`

Transforms an existing image file based on a text prompt.

```
transform_image_from_file(
  image_file_path: str,
  prompt: str,
  output_image_path: Optional[str] = None
) -> Tuple[bytes, str]
```

**Parameters:**
- `image_file_path`: Path to the image file to be transformed
- `prompt`: Text description of how you want to transform the image
- `output_image_path` (optional): Absolute directory path to save the transformed image. If omitted, uses `OUTPUT_IMAGE_PATH` or `~/gen_image`.

**Returns:**
- A tuple containing:
  - Raw transformed image data (bytes)
  - Path to the saved transformed image file (str)

**Examples:**
- "Add a llama next to the person in this image"
- "Make this daytime scene look like night time"

With custom output directory:

```
transform_image_from_file(
  image_file_path="/Users/username/Pictures/input.png",
  prompt="convert to watercolor style",
  output_image_path="/Users/username/Pictures/ai_out"
)
```

#### Example Transformation

Using the ginger cat image created above, we applied a transformation with the following prompt:

```
"Using the provided image of my cat, please add a small, knitted wizard hat on its head. Make it look like it's sitting comfortably and not falling off."
```

**Before:**
![fluffy ginger cat](examples/cat.png)

**After:**
![fluffy ginger cat with hat](examples/cat_with_hat.png)

*The original ginger cat with a hat*

## Setup

### Prerequisites

- Python 3.11+
- Google AI API key (Gemini)
- MCP host application (Claude Desktop App, Cursor, or other MCP-compatible clients)

### Getting a Gemini API Key

1. Visit [Google AI Studio API Keys page](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your new API key for use in the configuration
5. Note: The API key provides a certain quota of free usage per month. You can check your usage in the Google AI Studio

### Installation

### Manual Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/mcp-server-gemini-image-generator.git
cd mcp-server-gemini-image-generator
```

2. Create a virtual environment and install dependencies:
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or using regular venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

3. Set up environment variables (choose one method):

**Method A: Using .env file (optional)**
```bash
# Create .env file in the project root
cat > .env << 'EOF'
GEMINI_API_KEY=your-gemini-api-key-here
OUTPUT_IMAGE_PATH=/path/to/save/images
EOF
```

**Method B: Set directly in Claude Desktop config (recommended)**
- Set environment variables directly in the `claude_desktop_config.json` (shown in configuration section below)

### Configure Claude Desktop

Add the following to your `claude_desktop_config.json`:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
    "mcpServers": {
        "gemini-image-generator": {
            "command": "uv",
            "args": [
                "--directory",
                "/absolute/path/to/mcp-server-gemini-image-generator",
                "run",
                "mcp-server-gemini-image-generator"
            ],
            "env": {
                "GEMINI_API_KEY": "your-actual-gemini-api-key-here",
                "OUTPUT_IMAGE_PATH": "/absolute/path/to/your/images/directory"
            }
        }
    }
}
```

**Important Configuration Notes:**

1. **Replace paths with your actual paths:**
   - Change `/absolute/path/to/mcp-server-gemini-image-generator` to the actual location where you cloned this repository
   - Change `/absolute/path/to/your/images/directory` to where you want generated images to be saved

2. **Environment Variables:**
   - Replace `your-actual-gemini-api-key-here` with your real Gemini API key from Google AI Studio
   - Use absolute paths for `OUTPUT_IMAGE_PATH` to ensure images are saved correctly

3. **Example with real paths:**
```json
{
    "mcpServers": {
        "gemini-image-generator": {
            "command": "uv",
            "args": [
                "--directory",
                "/Users/username/Projects/mcp-server-gemini-image-generator",
                "run",
                "mcp-server-gemini-image-generator"
            ],
            "env": {
                "GEMINI_API_KEY": "GEMINI_API_KEY",
                "OUTPUT_IMAGE_PATH": "OUTPUT_IMAGE_PATH"
            }
        }
    }
}
```

## Usage

Once installed and configured, you can ask Claude to generate or transform images using prompts like:

### Generating New Images
- "Generate an image of a sunset over mountains"
- "Create an illustration of a futuristic cityscape"
- "Make a picture of a cat wearing sunglasses"

### Transforming Existing Images
- "Transform this image by adding snow to the scene"
- "Edit this photo to make it look like it was taken at night"
- "Add a dragon flying in the background of this picture"

The generated/transformed images will be saved to the provided `output_image_path` when specified, otherwise to your configured default path (`OUTPUT_IMAGE_PATH` or `~/gen_image`). With the updated return types, AI assistants can also work directly with the image data without needing to access the saved files.

## Testing

You can test the application by running the FastMCP development server:

```
fastmcp dev server.py
```

This command starts a local development server and makes the MCP Inspector available at http://localhost:5173/. 
The MCP Inspector provides a convenient web interface where you can directly test the image generation tool without needing to use Claude or another MCP client. 
You can enter text prompts, execute the tool, and see the results immediately, which is helpful for development and debugging.

## License

MIT License