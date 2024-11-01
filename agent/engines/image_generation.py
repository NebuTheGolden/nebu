import time
from typing import List, Dict
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline

def generate_image(
    short_term_context: str,
    image_references: List[Dict],
    style_preferences: Dict,
    output_path: Path,
    model_path: str = "runwayml/stable-diffusion-v1-5"
) -> Path:
    """
    Generate an image based on context, reference images, and style preferences using local resources.
    
    Args:
        short_term_context (str): Immediate context for image generation
        image_references (List[Dict]): List of reference images with metadata
        style_preferences (Dict): Dictionary containing style parameters
        output_path (Path): Where to save the generated image
        model_path (str): Path to the model or model identifier
    
    Returns:
        Path: Path to the generated image
    """
    
    # Initialize text-to-prompt pipeline for enhancing the image prompt
    prompt_generator = pipeline(
        "text-generation",
        model="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create base prompt from context and references
    base_prompt = f"""
    Context: {short_term_context}
    Style: {style_preferences.get('style', 'realistic')}
    Mood: {style_preferences.get('mood', 'neutral')}
    """
    
    # Generate enhanced prompt
    tries = 0
    max_tries = 3
    enhanced_prompt = ""
    
    while tries < max_tries:
        try:
            # Generate more detailed prompt
            response = prompt_generator(
                base_prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7
            )[0]['generated_text']
            
            if response and response.strip():
                enhanced_prompt = response
                print(f"Enhanced prompt generated: {enhanced_prompt}")
                break
                
        except Exception as e:
            print(f"Error on prompt generation attempt {tries + 1}: {str(e)}")
            tries += 1
            time.sleep(1)
    
    # Initialize Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # Apply style preferences
    generator_params = {
        "num_inference_steps": style_preferences.get("steps", 50),
        "guidance_scale": style_preferences.get("guidance_scale", 7.5),
        "negative_prompt": style_preferences.get("negative_prompt", "blurry, low quality, distorted"),
    }
    
    # Generate image
    tries = 0
    while tries < max_tries:
        try:
            # Generate the image
            image = pipe(
                enhanced_prompt,
                **generator_params
            ).images[0]
            
            # Save the image
            output_file = output_path / f"generated_{int(time.time())}.png"
            image.save(output_file)
            
            print(f"Image generated successfully at: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error on image generation attempt {tries + 1}: {str(e)}")
            tries += 1
            time.sleep(1)
    
    raise Exception("Failed to generate image after maximum attempts")

# Example usage
if __name__ == "__main__":
    # Example inputs
    context = "A serene landscape at sunset"
    references = [
        {"path": "ref1.jpg", "weight": 0.7},
        {"path": "ref2.jpg", "weight": 0.3}
    ]
    style_prefs = {
        "style": "realistic",
        "mood": "peaceful",
        "steps": 50,
        "guidance_scale": 7.5,
        "negative_prompt": "blurry, low quality, distorted"
    }
    
    output_dir = Path("./generated_images")
    output_dir.mkdir(exist_ok=True)
    
    try:
        generated_image_path = generate_image(
            context,
            references,
            style_prefs,
            output_dir
        )
        print(f"Image generated at: {generated_image_path}")
    except Exception as e:
        print(f"Failed to generate image: {str(e)}")
