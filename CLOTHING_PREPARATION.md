# Clothing Image Preparation Guide

This guide explains how to prepare clothing images for the Virtual Try-On application.

## Requirements

- **Format**: PNG with transparent background
- **Resolution**: Minimum 500x500 pixels (higher is better)
- **Orientation**: Front-facing view
- **Background**: Must be transparent (alpha channel)

## Step-by-Step Guide

### 1. Using Photoshop

1. Open your clothing image in Photoshop
2. Use the Quick Selection Tool or Magic Wand to select the background
3. Delete the background (Delete key)
4. Save as PNG with transparency (File > Export > Export As > PNG)

### 2. Using GIMP (Free Alternative)

1. Open your image in GIMP
2. Right-click the layer > Add Alpha Channel
3. Use Fuzzy Select Tool to select background
4. Press Delete to remove background
5. Export as PNG (File > Export As > filename.png)

### 3. Using Online Tools

Several free online tools can remove backgrounds:
- [remove.bg](https://www.remove.bg/)
- [Canva Background Remover](https://www.canva.com/features/background-remover/)
- [Adobe Express](https://www.adobe.com/express/feature/image/remove-background)

### 4. Using Python (Automated)

```python
from rembg import remove
from PIL import Image

input_path = 'input.jpg'
output_path = 'output.png'

input = Image.open(input_path)
output = remove(input)
output.save(output_path)
```

## Naming Convention

Use descriptive names for your clothing items:
- `vest1.png`, `vest2.png` for vests
- `shirt1.png`, `shirt2.png` for shirts
- `jacket1.png`, `jacket2.png` for jackets
- `bra1.png`, `bra2.png` for bras

## Tips for Best Results

1. **Lighting**: Ensure even lighting in the original photo
2. **Contrast**: High contrast between clothing and background makes removal easier
3. **Details**: Preserve fine details like lace, buttons, or patterns
4. **Shadows**: Remove any shadows from the clothing
5. **Edges**: Clean up edges to avoid white halos

## Testing Your Images

Before using in the application:
1. Open the PNG in an image viewer
2. Check that the background shows as checkered pattern (transparency)
3. Verify edges are clean and smooth
4. Ensure the clothing item is centered

## Default Images Required

The application expects these files in the `assets/` folder:
- `vest1.png`
- `vest2.png`
- `vest3.png`
- `vest4.png`
- `bra1.png`
- `bra2.png`

You can replace these with your own clothing items following the same naming pattern.