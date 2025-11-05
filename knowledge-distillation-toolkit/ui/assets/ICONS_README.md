# Application Icons

This directory should contain the application icons for different platforms:

## Required Icons:

### macOS (icon.icns)
- **Format**: .icns (Icon Composer format)
- **Sizes**: 16x16, 32x32, 128x128, 256x256, 512x512, 1024x1024
- **Tool**: Use `png2icns` or online converters like CloudConvert

### Windows (icon.ico)
- **Format**: .ico
- **Sizes**: 16x16, 32x32, 48x48, 64x64, 128x128, 256x256
- **Tool**: Use GIMP, ImageMagick, or online converters

### Linux (icon.png)
- **Format**: .png
- **Size**: 512x512 (recommended) or 1024x1024
- **Transparency**: Supported

## Creating Icons:

### Quick Method (Online):
1. Create a 1024x1024 PNG with your logo/design
2. Use https://cloudconvert.com/ or https://icon.kitchen/
3. Convert to .icns, .ico, and keep the .png

### Manual Method:

**For macOS (.icns):**
```bash
# Install iconutil (comes with Xcode)
mkdir icon.iconset
sips -z 16 16     icon.png --out icon.iconset/icon_16x16.png
sips -z 32 32     icon.png --out icon.iconset/icon_16x16@2x.png
sips -z 32 32     icon.png --out icon.iconset/icon_32x32.png
sips -z 64 64     icon.png --out icon.iconset/icon_32x32@2x.png
sips -z 128 128   icon.png --out icon.iconset/icon_128x128.png
sips -z 256 256   icon.png --out icon.iconset/icon_128x128@2x.png
sips -z 256 256   icon.png --out icon.iconset/icon_256x256.png
sips -z 512 512   icon.png --out icon.iconset/icon_256x256@2x.png
sips -z 512 512   icon.png --out icon.iconset/icon_512x512.png
sips -z 1024 1024 icon.png --out icon.iconset/icon_512x512@2x.png
iconutil -c icns icon.iconset
```

**For Windows (.ico):**
```bash
# Using ImageMagick
convert icon.png -define icon:auto-resize=256,128,64,48,32,16 icon.ico
```

## Temporary Placeholder:
Until you create custom icons, electron-builder will use default Electron icons.
The app will still build and run, but won't have a branded icon.

## Design Recommendations:
- **Simple & Clear**: Icon should be recognizable at small sizes (16x16)
- **High Contrast**: Ensure visibility on light and dark backgrounds
- **No Text**: Avoid text in icons as it becomes unreadable at small sizes
- **Square Format**: Design should work well in a square
- **Brand Colors**: Use your app's color scheme (pastels from your UI)

## Suggested Design for Zynthe:
- Neural network nodes/connections (representing knowledge distillation)
- Book/knowledge transfer symbol with arrows
- Gradient sphere (soft blue to green from your pastel palette)
- Abstract "Z" lettermark with distillation symbol
