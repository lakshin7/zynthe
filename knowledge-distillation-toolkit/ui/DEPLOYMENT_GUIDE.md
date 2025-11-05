# Zynthe Desktop App - Build & Deployment Guide

## 🚀 Quick Start

### Development Mode
```bash
cd ui
npm start
```
This will:
1. Start Vite dev server on http://localhost:5173
2. Launch Electron window automatically
3. Enable hot-reload for instant updates
4. Open DevTools for debugging

### Backend Server
In a separate terminal:
```bash
cd ui/backend
python api.py
```
Backend will run on http://localhost:8765

---

## 📦 Building for Distribution

### Prerequisites
- Node.js 18+ installed
- Python 3.10+ installed
- Xcode Command Line Tools (macOS builds)
- Windows SDK (Windows builds)

### Build Commands

#### Build for macOS (.dmg)
```bash
cd ui
npm run electron:build:mac
```
Output: `ui/release/Zynthe-1.0.0-mac-arm64.dmg` (Apple Silicon)
Output: `ui/release/Zynthe-1.0.0-mac-x64.dmg` (Intel)

#### Build for Windows (.exe)
```bash
cd ui
npm run electron:build:win
```
Output: `ui/release/Zynthe-1.0.0-win-x64.exe`

#### Build for Linux
```bash
cd ui
npm run electron:build:linux
```
Output: `ui/release/Zynthe-1.0.0-linux-x64.AppImage`
Output: `ui/release/Zynthe-1.0.0-linux-x64.deb`

#### Build for All Platforms
```bash
cd ui
npm run electron:build
```

---

## 📂 Project Structure

```
ui/
├── electron/
│   ├── main.js          # Electron main process
│   └── preload.js       # Secure IPC bridge
├── src/
│   ├── pages/           # React pages
│   ├── components/      # UI components
│   ├── index.css        # Tailwind styles
│   └── main.tsx         # React entry point
├── assets/
│   ├── icon.icns        # macOS icon (512x512+)
│   ├── icon.ico         # Windows icon
│   └── icon.png         # Linux icon
├── dist/                # Built React app
└── release/             # Final distributables
```

---

## 🎨 Creating Application Icons

### Quick Method
1. Create a 1024x1024 PNG with your logo
2. Use online converter: https://cloudconvert.com/ or https://icon.kitchen/
3. Place files in `ui/assets/`:
   - `icon.icns` (macOS)
   - `icon.ico` (Windows)
   - `icon.png` (Linux)

### Manual Method (macOS)
```bash
cd ui/assets
# Create iconset from 1024x1024 PNG
mkdir icon.iconset
sips -z 16 16     source.png --out icon.iconset/icon_16x16.png
sips -z 32 32     source.png --out icon.iconset/icon_16x16@2x.png
sips -z 32 32     source.png --out icon.iconset/icon_32x32.png
sips -z 64 64     source.png --out icon.iconset/icon_32x32@2x.png
sips -z 128 128   source.png --out icon.iconset/icon_128x128.png
sips -z 256 256   source.png --out icon.iconset/icon_128x128@2x.png
sips -z 256 256   source.png --out icon.iconset/icon_256x256.png
sips -z 512 512   source.png --out icon.iconset/icon_256x256@2x.png
sips -z 512 512   source.png --out icon.iconset/icon_512x512.png
sips -z 1024 1024 source.png --out icon.iconset/icon_512x512@2x.png
iconutil -c icns icon.iconset
rm -rf icon.iconset
```

---

## 🔧 Configuration

### package.json Build Settings
```json
{
  "build": {
    "appId": "com.zynthe.app",
    "productName": "Zynthe",
    "directories": {
      "output": "release"
    },
    "mac": {
      "category": "public.app-category.developer-tools",
      "target": ["dmg"],
      "icon": "assets/icon.icns"
    },
    "win": {
      "target": ["nsis"],
      "icon": "assets/icon.ico"
    },
    "linux": {
      "target": ["AppImage", "deb"],
      "icon": "assets/icon.png"
    }
  }
}
```

### Customization
Edit `ui/electron/main.js` to change:
- Window size: `width`, `height`
- Minimum size: `minWidth`, `minHeight`
- Title bar style: `titleBarStyle`
- Background color: `backgroundColor`

---

## 🚦 Testing Before Release

### 1. Test Development Build
```bash
npm start
```
✅ Check all pages load
✅ Verify API connection
✅ Test experiment creation
✅ Monitor live training updates

### 2. Test Production Build
```bash
npm run build
electron .
```
✅ Ensure no dev server dependency
✅ Check bundled assets load
✅ Verify routing works

### 3. Test Packaged App
```bash
npm run electron:build:mac  # or :win, :linux
```
✅ Install/open the .dmg/.exe
✅ Test all features end-to-end
✅ Check app launches without terminal

---

## 📤 Distribution

### macOS
**File**: `Zynthe-1.0.0-mac-arm64.dmg`
- Upload to website/GitHub Releases
- Users drag to Applications folder
- **Code Signing**: For Gatekeeper, run:
  ```bash
  codesign --deep --force --verify --verbose --sign "Developer ID" Zynthe.app
  ```

### Windows
**File**: `Zynthe-1.0.0-win-x64.exe`
- NSIS installer with wizard
- Creates Start Menu shortcuts
- Uninstaller included
- **Code Signing**: Use `signtool` for SmartScreen trust

### Linux
**Files**: 
- `Zynthe-1.0.0-linux-x64.AppImage` (portable)
- `Zynthe-1.0.0-linux-x64.deb` (Ubuntu/Debian)

---

## 🐛 Troubleshooting

### Build Fails
```bash
# Clear cache and rebuild
rm -rf node_modules dist release
npm install
npm run build
npm run electron:build:mac
```

### Icon Not Showing
- Verify icons exist in `ui/assets/`
- Check file formats (.icns, .ico, .png)
- Clear Electron cache: `rm -rf ~/Library/Application\ Support/zynthe-desktop/`

### App Won't Launch
- Check `electron/main.js` syntax
- Verify `dist/` folder exists after build
- Look at Console.app (macOS) or Event Viewer (Windows) for errors

### API Connection Failed
- Ensure backend is bundled or runs separately
- Update API URLs in production builds
- Check CORS settings in `backend/api.py`

---

## 📈 Next Steps

### Phase 2 Features (Future)
- [ ] Auto-update system (electron-updater)
- [ ] Crash reporting (Sentry)
- [ ] Analytics integration
- [ ] Multi-language support
- [ ] Dark mode toggle
- [ ] Native notifications
- [ ] System tray icon

### Bundling Python Backend
To create a truly standalone app with Python backend:
```bash
# Use PyInstaller to bundle backend
pip install pyinstaller
pyinstaller --onefile backend/api.py
```
Then modify `electron/main.js` to spawn the bundled Python executable.

---

## 📝 Release Checklist

- [ ] Update version in `package.json`
- [ ] Add custom application icons
- [ ] Test all features in development mode
- [ ] Build production version
- [ ] Test packaged app on target OS
- [ ] Code sign (macOS/Windows)
- [ ] Create release notes
- [ ] Upload to distribution platform
- [ ] Update documentation

---

## 🆘 Support

For issues, check:
- Electron Builder Docs: https://www.electron.build/
- Vite Docs: https://vitejs.dev/
- Project GitHub Issues: [your-repo-url]

---

**Built with ❤️ using Electron + React + Vite + TailwindCSS**
