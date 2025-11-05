import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type Theme = 'light' | 'dark' | 'auto';

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  glassIntensity: number;
  setGlassIntensity: (intensity: number) => void;
  increaseContrast: boolean;
  setIncreaseContrast: (value: boolean) => void;
  reduceTransparency: boolean;
  setReduceTransparency: (value: boolean) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>('auto');
  const [glassIntensity, setGlassIntensity] = useState(0.65);
  const [increaseContrast, setIncreaseContrast] = useState(false);
  const [reduceTransparency, setReduceTransparency] = useState(false);

  useEffect(() => {
    const root = document.documentElement;
    
    if (theme === 'auto') {
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.classList.toggle('dark', isDark);
      
      // Listen for system theme changes
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const handler = (e: MediaQueryListEvent) => {
        root.classList.toggle('dark', e.matches);
      };
      mediaQuery.addEventListener('change', handler);
      return () => mediaQuery.removeEventListener('change', handler);
    } else {
      root.classList.toggle('dark', theme === 'dark');
    }
  }, [theme]);

  useEffect(() => {
    const root = document.documentElement;
    // Set CSS variables for glass intensity
    root.style.setProperty('--glass-intensity', glassIntensity.toString());
    root.style.setProperty('--contrast-mode', increaseContrast ? '1' : '0');
    root.style.setProperty('--reduce-transparency', reduceTransparency ? '1' : '0');
  }, [glassIntensity, increaseContrast, reduceTransparency]);

  return (
    <ThemeContext.Provider
      value={{
        theme,
        setTheme,
        glassIntensity,
        setGlassIntensity,
        increaseContrast,
        setIncreaseContrast,
        reduceTransparency,
        setReduceTransparency,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within ThemeProvider');
  return context;
}
