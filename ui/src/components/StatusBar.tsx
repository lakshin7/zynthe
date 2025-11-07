import { useTheme } from './ThemeProvider';

export function StatusBar() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="absolute bottom-0 left-0 right-0 h-10 px-6 flex items-center justify-between glass-pastel border-t-2 border-gradient-pastel shadow-pastel-sm">
      {/* Left: Status */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 border border-green-200 dark:border-green-700/50">
          <div className="w-2 h-2 rounded-full bg-green-500 dark:bg-green-400 pulse-pastel shadow-pastel-sm" />
          <span className="text-green-700 dark:text-green-300 text-xs font-semibold">System Ready</span>
        </div>
      </div>

      {/* Right: Theme toggle */}
      <div className="flex items-center gap-3">
        <span className="text-slate-600 dark:text-slate-300 text-xs font-semibold">Theme:</span>
        <div className="flex gap-1.5 glass-pastel p-1.5 rounded-xl border border-purple-200 dark:border-purple-700/50 shadow-pastel-sm">
          {['light', 'auto', 'dark'].map((t) => (
            <button
              key={t}
              onClick={() => setTheme(t as any)}
              className={`px-3 py-1.5 rounded-lg text-xs capitalize transition-smooth font-semibold ${
                theme === t
                  ? 'bg-gradient-to-r from-purple-300 to-blue-300 text-white shadow-pastel-md'
                  : 'text-slate-600 dark:text-slate-300 hover:bg-gradient-to-r hover:from-purple-50 hover:to-blue-50 dark:hover:from-purple-900/30 dark:hover:to-blue-900/30'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
