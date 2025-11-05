/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './index.html',
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx,js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Pastel Status Colors
        status: {
          running: '#7FD4A8',      // Soft Green
          queued: '#FFD97D',       // Soft Yellow
          completed: '#5B9BD5',    // Soft Blue
          failed: '#9CA3AF',       // Soft Gray
        },
        // Primary Colors
        primary: {
          DEFAULT: '#5B9BD5',      // Soft Blue
          light: '#8CB9E5',
          dark: '#3A7AB8',
        },
        secondary: {
          DEFAULT: '#A59ACA',      // Lavender
          light: '#C4BBE0',
          dark: '#7D6FA8',
        },
        accent: {
          DEFAULT: '#7FD4A8',      // Mint Green
          light: '#A3E3C4',
          dark: '#5AB88D',
        },
        warning: {
          DEFAULT: '#FFB88C',      // Peachy
          light: '#FFD4B5',
          dark: '#FF9A5C',
        },
        error: {
          DEFAULT: '#FF8A80',      // Coral
          light: '#FFB3AD',
          dark: '#FF5A4D',
        },
        // Background Colors
        bg: {
          primary: '#F5F7FA',      // Off-white
          secondary: '#FFFFFF',    // Pure white
          tertiary: '#E8EDF2',     // Light gray-blue
        },
        // Text Colors
        text: {
          primary: '#2C3E50',      // Charcoal
          secondary: '#546E7A',    // Gray-blue
          muted: '#90A4AE',        // Light gray
        },
        // Border Colors
        border: {
          light: '#E0E7EF',
          medium: '#CBD5E1',
          dark: '#94A3B8',
        },
      },
      boxShadow: {
        'pastel': '0 4px 20px rgba(91, 155, 213, 0.15)',
        'pastel-lg': '0 10px 40px rgba(91, 155, 213, 0.2)',
        'card': '0 2px 8px rgba(0, 0, 0, 0.08)',
        'card-hover': '0 4px 16px rgba(0, 0, 0, 0.12)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
