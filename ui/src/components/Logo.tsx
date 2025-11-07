interface LogoProps {
  variant?: 'full' | 'icon';
  size?: number;
  className?: string;
}

export function Logo({ variant = 'full', size = 32, className = '' }: LogoProps) {
  if (variant === 'icon') {
    return (
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        fill="none"
        className={className}
      >
        {/* Simplified prism shape */}
        <path
          d="M50 15 L80 75 L20 75 Z"
          fill="currentColor"
          fillOpacity="0.15"
          stroke="currentColor"
          strokeWidth="1.5"
          className="text-amber-500 dark:text-cyan-400"
        />
        
        {/* Light beam entering */}
        <line
          x1="10"
          y1="45"
          x2="35"
          y2="45"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          opacity="0.6"
          className="text-amber-500"
        />
        
        {/* Refracted spectrum rays */}
        <line
          x1="65"
          y1="45"
          x2="90"
          y2="35"
          stroke="#FCD34D"
          strokeWidth="2"
          strokeLinecap="round"
          opacity="0.7"
        />
        <line
          x1="65"
          y1="45"
          x2="90"
          y2="45"
          stroke="#FB923C"
          strokeWidth="2"
          strokeLinecap="round"
          opacity="0.7"
        />
        <line
          x1="65"
          y1="45"
          x2="90"
          y2="55"
          stroke="#6EE7B7"
          strokeWidth="2"
          strokeLinecap="round"
          opacity="0.7"
        />
        
        {/* Z core - simplified */}
        <path
          d="M45 35 L55 35 L45 55 L55 55"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
          opacity="0.4"
          className="text-rose-500 dark:text-fuchsia-400"
        />
      </svg>
    );
  }

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <Logo variant="icon" size={size} />
      <span className="tracking-tight text-slate-800 dark:text-white">
        Zynthe
      </span>
    </div>
  );
}
