import React, { useRef } from "react";

interface SpotlightProps {
  className?: string;
  children: React.ReactNode;
}

const Spotlight: React.FC<SpotlightProps> = ({ className = "", children }) => {
  const ref = useRef<HTMLDivElement>(null);

  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    el.style.setProperty("--spot-x", `${x}px`);
    el.style.setProperty("--spot-y", `${y}px`);
  };

  return (
    <div ref={ref} onMouseMove={onMove} className={`ambient-spotlight ${className}`}>
      {children}
    </div>
  );
};

export default Spotlight;
