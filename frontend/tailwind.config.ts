import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        void: "#050816",
        panel: "rgba(10, 18, 38, 0.72)",
        cyanCore: "#27e8ff",
        dangerCore: "#ff3158",
        signalBlue: "#4f8cff"
      },
      boxShadow: {
        cyan: "0 0 38px rgba(39, 232, 255, 0.28)",
        danger: "0 0 42px rgba(255, 49, 88, 0.36)"
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "ui-monospace", "monospace"]
      }
    }
  },
  plugins: []
};

export default config;
