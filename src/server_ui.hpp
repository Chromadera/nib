#pragma once

namespace inkling {

inline constexpr const char * kIndexHtml = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>nib</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg" />
<style>
/* ---- Exact llama-swap / shadcn oklch Theme Tokens ---- */
:root {
  --radius: 0.5rem;
  --background: oklch(100% 0 0);
  --foreground: oklch(14.1% .005 285.823);
  --card: oklch(100% 0 0);
  --card-foreground: oklch(14.1% .005 285.823);
  --popover: oklch(100% 0 0);
  --popover-foreground: oklch(14.1% .005 285.823);
  --primary: #32b8c6;
  --primary-foreground: oklch(98.5% 0 0);
  --secondary: oklch(96.7% .001 286.375);
  --secondary-foreground: oklch(21% .006 285.885);
  --muted: oklch(96.7% .001 286.375);
  --muted-foreground: oklch(55.2% .016 285.938);
  --accent: oklch(96.7% .001 286.375);
  --accent-foreground: oklch(21% .006 285.885);
  --destructive: oklch(57.7% .245 27.325);
  --border: oklch(92% .004 286.32);
  --input: oklch(92% .004 286.32);
  --ring: #32b8c6;
  --chart-1: #32b8c6;
  --chart-2: oklch(60% .118 184.704);
  --chart-3: oklch(39.8% .07 227.392);
  --sidebar: oklch(98.5% 0 0);
  --sidebar-foreground: oklch(14.1% .005 285.823);
  --sidebar-primary: #32b8c6;
  --sidebar-primary-foreground: oklch(98.5% 0 0);
  --sidebar-accent: oklch(96.7% .001 286.375);
  --sidebar-accent-foreground: oklch(21% .006 285.885);
  --sidebar-border: oklch(92% .004 286.32);
  --success: #10b981;
  --warning: #f59e0b;
  --font-sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
.dark {
  --background: oklch(14.1% .005 285.823);
  --foreground: oklch(98.5% 0 0);
  --card: oklch(21% .006 285.885);
  --card-foreground: oklch(98.5% 0 0);
  --popover: oklch(21% .006 285.885);
  --popover-foreground: oklch(98.5% 0 0);
  --primary: #2da6b2;
  --primary-foreground: oklch(14.1% .005 285.823);
  --secondary: oklch(27.4% .006 286.033);
  --secondary-foreground: oklch(98.5% 0 0);
  --muted: oklch(27.4% .006 286.033);
  --muted-foreground: oklch(70.5% .015 286.067);
  --accent: oklch(27.4% .006 286.033);
  --accent-foreground: oklch(98.5% 0 0);
  --destructive: oklch(70.4% .191 22.216);
  --border: oklch(100% 0 0 / .1);
  --input: oklch(100% 0 0 / .15);
  --ring: #32b8c6;
  --chart-1: #32b8c6;
  --chart-2: oklch(69.6% .17 162.48);
  --chart-3: oklch(76.9% .188 70.08);
  --sidebar: oklch(21% .006 285.885);
  --sidebar-foreground: oklch(98.5% 0 0);
  --sidebar-primary: #32b8c6;
  --sidebar-primary-foreground: oklch(14.1% .005 285.823);
  --sidebar-accent: oklch(27.4% .006 286.033);
  --sidebar-accent-foreground: oklch(98.5% 0 0);
  --sidebar-border: oklch(100% 0 0 / .1);
}
[data-theme=ocean]{--background:oklch(99.1% .005 240);--foreground:oklch(21% .04 256);--card:oklch(100% 0 0);--card-foreground:oklch(21% .04 256);--popover:oklch(100% 0 0);--popover-foreground:oklch(21% .04 256);--primary:#3b82f6;--primary-foreground:oklch(98.5% 0 0);--secondary:oklch(96% .02 240);--secondary-foreground:oklch(28% .05 256);--muted:oklch(96% .02 240);--muted-foreground:oklch(52% .04 250);--accent:oklch(93% .04 240);--accent-foreground:oklch(28% .05 256);--border:oklch(90% .02 240);--input:oklch(90% .02 240);--ring:#3b82f6;--sidebar:oklch(97% .015 240);--sidebar-foreground:oklch(21% .04 256);--sidebar-accent:oklch(93% .04 240);--sidebar-accent-foreground:oklch(28% .05 256);--sidebar-border:oklch(90% .02 240);--chart-1:#3b82f6}
[data-theme=ocean].dark{--background:oklch(19% .03 256);--foreground:oklch(97% .01 240);--card:oklch(24% .035 256);--card-foreground:oklch(97% .01 240);--popover:oklch(24% .035 256);--popover-foreground:oklch(97% .01 240);--primary:#60a5fa;--primary-foreground:oklch(19% .03 256);--secondary:oklch(30% .04 256);--secondary-foreground:oklch(97% .01 240);--muted:oklch(30% .04 256);--muted-foreground:oklch(72% .03 250);--accent:oklch(34% .05 256);--accent-foreground:oklch(97% .01 240);--border:oklch(100% 0 0 / .1);--input:oklch(100% 0 0 / .15);--ring:#60a5fa;--sidebar:oklch(24% .035 256);--sidebar-foreground:oklch(97% .01 240);--sidebar-accent:oklch(34% .05 256);--sidebar-accent-foreground:oklch(97% .01 240);--sidebar-border:oklch(100% 0 0 / .1);--chart-1:#60a5fa}
[data-theme=violet]{--background:oklch(99.2% .005 300);--foreground:oklch(22% .04 300);--card:oklch(100% 0 0);--card-foreground:oklch(22% .04 300);--popover:oklch(100% 0 0);--popover-foreground:oklch(22% .04 300);--primary:#8b5cf6;--primary-foreground:oklch(98.5% 0 0);--secondary:oklch(96% .02 300);--secondary-foreground:oklch(30% .05 300);--muted:oklch(96% .02 300);--muted-foreground:oklch(53% .04 300);--accent:oklch(93% .04 300);--accent-foreground:oklch(30% .05 300);--border:oklch(90% .02 300);--input:oklch(90% .02 300);--ring:#8b5cf6;--sidebar:oklch(97% .015 300);--sidebar-foreground:oklch(22% .04 300);--sidebar-accent:oklch(93% .04 300);--sidebar-accent-foreground:oklch(30% .05 300);--sidebar-border:oklch(90% .02 300);--chart-1:#8b5cf6}
[data-theme=violet].dark{--background:oklch(20% .03 300);--foreground:oklch(97% .01 300);--card:oklch(25% .035 300);--card-foreground:oklch(97% .01 300);--popover:oklch(25% .035 300);--popover-foreground:oklch(97% .01 300);--primary:#a78bfa;--primary-foreground:oklch(20% .03 300);--secondary:oklch(31% .04 300);--secondary-foreground:oklch(97% .01 300);--muted:oklch(31% .04 300);--muted-foreground:oklch(73% .03 300);--accent:oklch(35% .05 300);--accent-foreground:oklch(97% .01 300);--border:oklch(100% 0 0 / .1);--input:oklch(100% 0 0 / .15);--ring:#a78bfa;--sidebar:oklch(25% .035 300);--sidebar-foreground:oklch(97% .01 300);--sidebar-accent:oklch(35% .05 300);--sidebar-accent-foreground:oklch(97% .01 300);--sidebar-border:oklch(100% 0 0 / .1);--chart-1:#a78bfa}
[data-theme=emerald]{--background:oklch(99.2% .006 165);--foreground:oklch(22% .04 165);--card:oklch(100% 0 0);--card-foreground:oklch(22% .04 165);--popover:oklch(100% 0 0);--popover-foreground:oklch(22% .04 165);--primary:#10b981;--primary-foreground:oklch(98.5% 0 0);--secondary:oklch(96% .02 165);--secondary-foreground:oklch(30% .05 165);--muted:oklch(96% .02 165);--muted-foreground:oklch(52% .04 165);--accent:oklch(93% .04 165);--accent-foreground:oklch(30% .05 165);--border:oklch(90% .02 165);--input:oklch(90% .02 165);--ring:#10b981;--sidebar:oklch(97% .015 165);--sidebar-foreground:oklch(22% .04 165);--sidebar-accent:oklch(93% .04 165);--sidebar-accent-foreground:oklch(30% .05 165);--sidebar-border:oklch(90% .02 165);--chart-1:#10b981}
[data-theme=emerald].dark{--background:oklch(19% .03 165);--foreground:oklch(97% .01 165);--card:oklch(24% .035 165);--card-foreground:oklch(97% .01 165);--popover:oklch(24% .035 165);--popover-foreground:oklch(97% .01 165);--primary:#34d399;--primary-foreground:oklch(19% .03 165);--secondary:oklch(30% .04 165);--secondary-foreground:oklch(97% .01 165);--muted:oklch(30% .04 165);--muted-foreground:oklch(72% .03 165);--accent:oklch(34% .05 165);--accent-foreground:oklch(97% .01 165);--border:oklch(100% 0 0 / .1);--input:oklch(100% 0 0 / .15);--ring:#34d399;--sidebar:oklch(24% .035 165);--sidebar-foreground:oklch(97% .01 165);--sidebar-accent:oklch(34% .05 165);--sidebar-accent-foreground:oklch(97% .01 165);--sidebar-border:oklch(100% 0 0 / .1);--chart-1:#34d399}
[data-theme=rose]{--background:oklch(99.2% .006 12);--foreground:oklch(22% .04 12);--card:oklch(100% 0 0);--card-foreground:oklch(22% .04 12);--popover:oklch(100% 0 0);--popover-foreground:oklch(22% .04 12);--primary:#f43f5e;--primary-foreground:oklch(98.5% 0 0);--secondary:oklch(96% .02 12);--secondary-foreground:oklch(30% .05 12);--muted:oklch(96% .02 12);--muted-foreground:oklch(52% .04 12);--accent:oklch(93% .04 12);--accent-foreground:oklch(30% .05 12);--border:oklch(90% .02 12);--input:oklch(90% .02 12);--ring:#f43f5e;--sidebar:oklch(97% .015 12);--sidebar-foreground:oklch(22% .04 12);--sidebar-accent:oklch(93% .04 12);--sidebar-accent-foreground:oklch(30% .05 12);--sidebar-border:oklch(90% .02 12);--chart-1:#f43f5e}
[data-theme=rose].dark{--background:oklch(19% .03 12);--foreground:oklch(97% .01 12);--card:oklch(24% .035 12);--card-foreground:oklch(97% .01 12);--popover:oklch(24% .035 12);--popover-foreground:oklch(97% .01 12);--primary:#fb7185;--primary-foreground:oklch(19% .03 12);--secondary:oklch(30% .04 12);--secondary-foreground:oklch(97% .01 12);--muted:oklch(30% .04 12);--muted-foreground:oklch(72% .03 12);--accent:oklch(34% .05 12);--accent-foreground:oklch(97% .01 12);--border:oklch(100% 0 0 / .1);--input:oklch(100% 0 0 / .15);--ring:#fb7185;--sidebar:oklch(24% .035 12);--sidebar-foreground:oklch(97% .01 12);--sidebar-accent:oklch(34% .05 12);--sidebar-accent-foreground:oklch(97% .01 12);--sidebar-border:oklch(100% 0 0 / .1);--chart-1:#fb7185}
[data-theme=amber]{--background:oklch(99.3% .008 80);--foreground:oklch(24% .04 70);--card:oklch(100% 0 0);--card-foreground:oklch(24% .04 70);--popover:oklch(100% 0 0);--popover-foreground:oklch(24% .04 70);--primary:#f59e0b;--primary-foreground:oklch(24% .04 70);--secondary:oklch(96% .03 80);--secondary-foreground:oklch(32% .05 70);--muted:oklch(96% .03 80);--muted-foreground:oklch(53% .05 70);--accent:oklch(92% .06 80);--accent-foreground:oklch(32% .05 70);--border:oklch(90% .03 80);--input:oklch(90% .03 80);--ring:#f59e0b;--sidebar:oklch(97% .025 80);--sidebar-foreground:oklch(24% .04 70);--sidebar-accent:oklch(92% .06 80);--sidebar-accent-foreground:oklch(32% .05 70);--sidebar-border:oklch(90% .03 80);--chart-1:#f59e0b}
[data-theme=amber].dark{--background:oklch(20% .025 60);--foreground:oklch(97% .02 80);--card:oklch(25% .03 60);--card-foreground:oklch(97% .02 80);--popover:oklch(25% .03 60);--popover-foreground:oklch(97% .02 80);--primary:#fbbf24;--primary-foreground:oklch(20% .025 60);--secondary:oklch(31% .035 60);--secondary-foreground:oklch(97% .02 80);--muted:oklch(31% .035 60);--muted-foreground:oklch(74% .04 75);--accent:oklch(35% .045 60);--accent-foreground:oklch(97% .02 80);--border:oklch(100% 0 0 / .1);--input:oklch(100% 0 0 / .15);--ring:#fbbf24;--sidebar:oklch(25% .03 60);--sidebar-foreground:oklch(97% .02 80);--sidebar-accent:oklch(35% .045 60);--sidebar-accent-foreground:oklch(97% .02 80);--sidebar-border:oklch(100% 0 0 / .1);--chart-1:#fbbf24}
[data-theme=slate]{--background:oklch(98.5% .003 256);--foreground:oklch(21% .02 256);--card:oklch(100% 0 0);--card-foreground:oklch(21% .02 256);--popover:oklch(100% 0 0);--popover-foreground:oklch(21% .02 256);--primary:#475569;--primary-foreground:oklch(98.5% 0 0);--secondary:oklch(95% .008 256);--secondary-foreground:oklch(28% .02 256);--muted:oklch(95% .008 256);--muted-foreground:oklch(52% .02 256);--accent:oklch(92% .012 256);--accent-foreground:oklch(28% .02 256);--border:oklch(89% .01 256);--input:oklch(89% .01 256);--ring:#64748b;--sidebar:oklch(96% .006 256);--sidebar-foreground:oklch(21% .02 256);--sidebar-accent:oklch(92% .012 256);--sidebar-accent-foreground:oklch(28% .02 256);--sidebar-border:oklch(89% .01 256);--chart-1:#475569}
[data-theme=slate].dark{--background:oklch(20% .015 256);--foreground:oklch(96% .005 256);--card:oklch(25% .018 256);--card-foreground:oklch(96% .005 256);--popover:oklch(25% .018 256);--popover-foreground:oklch(96% .005 256);--primary:#94a3b8;--primary-foreground:oklch(20% .015 256);--secondary:oklch(31% .02 256);--secondary-foreground:oklch(96% .005 256);--muted:oklch(31% .02 256);--muted-foreground:oklch(72% .015 256);--accent:oklch(35% .025 256);--accent-foreground:oklch(96% .005 256);--border:oklch(100% 0 0 / .1);--input:oklch(100% 0 0 / .15);--ring:#94a3b8;--sidebar:oklch(25% .018 256);--sidebar-foreground:oklch(96% .005 256);--sidebar-accent:oklch(35% .025 256);--sidebar-accent-foreground:oklch(96% .005 256);--sidebar-border:oklch(100% 0 0 / .1);--chart-1:#94a3b8}
[data-theme=sunset]{--background:oklch(99% .015 55);--foreground:oklch(26% .06 35);--card:oklch(100% 0 0);--card-foreground:oklch(26% .06 35);--popover:oklch(100% 0 0);--popover-foreground:oklch(26% .06 35);--primary:#f97316;--primary-foreground:oklch(98.5% 0 0);--secondary:oklch(95% .03 50);--secondary-foreground:oklch(34% .06 35);--muted:oklch(95% .03 50);--muted-foreground:oklch(54% .06 35);--accent:oklch(92% .05 30);--accent-foreground:oklch(34% .06 35);--border:oklch(90% .03 45);--input:oklch(90% .03 45);--ring:#f97316;--sidebar:oklch(97% .03 45);--sidebar-foreground:oklch(26% .06 35);--sidebar-accent:oklch(92% .05 30);--sidebar-accent-foreground:oklch(34% .06 35);--sidebar-border:oklch(90% .03 45);--chart-1:#f97316}
[data-theme=sunset].dark{--background:oklch(20% .045 320);--foreground:oklch(96% .02 50);--card:oklch(25% .05 320);--card-foreground:oklch(96% .02 50);--popover:oklch(25% .05 320);--popover-foreground:oklch(96% .02 50);--primary:#fb923c;--primary-foreground:oklch(20% .045 320);--secondary:oklch(31% .05 320);--secondary-foreground:oklch(96% .02 50);--muted:oklch(31% .05 320);--muted-foreground:oklch(74% .04 40);--accent:oklch(36% .08 350);--accent-foreground:oklch(96% .02 50);--border:oklch(100% 0 0 / .1);--input:oklch(100% 0 0 / .15);--ring:#fb923c;--sidebar:oklch(25% .05 320);--sidebar-foreground:oklch(96% .02 50);--sidebar-accent:oklch(36% .08 350);--sidebar-accent-foreground:oklch(96% .02 50);--sidebar-border:oklch(100% 0 0 / .1);--chart-1:#fb923c}
[data-theme=terminal]{--background:#e8f5e9;--foreground:#14401f;--card:#f1faf2;--card-foreground:#14401f;--popover:#f1faf2;--popover-foreground:#14401f;--primary:#168f3f;--primary-foreground:#f1faf2;--secondary:#d4ead6;--secondary-foreground:#1a4d24;--muted:#d4ead6;--muted-foreground:#4a7a52;--accent:#c8e6cb;--accent-foreground:#1a4d24;--destructive:#b3261e;--border:#b8d8bb;--input:#b8d8bb;--ring:#168f3f;--sidebar:#dcefdd;--sidebar-foreground:#14401f;--sidebar-accent:#c8e6cb;--sidebar-accent-foreground:#1a4d24;--sidebar-border:#b8d8bb;--chart-1:#168f3f}
[data-theme=terminal].dark{--background:#0a0f0a;--foreground:#3f6;--card:#0d160d;--card-foreground:#3f6;--popover:#0d160d;--popover-foreground:#3f6;--primary:#0f6;--primary-foreground:#001a0a;--secondary:#12241a;--secondary-foreground:#6f9;--muted:#12241a;--muted-foreground:#4d995f;--accent:#14301f;--accent-foreground:#6f9;--destructive:#f55;--border:#1f4d33;--input:#1f4d33;--ring:#0f6;--sidebar:#0a0f0a;--sidebar-foreground:#3f6;--sidebar-accent:#14301f;--sidebar-accent-foreground:#6f9;--sidebar-border:#1f4d33;--chart-1:#0f6}
[data-theme=mc]{--background:#d4d4d4;--foreground:#161616;--card:#e4e4e4;--card-foreground:#161616;--popover:#e4e4e4;--popover-foreground:#161616;--primary:#007a7a;--primary-foreground:#fff;--secondary:#c4c4cc;--secondary-foreground:#1a1a4a;--muted:#c4c4cc;--muted-foreground:#4a4a6a;--accent:#bcdcdc;--accent-foreground:#003a3a;--destructive:#a00;--border:#a8a8b4;--input:#a8a8b4;--ring:#007a7a;--sidebar:#c8c8d0;--sidebar-foreground:#161616;--sidebar-accent:#bcdcdc;--sidebar-accent-foreground:#003a3a;--sidebar-border:#a8a8b4;--chart-1:#007a7a}
[data-theme=mc].dark{--background:#00a;--foreground:#d4d4d4;--card:#1818b4;--card-foreground:#fff;--popover:#1818b4;--popover-foreground:#fff;--primary:#0aa;--primary-foreground:#000;--secondary:#1818b4;--secondary-foreground:#fff;--muted:#1818b4;--muted-foreground:#9a9ad6;--accent:#0aa;--accent-foreground:#000;--destructive:#d40000;--border:#0aa;--input:#1818b4;--ring:#ff5;--sidebar:#00008f;--sidebar-foreground:#d4d4d4;--sidebar-accent:#0aa;--sidebar-accent-foreground:#000;--sidebar-border:#0aa;--chart-1:#0aa}
[data-theme=solarized]{--background:#fdf6e3;--foreground:#657b83;--card:#fdf6e3;--card-foreground:#657b83;--popover:#fdf6e3;--popover-foreground:#657b83;--primary:#268bd2;--primary-foreground:#fdf6e3;--secondary:#eee8d5;--secondary-foreground:#586e75;--muted:#eee8d5;--muted-foreground:#93a1a1;--accent:#eee8d5;--accent-foreground:#586e75;--destructive:#dc322f;--border:#ddd6c1;--input:#ddd6c1;--ring:#268bd2;--sidebar:#eee8d5;--sidebar-foreground:#657b83;--sidebar-accent:#e3ddc8;--sidebar-accent-foreground:#586e75;--sidebar-border:#ddd6c1;--chart-1:#268bd2}
[data-theme=solarized].dark{--background:#002b36;--foreground:#839496;--card:#073642;--card-foreground:#839496;--popover:#073642;--popover-foreground:#839496;--primary:#268bd2;--primary-foreground:#fdf6e3;--secondary:#073642;--secondary-foreground:#93a1a1;--muted:#073642;--muted-foreground:#586e75;--accent:#0a4453;--accent-foreground:#93a1a1;--destructive:#dc322f;--border:#0f4a59;--input:#0f4a59;--ring:#268bd2;--sidebar:#073642;--sidebar-foreground:#839496;--sidebar-accent:#0a4453;--sidebar-accent-foreground:#93a1a1;--sidebar-border:#0f4a59;--chart-1:#268bd2}

/* ---- Reset & Base ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  height: 100%;
  background: var(--background);
  color: var(--foreground);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}
button, input, select, textarea { font: inherit; color: inherit; }
button { cursor: pointer; background: none; border: none; }
a { color: inherit; text-decoration: none; }
svg { display: block; flex-shrink: 0; }

/* Subtle scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted-foreground); }

/* ---- App Shell ---- */
.app-shell {
  display: flex;
  width: 100%;
  height: 100vh;
  overflow: hidden;
}

/* ---- Left Sidebar (exact llama-swap proportions) ---- */
.sidebar {
  width: 256px;
  min-width: 256px;
  background: var(--sidebar);
  color: var(--sidebar-foreground);
  border-right: 1px solid var(--sidebar-border);
  display: flex;
  flex-direction: column;
  transition: width 160ms ease, min-width 160ms ease;
  user-select: none;
  z-index: 20;
}
.sidebar.collapsed {
  width: 56px;
  min-width: 56px;
}
.sidebar-header {
  height: 56px;
  padding: 0 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  position: relative;
}
.sidebar.collapsed .sidebar-header {
  justify-content: center;
  padding: 0;
}
.status-dot {
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: #10b981;
  flex-shrink: 0;
}
.sidebar-header .status-dot {
  margin-left: auto;
}
.sidebar.collapsed .sidebar-header .status-dot {
  position: absolute;
  right: 11px;
  bottom: 11px;
  width: 8px;
  height: 8px;
  margin-left: 0;
  box-shadow: 0 0 0 2px var(--sidebar);
}
.status-dot.offline { background: #ef4444; }
.status-dot.busy {
  background: #f59e0b;
  animation: pulse 1.3s ease-in-out infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.brand-link {
  display: inline-flex;
  align-items: center;
  border-radius: 6px;
  padding: 2px 4px;
  margin-left: -4px;
  transition: opacity 120ms ease;
}
.brand-link:hover { opacity: 0.9; }
.brand-logo {
  height: 30px;
  width: auto;
  display: block;
  filter: drop-shadow(0 1px 1.5px rgba(15, 23, 42, 0.28));
}
.brand-mark {
  display: none;
  width: 26px;
  height: 26px;
  filter: drop-shadow(0 1px 1.5px rgba(15, 23, 42, 0.24));
}
.dark .brand-logo,
.dark .brand-mark {
  filter: none;
}
.sidebar.collapsed .brand-link { margin-left: 0; padding: 0; }
.sidebar.collapsed .brand-logo { display: none; }
.sidebar.collapsed .brand-mark { display: block; }

.sidebar-nav {
  flex: 1;
  padding: 8px 10px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.nav-item {
  width: 100%;
  height: 36px;
  padding: 0 10px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  font-weight: 450;
  color: var(--sidebar-foreground);
  transition: background 120ms ease;
  text-align: left;
}
.nav-item:hover { background: var(--sidebar-accent); }
.nav-item.active {
  background: var(--sidebar-accent);
  font-weight: 550;
}
.nav-item svg { width: 16px; height: 16px; opacity: 0.85; }
.nav-item .nav-label { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.nav-item .nav-chevron { width: 14px; height: 14px; opacity: 0.6; transition: transform 150ms ease; }
.nav-item.open .nav-chevron { transform: rotate(180deg); }
.sidebar.collapsed .nav-item { justify-content: center; padding: 0; }
.sidebar.collapsed .nav-label,
.sidebar.collapsed .nav-chevron,
.sidebar.collapsed .nav-sublist { display: none !important; }

.nav-sublist {
  margin: 2px 0 4px 18px;
  padding-left: 14px;
  border-left: 1px solid var(--sidebar-border);
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.nav-sublist.hidden { display: none; }
.nav-subitem {
  height: 32px;
  padding: 0 8px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--sidebar-foreground);
  opacity: 0.9;
  transition: background 120ms ease;
}
.nav-subitem:hover { background: var(--sidebar-accent); opacity: 1; }
.sub-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: #10b981;
  flex-shrink: 0;
}
.sub-dot.busy { background: #f59e0b; animation: pulse 1.3s ease-in-out infinite; }
.sub-dot.stopped { background: var(--muted-foreground); opacity: 0.5; }
.nav-subitem span:last-child {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.sidebar-footer {
  padding: 10px 10px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 4px;
}
.sidebar.collapsed .sidebar-footer {
  flex-direction: column;
  padding: 10px 8px;
}
.sidebar-footer .nav-item { width: auto; flex: 1; }
.icon-btn {
  width: 34px;
  height: 34px;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--foreground);
  transition: background 120ms ease, border-color 120ms ease;
}
.icon-btn:hover { background: var(--accent); }
.icon-btn.bordered {
  border: 1px solid var(--border);
  background: var(--background);
}
.icon-btn.bordered:hover { background: var(--accent); }
.icon-btn svg { width: 16px; height: 16px; }

/* ---- Main Column ---- */
.main-area {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  background: var(--background);
}

/* ---- Sticky Top Header (exact h-14 llama-swap style) ---- */
.topbar {
  height: 56px;
  min-height: 56px;
  padding: 0 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  background: var(--background);
}
.topbar-left {
  display: flex;
  align-items: center;
  gap: 10px;
}
.topbar-sep {
  width: 1px;
  height: 16px;
  background: var(--border);
}
.breadcrumb {
  font-size: 14px;
  font-weight: 600;
  color: var(--foreground);
  letter-spacing: -0.01em;
}
.topbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.live-pill {
  display: none;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  background: var(--secondary);
  color: var(--muted-foreground);
  border: 1px solid var(--border);
}
.live-pill.visible { display: inline-flex; }

/* ---- Page Container ---- */
.page-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 20px 24px 24px;
}
.page-body.full-height {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 16px 24px 20px;
}
.view { display: none; }
.view.active { display: block; }
.view.flex-view.active {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

/* ---- Shared Cards, Buttons, Inputs, Underline Tabs ---- */
.card {
  background: var(--card);
  color: var(--card-foreground);
  border: 1px solid var(--border);
  border-radius: 12px;
}
.card + .card { margin-top: 16px; }
.card-pad { padding: 20px; }

.btn {
  height: 36px;
  padding: 0 14px;
  border-radius: 8px;
  font-size: 13.5px;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  white-space: nowrap;
  transition: background 120ms ease, opacity 120ms ease, border-color 120ms ease;
}
.btn-outline {
  border: 1px solid var(--border);
  background: var(--background);
  color: var(--foreground);
}
.btn-outline:hover { background: var(--accent); }
.btn-primary {
  background: var(--primary);
  color: var(--primary-foreground);
  font-weight: 550;
}
.btn-primary:hover { filter: brightness(0.95); }
.btn-primary:disabled { opacity: 0.45; cursor: not-allowed; }
.btn-danger {
  background: var(--destructive);
  color: #fff;
}
.btn-danger:hover { filter: brightness(0.95); }
.btn-sm { height: 32px; padding: 0 12px; font-size: 13px; }
.btn svg { width: 15px; height: 15px; }

.select, .input {
  height: 36px;
  padding: 0 12px;
  border-radius: 8px;
  border: 1px solid var(--input);
  background: var(--background);
  color: var(--foreground);
  font-size: 13.5px;
  outline: none;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}
.select:focus, .input:focus {
  border-color: var(--ring);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--ring) 20%, transparent);
}
.select-sm { height: 30px; padding: 0 8px; font-size: 12.5px; border-radius: 6px; }

/* Underline Tabs (exact llama-swap Playground & Logs style) */
.underline-tabs {
  display: flex;
  align-items: center;
  gap: 4px;
  border-bottom: 1px solid var(--border);
  padding: 0 16px;
}
.underline-tab {
  position: relative;
  height: 44px;
  padding: 0 12px;
  font-size: 13.5px;
  font-weight: 500;
  color: var(--muted-foreground);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: color 120ms ease;
}
.underline-tab:hover { color: var(--foreground); }
.underline-tab.active {
  color: var(--foreground);
  font-weight: 600;
}
.underline-tab.active::after {
  content: "";
  position: absolute;
  left: 8px;
  right: 8px;
  bottom: -1px;
  height: 2px;
  background: var(--foreground);
  border-radius: 2px 2px 0 0;
}

/* ---- Activity Page (exact llama-swap 3-card layout) ---- */
.activity-charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  padding: 18px 20px 14px;
}
.chart-col-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--muted-foreground);
  margin-bottom: 10px;
}
.mini-chart-box {
  height: 86px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted-foreground);
  font-size: 13px;
  position: relative;
}
.mini-chart-box svg { width: 100%; height: 100%; }

.activity-kpis {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  padding: 12px 20px 18px;
  border-top: 1px solid color-mix(in srgb, var(--border) 60%, transparent);
}
.kpi-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted-foreground);
  margin-bottom: 2px;
}
.kpi-val {
  font-size: 14px;
  color: var(--foreground);
}
.kpi-val strong { font-weight: 650; }

.section-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
}
.section-kicker {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted-foreground);
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-kicker strong {
  color: var(--foreground);
  text-transform: none;
  letter-spacing: 0;
  font-size: 13px;
}

.table-wrap { width: 100%; overflow-x: auto; }
table.swap-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.swap-table th {
  text-align: left;
  font-weight: 500;
  color: var(--muted-foreground);
  padding: 10px 16px;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  font-size: 12.5px;
}
table.swap-table td {
  padding: 10px 16px;
  border-bottom: 1px solid color-mix(in srgb, var(--border) 60%, transparent);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
table.swap-table tbody tr:last-child td { border-bottom: none; }
table.swap-table tbody tr:hover td { background: color-mix(in srgb, var(--muted) 45%, transparent); }
.table-empty {
  text-align: center;
  padding: 30px 16px !important;
  color: var(--muted-foreground);
  font-size: 13.5px;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 7px;
  border-radius: 6px;
  font-size: 11.5px;
  font-weight: 550;
  background: var(--secondary);
  color: var(--secondary-foreground);
}
.badge-ok {
  background: color-mix(in srgb, #10b981 14%, transparent);
  color: #10b981;
}
.badge-primary {
  background: color-mix(in srgb, var(--primary) 15%, transparent);
  color: var(--primary);
}
.mono { font-family: var(--font-mono); }

/* ---- Playground (exact llama-swap framed card + popover settings) ---- */
.playground-card {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
}
.pg-pane {
  display: none;
  flex: 1;
  min-height: 0;
  flex-direction: column;
}
.pg-pane.active { display: flex; }

.pg-toolbar {
  padding: 14px 16px 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  position: relative;
}
.model-select-wrap {
  flex: 1;
  min-width: 0;
  position: relative;
}
.model-select-wrap select {
  width: 100%;
  height: 38px;
  padding: 0 32px 0 12px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--background);
  color: var(--foreground);
  font-size: 13.5px;
  appearance: none;
  cursor: pointer;
}
.model-select-wrap::after {
  content: "";
  position: absolute;
  right: 12px;
  top: 50%;
  width: 6px;
  height: 6px;
  border-right: 1.5px solid var(--muted-foreground);
  border-bottom: 1.5px solid var(--muted-foreground);
  transform: translateY(-65%) rotate(45deg);
  pointer-events: none;
}

/* Floating Parameters Popover (hidden by default so chat canvas breathes!) */
.params-popover {
  display: none;
  position: absolute;
  top: 58px;
  right: 16px;
  width: 340px;
  max-height: calc(100vh - 200px);
  overflow-y: auto;
  background: var(--popover);
  color: var(--popover-foreground);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.14);
  z-index: 40;
}
.params-popover.open { display: block; }
.popover-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.popover-title { font-size: 13.5px; font-weight: 600; }
.field { margin-bottom: 14px; }
.field:last-child { margin-bottom: 0; }
.field-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}
.field label {
  font-size: 12.5px;
  font-weight: 500;
  color: var(--foreground);
}
.field-val {
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--muted-foreground);
}
.field-hint {
  font-size: 11.5px;
  color: var(--muted-foreground);
  margin-top: 4px;
}
.field textarea {
  width: 100%;
  min-height: 70px;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid var(--input);
  background: var(--background);
  font-size: 12.5px;
  resize: vertical;
  outline: none;
}
.field textarea:focus { border-color: var(--ring); }
.field input[type="range"] {
  width: 100%;
  accent-color: var(--primary);
  cursor: pointer;
}
.switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 0;
  border-top: 1px solid color-mix(in srgb, var(--border) 60%, transparent);
}
.switch {
  position: relative;
  width: 36px;
  height: 20px;
  display: inline-block;
}
.switch input { opacity: 0; width: 0; height: 0; }
.switch-track {
  position: absolute;
  inset: 0;
  border-radius: 999px;
  background: var(--input);
  transition: background 140ms ease;
  cursor: pointer;
}
.switch-track::after {
  content: "";
  position: absolute;
  top: 2px;
  left: 2px;
  width: 16px;
  height: 16px;
  border-radius: 999px;
  background: var(--background);
  box-shadow: 0 1px 2px rgba(0,0,0,0.2);
  transition: transform 140ms ease;
}
.switch input:checked + .switch-track { background: var(--primary); }
.switch input:checked + .switch-track::after { transform: translateX(16px); }

/* Chat Transcript Area */
.chat-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 12px 24px 20px;
  display: flex;
  flex-direction: column;
}
.chat-empty {
  margin: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  text-align: center;
  color: var(--muted-foreground);
  font-size: 14px;
  user-select: none;
}
.chat-empty-logo {
  width: 144px;
  height: auto;
  opacity: 0.92;
  filter: drop-shadow(0 2px 4px rgba(15, 23, 42, 0.24));
}
.dark .chat-empty-logo {
  filter: none;
}
.chat-thread {
  width: 100%;
  max-width: 820px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.msg-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.msg-row.user { align-items: flex-end; }
.msg-row.assistant { align-items: flex-start; }

.msg-user-bubble {
  max-width: 78%;
  padding: 10px 15px;
  border-radius: 14px;
  background: var(--secondary);
  color: var(--secondary-foreground);
  font-size: 14px;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}
.msg-user-imgs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 6px;
}
.msg-user-imgs img {
  max-width: 220px;
  max-height: 160px;
  border-radius: 8px;
  border: 1px solid var(--border);
  object-fit: cover;
}
.msg-assistant-body {
  width: 100%;
  font-size: 14.5px;
  line-height: 1.65;
  color: var(--foreground);
  word-break: break-word;
}
.msg-assistant-body p { margin-bottom: 10px; }
.msg-assistant-body p:last-child { margin-bottom: 0; }
.msg-assistant-body ul, .msg-assistant-body ol { margin: 6px 0 10px 22px; }
.msg-assistant-body li { margin-bottom: 4px; }
.msg-assistant-body h1, .msg-assistant-body h2, .msg-assistant-body h3 {
  font-weight: 650;
  letter-spacing: -0.015em;
  margin: 14px 0 6px;
}
.msg-assistant-body h1 { font-size: 18px; }
.msg-assistant-body h2 { font-size: 16px; }
.msg-assistant-body h3 { font-size: 14.5px; }
.msg-assistant-body code {
  font-family: var(--font-mono);
  font-size: 12.5px;
  padding: 2px 5px;
  border-radius: 5px;
  background: var(--muted);
}
.code-block {
  margin: 10px 0;
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  background: var(--muted);
}
.code-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 6px 12px;
  font-size: 11.5px;
  color: var(--muted-foreground);
  border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
}
.code-actions {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
}
.code-btn, .code-copy {
  font-family: var(--font-sans);
  font-size: 11.5px;
  padding: 2px 8px;
  border-radius: 5px;
  color: var(--muted-foreground);
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
}
.code-btn:hover, .code-copy:hover { background: var(--accent); color: var(--foreground); }
.code-btn.saved { color: var(--green, #16a34a); font-weight: 500; }
.code-btn.code-open { color: var(--primary); font-weight: 500; }
.code-block pre {
  margin: 0;
  padding: 12px 14px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.55;
}
.code-block pre code { padding: 0; background: none; font-size: inherit; }

/* Saved Files Popover */
.files-popover {
  position: absolute;
  top: 52px;
  right: 16px;
  width: 380px;
  max-width: calc(100% - 32px);
  max-height: 440px;
  display: none;
  flex-direction: column;
  background: var(--popover);
  color: var(--popover-foreground);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
  padding: 14px;
  z-index: 40;
}
.files-popover.open { display: flex; }
.files-dir-hint {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--muted-foreground);
  margin-bottom: 10px;
  word-break: break-all;
}
.files-list {
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.file-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 7px 9px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--muted);
  font-size: 12px;
}
.file-row-meta {
  min-width: 0;
  flex: 1;
}
.file-row-name {
  font-family: var(--font-mono);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.file-row-size {
  font-size: 10.5px;
  color: var(--muted-foreground);
}
.file-row-actions {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

/* Live HTML Preview Modal */
.preview-modal {
  position: fixed;
  inset: 0;
  z-index: 100;
  background: rgba(0, 0, 0, 0.68);
  backdrop-filter: blur(4px);
  display: none;
  align-items: center;
  justify-content: center;
  padding: 20px;
}
.preview-modal.open { display: flex; }
.preview-card {
  width: min(1280px, 96vw);
  height: min(860px, 92vh);
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.45);
}
.preview-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  background: var(--card);
}
.preview-title {
  font-family: var(--font-mono);
  font-size: 12.5px;
  color: var(--foreground);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.preview-actions {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}
.preview-frame {
  flex: 1;
  width: 100%;
  border: none;
  background: #fff;
}

/* Save File Modal */
.save-modal {
  position: fixed;
  inset: 0;
  z-index: 120;
  background: rgba(0, 0, 0, 0.58);
  backdrop-filter: blur(4px);
  display: none;
  align-items: center;
  justify-content: center;
  padding: 20px;
}
.save-modal.open { display: flex; }
.save-card {
  width: min(500px, 94vw);
  background: var(--card);
  color: var(--foreground);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.42);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.save-body {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.save-presets {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
}
.save-chip {
  font-family: var(--font-sans);
  font-size: 11.5px;
  padding: 3px 9px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--muted);
  color: var(--muted-foreground);
  cursor: pointer;
}
.save-chip:hover, .save-chip.active {
  background: var(--accent);
  color: var(--foreground);
  border-color: var(--ring);
}
.save-err {
  font-size: 12px;
  color: var(--red, #ef4444);
  display: none;
}
.save-err.show { display: block; }
.save-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  background: color-mix(in srgb, var(--muted) 40%, transparent);
}
.save-foot-right {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.think-box {
  width: 100%;
  margin-bottom: 8px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: color-mix(in srgb, var(--muted) 40%, transparent);
}
.think-box summary {
  padding: 7px 12px;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--muted-foreground);
  cursor: pointer;
  user-select: none;
}
.think-box pre {
  padding: 0 12px 10px;
  white-space: pre-wrap;
  font-family: var(--font-sans);
  font-size: 12.5px;
  line-height: 1.55;
  color: var(--muted-foreground);
}
.msg-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  font-size: 11.5px;
  font-family: var(--font-mono);
  color: var(--muted-foreground);
  margin-top: 2px;
}

/* Composer (exact llama-swap soft rounded box + stacked attach/send on right) */
.composer-area {
  padding: 12px 16px 16px;
}
.composer-attachments {
  display: none;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}
.composer-attachments.has-items { display: flex; }
.att-thumb {
  position: relative;
  width: 60px;
  height: 60px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
}
.att-thumb img { width: 100%; height: 100%; object-fit: cover; }
.att-file {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 26px 6px 10px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--muted);
  font-family: var(--font-mono);
  font-size: 11.5px;
  max-width: 240px;
}
.att-file-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.att-remove {
  position: absolute;
  top: 2px;
  right: 2px;
  width: 18px;
  height: 18px;
  border-radius: 999px;
  background: rgba(0,0,0,0.7);
  color: #fff;
  font-size: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.msg-user-files {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: flex-end;
}
.msg-user-file-chip {
  font-family: var(--font-mono);
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--muted);
  color: var(--muted-foreground);
}
.composer-row {
  display: flex;
  align-items: flex-end;
  gap: 10px;
}
.composer-box {
  flex: 1;
  position: relative;
  background: var(--muted);
  border: 1px solid transparent;
  border-radius: 10px;
  transition: border-color 140ms ease, background 140ms ease;
}
.composer-box:focus-within {
  border-color: var(--ring);
  background: var(--background);
}
.composer-box textarea {
  width: 100%;
  min-height: 76px;
  max-height: 220px;
  padding: 12px 38px 12px 14px;
  background: transparent;
  border: none;
  outline: none;
  resize: none;
  font-size: 14px;
  line-height: 1.5;
  display: block;
}
.composer-box.expanded textarea { min-height: 180px; }
.composer-box textarea::placeholder { color: var(--muted-foreground); }
.expand-btn {
  position: absolute;
  right: 8px;
  bottom: 8px;
  width: 26px;
  height: 26px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted-foreground);
}
.expand-btn:hover { background: var(--accent); color: var(--foreground); }
.composer-actions {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
}
.composer-actions .btn-send {
  min-width: 64px;
  height: 34px;
  padding: 0 14px;
  border-radius: 8px;
}

/* ---- Load Test Pane ---- */
.loadtest-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.lt-form {
  display: grid;
  grid-template-columns: 1fr 130px 130px auto;
  gap: 12px;
  align-items: end;
}

/* ---- Models Page (exact llama-swap list card) ---- */
.models-header-card {
  overflow: hidden;
}
.models-header-top {
  padding: 16px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--border);
}
.models-title-group {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.models-title-group h2 {
  font-size: 16px;
  font-weight: 650;
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.models-meta-count {
  font-size: 13px;
  color: var(--muted-foreground);
}
.models-header-sub {
  padding: 12px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  font-size: 13px;
  color: var(--muted-foreground);
  flex-wrap: wrap;
}
.kv-bar-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 240px;
}
.progress-track {
  flex: 1;
  height: 6px;
  border-radius: 999px;
  background: var(--secondary);
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--primary);
  border-radius: 999px;
  transition: width 220ms ease;
}

.model-row {
  padding: 16px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid var(--border);
}
.model-row:last-child { border-bottom: none; }
.model-row-left {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}
.model-row-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--foreground);
}
.model-row-sub {
  font-size: 12.5px;
  color: var(--muted-foreground);
  margin-top: 2px;
}
.model-row-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}
.state-tag {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.03em;
  color: #10b981;
  margin-right: 4px;
}
.state-tag.busy { color: #f59e0b; }
.state-tag.offline { color: var(--muted-foreground); }

/* ---- Logs Page (exact llama-swap underline tabs + log panel) ---- */
.logs-shell {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.logs-panel {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.logs-panel-head {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.logs-panel-title { font-size: 13.5px; font-weight: 600; }
.logs-tools {
  display: flex;
  align-items: center;
  gap: 4px;
}
.logs-search {
  width: 200px;
  height: 30px;
  padding: 0 10px;
  font-size: 12.5px;
  display: none;
}
.logs-search.open { display: block; }
.logs-pre {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 14px 16px;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.6;
  white-space: pre;
  color: var(--foreground);
}
.logs-pre.wrap { white-space: pre-wrap; word-break: break-word; }
.log-ts { color: var(--muted-foreground); margin-right: 8px; }
.log-lvl-INFO { color: var(--primary); font-weight: 600; margin-right: 8px; }
.log-lvl-REQ { color: #10b981; font-weight: 600; margin-right: 8px; }

/* ---- Performance Page (exact llama-swap charts + header) ---- */
.perf-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 18px;
  flex-wrap: wrap;
}
.perf-head h1 {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.seg-group {
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  background: var(--card);
}
.seg-btn {
  height: 32px;
  padding: 0 11px;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--muted-foreground);
  border-right: 1px solid var(--border);
}
.seg-btn:last-child { border-right: none; }
.seg-btn:hover { background: var(--accent); color: var(--foreground); }
.seg-btn.active {
  background: var(--primary);
  color: var(--primary-foreground);
}
.perf-sec-title {
  font-size: 15px;
  font-weight: 600;
  margin: 18px 0 10px;
  color: var(--foreground);
}
.perf-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}
.perf-stat-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
}
.perf-grid > .card, .perf-stat-row > .card { margin-top: 0; }
.perf-stat {
  padding: 16px 18px;
}
.perf-stat-label {
  font-size: 12px;
  color: var(--muted-foreground);
  margin-bottom: 4px;
}
.perf-stat-val {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}
.perf-stat-sub {
  font-size: 12px;
  color: var(--muted-foreground);
  margin-top: 2px;
}
.chart-card {
  padding: 18px 20px;
}
.chart-card-title {
  text-align: center;
  font-size: 13.5px;
  font-weight: 600;
  margin-bottom: 12px;
}
.chart-svg-wrap {
  width: 100%;
  height: 210px;
}
.chart-svg-wrap svg { width: 100%; height: 100%; }

/* ---- Settings Page (exact llama-swap narrow clean column) ---- */
.settings-col {
  max-width: 520px;
}
.settings-col h1 {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 18px;
}
.settings-card-title {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--muted-foreground);
  margin-bottom: 16px;
}
.settings-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 7px 0;
}
.settings-row label {
  font-size: 14px;
  color: var(--foreground);
}
.settings-row select { width: 190px; }
.kv-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 5px 0;
  font-size: 13.5px;
}
.kv-row .k { color: var(--muted-foreground); }
.kv-row .v { font-family: var(--font-mono); font-size: 13px; text-align: right; }

@media (max-width: 960px) {
  .sidebar { width: 56px; min-width: 56px; }
  .sidebar .sidebar-header { justify-content: center; padding: 0; }
  .sidebar .brand-link { margin-left: 0; padding: 0; }
  .sidebar .brand-logo, .sidebar .nav-label, .sidebar .nav-chevron, .sidebar .nav-sublist { display: none !important; }
  .sidebar .brand-mark { display: block !important; }
  .sidebar .sidebar-header .status-dot { position: absolute; right: 11px; bottom: 11px; width: 8px; height: 8px; margin-left: 0; box-shadow: 0 0 0 2px var(--sidebar); }
  .activity-charts-grid, .perf-grid { grid-template-columns: 1fr; }
  .activity-kpis, .perf-stat-row { grid-template-columns: repeat(2, 1fr); }
  .lt-form { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="app-shell">
  <!-- Left Sidebar -->
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <a class="brand-link" href="#/playground" title="nib">
        <img class="brand-logo" src="/logo.svg" alt="nib" />
        <img class="brand-mark" src="/favicon.svg" alt="nib" />
      </a>
      <span class="status-dot" id="brandDot" title="Server Online"></span>
    </div>

    <nav class="sidebar-nav">
      <button class="nav-item" data-route="activity" title="Activity">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>
        <span class="nav-label">Activity</span>
      </button>

      <button class="nav-item" data-route="playground" title="Playground">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5h.01"/><path d="M14.5 9.5h.01"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/></svg>
        <span class="nav-label">Playground</span>
      </button>

      <button class="nav-item open" data-route="models" id="navModelsBtn" title="Models">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>
        <span class="nav-label">Models</span>
        <svg class="nav-chevron" id="modelsChevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>
      </button>
      <div class="nav-sublist" id="modelsSublist">
        <button class="nav-subitem" id="subModelBtn" title="Open inkling-2-mini in Playground">
          <span class="sub-dot" id="subModelDot"></span>
          <span id="subModelName">inkling-2-mini</span>
        </button>
      </div>

      <button class="nav-item" data-route="logs" title="Logs">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>
        <span class="nav-label">Logs</span>
      </button>

      <button class="nav-item" data-route="performance" title="Performance">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>
        <span class="nav-label">Performance</span>
      </button>
    </nav>

    <div class="sidebar-footer">
      <button class="nav-item" data-route="settings" title="Settings">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
        <span class="nav-label">Settings</span>
      </button>
      <button class="icon-btn" id="modeCycleBtn" title="Color mode: System"></button>
    </div>
  </aside>

  <!-- Main Workspace -->
  <div class="main-area">
    <header class="topbar">
      <div class="topbar-left">
        <button class="icon-btn" id="sidebarToggleBtn" title="Toggle sidebar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/></svg>
        </button>
        <span class="topbar-sep"></span>
        <span class="breadcrumb" id="breadcrumb">Playground / Chat</span>
      </div>
      <div class="topbar-right">
        <span class="live-pill" id="liveBusyPill">
          <span class="status-dot busy" style="width:7px;height:7px;"></span>
          <span id="liveBusyText">Generating</span>
        </span>
      </div>
    </header>

    <main class="page-body" id="pageBody">
      <!-- 1. ACTIVITY -->
      <section class="view" id="view-activity">
        <div class="card">
          <div class="activity-charts-grid">
            <div>
              <div class="chart-col-title">Prompt Processing</div>
              <div class="mini-chart-box" id="actPromptChart">No prompt speed data yet</div>
            </div>
            <div>
              <div class="chart-col-title">Token Generation</div>
              <div class="mini-chart-box" id="actGenChart">No generation speed data yet</div>
            </div>
          </div>
          <div class="activity-kpis">
            <div>
              <div class="kpi-label">Requests</div>
              <div class="kpi-val"><strong id="kpiReqs">0</strong> completed</div>
            </div>
            <div>
              <div class="kpi-label">Cached</div>
              <div class="kpi-val"><strong id="kpiCached">0</strong> tokens</div>
            </div>
            <div>
              <div class="kpi-label">Processed</div>
              <div class="kpi-val"><strong id="kpiProcessed">0</strong> tokens</div>
            </div>
            <div>
              <div class="kpi-label">Generated</div>
              <div class="kpi-val"><strong id="kpiGenerated">0</strong> tokens</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="section-card-head">
            <div class="section-kicker">IN-FLIGHT REQUESTS <strong id="inflightCount">0 active</strong></div>
          </div>
          <div class="table-wrap">
            <table class="swap-table">
              <thead>
                <tr>
                  <th>Slot</th>
                  <th>State</th>
                  <th>Request</th>
                  <th>Prompt</th>
                  <th>Generated</th>
                  <th>Elapsed</th>
                </tr>
              </thead>
              <tbody id="inflightBody">
                <tr><td colspan="6" class="table-empty">No in-flight requests</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="section-card-head">
            <div class="section-kicker">REQUEST HISTORY</div>
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="font-size:12.5px;color:var(--muted-foreground);">Rows</span>
              <select class="select select-sm" id="actRowsSelect">
                <option value="25" selected>25</option>
                <option value="50">50</option>
                <option value="100">100</option>
              </select>
              <button class="icon-btn" id="clearActivityBtn" title="Clear activity history">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
              </button>
            </div>
          </div>
          <div class="table-wrap">
            <table class="swap-table">
              <thead>
                <tr>
                  <th>ID ↓</th>
                  <th>Time</th>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Cached</th>
                  <th>Prompt</th>
                  <th>Generated</th>
                  <th>Prompt Speed</th>
                  <th>Gen Speed</th>
                  <th>MTP</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody id="activityTableBody">
                <tr><td colspan="11" class="table-empty">No activity recorded</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <!-- 2. PLAYGROUND -->
      <section class="view flex-view" id="view-playground">
        <div class="card playground-card">
          <div class="underline-tabs">
            <button class="underline-tab active" data-pgtab="chat">Chat</button>
            <button class="underline-tab" data-pgtab="loadtest">Load Test</button>
          </div>

          <!-- Chat Pane -->
          <div class="pg-pane active" id="pg-chat">
            <div class="pg-toolbar">
              <div class="model-select-wrap">
                <select id="modelSelect">
                  <option value="inkling-2-mini">inkling-2-mini</option>
                </select>
              </div>
              <button class="icon-btn bordered" id="paramsBtn" title="Generation parameters">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
              </button>
              <button class="btn btn-outline" id="filesBtn" title="Saved output files (open or preload into chat)">Files</button>
              <button class="btn btn-outline" id="newChatBtn">New Chat</button>

              <!-- Floating Saved Files Popover -->
              <div class="files-popover" id="filesPopover">
                <div class="popover-head">
                  <span class="popover-title">Saved Files</span>
                  <div style="display:inline-flex;gap:4px;align-items:center;">
                    <button class="btn btn-outline btn-sm" id="refreshFilesBtn" type="button">Refresh</button>
                    <button class="icon-btn" id="closeFilesBtn" style="width:26px;height:26px;" title="Close">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                    </button>
                  </div>
                </div>
                <div class="files-dir-hint" id="filesDirHint">outputs/</div>
                <div class="files-list" id="filesList">
                  <div style="font-size:12px;color:var(--muted-foreground);padding:8px 0;">No saved files yet. Click Save on any code block in chat.</div>
                </div>
              </div>

              <!-- Floating Parameters Popover -->
              <div class="params-popover" id="paramsPopover">
                <div class="popover-head">
                  <span class="popover-title">Parameters</span>
                  <button class="icon-btn" id="closeParamsBtn" style="width:26px;height:26px;" title="Close">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                  </button>
                </div>

                <div class="field">
                  <div class="field-row"><label for="sysPrompt">System Prompt</label></div>
                  <textarea id="sysPrompt" placeholder="Optional system instructions..."></textarea>
                </div>

                <div class="field">
                  <div class="field-row">
                    <label for="tempRange">Temperature</label>
                    <span class="field-val" id="tempVal">0.00</span>
                  </div>
                  <input type="range" id="tempRange" min="0" max="1.5" step="0.05" value="0" />
                  <div class="field-hint">Lossless DSpark MTP runs at any temperature (0.00 = greedy)</div>
                </div>

                <div class="field">
                  <div class="field-row">
                    <label for="topPRange">Top-P</label>
                    <span class="field-val" id="topPVal">0.95</span>
                  </div>
                  <input type="range" id="topPRange" min="0.1" max="1" step="0.05" value="0.95" />
                </div>

                <div class="field">
                  <div class="field-row">
                    <label for="maxTokRange">Max Tokens</label>
                    <span class="field-val" id="maxTokVal">131,072</span>
                  </div>
                  <input type="range" id="maxTokRange" min="256" max="131072" step="256" value="131072" />
                </div>

                <div class="field">
                  <div class="field-row">
                    <label for="thinkRange">Thinking Effort</label>
                    <span class="field-val" id="thinkVal">0 (off)</span>
                  </div>
                  <input type="range" id="thinkRange" min="0" max="3" step="1" value="0" />
                </div>

                <div class="switch-row">
                  <label for="mtpToggle" style="font-size:13px;">DSpark Speculative MTP</label>
                  <label class="switch">
                    <input type="checkbox" id="mtpToggle" checked />
                    <span class="switch-track"></span>
                  </label>
                </div>

                <div class="switch-row">
                  <label for="streamToggle" style="font-size:13px;">Stream Response (SSE)</label>
                  <label class="switch">
                    <input type="checkbox" id="streamToggle" checked />
                    <span class="switch-track"></span>
                  </label>
                </div>

                <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;">
                  <span class="mono" style="font-size:11.5px;color:var(--muted-foreground);" id="popoverKvText">KV: 0 / 262k</span>
                  <button class="btn btn-outline btn-sm" id="popoverResetKvBtn">Reset KV</button>
                </div>
              </div>
            </div>

            <div class="chat-scroll" id="chatScroll">
              <div class="chat-empty" id="chatEmpty">
                <img class="chat-empty-logo" src="/logo.svg" alt="nib" />
                <span>Start a conversation by typing a message below.</span>
              </div>
              <div class="chat-thread" id="chatThread"></div>
            </div>

            <div class="composer-area" id="dropZone">
              <div class="composer-attachments" id="attList"></div>
              <div class="composer-row">
                <div class="composer-box" id="composerBox">
                  <textarea id="chatInput" placeholder="Type a message..." rows="3"></textarea>
                  <button class="expand-btn" id="expandInputBtn" title="Expand editor">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" x2="14" y1="3" y2="10"/><line x1="3" x2="10" y1="21" y2="14"/></svg>
                  </button>
                </div>
                <div class="composer-actions">
                  <input type="file" id="imgFileInput" accept="image/*,.html,.htm,.svg,.js,.mjs,.ts,.jsx,.tsx,.css,.json,.py,.c,.cc,.cpp,.h,.hpp,.rs,.go,.sh,.md,.txt,.yaml,.yml,.xml" multiple style="display:none;" />
                  <button class="icon-btn bordered" id="attachBtn" title="Attach image or code/text file (or paste / drag-and-drop)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
                  </button>
                  <button class="btn btn-primary btn-send" id="sendBtn">Send</button>
                </div>
              </div>
            </div>
          </div>

          <!-- Load Test Pane -->
          <div class="pg-pane" id="pg-loadtest">
            <div class="loadtest-body">
              <div class="lt-form">
                <div>
                  <div class="kpi-label" style="margin-bottom:6px;">Prompt</div>
                  <input class="input" id="ltPrompt" style="width:100%;" value="Write a clean, self-contained C++17 intrusive ring buffer with push, pop, and size." />
                </div>
                <div>
                  <div class="kpi-label" style="margin-bottom:6px;">Runs</div>
                  <input class="input" id="ltRuns" type="number" min="1" max="20" value="3" style="width:100%;" />
                </div>
                <div>
                  <div class="kpi-label" style="margin-bottom:6px;">Max Tokens</div>
                  <input class="input" id="ltMaxTok" type="number" min="32" max="131072" value="256" style="width:100%;" />
                </div>
                <button class="btn btn-primary" id="ltRunBtn">Run Benchmark</button>
              </div>
              <div class="card" style="overflow:hidden;">
                <div class="table-wrap">
                  <table class="swap-table">
                    <thead>
                      <tr>
                        <th>Run</th>
                        <th>Prompt</th>
                        <th>Generated</th>
                        <th>Prefill</th>
                        <th>Decode Speed</th>
                        <th>MTP tok/rnd</th>
                        <th>Duration</th>
                      </tr>
                    </thead>
                    <tbody id="ltTableBody">
                      <tr><td colspan="7" class="table-empty">Configure and run a benchmark to measure live throughput.</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- 3. MODELS -->
      <section class="view" id="view-models">
        <div class="card models-header-card">
          <div class="models-header-top">
            <div class="models-title-group">
              <h2>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="width:18px;height:18px;"><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>
                Models
              </h2>
              <span class="models-meta-count">(1 of 1) &nbsp; <span id="modelsReadyLabel">1 READY</span></span>
            </div>
            <button class="btn btn-outline btn-sm" id="modelsResetKvBtn">Reset KV Cache</button>
          </div>
          <div class="models-header-sub">
            <span id="modelsEngineSummary">RX 7900 XTX · 262,848 ctx · Q8_0 sparse-112 · DSpark MTP · Vision</span>
            <div class="kv-bar-wrap">
              <span class="mono" style="font-size:12px;" id="modelsKvLabel">KV 0 / 262,848</span>
              <div class="progress-track"><div class="progress-fill" id="modelsKvBar" style="width:0%"></div></div>
            </div>
          </div>
        </div>

        <div class="card" style="overflow:hidden;">
          <div class="model-row">
            <div class="model-row-left">
              <span class="status-dot" id="modelRowDot"></span>
              <div>
                <div class="model-row-name" id="modelRowTitle">Inkling 2 Mini NVFP4-Mixed (RDNA3 HIP, 262k, MTP, Vision)</div>
                <div class="model-row-sub mono" id="modelRowSub">inkling-2-mini · VRAM 21.50 GiB (LLM 21.18 GiB, Vision 0.32 GiB) · Draft Ring 8,192</div>
              </div>
            </div>
            <div class="model-row-right">
              <span class="state-tag" id="modelRowState">READY</span>
              <button class="icon-btn" id="modelOpenPgBtn" title="Open in Playground">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>
              </button>
              <button class="icon-btn" id="modelRowResetBtn" title="Reset KV Cache">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
              </button>
            </div>
          </div>
        </div>
      </section>

      <!-- 4. LOGS -->
      <section class="view flex-view" id="view-logs">
        <div class="logs-shell">
          <div class="underline-tabs" style="padding:0;">
            <button class="underline-tab active" data-logtab="ALL">All</button>
            <button class="underline-tab" data-logtab="REQ">Requests</button>
            <button class="underline-tab" data-logtab="INFO">Engine</button>
          </div>
          <div class="card logs-panel">
            <div class="logs-panel-head">
              <span class="logs-panel-title">Server Logs</span>
              <div class="logs-tools">
                <input class="input logs-search" id="logsSearch" placeholder="Filter logs..." />
                <button class="icon-btn" id="logsWrapBtn" title="Toggle line wrap">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><line x1="3" x2="21" y1="6" y2="6"/><path d="M3 12h15a3 3 0 1 1 0 6h-4"/><polyline points="16 16 14 18 16 20"/><line x1="3" x2="10" y1="18" y2="18"/></svg>
                </button>
                <button class="icon-btn" id="logsSearchBtn" title="Search logs">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                </button>
                <button class="icon-btn" id="logsClearBtn" title="Clear logs">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
                </button>
              </div>
            </div>
            <div class="logs-pre" id="logsPre"></div>
          </div>
        </div>
      </section>

      <!-- 5. PERFORMANCE -->
      <section class="view" id="view-performance">
        <div class="perf-head">
          <h1>Performance</h1>
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:12.5px;color:var(--muted-foreground);">Refresh:</span>
            <div class="seg-group" id="perfRefreshGroup">
              <button class="seg-btn" data-ms="0">Off</button>
              <button class="seg-btn active" data-ms="2000">2s</button>
              <button class="seg-btn" data-ms="5000">5s</button>
              <button class="seg-btn" data-ms="10000">10s</button>
            </div>
            <button class="icon-btn bordered" id="perfRefreshNow" title="Refresh now">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
            </button>
          </div>
        </div>

        <div class="perf-sec-title" style="margin-top:0;">GPU &amp; Engine</div>
        <div class="perf-stat-row">
          <div class="card perf-stat">
            <div class="perf-stat-label">VRAM Allocated</div>
            <div class="perf-stat-val" id="perfVram">--</div>
            <div class="perf-stat-sub" id="perfVramSub">RX 7900 XTX 24 GiB</div>
          </div>
          <div class="card perf-stat">
            <div class="perf-stat-label">KV Cache Position</div>
            <div class="perf-stat-val" id="perfKvPos">0</div>
            <div class="perf-stat-sub" id="perfKvSub">of 262,848 tokens</div>
          </div>
          <div class="card perf-stat">
            <div class="perf-stat-label">Avg Decode Speed</div>
            <div class="perf-stat-val" id="perfAvgDecode">--</div>
            <div class="perf-stat-sub">tok/s (recent)</div>
          </div>
          <div class="card perf-stat">
            <div class="perf-stat-label">Avg MTP Yield</div>
            <div class="perf-stat-val" id="perfAvgMtp">--</div>
            <div class="perf-stat-sub">tok/round (DSpark)</div>
          </div>
        </div>

        <div class="perf-sec-title">Telemetry</div>
        <div class="perf-grid">
          <div class="card chart-card">
            <div class="chart-card-title">Decode Throughput (tok/s)</div>
            <div class="chart-svg-wrap" id="perfDecodeChart"></div>
          </div>
          <div class="card chart-card">
            <div class="chart-card-title">MTP Acceptance (tok/round)</div>
            <div class="chart-svg-wrap" id="perfMtpChart"></div>
          </div>
          <div class="card chart-card">
            <div class="chart-card-title">Prompt Prefill Speed (tok/s)</div>
            <div class="chart-svg-wrap" id="perfPrefillChart"></div>
          </div>
          <div class="card chart-card">
            <div class="chart-card-title">Draft vs Verify Latency (ms/round)</div>
            <div class="chart-svg-wrap" id="perfLatChart"></div>
          </div>
        </div>
      </section>

      <!-- 6. SETTINGS -->
      <section class="view" id="view-settings">
        <div class="settings-col">
          <h1>Settings</h1>

          <div class="card card-pad">
            <div class="settings-card-title">Appearance</div>
            <div class="settings-row">
              <label for="themeSelect">Theme</label>
              <select class="select" id="themeSelect">
                <option value="default">Default</option>
                <option value="ocean">Ocean</option>
                <option value="violet">Violet</option>
                <option value="emerald">Emerald</option>
                <option value="rose">Rose</option>
                <option value="amber">Amber</option>
                <option value="slate">Slate</option>
                <option value="sunset">Sunset</option>
                <option value="terminal">Terminal</option>
                <option value="mc">Midnight Commander</option>
                <option value="solarized">Solarized</option>
              </select>
            </div>
            <div class="settings-row">
              <label for="modeSelect">Mode</label>
              <select class="select" id="modeSelect">
                <option value="system">System</option>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
            </div>
          </div>

          <div class="card card-pad">
            <div class="settings-card-title">Engine Information</div>
            <div class="kv-row"><span class="k">Status</span><span class="v" id="setStatus">connected</span></div>
            <div class="kv-row"><span class="k">OpenAI Base URL</span><span class="v" id="setEndpoint">-</span></div>
            <div class="kv-row"><span class="k">Model</span><span class="v" id="setModel">-</span></div>
            <div class="kv-row"><span class="k">Context Window</span><span class="v" id="setCtx">-</span></div>
            <div class="kv-row"><span class="k">KV Cache</span><span class="v" id="setKv">-</span></div>
            <div class="kv-row"><span class="k">DSpark MTP</span><span class="v" id="setMtp">-</span></div>
            <div class="kv-row"><span class="k">Vision Encoder</span><span class="v" id="setVision">-</span></div>
            <div class="kv-row"><span class="k">VRAM Allocated</span><span class="v" id="setVram">-</span></div>
            <div class="kv-row"><span class="k">Uptime</span><span class="v" id="setUptime">-</span></div>
          </div>
        </div>
      </section>
    </main>
  </div>
</div>

<!-- Live HTML Preview Modal -->
<div class="preview-modal" id="previewModal">
  <div class="preview-card">
    <div class="preview-head">
      <span class="preview-title" id="previewTitle">Preview</span>
      <div class="preview-actions">
        <button class="btn btn-outline btn-sm" id="previewSaveOpenBtn" type="button">Save As…</button>
        <button class="btn btn-outline btn-sm" id="previewDownloadBtn" type="button">Download</button>
        <button class="icon-btn" id="previewCloseBtn" style="width:28px;height:28px;" title="Close preview">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
        </button>
      </div>
    </div>
    <iframe class="preview-frame" id="previewFrame" sandbox="allow-scripts allow-same-origin allow-pointer-lock allow-popups allow-forms"></iframe>
  </div>
</div>

<!-- Save File As Modal -->
<div class="save-modal" id="saveModal">
  <div class="save-card">
    <div class="preview-head">
      <span class="popover-title">Save File</span>
      <button class="icon-btn" id="saveCloseBtn" style="width:28px;height:28px;" title="Close">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      </button>
    </div>
    <div class="save-body">
      <div class="field" style="margin:0;">
        <div class="field-row"><label for="saveFilenameInput">Filename</label></div>
        <input class="input mono" id="saveFilenameInput" type="text" style="width:100%;" placeholder="example.html" />
      </div>
      <div class="field" style="margin:0;">
        <div class="field-row"><label for="saveDirInput">Save Folder</label></div>
        <input class="input mono" id="saveDirInput" type="text" style="width:100%;" placeholder="./outputs" />
        <div class="save-presets" id="savePresets"></div>
        <div class="field-hint">Any folder inside your home directory (missing subfolders are created automatically).</div>
      </div>
      <div class="switch-row" id="saveAlsoRow" style="margin:0;">
        <label for="saveAlsoOutputs" style="font-size:12.5px;">Also keep a copy in Nib Files (for Open / Preload)</label>
        <label class="switch">
          <input type="checkbox" id="saveAlsoOutputs" checked />
          <span class="switch-track"></span>
        </label>
      </div>
      <div class="save-err" id="saveErr"></div>
    </div>
    <div class="save-foot">
      <button class="btn btn-outline btn-sm" id="savePickerBtn" type="button" title="Open native OS Save As dialog">Browse (OS Dialog)…</button>
      <div class="save-foot-right">
        <button class="btn btn-outline btn-sm" id="saveCancelBtn" type="button">Cancel</button>
        <button class="btn btn-primary btn-sm" id="saveConfirmBtn" type="button">Save</button>
      </div>
    </div>
  </div>
</div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const ROUTES = ["activity", "playground", "models", "logs", "performance", "settings"];

  const state = {
    route: "playground",
    pgTab: "chat",
    logTab: "ALL",
    logWrap: false,
    theme: localStorage.getItem("inkling_theme") || "default",
    mode: localStorage.getItem("inkling_mode") || "system",
    collapsed: localStorage.getItem("inkling_sidebar_collapsed") === "1",
    health: null,
    activity: [],
    logs: [],
    outputs: [],
    outputsDir: "outputs/",
    outputPresets: [],
    allowCustomDir: true,
    preview: null,
    saveTarget: null,
    messages: [],
    attachments: [],
    generating: false,
    abortCtrl: null,
    renderRaf: 0,
    pollMs: 2000,
    pollTimer: null
  };

  /* ---- Theme & Color Mode (exact llama-swap behavior) ---- */
  const mqDark = window.matchMedia("(prefers-color-scheme: dark)");
  const MODE_ICONS = {
    system: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/></svg>',
    light:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>',
    dark:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>'
  };

  function applyTheme() {
    const html = document.documentElement;
    const isDark = state.mode === "dark" || (state.mode === "system" && mqDark.matches);
    html.classList.toggle("dark", isDark);
    if (state.theme === "default") html.removeAttribute("data-theme");
    else html.setAttribute("data-theme", state.theme);
    $("themeSelect").value = state.theme;
    $("modeSelect").value = state.mode;
    const btn = $("modeCycleBtn");
    btn.innerHTML = MODE_ICONS[state.mode];
    btn.title = "Color mode: " + state.mode[0].toUpperCase() + state.mode.slice(1);
    localStorage.setItem("inkling_theme", state.theme);
    localStorage.setItem("inkling_mode", state.mode);
  }
  mqDark.addEventListener("change", () => { if (state.mode === "system") applyTheme(); });

  $("themeSelect").addEventListener("change", (e) => { state.theme = e.target.value; applyTheme(); });
  $("modeSelect").addEventListener("change", (e) => { state.mode = e.target.value; applyTheme(); });
  $("modeCycleBtn").addEventListener("click", () => {
    const order = ["system", "light", "dark"];
    state.mode = order[(order.indexOf(state.mode) + 1) % order.length];
    applyTheme();
  });

  /* ---- Sidebar Collapse ---- */
  function applySidebar() {
    $("sidebar").classList.toggle("collapsed", state.collapsed);
    localStorage.setItem("inkling_sidebar_collapsed", state.collapsed ? "1" : "0");
  }
  $("sidebarToggleBtn").addEventListener("click", () => { state.collapsed = !state.collapsed; applySidebar(); });

  $("modelsChevron").addEventListener("click", (e) => {
    e.stopPropagation();
    const open = $("navModelsBtn").classList.toggle("open");
    $("modelsSublist").classList.toggle("hidden", !open);
  });
  $("subModelBtn").addEventListener("click", () => navigate("playground"));

  function updateDocTitle() {
    const online = state.health && state.health.status === "ok";
    document.title = (online ? "🟢 " : "🔴 ") + "nib";
  }

  /* ---- Router (supports #/route and /ui/route) ---- */
  function parseInitialRoute() {
    const h = (location.hash || "").replace(/^#\/?/, "").split("/")[0];
    if (ROUTES.includes(h)) return h;
    const p = location.pathname.replace(/^\/ui\/?/, "").replace(/^\//, "").split("/")[0];
    if (ROUTES.includes(p)) return p;
    return "playground";
  }

  function updateBreadcrumb() {
    const map = {
      activity: "Activity",
      playground: state.pgTab === "chat" ? "Playground / Chat" : "Playground / Load Test",
      models: "Models",
      logs: "Logs",
      performance: "Performance",
      settings: "Settings"
    };
    $("breadcrumb").textContent = map[state.route] || "Playground";
  }

  function navigate(route, pushHash = true) {
    if (!ROUTES.includes(route)) route = "playground";
    state.route = route;
    document.querySelectorAll(".view").forEach((el) => el.classList.remove("active"));
    $("view-" + route).classList.add("active");
    document.querySelectorAll(".nav-item[data-route]").forEach((el) => {
      el.classList.toggle("active", el.dataset.route === route);
    });
    const fullH = (route === "playground" || route === "logs");
    $("pageBody").classList.toggle("full-height", fullH);
    updateBreadcrumb();
    if (pushHash) history.replaceState(null, "", "#/" + route);
    if (route === "activity" || route === "performance") renderActivityAndPerf();
    if (route === "logs") renderLogs();
  }

  document.querySelectorAll(".nav-item[data-route]").forEach((btn) => {
    btn.addEventListener("click", () => navigate(btn.dataset.route));
  });
  window.addEventListener("hashchange", () => navigate(parseInitialRoute(), false));

  /* ---- Playground Tabs & Parameters Popover ---- */
  document.querySelectorAll(".underline-tab[data-pgtab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.pgTab = tab.dataset.pgtab;
      document.querySelectorAll(".underline-tab[data-pgtab]").forEach((t) => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".pg-pane").forEach((p) => p.classList.toggle("active", p.id === "pg-" + state.pgTab));
      updateBreadcrumb();
    });
  });

  $("paramsBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    $("filesPopover").classList.remove("open");
    $("paramsPopover").classList.toggle("open");
  });
  $("closeParamsBtn").addEventListener("click", () => $("paramsPopover").classList.remove("open"));
  $("filesBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    $("paramsPopover").classList.remove("open");
    const open = $("filesPopover").classList.toggle("open");
    if (open) refreshOutputs();
  });
  $("closeFilesBtn").addEventListener("click", () => $("filesPopover").classList.remove("open"));
  $("refreshFilesBtn").addEventListener("click", () => refreshOutputs());
  document.addEventListener("click", (e) => {
    const pop = $("paramsPopover");
    if (pop.classList.contains("open") && !pop.contains(e.target) && !$("paramsBtn").contains(e.target)) {
      pop.classList.remove("open");
    }
    const fpop = $("filesPopover");
    if (fpop.classList.contains("open") && !fpop.contains(e.target) && !$("filesBtn").contains(e.target)) {
      fpop.classList.remove("open");
    }
  });

  const THINK_LABELS = ["0 (off)", "1 (low)", "2 (medium)", "3 (high)"];
  $("tempRange").addEventListener("input", () => { $("tempVal").textContent = Number($("tempRange").value).toFixed(2); });
  $("topPRange").addEventListener("input", () => { $("topPVal").textContent = Number($("topPRange").value).toFixed(2); });
  $("maxTokRange").addEventListener("input", () => { $("maxTokVal").textContent = Number($("maxTokRange").value).toLocaleString(); });
  $("thinkRange").addEventListener("input", () => { $("thinkVal").textContent = THINK_LABELS[Number($("thinkRange").value)] || "0"; });

  /* ---- Markdown Renderer (clean, safe, preview/save/download/copy code blocks) ---- */
  function escHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function renderInline(text) {
    let out = escHtml(text);
    out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return out;
  }

  const LANG_EXT = {
    html: "html", htm: "html", svg: "svg",
    js: "js", javascript: "js", mjs: "mjs",
    ts: "ts", typescript: "ts", jsx: "jsx", tsx: "tsx",
    css: "css", json: "json", py: "py", python: "py",
    cpp: "cpp", "c++": "cpp", cc: "cpp", c: "c", h: "h", hpp: "hpp",
    rs: "rs", rust: "rs", go: "go", sh: "sh", bash: "sh",
    md: "md", markdown: "md", yaml: "yaml", yml: "yml", xml: "xml", sql: "sql"
  };

  function isPreviewableHtml(lang, code) {
    const l = (lang || "").toLowerCase();
    if (l === "html" || l === "htm" || l === "svg") return true;
    const head = (code || "").trimStart().slice(0, 120).toLowerCase();
    return head.startsWith("<!doctype html") || head.startsWith("<html") || head.startsWith("<svg");
  }

  function guessFilename(lang, code) {
    const l = (lang || "").toLowerCase();
    const isHtml = isPreviewableHtml(l, code);
    const ext = LANG_EXT[l] || (isHtml ? ((code || "").trimStart().toLowerCase().startsWith("<svg") ? "svg" : "html") : "txt");
    if (ext === "html") {
      const m = (code || "").match(/<title[^>]*>([^<]{1,80})<\/title>/i);
      if (m && m[1]) {
        const slug = m[1].trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
        if (slug) return slug + ".html";
      }
    }
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const stamp = `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    return `nib-${ext}-${stamp}.${ext}`;
  }

  function downloadTextFile(filename, content) {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function saveCodeToServer(filename, content, dir, alsoOutputs) {
    const resp = await fetch("/api/outputs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename,
        content,
        dir: dir || "",
        also_outputs: !!alsoOutputs
      })
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error((data.error && data.error.message) || "Save failed");
    refreshOutputs();
    return data;
  }

  function fmtBytes(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / (1024 * 1024)).toFixed(2) + " MB";
  }

  async function refreshOutputs() {
    try {
      const resp = await fetch("/api/outputs");
      if (!resp.ok) return;
      const data = await resp.json();
      state.outputs = data.files || [];
      state.outputsDir = data.dir || "outputs/";
      state.outputPresets = data.presets || [];
      state.allowCustomDir = data.allow_custom_dir !== false;
      renderOutputs();
    } catch (_) {}
  }

  function renderOutputs() {
    const btn = $("filesBtn");
    btn.textContent = state.outputs.length ? `Files (${state.outputs.length})` : "Files";
    $("filesDirHint").textContent = state.outputsDir;
    const list = $("filesList");
    if (!state.outputs.length) {
      list.innerHTML = '<div style="font-size:12px;color:var(--muted-foreground);padding:8px 0;">No saved files yet. Click Save on any code block in chat.</div>';
      return;
    }
    list.innerHTML = state.outputs.map((f) => `
      <div class="file-row" data-name="${escHtml(f.filename)}" data-url="${escHtml(f.url)}">
        <div class="file-row-meta">
          <div class="file-row-name" title="${escHtml(f.filename)}">${escHtml(f.filename)}</div>
          <div class="file-row-size">${fmtBytes(f.bytes)}</div>
        </div>
        <div class="file-row-actions">
          <a class="code-btn code-open" href="${escHtml(f.url)}" target="_blank" rel="noopener">Open</a>
          <button class="code-btn file-preload" type="button" title="Attach file to composer">Preload</button>
          <button class="code-btn file-del" type="button" title="Delete file">&times;</button>
        </div>
      </div>`).join("");
  }

  $("filesList").addEventListener("click", async (e) => {
    const row = e.target.closest(".file-row");
    if (!row) return;
    const fname = row.dataset.name;
    const url = row.dataset.url;
    if (e.target.closest(".file-preload")) {
      const btn = e.target.closest(".file-preload");
      btn.textContent = "...";
      try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error("Failed to load");
        const text = await resp.text();
        state.attachments.push({ kind: "text", name: fname, content: text, size: text.length });
        renderAttachments();
        $("filesPopover").classList.remove("open");
        $("chatInput").focus();
      } catch (_) {
        btn.textContent = "Err";
        setTimeout(() => { btn.textContent = "Preload"; }, 1200);
      }
      return;
    }
    if (e.target.closest(".file-del")) {
      await fetch("/api/outputs/" + encodeURIComponent(fname), { method: "DELETE" });
      refreshOutputs();
    }
  });

  /* ---- Save File As Modal ---- */
  function syncSaveModalUi() {
    const curDir = $("saveDirInput").value.trim().replace(/\/+$/, "");
    const defDir = (state.outputsDir || "").replace(/\/+$/, "");
    const sameAsDef = !curDir || curDir === defDir;
    $("saveAlsoRow").style.display = sameAsDef ? "none" : "flex";
    $("savePresets").querySelectorAll(".save-chip").forEach((chip) => {
      chip.classList.toggle("active", chip.dataset.path.replace(/\/+$/, "") === curDir);
    });
  }

  function showSaveError(msg) {
    const el = $("saveErr");
    el.textContent = msg || "";
    el.classList.toggle("show", !!msg);
  }

  async function openSaveModal(opts) {
    if (!state.outputPresets.length) await refreshOutputs();
    state.saveTarget = opts;
    const fname = opts.filename || guessFilename(opts.lang, opts.code);
    const lastDir = localStorage.getItem("nib_last_save_dir");
    const initDir = (state.allowCustomDir && lastDir) ? lastDir : state.outputsDir;

    $("saveFilenameInput").value = fname;
    $("saveDirInput").value = initDir;
    $("saveDirInput").disabled = !state.allowCustomDir;
    $("savePresets").innerHTML = state.outputPresets.map((p) =>
      `<button class="save-chip" type="button" data-path="${escHtml(p.path)}" title="${escHtml(p.path)}">${escHtml(p.label)}</button>`
    ).join("");
    $("savePickerBtn").style.display = (typeof window.showSaveFilePicker === "function") ? "inline-flex" : "none";
    showSaveError("");
    syncSaveModalUi();
    $("saveModal").classList.add("open");
    const inp = $("saveFilenameInput");
    inp.focus();
    const dot = fname.lastIndexOf(".");
    inp.setSelectionRange(0, dot > 0 ? dot : fname.length);
  }

  function closeSaveModal() {
    $("saveModal").classList.remove("open");
    state.saveTarget = null;
    showSaveError("");
  }

  function markSaveResult(target, saved) {
    const btn = target && target.triggerBtn;
    if (!btn) return;
    if (btn.classList.contains("code-save")) {
      btn.textContent = "Saved ✓";
      btn.classList.add("saved");
      if (saved.path) btn.title = "Saved to " + saved.path;
      if (saved.url) {
        const actions = btn.closest(".code-actions");
        if (actions) {
          let openLink = actions.querySelector(".code-open");
          if (!openLink) {
            openLink = document.createElement("a");
            openLink.className = "code-btn code-open";
            openLink.target = "_blank";
            openLink.rel = "noopener";
            actions.insertBefore(openLink, btn);
          }
          openLink.href = saved.url;
          openLink.textContent = "Open ↗";
          openLink.title = saved.path || saved.url;
        }
      }
    } else {
      const prev = btn.dataset.origLabel || btn.textContent;
      btn.dataset.origLabel = prev;
      btn.textContent = "Saved ✓";
      setTimeout(() => { btn.textContent = prev; }, 1600);
    }
  }

  async function confirmSaveModal() {
    if (!state.saveTarget) return;
    const filename = $("saveFilenameInput").value.trim();
    const dir = $("saveDirInput").value.trim();
    if (!filename) { showSaveError("Please enter a filename."); return; }
    const alsoOutputs = $("saveAlsoRow").style.display !== "none" && $("saveAlsoOutputs").checked;
    const btn = $("saveConfirmBtn");
    btn.disabled = true;
    btn.textContent = "Saving...";
    showSaveError("");
    try {
      const saved = await saveCodeToServer(filename, state.saveTarget.code, dir, alsoOutputs);
      if (saved.dir) localStorage.setItem("nib_last_save_dir", saved.dir);
      const target = state.saveTarget;
      closeSaveModal();
      markSaveResult(target, saved);
      if (target.openAfter && saved.url) window.open(saved.url, "_blank", "noopener");
    } catch (err) {
      showSaveError(err.message || "Failed to save file.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Save";
    }
  }

  $("savePresets").addEventListener("click", (e) => {
    const chip = e.target.closest(".save-chip");
    if (!chip) return;
    $("saveDirInput").value = chip.dataset.path;
    syncSaveModalUi();
  });
  $("saveDirInput").addEventListener("input", syncSaveModalUi);
  $("saveCloseBtn").addEventListener("click", closeSaveModal);
  $("saveCancelBtn").addEventListener("click", closeSaveModal);
  $("saveConfirmBtn").addEventListener("click", confirmSaveModal);
  $("saveModal").addEventListener("click", (e) => {
    if (e.target === $("saveModal")) closeSaveModal();
  });
  [$("saveFilenameInput"), $("saveDirInput")].forEach((inp) => {
    inp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); confirmSaveModal(); }
    });
  });
  $("savePickerBtn").addEventListener("click", async () => {
    if (!state.saveTarget || typeof window.showSaveFilePicker !== "function") return;
    const suggestedName = $("saveFilenameInput").value.trim() || state.saveTarget.filename || "output.txt";
    try {
      const handle = await window.showSaveFilePicker({ suggestedName });
      const writable = await handle.createWritable();
      await writable.write(state.saveTarget.code);
      await writable.close();
      const target = state.saveTarget;
      closeSaveModal();
      markSaveResult(target, { path: handle.name, url: "" });
    } catch (err) {
      if (err && err.name !== "AbortError") showSaveError(err.message || "OS Save dialog failed.");
    }
  });

  function openPreviewModal(code, lang) {
    const fname = guessFilename(lang, code);
    state.preview = { code, lang, filename: fname };
    $("previewTitle").textContent = fname;
    const frame = $("previewFrame");
    const isSvg = (lang || "").toLowerCase() === "svg" || code.trimStart().toLowerCase().startsWith("<svg");
    frame.srcdoc = isSvg
      ? `<!doctype html><html><body style="margin:0;display:flex;align-items:center;justify-content:center;min-height:100vh;background:#111;">${code}</body></html>`
      : code;
    $("previewModal").classList.add("open");
  }

  function closePreviewModal() {
    $("previewModal").classList.remove("open");
    $("previewFrame").srcdoc = "";
    state.preview = null;
  }

  $("previewCloseBtn").addEventListener("click", closePreviewModal);
  $("previewModal").addEventListener("click", (e) => {
    if (e.target === $("previewModal")) closePreviewModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if ($("saveModal").classList.contains("open")) { closeSaveModal(); return; }
    if ($("previewModal").classList.contains("open")) closePreviewModal();
  });
  $("previewDownloadBtn").addEventListener("click", () => {
    if (!state.preview) return;
    downloadTextFile(state.preview.filename, state.preview.code);
  });
  $("previewSaveOpenBtn").addEventListener("click", () => {
    if (!state.preview) return;
    openSaveModal({
      code: state.preview.code,
      lang: state.preview.lang,
      filename: state.preview.filename,
      triggerBtn: $("previewSaveOpenBtn"),
      openAfter: true
    });
  });

  function renderMarkdown(md) {
    if (!md) return "";
    const parts = md.split(/(```[\s\S]*?```)/g);
    let html = "";
    for (const part of parts) {
      if (part.startsWith("```")) {
        const inner = part.slice(3, part.endsWith("```") ? -3 : undefined);
        const nl = inner.indexOf("\n");
        const lang = nl >= 0 ? inner.slice(0, nl).trim() : "";
        const code = nl >= 0 ? inner.slice(nl + 1) : inner;
        const canPreview = isPreviewableHtml(lang, code);
        const previewBtn = canPreview ? `<button class="code-btn code-preview" type="button">Preview</button>` : "";
        html += `<div class="code-block" data-lang="${escHtml(lang)}"><div class="code-head"><span>${escHtml(lang || "code")}</span><div class="code-actions">${previewBtn}<button class="code-btn code-save" type="button" title="Save file to a chosen folder">Save</button><button class="code-btn code-download" type="button">Download</button><button class="code-btn code-copy" type="button">Copy</button></div></div><pre><code>${escHtml(code)}</code></pre></div>`;
      } else {
        const lines = part.split("\n");
        let inUl = false, para = [];
        const flushPara = () => {
          if (para.length) { html += "<p>" + para.map(renderInline).join("<br>") + "</p>"; para = []; }
        };
        const closeUl = () => { if (inUl) { html += "</ul>"; inUl = false; } };
        for (const line of lines) {
          const h = line.match(/^(#{1,3})\s+(.*)$/);
          const li = line.match(/^\s*[-*]\s+(.*)$/);
          if (h) {
            flushPara(); closeUl();
            const lvl = h[1].length;
            html += `<h${lvl}>${renderInline(h[2])}</h${lvl}>`;
          } else if (li) {
            flushPara();
            if (!inUl) { html += "<ul>"; inUl = true; }
            html += `<li>${renderInline(li[1])}</li>`;
          } else if (!line.trim()) {
            flushPara(); closeUl();
          } else {
            closeUl();
            para.push(line);
          }
        }
        flushPara(); closeUl();
      }
    }
    return html;
  }

  $("chatThread").addEventListener("click", (e) => {
    const block = e.target.closest(".code-block");
    if (!block) return;
    const code = block.querySelector("pre").textContent;
    const lang = block.dataset.lang || "";

    const copyBtn = e.target.closest(".code-copy");
    if (copyBtn) {
      navigator.clipboard.writeText(code).then(() => {
        copyBtn.textContent = "Copied";
        setTimeout(() => { copyBtn.textContent = "Copy"; }, 1400);
      });
      return;
    }
    const dlBtn = e.target.closest(".code-download");
    if (dlBtn) {
      downloadTextFile(guessFilename(lang, code), code);
      return;
    }
    const prevBtn = e.target.closest(".code-preview");
    if (prevBtn) {
      openPreviewModal(code, lang);
      return;
    }
    const saveBtn = e.target.closest(".code-save");
    if (saveBtn) {
      openSaveModal({
        code,
        lang,
        filename: guessFilename(lang, code),
        triggerBtn: saveBtn,
        openAfter: false
      });
    }
  });

  /* ---- Attachments: Images & Text/Code Files (Pick, Paste, Drag & Drop) ---- */
  function addFiles(files) {
    for (const file of files) {
      if (file.type && file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = () => {
          state.attachments.push({ kind: "image", dataUrl: reader.result, name: file.name || "image.png" });
          renderAttachments();
        };
        reader.readAsDataURL(file);
      } else if (file.size <= 2 * 1024 * 1024) {
        const reader = new FileReader();
        reader.onload = () => {
          state.attachments.push({ kind: "text", name: file.name || "file.txt", content: reader.result || "", size: file.size });
          renderAttachments();
        };
        reader.readAsText(file);
      }
    }
  }
  function renderAttachments() {
    const wrap = $("attList");
    wrap.classList.toggle("has-items", state.attachments.length > 0);
    wrap.innerHTML = state.attachments.map((a, i) => {
      if (a.kind === "text") {
        return `
          <div class="att-file" title="${escHtml(a.name)}">
            <span class="att-file-name">${escHtml(a.name)} (${fmtBytes(a.size || a.content.length)})</span>
            <button class="att-remove" data-i="${i}" title="Remove">&times;</button>
          </div>`;
      }
      return `
        <div class="att-thumb">
          <img src="${a.dataUrl}" alt="${escHtml(a.name)}" />
          <button class="att-remove" data-i="${i}" title="Remove">&times;</button>
        </div>`;
    }).join("");
  }
  $("attList").addEventListener("click", (e) => {
    const btn = e.target.closest(".att-remove");
    if (!btn) return;
    state.attachments.splice(Number(btn.dataset.i), 1);
    renderAttachments();
  });
  $("attachBtn").addEventListener("click", () => $("imgFileInput").click());
  $("imgFileInput").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
  $("chatInput").addEventListener("paste", (e) => {
    const items = (e.clipboardData && e.clipboardData.items) || [];
    const imgs = [];
    for (const it of items) { if (it.type.startsWith("image/")) imgs.push(it.getAsFile()); }
    if (imgs.length) addFiles(imgs);
  });
  $("dropZone").addEventListener("dragover", (e) => { e.preventDefault(); });
  $("dropZone").addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
  });
  $("expandInputBtn").addEventListener("click", () => $("composerBox").classList.toggle("expanded"));

  /* ---- Chat Rendering & Streaming ---- */
  function renderChat(forceScroll = false) {
    const sc = $("chatScroll");
    const nearBottom = forceScroll || (sc.scrollHeight - sc.scrollTop - sc.clientHeight < 120);
    const thread = $("chatThread");
    $("chatEmpty").style.display = state.messages.length ? "none" : "flex";
    thread.innerHTML = "";
    for (const m of state.messages) {
      const row = document.createElement("div");
      row.className = "msg-row " + m.role;
      if (m.role === "user") {
        let imgHtml = "";
        if (m.images && m.images.length) {
          imgHtml = `<div class="msg-user-imgs">${m.images.map((u) => `<img src="${u}" />`).join("")}</div>`;
        }
        let fileHtml = "";
        if (m.files && m.files.length) {
          fileHtml = `<div class="msg-user-files">${m.files.map((f) => `<span class="msg-user-file-chip">${escHtml(f.name)} (${fmtBytes(f.content.length)})</span>`).join("")}</div>`;
        }
        const shownText = m.displayContent !== undefined ? m.displayContent : (m.content || "");
        row.innerHTML = `${imgHtml}${fileHtml}${shownText ? `<div class="msg-user-bubble">${escHtml(shownText)}</div>` : ""}`;
      } else {
        let thinkHtml = "";
        if (m.reasoning) {
          thinkHtml = `<details class="think-box"${m.streaming && !m.content ? " open" : ""}><summary>Thought process</summary><pre>${escHtml(m.reasoning)}</pre></details>`;
        }
        const bodyHtml = m.content ? renderMarkdown(m.content) : (m.streaming ? '<span style="color:var(--muted-foreground);">Thinking...</span>' : "");
        let metaHtml = "";
        if (m.timings) {
          const t = m.timings;
          const bits = [];
          if (t.predicted_per_second) bits.push(`${t.predicted_per_second.toFixed(1)} tok/s`);
          if (t.mtp_used && t.mtp_tok_per_round) bits.push(`MTP ${t.mtp_tok_per_round.toFixed(2)} tok/rnd`);
          if (t.prompt_ms) bits.push(`TTFT ${Math.round(t.prompt_ms)} ms`);
          if (t.cache_hit_tokens) bits.push(`${t.cache_hit_tokens} cached`);
          if (m.completionTokens) bits.push(`${m.completionTokens} tok`);
          if (bits.length) metaHtml = `<div class="msg-meta">${bits.join(" · ")}</div>`;
        }
        row.innerHTML = `${thinkHtml}<div class="msg-assistant-body">${bodyHtml}</div>${metaHtml}`;
      }
      thread.appendChild(row);
    }
    if (nearBottom) sc.scrollTop = sc.scrollHeight;
  }

  function scheduleRenderChat() {
    if (state.renderRaf) return;
    state.renderRaf = requestAnimationFrame(() => {
      state.renderRaf = 0;
      renderChat(false);
    });
  }

  function setGenerating(on) {
    state.generating = on;
    const btn = $("sendBtn");
    btn.textContent = on ? "Stop" : "Send";
    btn.classList.toggle("btn-danger", on);
    btn.classList.toggle("btn-primary", !on);
  }

  async function sendChat() {
    if (state.generating) {
      if (state.abortCtrl) state.abortCtrl.abort();
      return;
    }
    const text = $("chatInput").value.trim();
    if (!text && state.attachments.length === 0) return;

    const imgs = state.attachments.filter((a) => a.kind !== "text").map((a) => a.dataUrl);
    const textFiles = state.attachments.filter((a) => a.kind === "text");
    let fullContent = text;
    if (textFiles.length) {
      const blocks = textFiles.map((f) => {
        const ext = (f.name.split(".").pop() || "").toLowerCase();
        return `File: ${f.name}\n\`\`\`${ext}\n${f.content}\n\`\`\``;
      }).join("\n\n");
      fullContent = text ? `${blocks}\n\n${text}` : blocks;
    }
    state.messages.push({ role: "user", content: fullContent, displayContent: text, images: imgs, files: textFiles });
    const asst = { role: "assistant", content: "", reasoning: "", streaming: true, timings: null, completionTokens: 0 };
    state.messages.push(asst);
    $("chatInput").value = "";
    state.attachments = [];
    renderAttachments();
    renderChat(true);

    const apiMsgs = [];
    const sys = $("sysPrompt").value.trim();
    if (sys) apiMsgs.push({ role: "system", content: sys });
    for (let i = 0; i < state.messages.length - 1; i++) {
      const m = state.messages[i];
      if (m.role === "user" && m.images && m.images.length) {
        const parts = [];
        for (const u of m.images) parts.push({ type: "image_url", image_url: { url: u } });
        if (m.content) parts.push({ type: "text", text: m.content });
        apiMsgs.push({ role: "user", content: parts });
      } else {
        apiMsgs.push({ role: m.role, content: m.content });
      }
    }

    const useStream = $("streamToggle").checked;
    const payload = {
      model: $("modelSelect").value || "inkling-2-mini",
      messages: apiMsgs,
      temperature: Number($("tempRange").value),
      top_p: Number($("topPRange").value),
      max_tokens: Number($("maxTokRange").value),
      thinking_effort: Number($("thinkRange").value),
      mtp: $("mtpToggle").checked,
      stream: useStream
    };

    state.abortCtrl = new AbortController();
    setGenerating(true);
    try {
      const resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: state.abortCtrl.signal
      });
      if (!resp.ok) {
        const errTxt = await resp.text();
        asst.content = "Error " + resp.status + ": " + errTxt;
      } else if (!useStream) {
        const data = await resp.json();
        const ch = data.choices && data.choices[0] && data.choices[0].message;
        asst.content = (ch && ch.content) || "";
        asst.reasoning = (ch && ch.reasoning_content) || "";
        asst.timings = data.timings || null;
        asst.completionTokens = (data.usage && data.usage.completion_tokens) || 0;
      } else {
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() || "";
          for (const raw of lines) {
            const line = raw.trim();
            if (!line.startsWith("data:")) continue;
            const d = line.slice(5).trim();
            if (!d || d === "[DONE]") continue;
            try {
              const obj = JSON.parse(d);
              const delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
              if (delta) {
                if (delta.reasoning_content) asst.reasoning += delta.reasoning_content;
                if (delta.content) asst.content += delta.content;
              }
              if (obj.timings) asst.timings = obj.timings;
              if (obj.usage && obj.usage.completion_tokens) asst.completionTokens = obj.usage.completion_tokens;
              scheduleRenderChat();
            } catch (_) {}
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") asst.content = (asst.content ? asst.content + "\n\n" : "") + "[Error: " + err.message + "]";
    } finally {
      if (state.renderRaf) { cancelAnimationFrame(state.renderRaf); state.renderRaf = 0; }
      asst.streaming = false;
      setGenerating(false);
      state.abortCtrl = null;
      renderChat(false);
      pollAll();
    }
  }

  $("sendBtn").addEventListener("click", sendChat);
  $("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });
  $("newChatBtn").addEventListener("click", () => {
    if (state.generating && state.abortCtrl) state.abortCtrl.abort();
    state.messages = [];
    state.attachments = [];
    renderAttachments();
    renderChat(true);
  });

  /* ---- Load Test Runner ---- */
  $("ltRunBtn").addEventListener("click", async () => {
    const runs = Math.max(1, Math.min(20, Number($("ltRuns").value) || 3));
    const maxTok = Math.max(32, Math.min(131072, Number($("ltMaxTok").value) || 256));
    const prompt = $("ltPrompt").value.trim() || "Write a binary search in C++17.";
    const btn = $("ltRunBtn");
    btn.disabled = true;
    btn.textContent = "Running...";
    const tbody = $("ltTableBody");
    tbody.innerHTML = "";
    try {
      for (let i = 1; i <= runs; i++) {
        await fetch("/api/reset-kv", { method: "POST" });
        const resp = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: $("modelSelect").value || "inkling-2-mini",
            messages: [{ role: "user", content: prompt }],
            max_tokens: maxTok,
            temperature: 0.0,
            stream: false
          })
        });
        const data = await resp.json();
        const t = data.timings || {};
        const u = data.usage || {};
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>#${i}</td>
          <td>${u.prompt_tokens || 0}</td>
          <td>${u.completion_tokens || 0}</td>
          <td>${t.prompt_per_second ? t.prompt_per_second.toFixed(1) + " tok/s" : "-"}</td>
          <td><strong>${t.predicted_per_second ? t.predicted_per_second.toFixed(1) + " tok/s" : "-"}</strong></td>
          <td>${t.mtp_used && t.mtp_tok_per_round ? t.mtp_tok_per_round.toFixed(2) : "-"}</td>
          <td>${t.predicted_ms ? ((t.predicted_ms + (t.prompt_ms || 0)) / 1000).toFixed(2) + " s" : "-"}</td>`;
        tbody.appendChild(tr);
        pollAll();
      }
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="7" class="table-empty">Benchmark failed: ${escHtml(e.message)}</td></tr>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Benchmark";
    }
  });

  /* ---- KV Reset Actions ---- */
  async function resetKv() {
    await fetch("/api/reset-kv", { method: "POST" });
    pollAll();
  }
  $("popoverResetKvBtn").addEventListener("click", resetKv);
  $("modelsResetKvBtn").addEventListener("click", resetKv);
  $("modelRowResetBtn").addEventListener("click", resetKv);
  $("modelOpenPgBtn").addEventListener("click", () => navigate("playground"));

  /* ---- SVG Chart Helpers (clean llama-swap gridline style) ---- */
  function renderSparkline(container, values, unit, emptyText) {
    if (!values.length) {
      container.innerHTML = emptyText;
      return;
    }
    const w = 520, h = 84, padX = 10, padY = 14;
    const min = 0;
    const max = Math.max(...values, 1) * 1.18;
    const pts = values.map((v, i) => {
      const x = values.length === 1 ? w / 2 : padX + (i / (values.length - 1)) * (w - padX * 2);
      const y = h - padY - ((v - min) / (max - min)) * (h - padY * 2);
      return [x, y, v];
    });
    const lineD = pts.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
    const areaD = pts.length > 1
      ? `${lineD} L${pts[pts.length - 1][0].toFixed(1)},${h - 4} L${pts[0][0].toFixed(1)},${h - 4} Z`
      : "";
    const last = values[values.length - 1];
    container.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}">
        ${areaD ? `<path d="${areaD}" fill="var(--chart-1)" fill-opacity="0.11" />` : ""}
        ${pts.length > 1 ? `<path d="${lineD}" fill="none" stroke="var(--chart-1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />` : ""}
        ${pts.map((p) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.2" fill="var(--chart-1)"><title>${p[2].toFixed(1)} ${unit}</title></circle>`).join("")}
      </svg>
      <span class="mono" style="position:absolute;top:0;right:2px;font-size:12px;color:var(--foreground);font-weight:600;">${last.toFixed(1)} ${unit}</span>`;
  }

  function renderGridChart(container, series, yLabel) {
    const hasLegend = series.length > 1;
    const w = 520, h = 205, padL = 44, padR = 14, padT = 12, padB = hasLegend ? 36 : 18;
    const allVals = series.flatMap((s) => s.values);
    const maxRaw = allVals.length ? Math.max(...allVals) : 100;
    const maxY = Math.max(1, Math.ceil(maxRaw * 1.15));
    const steps = 5;
    let grid = "";
    for (let i = 0; i <= steps; i++) {
      const val = (maxY * i) / steps;
      const y = h - padB - (i / steps) * (h - padT - padB);
      grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${w - padR}" y2="${y.toFixed(1)}" stroke="var(--border)" stroke-width="1" />`;
      grid += `<text x="${padL - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" fill="var(--muted-foreground)" font-size="10.5" font-family="var(--font-mono)">${val >= 10 ? Math.round(val) : val.toFixed(1)}</text>`;
    }

    let paths = "";
    for (const s of series) {
      if (!s.values.length) continue;
      const pts = s.values.map((v, i) => {
        const x = s.values.length === 1
          ? (padL + w - padR) / 2
          : padL + (i / (s.values.length - 1)) * (w - padL - padR);
        const y = h - padB - (v / maxY) * (h - padT - padB);
        return [x, y, v];
      });
      const d = pts.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
      if (pts.length > 1) {
        paths += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />`;
      }
      paths += pts.map((p) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.2" fill="${s.color}"><title>${s.name}: ${p[2].toFixed(2)}</title></circle>`).join("");
    }

    let legend = "";
    if (hasLegend) {
      legend = series.map((s, idx) =>
        `<g transform="translate(${padL + idx * 125}, ${h - 8})"><rect width="8" height="8" y="-7" rx="2" fill="${s.color}"/><text x="13" y="0" fill="var(--muted-foreground)" font-size="11">${s.name}</text></g>`
      ).join("");
    }

    const midY = (padT + (h - padB)) / 2;
    container.innerHTML = `<svg viewBox="0 0 ${w} ${h}">
      ${grid}
      ${paths}
      ${legend}
      <text x="12" y="${midY.toFixed(1)}" transform="rotate(-90 12 ${midY.toFixed(1)})" text-anchor="middle" fill="var(--muted-foreground)" font-size="10.5">${yLabel}</text>
    </svg>`;
  }

  /* ---- Render Activity, Performance, Models, Settings, Logs ---- */
  function fmtNum(n) { return Number(n || 0).toLocaleString(); }
  function fmtTime(ts) {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], { hour12: false });
  }
  function fmtUptime(s) {
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
    if (h > 0) return `${h}h ${m}m ${sec}s`;
    if (m > 0) return `${m}m ${sec}s`;
    return `${sec}s`;
  }

  function renderHealth() {
    const h = state.health;
    const ok = h && h.status === "ok";
    const busy = ok && !!h.busy;

    $("brandDot").className = "status-dot" + (!ok ? " offline" : busy ? " busy" : "");
    $("subModelDot").className = "sub-dot" + (!ok ? " stopped" : busy ? " busy" : "");
    $("modelRowDot").className = "status-dot" + (!ok ? " offline" : busy ? " busy" : "");
    $("liveBusyPill").classList.toggle("visible", busy);
    updateDocTitle();

    if (!ok) {
      $("modelRowState").textContent = "OFFLINE";
      $("modelRowState").className = "state-tag offline";
      $("setStatus").textContent = "disconnected";
      return;
    }

    $("subModelName").textContent = h.model;
    const sel = $("modelSelect");
    const optLabel = `${h.model}  (${Math.round(h.max_ctx / 1024)}k ${h.kv_type} sp${h.sparse_pages}${h.mtp ? " · MTP" : ""}${h.vision ? " · Vision" : ""})`;
    if (sel.options[0]) {
      sel.options[0].value = h.model;
      sel.options[0].textContent = optLabel;
    }

    const maxSlots = h.max_slots || 1;
    const slotCtx = h.slot_ctx || h.max_ctx;
    const reqMaxCtx = h.request_max_ctx || h.max_ctx;
    const usedSegs = (h.used_segments !== undefined) ? h.used_segments : (h.active_requests || 0);
    const activeReqs = h.active_requests || 0;
    const queuedReqs = h.queued_requests || 0;
    const inFlight = Array.isArray(h.in_flight) ? h.in_flight : [];
    const elasticMode = maxSlots > 1 && reqMaxCtx > slotCtx;

    $("liveBusyText").textContent = activeReqs > 1
      ? `Generating · ${activeReqs} req (${usedSegs}/${maxSlots} segs)`
      : (queuedReqs > 0 ? `Generating (+${queuedReqs} queued)` : "Generating");

    // Popover & Models KV
    const pct = h.max_ctx > 0 ? Math.min(100, (h.cur_pos / h.max_ctx) * 100) : 0;
    $("popoverKvText").textContent = `KV: ${fmtNum(h.cur_pos)} / ${Math.round(h.max_ctx / 1024)}k`;
    $("modelsKvLabel").textContent = `KV ${fmtNum(h.cur_pos)} / ${fmtNum(h.max_ctx)}`;
    $("modelsKvBar").style.width = pct.toFixed(1) + "%";
    $("modelsReadyLabel").textContent = busy ? `${usedSegs}/${maxSlots} SEGS BUSY` : `${maxSlots} SLOTS READY`;
    $("modelRowState").textContent = busy ? "BUSY" : "READY";
    $("modelRowState").className = "state-tag" + (busy ? " busy" : "");
    const slotInfo = maxSlots > 1
      ? `${maxSlots}×${Math.round(slotCtx / 1024)}k ${elasticMode ? "elastic" : "strict"} (max ${Math.round(reqMaxCtx / 1024)}k)`
      : `${fmtNum(reqMaxCtx)} ctx`;
    $("modelsEngineSummary").textContent = `RX 7900 XTX · ${slotInfo} · ${h.kv_type} sparse-${h.sparse_pages} · ${h.mtp ? "DSpark MTP (" + Math.round(h.draft_ctx / 1024) + "k ring)" : "MTP off"} · ${h.vision ? "Vision" : "Text-only"}`;
    $("modelRowSub").textContent = `${h.model} · VRAM ${h.vram_gib.toFixed(2)} GiB (LLM ${h.llm_vram_gib.toFixed(2)} GiB, Vision ${h.vision_vram_gib.toFixed(2)} GiB)`;

    // In-flight table (live concurrent slots)
    $("inflightCount").textContent = queuedReqs > 0
      ? `${activeReqs} active (${usedSegs}/${maxSlots} segs) · ${queuedReqs} queued`
      : `${activeReqs} active (${usedSegs}/${maxSlots} segs)`;
    $("inflightBody").innerHTML = inFlight.length
      ? inFlight.map((f) => {
          const sp = f.span || 1;
          const slotTxt = f.slot >= 0
            ? (sp > 1 ? `S${f.slot}..${f.slot + sp - 1}` : `S${f.slot}`)
            : "Q";
          const limTxt = (f.slot >= 0 && f.ctx_limit) ? ` ${Math.round(f.ctx_limit / 1024)}k` : "";
          const slotBadge = `<span class="badge">${slotTxt}${limTxt}</span>`;
          const stLabel = f.waiting_expand ? "Waiting expand"
                        : f.state === "decoding" ? (f.mtp ? "Decoding · MTP" : "Decoding")
                        : f.state === "prefilling" ? "Prefilling" : "Queued";
          const stCls = (f.state === "queued" || f.waiting_expand) ? "badge" : "badge badge-primary";
          return `<tr>
            <td class="mono">${slotBadge}</td>
            <td><span class="${stCls}">${escHtml(stLabel)}</span></td>
            <td class="mono">${escHtml(f.id || "")}</td>
            <td class="mono">${fmtNum(f.prompt_tokens)} tok</td>
            <td class="mono">${fmtNum(f.completion_tokens)} / ${fmtNum(f.max_tokens)}</td>
            <td class="mono">${((f.elapsed_ms || 0) / 1000).toFixed(2)}s</td>
          </tr>`;
        }).join("")
      : `<tr><td colspan="6" class="table-empty">No in-flight requests</td></tr>`;

    // Performance KPIs
    $("perfVram").textContent = `${h.vram_gib.toFixed(2)} GiB`;
    $("perfVramSub").textContent = `LLM ${h.llm_vram_gib.toFixed(2)} + Vision ${h.vision_vram_gib.toFixed(2)} GiB`;
    $("perfKvPos").textContent = fmtNum(h.cur_pos);
    $("perfKvSub").textContent = `of ${fmtNum(h.max_ctx)} (${pct.toFixed(1)}%)`;

    // Settings Info
    $("setStatus").textContent = busy ? `busy (${activeReqs} req, ${usedSegs}/${maxSlots} segs)` : "connected";
    $("setEndpoint").textContent = `${location.origin}/v1`;
    $("setModel").textContent = h.model;
    $("setCtx").textContent = maxSlots > 1
      ? `${fmtNum(reqMaxCtx)} max · ${maxSlots} × ${fmtNum(slotCtx)} (${elasticMode ? "elastic multi-segment" : "strict per-slot"})`
      : `${fmtNum(reqMaxCtx)} tokens`;
    $("setKv").textContent = `${h.kv_type} (sparse ${h.sparse_pages} pages)`;
    $("setMtp").textContent = h.mtp ? `enabled (ctx ${fmtNum(h.draft_ctx)}, thr ${h.spec_threshold.toFixed(2)})` : "disabled";
    $("setVision").textContent = h.vision ? "enabled (mmproj Q8_0)" : "disabled";
    $("setVram").textContent = `${h.vram_gib.toFixed(2)} GiB`;
    $("setUptime").textContent = fmtUptime(h.uptime_s || 0);
  }

  function renderActivityAndPerf() {
    const items = state.activity || [];
    const totalCached = items.reduce((a, x) => a + (x.cached_tokens || 0), 0);
    const totalPrompt = items.reduce((a, x) => a + (x.prompt_tokens || 0), 0);
    const totalGen = items.reduce((a, x) => a + (x.completion_tokens || 0), 0);

    $("kpiReqs").textContent = fmtNum(items.length);
    $("kpiCached").textContent = fmtNum(totalCached);
    $("kpiProcessed").textContent = fmtNum(totalPrompt);
    $("kpiGenerated").textContent = fmtNum(totalGen);

    const promptSpeeds = items.map((x) => x.prompt_tps).filter((v) => v > 0);
    const decodeSpeeds = items.map((x) => x.decode_tps).filter((v) => v > 0);
    const mtpVals = items.map((x) => x.mtp_tok_per_round).filter((v) => v > 0);
    const draftMs = items.map((x) => x.mtp_draft_ms).filter((v) => v > 0);
    const verifyMs = items.map((x) => x.mtp_verify_ms).filter((v) => v > 0);

    renderSparkline($("actPromptChart"), promptSpeeds.slice(-30), "tok/s", "No prompt speed data yet");
    renderSparkline($("actGenChart"), decodeSpeeds.slice(-30), "tok/s", "No generation speed data yet");

    const limit = Number($("actRowsSelect").value) || 25;
    const rows = items.slice(-limit).reverse();
    $("activityTableBody").innerHTML = rows.length
      ? rows.map((r) => `
        <tr>
          <td class="mono">#${r.id}</td>
          <td class="mono">${fmtTime(r.timestamp)}</td>
          <td class="mono">${escHtml(r.model)}</td>
          <td><span class="badge badge-ok">${escHtml(r.finish_reason || "stop")}</span></td>
          <td>${fmtNum(r.cached_tokens)}</td>
          <td>${fmtNum(r.prompt_tokens)}</td>
          <td>${fmtNum(r.completion_tokens)}</td>
          <td>${r.prompt_tps ? r.prompt_tps.toFixed(1) + " t/s" : "-"}</td>
          <td><strong>${r.decode_tps ? r.decode_tps.toFixed(1) + " t/s" : "-"}</strong></td>
          <td>${r.mtp && r.mtp_tok_per_round ? `<span class="badge badge-primary">${r.mtp_tok_per_round.toFixed(2)}x</span>` : "-"}</td>
          <td>${((r.prompt_ms + r.decode_ms) / 1000).toFixed(2)}s</td>
        </tr>`).join("")
      : `<tr><td colspan="11" class="table-empty">No activity recorded</td></tr>`;

    // Performance averages & charts
    const avgDecode = decodeSpeeds.length ? (decodeSpeeds.reduce((a, b) => a + b, 0) / decodeSpeeds.length) : 0;
    const avgMtp = mtpVals.length ? (mtpVals.reduce((a, b) => a + b, 0) / mtpVals.length) : 0;
    $("perfAvgDecode").textContent = avgDecode ? avgDecode.toFixed(1) : "--";
    $("perfAvgMtp").textContent = avgMtp ? avgMtp.toFixed(2) : "--";

    renderGridChart($("perfDecodeChart"), [{ name: "Decode", color: "var(--chart-1)", values: decodeSpeeds.slice(-30) }], "tok/s");
    renderGridChart($("perfMtpChart"), [{ name: "MTP", color: "var(--chart-2)", values: mtpVals.slice(-30) }], "tok/round");
    renderGridChart($("perfPrefillChart"), [{ name: "Prefill", color: "var(--chart-3)", values: promptSpeeds.slice(-30) }], "tok/s");
    renderGridChart($("perfLatChart"), [
      { name: "Verify (ms)", color: "var(--chart-1)", values: verifyMs.slice(-30) },
      { name: "Draft (ms)", color: "var(--chart-2)", values: draftMs.slice(-30) }
    ], "ms");
  }

  function renderLogs() {
    const pre = $("logsPre");
    const q = ($("logsSearch").value || "").toLowerCase();
    const filtered = (state.logs || []).filter((l) => {
      if (state.logTab !== "ALL" && l.level !== state.logTab) return false;
      if (q && !(l.msg.toLowerCase().includes(q) || l.level.toLowerCase().includes(q))) return false;
      return true;
    });
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
    pre.innerHTML = filtered.length
      ? filtered.map((l) => `<div><span class="log-ts">${escHtml(l.ts)}</span><span class="log-lvl-${escHtml(l.level)}">[${escHtml(l.level)}]</span><span>${escHtml(l.msg)}</span></div>`).join("")
      : `<div style="color:var(--muted-foreground);">No log entries.</div>`;
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }

  /* ---- Log & Activity Toolbar Wiring ---- */
  document.querySelectorAll(".underline-tab[data-logtab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.logTab = tab.dataset.logtab;
      document.querySelectorAll(".underline-tab[data-logtab]").forEach((t) => t.classList.toggle("active", t === tab));
      renderLogs();
    });
  });
  $("logsWrapBtn").addEventListener("click", () => {
    state.logWrap = !state.logWrap;
    $("logsPre").classList.toggle("wrap", state.logWrap);
  });
  $("logsSearchBtn").addEventListener("click", () => {
    const inp = $("logsSearch");
    inp.classList.toggle("open");
    if (inp.classList.contains("open")) inp.focus();
  });
  $("logsSearch").addEventListener("input", renderLogs);
  $("logsClearBtn").addEventListener("click", async () => {
    await fetch("/api/clear-logs", { method: "POST" });
    state.logs = [];
    renderLogs();
  });
  $("clearActivityBtn").addEventListener("click", async () => {
    await fetch("/api/clear-activity", { method: "POST" });
    state.activity = [];
    renderActivityAndPerf();
  });
  $("actRowsSelect").addEventListener("change", renderActivityAndPerf);

  document.querySelectorAll("#perfRefreshGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#perfRefreshGroup .seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
      state.pollMs = Number(btn.dataset.ms) || 0;
      restartPoll();
    });
  });
  $("perfRefreshNow").addEventListener("click", pollAll);

  /* ---- Live Polling ---- */
  async function pollAll() {
    try {
      const [hr, ar, lr] = await Promise.all([
        fetch("/health"),
        fetch("/api/activity"),
        fetch("/api/logs")
      ]);
      if (hr.ok) state.health = await hr.json();
      if (ar.ok) { const a = await ar.json(); state.activity = a.items || []; }
      if (lr.ok) { const l = await lr.json(); state.logs = l.lines || []; }
      renderHealth();
      renderActivityAndPerf();
      renderLogs();
    } catch (_) {
      state.health = null;
      renderHealth();
    }
  }

  function restartPoll() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    if (state.pollMs > 0) state.pollTimer = setInterval(pollAll, state.pollMs);
  }

  /* ---- Boot ---- */
  applyTheme();
  applySidebar();
  navigate(parseInitialRoute(), false);
  pollAll();
  refreshOutputs();
  restartPoll();
})();
</script>
</body>
</html>
)HTML";

} // namespace inkling
