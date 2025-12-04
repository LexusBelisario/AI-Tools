export const FONT_FAMILY = "Plus Jakarta Sans, system-ui, sans-serif";

// Common colors - Gold themed!
export const COLORS = {
  primary: "#f59e0b", // amber-500 (gold)
  primaryDark: "#92400e", // amber-900 (dark brown outline)
  secondary: "#fbbf24", // amber-400 (lighter gold)
  secondaryDark: "#78350f", // amber-950 (very dark outline)
  accent: "#fcd34d", // amber-300 (bright gold)
  accentDark: "#451a03", // dark brown for accents
  danger: "#ef4444", // red for errors
  dangerDark: "#7f1d1d", // dark red outline
  gray: "#334155", // slate-700 (for grid lines)
  gridColor: "#1e293b", // slate-800 (darker grid)
  textLight: "#cbd5e1", // light text
  textBase: "#e2e8f0", // base text
};

// Base layout for all plots
export const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {
    color: COLORS.textBase,
    family: FONT_FAMILY,
    size: 12,
  },
  hovermode: "closest",
};

// Common axis style
export const AXIS_STYLE = {
  gridcolor: COLORS.gridColor,
  tickfont: {
    size: 11,
    family: FONT_FAMILY,
  },
};

// Axis title style
export const getAxisTitle = (text, color = COLORS.textLight) => ({
  text,
  font: {
    size: 14,
    color,
    family: FONT_FAMILY,
  },
});

// Legend style
export const LEGEND_STYLE = {
  x: 0.02,
  y: 0.98,
  bgcolor: "rgba(15, 23, 42, 0.8)",
  bordercolor: COLORS.gray,
  borderwidth: 1,
  font: {
    size: 11,
    family: FONT_FAMILY,
  },
};

// Plot config (hides toolbar)
export const PLOT_CONFIG = {
  displayModeBar: false,
  displaylogo: false,
};

// ========================================
// 🎨 MARKER STYLES
// ========================================

export const SCATTER_MARKER = {
  predictions: {
    color: COLORS.primary, // gold circles
    size: 8,
    line: {
      color: COLORS.primaryDark, // dark brown outline
      width: 1.5,
    },
  },
  residuals: {
    color: COLORS.secondary, // lighter gold circles
    size: 7,
    line: {
      color: COLORS.secondaryDark, // very dark outline
      width: 1.5,
    },
  },
};

export const BAR_MARKER = {
  residual: {
    color: COLORS.secondary, // gold bars
    line: {
      color: COLORS.secondaryDark, // dark outline
      width: 1,
    },
  },
  distribution: {
    color: COLORS.accent, // bright gold bars
    line: {
      color: COLORS.accentDark, // dark brown outline
      width: 1,
    },
  },
};

// ========================================
// 📈 COMPLETE PLOT LAYOUTS
// ========================================

export const SCATTER_LAYOUT = {
  ...BASE_LAYOUT,
  height: 400,
  margin: { t: 40, l: 60, r: 20, b: 60 },
  xaxis: AXIS_STYLE,
  yaxis: AXIS_STYLE,
  legend: LEGEND_STYLE,
};

export const BAR_LAYOUT = {
  ...BASE_LAYOUT,
  height: 350,
  margin: { t: 40, l: 60, r: 20, b: 60 },
  xaxis: AXIS_STYLE,
  yaxis: AXIS_STYLE,
  showlegend: false,
  bargap: 0.1,
};

export const DISTRIBUTION_LAYOUT = {
  ...BASE_LAYOUT,
  height: 300,
  margin: { t: 40, l: 60, r: 20, b: 60 },
  xaxis: AXIS_STYLE,
  yaxis: AXIS_STYLE,
  showlegend: false,
  bargap: 0.1,
};

// ========================================
// 🎯 HELPER FUNCTIONS
// ========================================

export const getDashedLine = (
  xMin,
  xMax,
  yMin,
  yMax,
  name = "Reference Line"
) => ({
  x: [xMin, xMax],
  y: [yMin, yMax],
  mode: "lines",
  name,
  line: {
    dash: "dash",
    color: COLORS.gray, // gray dashed line
    width: 2,
  },
});

export const getFeatureImportanceMarker = (values) => ({
  color: values,
  colorscale: [
    [0, "#fef3c7"], // amber-100 (lightest gold)
    [0.5, "#fbbf24"], // amber-400 (medium gold)
    [1, "#f59e0b"], // amber-500 (rich gold)
  ],
  line: {
    color: COLORS.accentDark, // dark brown outline
    width: 1,
  },
});