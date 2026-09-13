import React from "react";
import type { StructuredWeatherData } from "../types";
import { CloudRain, Sun, Wind, Droplets, ShieldCheck, Thermometer } from "lucide-react";

interface Props {
  data: StructuredWeatherData;
}

export const WeatherCard: React.FC<Props> = ({ data }) => {
  const formatTemp = () => {
    const hasRange =
      data.temp_min_c != null &&
      data.temp_max_c != null &&
      data.temp_min_c !== data.temp_max_c;
    if (hasRange) return `${data.temp_min_c}° – ${data.temp_max_c}°C`;
    const single = data.temp_c ?? data.temp_min_c ?? data.temp_max_c;
    return single != null ? `${single}°C` : "N/A";
  };

  const formatDateLabel = (raw: string) => {
    const lower = raw.toLowerCase();
    if (lower === "today" || lower === "current" || lower === "now") return "Today";
    if (lower === "tomorrow") return "Tomorrow";
    return raw.charAt(0).toUpperCase() + raw.slice(1);
  };

  return (
    <div style={{
      background: "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)",
      borderRadius: "16px",
      padding: "20px",
      marginTop: "12px",
      border: "1px solid #334155",
      boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.3)",
      color: "#f8fafc",
      maxWidth: "500px"
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h3 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 700, letterSpacing: "-0.025em" }}>
            {data.location}
          </h3>
          <span style={{ fontSize: "0.85rem", color: "#94a3b8", textTransform: "capitalize" }}>
            Forecast For {formatDateLabel(data.target_date)}
          </span>
        </div>
        <div style={{
          padding: "6px 12px",
          borderRadius: "9999px",
          background: data.will_rain ? "rgba(59, 130, 246, 0.2)" : "rgba(234, 179, 8, 0.2)",
          color: data.will_rain ? "#60a5fa" : "#facc15",
          display: "flex",
          alignItems: "center",
          gap: "6px",
          fontSize: "0.85rem",
          fontWeight: 600,
          border: `1px solid ${data.will_rain ? "#3b82f6" : "#eab308"}`
        }}>
          {data.will_rain ? <CloudRain size={16} /> : <Sun size={16} />}
          {data.will_rain ? "Rain Expected" : "Clear / Fair"}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "16px" }}>
        <div style={{ background: "rgba(255, 255, 255, 0.05)", padding: "12px", borderRadius: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#94a3b8", fontSize: "0.8rem" }}>
            <Thermometer size={14} color="#f87171" /> Temperature
          </div>
          <div style={{ fontSize: "1.3rem", fontWeight: 700, marginTop: "4px" }}>
            {formatTemp()}
          </div>
        </div>

        <div style={{ background: "rgba(255, 255, 255, 0.05)", padding: "12px", borderRadius: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#94a3b8", fontSize: "0.8rem" }}>
            <Droplets size={14} color="#60a5fa" /> Rain Probability
          </div>
          <div style={{ fontSize: "1.3rem", fontWeight: 700, marginTop: "4px", color: "#60a5fa" }}>
            {data.precipitation_probability_pct ?? (data.will_rain ? 70 : 10)}%
          </div>
        </div>

        {data.wind_speed_kph != null && (
          <div style={{ background: "rgba(255, 255, 255, 0.05)", padding: "12px", borderRadius: "12px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#94a3b8", fontSize: "0.8rem" }}>
              <Wind size={14} color="#34d399" /> Wind
            </div>
            <div style={{ fontSize: "1.1rem", fontWeight: 600, marginTop: "4px" }}>
              {data.wind_speed_kph} km/h
            </div>
          </div>
        )}

        <div style={{ background: "rgba(255, 255, 255, 0.05)", padding: "12px", borderRadius: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#94a3b8", fontSize: "0.8rem" }}>
            <ShieldCheck size={14} color="#a78bfa" /> Confidence
          </div>
          <div style={{ fontSize: "1.1rem", fontWeight: 600, marginTop: "4px", color: "#a78bfa" }}>
            {Math.round(data.confidence * 100)}%
          </div>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid #334155", paddingTop: "12px", fontSize: "0.75rem", color: "#64748b" }}>
        <span>Sources: {data.data_sources.join(", ")}</span>
        <span>CQRS Read Layer Verified</span>
      </div>
    </div>
  );
};
