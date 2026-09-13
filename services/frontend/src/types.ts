export interface StructuredWeatherData {
  location: string;
  target_date: string;
  summary: string;
  will_rain: boolean;
  precipitation_probability_pct?: number;
  temp_c?: number;
  temp_min_c?: number;
  temp_max_c?: number;
  humidity_pct?: number;
  wind_speed_kph?: number;
  conditions: string;
  confidence: number;
  data_sources: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  structured?: StructuredWeatherData;
  timestamp: string;
}
