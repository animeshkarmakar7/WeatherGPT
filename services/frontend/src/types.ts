export interface StructuredWeatherData {
  location: string;
  target_date: string;
  summary: string;
  will_rain: boolean;
  precipitation_probability_pct?: number | null;
  temp_c?: number | null;
  temp_min_c?: number | null;
  temp_max_c?: number | null;
  humidity_pct?: number | null;
  wind_speed_kph?: number | null;
  conditions: string;
  confidence: number;
  data_sources: string[];
  quality_flags: string[];
  observed_at?: string | null;
  fetched_at?: string | null;
  freshness: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  structured?: StructuredWeatherData;
  timestamp: string;
}
