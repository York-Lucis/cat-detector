// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    message: string;
    stack?: string;
  };
}

// Cat Detection Types
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CatDetection {
  id: string;
  confidence: number;
  boundingBox: BoundingBox;
  breed?: string;
}

export interface DetectionResult {
  detected: boolean;
  cats: CatDetection[];
  processingTime?: number;
}

export interface DetectionRequest {
  imageUrl?: string;
  imageBase64?: string;
}

// Environment Variables
export interface EnvConfig {
  PORT: string;
  NODE_ENV: string;
  CORS_ORIGIN: string;
  JWT_SECRET: string;
  LOG_LEVEL: string;
}

// Log Entry Types
export type LogLevel = 'info' | 'warn' | 'error' | 'debug';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  data?: any;
}

// Express Request Extensions
declare global {
  namespace Express {
    interface Request {
      user?: any;
      startTime?: number;
    }
  }
}

// Health Check Response
export interface HealthResponse {
  status: string;
  timestamp: string;
  uptime: number;
  environment: string;
} 