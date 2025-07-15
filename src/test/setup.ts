// Test setup file for Jest
import dotenv from 'dotenv';

// Load test environment variables
dotenv.config();

// Set test environment
process.env.NODE_ENV = 'test';

// Mock console methods in tests if needed
global.console = {
  ...console,
  // Suppress logs in tests by default
  log: jest.fn(),
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

// Add any global test utilities here
beforeEach(() => {
  // Reset all mocks before each test
  jest.clearAllMocks();
});

afterEach(() => {
  // Clean up after each test
  jest.restoreAllMocks();
});

// Increase timeout for async operations
jest.setTimeout(10000); 