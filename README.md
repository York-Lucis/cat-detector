# Cat Detector API

A Node.js TypeScript backend for cat detection using Express framework.

## Features

- **TypeScript** - Full TypeScript support with strict type checking
- **Express.js** - Fast, unopinionated web framework
- **Security** - Helmet.js for security headers, CORS configuration
- **Logging** - Structured logging with custom logger
- **Error Handling** - Comprehensive error handling middleware
- **Testing** - Jest testing framework with supertest
- **Code Quality** - ESLint for code linting
- **Development** - Hot reload with ts-node-dev

## Project Structure

```
cat-detector/
├── src/
│   ├── __tests__/          # Test files
│   ├── middleware/         # Express middleware
│   ├── routes/            # API routes
│   ├── types/             # TypeScript type definitions
│   ├── utils/             # Utility functions
│   ├── app.ts             # Express app configuration
│   └── index.ts           # Application entry point
├── dist/                  # Compiled JavaScript (generated)
├── .eslintrc.js          # ESLint configuration
├── .gitignore            # Git ignore rules
├── jest.config.js        # Jest configuration
├── package.json          # Dependencies and scripts
├── tsconfig.json         # TypeScript configuration
└── README.md             # This file
```

## API Endpoints

### Health Check
- `GET /health` - Server health status

### API Info
- `GET /api` - API information and available endpoints

### Cat Detection
- `GET /api/cats` - Get all cats (placeholder)
- `POST /api/cats/detect` - Detect cats in image
- `GET /api/cats/:id` - Get cat detection by ID

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cat-detector
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create environment file:
   ```bash
   # Create .env file with the following content:
   PORT=3000
   NODE_ENV=development
   CORS_ORIGIN=*
   JWT_SECRET=your-secret-key-here-change-in-production
   LOG_LEVEL=debug
   ```

### Development

Start the development server with hot reload:
```bash
npm run dev
```

The server will start on `http://localhost:3000`

### Building for Production

1. Build the TypeScript code:
   ```bash
   npm run build
   ```

2. Start the production server:
   ```bash
   npm start
   ```

### Testing

Run tests:
```bash
npm test
```

Run tests with coverage:
```bash
npm test -- --coverage
```

### Code Quality

Run ESLint:
```bash
npm run lint
```

Fix ESLint issues automatically:
```bash
npm run lint:fix
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `3000` |
| `NODE_ENV` | Environment mode | `development` |
| `CORS_ORIGIN` | CORS origin | `*` |
| `JWT_SECRET` | JWT secret key | Required |
| `LOG_LEVEL` | Logging level | `debug` |

## API Usage Examples

### Health Check
```bash
curl http://localhost:3000/health
```

### Cat Detection
```bash
curl -X POST http://localhost:3000/api/cats/detect \
  -H "Content-Type: application/json" \
  -d '{"imageUrl": "https://example.com/cat.jpg"}'
```

Or with base64 image:
```bash
curl -X POST http://localhost:3000/api/cats/detect \
  -H "Content-Type: application/json" \
  -d '{"imageBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD..."}'
```

## Development Notes

- The cat detection logic is currently mocked and returns sample data
- Database integration is prepared but not implemented
- The project follows TypeScript strict mode for better type safety
- All API responses follow a consistent format with `success` and `data`/`error` fields

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License 