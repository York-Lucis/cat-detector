import { Router } from 'express';
import { catRoutes } from './catRoutes';

const router = Router();

// Version endpoint
router.get('/', (_req, res) => {
  res.json({
    message: 'Cat Detector API',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      cats: '/api/cats',
    },
  });
});

// Cat detection routes
router.use('/cats', catRoutes);

export { router as apiRoutes }; 