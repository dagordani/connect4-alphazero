import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In development the API lives on the Node server (server/), so proxy /api there.
// Override with API_TARGET if the server runs on another port.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': process.env.API_TARGET ?? 'http://127.0.0.1:3001',
    },
  },
});
