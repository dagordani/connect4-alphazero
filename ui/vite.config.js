import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The engine runs in the browser (src/engine/worker.js). Vite bundles onnxruntime-web's
// WASM runtime as an asset, so the site needs no CDN and no server.
export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
