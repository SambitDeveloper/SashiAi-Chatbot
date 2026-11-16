import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

// WSL2 does not deliver file-change events for projects on a Windows drive (/mnt/c/...), so Vite never notices
// edits and the page does not reload. Poll for changes there instead (a small CPU cost, only in that case).
const onWindowsDriveUnderWsl = process.platform === 'linux' && path.resolve('.').startsWith('/mnt/');

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(import.meta.dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch:
        process.env.DISABLE_HMR === 'true'
          ? null
          : onWindowsDriveUnderWsl
            ? { usePolling: true, interval: 300 }
            : {},
      // Forward API calls to the Python backend (uvicorn) during development.
      proxy: {
        '/api': {
          target: process.env.API_PROXY_TARGET || 'http://localhost:8000',
          changeOrigin: true,
        },
      },
    },
  };
});
