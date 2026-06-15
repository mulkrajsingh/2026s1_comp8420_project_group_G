import { defineConfig } from 'vite'

// Lightweight dev server. Proxies /api/* to the local FastAPI backend so the
// browser talks to the UI origin only (no CORS setup needed).
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true }
    }
  }
})
