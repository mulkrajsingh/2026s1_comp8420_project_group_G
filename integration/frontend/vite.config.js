import { defineConfig } from 'vite'

// Dev server on :5173; proxy /api to the FastAPI backend so the browser stays same-origin.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true }
    }
  }
})
